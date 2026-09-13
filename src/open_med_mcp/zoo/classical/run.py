"""Classical segmentation adapter (thresholds, Otsu, region growing). CPU only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_sdk"))
import omm_job  # noqa: E402
from scipy import ndimage  # noqa: E402

_JOB: omm_job.Job | None = None


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist).astype(np.float64)
    w2 = w[-1] - w
    m = np.cumsum(hist * centers)
    mean1 = np.divide(m, w, out=np.zeros_like(m), where=w > 0)
    mean2 = np.divide(m[-1] - m, w2, out=np.zeros_like(m), where=w2 > 0)
    var = w * w2 * (mean1 - mean2) ** 2
    return float(centers[int(np.argmax(var))])


def _seed_index(seed: list[float], ndim: int) -> tuple[int, ...]:
    """native (x, y[, z]) -> numpy index (z, y, x)."""
    coords = [int(round(float(c))) for c in seed]
    if len(coords) != ndim:
        raise ValueError(f"seed {seed} must have {ndim} coordinates")
    return tuple(reversed(coords))


def _components(binary: np.ndarray, keep: str, seeds: list[list[float]], min_voxels: int) -> np.ndarray:
    labeled, n = ndimage.label(binary)
    if n == 0:
        return binary.astype(bool)
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
    selected = set(range(1, n + 1))
    if keep == "largest":
        selected = {int(np.argmax(sizes)) + 1}
    elif keep == "seeds":
        selected = set()
        for s in seeds:
            idx = _seed_index(s, binary.ndim)
            lab = int(labeled[idx])
            if lab:
                selected.add(lab)
            elif _JOB is not None:
                _JOB.warn(
                    f"seed {s} (native x,y,z) is not inside the selected intensity range; it selects nothing"
                )
    if min_voxels > 0:
        selected = {i for i in selected if sizes[i - 1] >= min_voxels}
    return np.isin(labeled, sorted(selected))


def run(job: omm_job.Job) -> None:
    global _JOB
    _JOB = job
    arr, geom = omm_job.load_image(job.input_path("image"))
    data = omm_job.scalar(arr, geom).astype(np.float32)
    sigma = float(job.param("smooth_sigma", 0) or 0)
    if sigma > 0:
        data = ndimage.gaussian_filter(data, sigma)
    task = job.task or "threshold"
    keep = str(job.param("keep", "all"))
    seeds = list(job.param("seeds", []) or [])
    min_voxels = int(job.param("min_voxels", 0) or 0)
    job.log(f"task={task} shape={data.shape} range=[{data.min():.1f}, {data.max():.1f}]")

    if task == "threshold":
        lower = job.param("lower", float(data.min()))
        upper = job.param("upper", float(data.max()))
        binary = (data >= float(lower)) & (data <= float(upper))
        mask = _components(binary, keep, seeds, min_voxels).astype(np.uint8)
    elif task == "otsu":
        t = _otsu(data.reshape(-1))
        binary = data <= t if job.param("invert", False) else data > t
        job.log(f"otsu threshold = {t:.3f}")
        mask = _components(binary, keep, seeds, min_voxels).astype(np.uint8)
    elif task == "multi_threshold":
        ranges = job.param("ranges")
        if not ranges:
            raise ValueError("multi_threshold needs `ranges: [[lower, upper, label], ...]`")
        mask = np.zeros(data.shape, dtype=np.uint8)
        for lo, hi, lab in ranges:
            mask[(data >= float(lo)) & (data <= float(hi))] = int(lab)
    elif task == "region_grow":
        if not seeds:
            raise ValueError("region_grow needs `seeds`")
        if job.param("lower") is not None and job.param("upper") is not None:
            lower, upper = float(job.param("lower")), float(job.param("upper"))
        else:
            tol = float(job.param("tolerance", 0) or 0)
            if tol <= 0:
                raise ValueError("region_grow needs lower+upper or tolerance")
            vals = [float(data[_seed_index(s, data.ndim)]) for s in seeds]
            mean = float(np.mean(vals))
            lower, upper = mean - tol, mean + tol
        job.log(f"region_grow bounds [{lower:.2f}, {upper:.2f}] from {len(seeds)} seed(s)")
        binary = (data >= lower) & (data <= upper)
        mask = _components(binary, "seeds", seeds, min_voxels).astype(np.uint8)
    else:
        raise ValueError(f"unknown task {task!r}")

    if job.param("fill_holes", False):
        out = np.zeros_like(mask)
        for lab in np.unique(mask):
            if lab:
                out[omm_job.fill_holes(mask == lab)] = lab
        mask = out
    job.mark("segment")
    suffix = ".png" if geom["is_2d"] and str(job.input_path("image")).lower().endswith(".png") else ".nii.gz"
    out_path = omm_job.save_mask(mask, geom, job.output_path("mask" + suffix))
    labels = {
        int(v): (f"label_{int(v)}" if task == "multi_threshold" else "object") for v in np.unique(mask) if v
    }
    job.finish(
        {"mask": out_path},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={"adapter": "classical", "task": task},
    )


if __name__ == "__main__":
    omm_job.main(run)
