"""SAM 2.1 / MedSAM2 adapter: promptable 2D segmentation and 3D slice propagation.

Prompt wire format (produced by ``open_med_mcp.core.prompts.normalize_prompts``)::

    {"type": "point"|"box", "label": 1|0, "object_id": int, "slice": int,
     "coords": [col, row] | [col0, row0, col1, row1]}   # pixel coords of the 2D slice

``params.axis`` is the NumPy axis of the volume along which slices are taken (0 for axial in a
standard ``[z, y, x]`` array).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402

VARIANTS: dict[str, dict[str, str]] = {
    "sam2.1_hiera_tiny": {"config": "sam2.1_hiera_t.yaml", "file": "sam2.1_hiera_tiny.pt"},
    "sam2.1_hiera_small": {"config": "sam2.1_hiera_s.yaml", "file": "sam2.1_hiera_small.pt"},
    "sam2.1_hiera_base_plus": {"config": "sam2.1_hiera_b+.yaml", "file": "sam2.1_hiera_base_plus.pt"},
    "sam2.1_hiera_large": {"config": "sam2.1_hiera_l.yaml", "file": "sam2.1_hiera_large.pt"},
    "medsam2_latest": {"config": "sam2.1_hiera_t512.yaml", "file": "MedSAM2_latest.pt"},
    "medsam2_ct_lesion": {"config": "sam2.1_hiera_t512.yaml", "file": "MedSAM2_CTLesion.pt"},
    "medsam2_mri_liver_lesion": {"config": "sam2.1_hiera_t512.yaml", "file": "MedSAM2_MRI_LiverLesion.pt"},
    "medsam2_us_heart": {"config": "sam2.1_hiera_t512.yaml", "file": "MedSAM2_US_Heart.pt"},
    "medsam2_2411": {"config": "sam2.1_hiera_t512.yaml", "file": "MedSAM2_2411.pt"},
}


def _init_hydra() -> None:
    """Point Hydra at our bundled config directory (works with the stock ``sam2`` package)."""
    import sam2  # noqa: F401  (registers sam2's own config module first)
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(HERE / "configs"), version_base="1.2")


def _checkpoint(job: omm_job.Job, variant: str) -> Path:
    spec = VARIANTS[variant]
    ckpt = job.weights_dir / spec["file"]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"checkpoint {ckpt} not found. Download it with `open-med-mcp models download sam2 --weights {variant}`"
            " (or mount your weights directory at /weights when using containers)."
        )
    return ckpt


def _group_prompts(prompts: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        groups[int(p.get("object_id", 1))].append(p)
    return dict(sorted(groups.items()))


def _prompt_arrays(
    items: list[dict[str, Any]],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    pts = [p["coords"][:2] for p in items if p["type"] == "point"]
    lbl = [int(p.get("label", 1)) for p in items if p["type"] == "point"]
    boxes = [p["coords"][:4] for p in items if p["type"] == "box"]
    points = np.asarray(pts, dtype=np.float32) if pts else None
    labels = np.asarray(lbl, dtype=np.int32) if lbl else None
    box = np.asarray(boxes[0], dtype=np.float32) if boxes else None
    return points, labels, box


def _to_rgb_uint8(slice2d: np.ndarray, window: Any, modality: str) -> np.ndarray:
    if slice2d.ndim == 3:  # already RGB
        return slice2d[..., :3].astype(np.uint8)
    u8, _ = omm_job.to_uint8(slice2d, window, modality)
    return np.stack([u8, u8, u8], axis=-1)


def _postprocess(mask: np.ndarray, job: omm_job.Job) -> np.ndarray:
    out = mask
    if job.param("largest_component", False) or job.param("fill_holes", False):
        out = np.zeros_like(mask)
        for lab in np.unique(mask):
            if not lab:
                continue
            b = mask == lab
            if job.param("largest_component", False):
                b = omm_job.largest_component(b)
            if job.param("fill_holes", False):
                b = omm_job.fill_holes(b)
            out[b] = lab
    return out


def segment_2d(
    job: omm_job.Job,
    image: np.ndarray,
    geom: dict[str, Any],
    prompts: list[dict[str, Any]],
    device: str,
    variant: str,
) -> np.ndarray:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _init_hydra()
    model = build_sam2(VARIANTS[variant]["config"], str(_checkpoint(job, variant)), device=device)
    predictor = SAM2ImagePredictor(model)
    # SAM2ImagePredictor hardcodes backbone feature sizes for 1024 px inputs; MedSAM2 runs at 512 px.
    size = int(getattr(model, "image_size", 1024))
    predictor._bb_feat_sizes = [(size // 4, size // 4), (size // 8, size // 8), (size // 16, size // 16)]
    modality = omm_job.guess_modality(image, geom)
    rgb = _to_rgb_uint8(image, job.param("window"), modality)
    job.mark("model_loaded")
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    with torch.inference_mode():
        predictor.set_image(rgb)
        for obj_id, items in _group_prompts(prompts).items():
            points, labels, box = _prompt_arrays(items)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=bool(job.param("multimask", False)),
            )
            best = int(np.argmax(scores))
            m = masks[best].astype(bool)
            job.log(f"object {obj_id}: score={float(scores[best]):.3f} voxels={int(m.sum())}")
            mask[m] = obj_id
    return mask


def segment_3d(
    job: omm_job.Job,
    volume: np.ndarray,
    geom: dict[str, Any],
    prompts: list[dict[str, Any]],
    device: str,
    variant: str,
) -> np.ndarray:
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    axis = int(job.param("axis", 0))
    n = volume.shape[axis]
    modality = omm_job.guess_modality(volume, geom)
    window = job.param("window")
    if window is None and modality == "CT":
        window = "soft-tissue"
    propagate = bool(job.param("propagate", True))
    max_slices = int(job.param("max_slices", 0) or 0)
    stop_after_empty = int(job.param("stop_after_empty", 3) or 0)

    groups = _group_prompts(prompts)
    prompt_slices = sorted({int(p["slice"]) for items in groups.values() for p in items})
    if not prompt_slices:
        raise ValueError("no prompts given")
    for s in prompt_slices:
        if not 0 <= s < n:
            raise ValueError(f"prompt slice {s} out of range [0, {n})")

    # Determine the frame window we actually feed to the model (saves time on long stacks)
    if propagate:
        lo = max(0, min(prompt_slices) - max_slices) if max_slices else 0
        hi = min(n, max(prompt_slices) + max_slices + 1) if max_slices else n
    else:
        lo, hi = min(prompt_slices), max(prompt_slices) + 1
    frames_dir = Path(tempfile.mkdtemp(prefix="omm_sam2_frames_"))
    try:
        # window once on the whole volume for consistent contrast across slices
        if volume.ndim == 3:
            u8, wlabel = omm_job.to_uint8(volume, window, modality)
        else:  # RGB volume [z, y, x, 3]
            u8, wlabel = volume[..., :3].astype(np.uint8), "rgb"
        for i, s in enumerate(range(lo, hi)):
            sl = np.take(u8, s, axis=axis)
            rgb = sl if sl.ndim == 3 else np.stack([sl, sl, sl], axis=-1)
            Image.fromarray(np.ascontiguousarray(rgb)).save(frames_dir / f"{i:05d}.jpg", quality=95)
        job.log(f"wrote {hi - lo} frames (slices {lo}..{hi - 1} of {n}, axis={axis}, window={wlabel})")
        job.mark("frames_written")

        _init_hydra()
        predictor = build_sam2_video_predictor(
            VARIANTS[variant]["config"], str(_checkpoint(job, variant)), device=device
        )
        job.mark("model_loaded")
        mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            state = predictor.init_state(
                video_path=str(frames_dir), offload_video_to_cpu=(hi - lo) > 400, async_loading_frames=False
            )
            predictor.reset_state(state)
            for obj_id, items in groups.items():
                by_slice: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for p in items:
                    by_slice[int(p["slice"])].append(p)
                for s, sitems in by_slice.items():
                    points, labels, box = _prompt_arrays(sitems)
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=s - lo,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        box=box,
                        clear_old_points=True,  # SAM 2 requires boxes first; box + points together is allowed,
                    )
            start = min(prompt_slices) - lo

            def collect(reverse: bool) -> None:
                empty_streak = 0
                for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=reverse
                ):
                    any_fg = False
                    for k, obj_id in enumerate(obj_ids):
                        m = (logits[k] > 0.0).squeeze(0).cpu().numpy().astype(bool)
                        if m.any():
                            any_fg = True
                            idx = [slice(None)] * 3
                            idx[axis] = frame_idx + lo
                            view = mask[tuple(idx)]
                            view[m] = int(obj_id)
                            mask[tuple(idx)] = view
                    if propagate and stop_after_empty and frame_idx != start:
                        empty_streak = 0 if any_fg else empty_streak + 1
                        if empty_streak >= stop_after_empty:
                            job.log(
                                f"stopping {'backward' if reverse else 'forward'} propagation at frame {frame_idx + lo}: {empty_streak} empty slices"
                            )
                            break
                    if not propagate:
                        break

            collect(reverse=False)
            if propagate and start > 0:
                collect(reverse=True)
        job.mark("propagated")
        return mask
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def run(job: omm_job.Job) -> None:
    variant = str(job.param("variant", "sam2.1_hiera_small"))
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    prompts = list(job.param("prompts", []) or [])
    if not prompts:
        raise ValueError("SAM 2 needs at least one point or box prompt")
    device = omm_job.resolve_device(job.device_pref)
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    image, geom = omm_job.load_image(job.input_path("image"))
    job.log(f"variant={variant} device={device} shape={image.shape} prompts={len(prompts)}")
    if geom["is_2d"]:
        mask = segment_2d(job, image, geom, prompts, device, variant)
    else:
        mask = segment_3d(job, image, geom, prompts, device, variant)
    mask = _postprocess(mask, job)
    suffix = ".png" if geom["is_2d"] and str(job.input_path("image")).lower().endswith(".png") else ".nii.gz"
    out = omm_job.save_mask(mask, geom, job.output_path("mask" + suffix))
    labels = {int(v): f"object_{int(v)}" for v in np.unique(mask) if v}
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={
            "adapter": "sam2",
            "variant": variant,
            "device": device,
            "config": VARIANTS[variant]["config"],
        },
    )


if __name__ == "__main__":
    omm_job.main(run)
