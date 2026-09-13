"""omm_job: a tiny, dependency-light helper for open-med-mcp model adapters.

An adapter is a ``run.py`` that is executed as ``python run.py --job <dir>`` (locally or inside a
container where the job directory is mounted at ``/job``). It reads ``request.json``, writes its
outputs into ``outputs/`` and finishes by writing ``response.json``.

This module deliberately does **not** import ``open_med_mcp`` so that containers only need
``numpy`` (+ ``SimpleITK``/``Pillow`` for image I/O).

Job contract v1
---------------
request.json::

    {"contract_version": 1, "model": "...", "adapter": "...", "task": "segment",
     "inputs": {"image": "inputs/image.nii.gz"}, "params": {...}, "output_dir": "outputs",
     "resources": {"device": "cuda|cpu", "weights_dir": "..."}}

response.json::

    {"contract_version": 1, "status": "ok", "outputs": {"mask": "outputs/mask.nii.gz"},
     "labels": {"1": "object"}, "stats": {...}, "model_info": {...}, "warnings": [], "timing": {...}}
    {"contract_version": 1, "status": "error", "error": {"type": "...", "message": "...", "traceback": "..."}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = 1

CT_WINDOWS: dict[str, tuple[float, float]] = {
    "soft-tissue": (40.0, 400.0),
    "abdomen": (60.0, 400.0),
    "liver": (60.0, 160.0),
    "lung": (-600.0, 1500.0),
    "bone": (400.0, 1800.0),
    "brain": (40.0, 80.0),
    "subdural": (75.0, 215.0),
    "stroke": (35.0, 40.0),
    "mediastinum": (50.0, 350.0),
    "angio": (300.0, 600.0),
}

_IMAGE2D = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class Job:
    """Access to the request, inputs, outputs and logging of one job directory."""

    def __init__(self, job_dir: str | os.PathLike[str]):
        self.dir = Path(job_dir).resolve()
        self.request: dict[str, Any] = json.loads((self.dir / "request.json").read_text(encoding="utf-8"))
        self.task: str = str(self.request.get("task") or "")
        self.params: dict[str, Any] = dict(self.request.get("params") or {})
        self.inputs: dict[str, str] = dict(self.request.get("inputs") or {})
        self.output_dir = self.dir / (self.request.get("output_dir") or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        res = self.request.get("resources") or {}
        self.weights_dir = Path(
            os.environ.get("OMM_WEIGHTS_DIR") or res.get("weights_dir") or (self.dir / "weights")
        )
        self.device_pref = os.environ.get("OMM_DEVICE") or res.get("device") or "auto"
        self._log_fh = (self.dir / "log.txt").open("a", encoding="utf-8")
        self._t0 = time.time()
        self._timing: dict[str, float] = {}
        self.warnings: list[str] = []

    # ----------------------------------------------------------------- misc
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:8.2f}s] {msg}"
        print(line, flush=True)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log("WARNING: " + msg)

    def mark(self, name: str) -> None:
        self._timing[name] = round(time.time() - self._t0, 3)

    def param(self, name: str, default: Any = None) -> Any:
        value = self.params.get(name, default)
        return default if value is None else value

    def input_path(self, key: str = "image", required: bool = True) -> Path | None:
        rel = self.inputs.get(key)
        if rel is None:
            if required:
                raise KeyError(f"missing input {key!r}")
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.dir / p

    def output_path(self, name: str) -> Path:
        return self.output_dir / name

    # ----------------------------------------------------------------- done
    def finish(
        self,
        outputs: dict[str, str | os.PathLike[str]],
        stats: dict[str, Any] | None = None,
        labels: dict[int | str, str] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        rel = {}
        for key, path in outputs.items():
            p = Path(path).resolve()
            try:
                rel[key] = str(p.relative_to(self.dir))
            except ValueError:
                rel[key] = str(p)
        resp = {
            "contract_version": CONTRACT_VERSION,
            "status": "ok",
            "outputs": rel,
            "labels": {str(k): str(v) for k, v in (labels or {}).items()},
            "stats": _jsonable(stats or {}),
            "model_info": _jsonable(model_info or {}),
            "warnings": list(self.warnings),
            "timing": {**self._timing, "total_s": round(time.time() - self._t0, 3)},
        }
        (self.dir / "response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")
        self.log("done")

    def fail(self, exc: BaseException) -> None:
        resp = {
            "contract_version": CONTRACT_VERSION,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "warnings": list(self.warnings),
            "timing": {**self._timing, "total_s": round(time.time() - self._t0, 3)},
        }
        (self.dir / "response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")


def main(fn: Callable[[Job], None], argv: list[str] | None = None) -> None:
    """Standard entry point: ``if __name__ == "__main__": omm_job.main(run)``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="job directory containing request.json")
    args = parser.parse_args(argv)
    job = Job(args.job)
    try:
        fn(job)
    except Exception as exc:  # noqa: BLE001 - report every failure through response.json
        job.log(f"ERROR: {type(exc).__name__}: {exc}")
        job.fail(exc)
        sys.exit(1)


# ------------------------------------------------------------------ image I/O
def load_image(path: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(array, geometry)``. Arrays are ``[z, y, x]`` (3D) or ``[y, x(, c)]`` (2D)."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(_IMAGE2D):
        from PIL import Image

        with Image.open(path) as im:
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                arr = np.asarray(im).astype(np.uint16 if "16" in im.mode else np.int32)
                channels_last = False
            elif im.mode in ("L", "1"):
                arr = np.asarray(im.convert("L"))
                channels_last = False
            else:
                arr = np.asarray(im.convert("RGB"))
                channels_last = True
        geom = {
            "is_2d": True,
            "channels_last": channels_last,
            "spacing": [1.0, 1.0],
            "origin": [0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 1.0],
            "format": "png",
        }
        return arr, geom
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    if img.GetDimension() == 3 and img.GetSize()[2] == 1:
        img = img[:, :, 0]
    geom = {
        "is_2d": img.GetDimension() == 2,
        "channels_last": img.GetNumberOfComponentsPerPixel() > 1,
        "spacing": list(img.GetSpacing()),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
        "format": "sitk",
    }
    return sitk.GetArrayFromImage(img), geom


def save_mask(mask: np.ndarray, geom: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write an integer label map next to the geometry of the input image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask)
    mask = (
        mask.astype(np.uint8)
        if mask.dtype == bool or (mask.min() >= 0 and mask.max() <= 255)
        else mask.astype(np.int32)
    )
    if path.name.lower().endswith(".png"):
        from PIL import Image

        Image.fromarray(mask if mask.dtype == np.uint8 else mask.astype(np.int32)).save(path)
        return path
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(mask))
    dim = img.GetDimension()
    if len(geom.get("spacing", [])) >= dim:
        img.SetSpacing([float(v) for v in geom["spacing"][:dim]])
    if len(geom.get("origin", [])) >= dim:
        img.SetOrigin([float(v) for v in geom["origin"][:dim]])
    if len(geom.get("direction", [])) == dim * dim:
        img.SetDirection([float(v) for v in geom["direction"]])
    sitk.WriteImage(img, str(path), True)
    return path


def scalar(arr: np.ndarray, geom: dict[str, Any]) -> np.ndarray:
    if geom.get("channels_last"):
        return arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr


def guess_modality(arr: np.ndarray, geom: dict[str, Any]) -> str:
    if geom.get("channels_last"):
        return "RGB"
    if not geom.get("is_2d") and float(arr.min()) <= -500 and float(arr.max()) >= 300:
        return "CT"
    return "unknown"


def to_uint8(arr: np.ndarray, window: Any = None, modality: str = "") -> tuple[np.ndarray, str]:
    """Window/level to 8-bit. ``window``: preset name, ``{"center","width"}``, ``{"lower","upper"}``,
    ``[lower, upper]``, ``"auto"`` or ``None`` (soft-tissue for CT, robust percentiles otherwise)."""
    if arr.dtype == np.uint8:
        return arr, "uint8"
    if window is None:
        window = "soft-tissue" if modality == "CT" else "auto"
    if isinstance(window, str):
        key = window.lower().replace("_", "-")
        if key == "auto":
            lo, hi = np.percentile(arr.astype(np.float32), [0.5, 99.5])
        elif key == "full":
            lo, hi = float(arr.min()), float(arr.max())
        elif key in CT_WINDOWS:
            c, w = CT_WINDOWS[key]
            lo, hi = c - w / 2, c + w / 2
        else:
            raise ValueError(f"unknown window {window!r}")
        label = key
    elif isinstance(window, dict):
        if "center" in window:
            c, w = float(window["center"]), float(window["width"])
            lo, hi = c - w / 2, c + w / 2
        else:
            lo, hi = float(window["lower"]), float(window["upper"])
        label = f"[{lo:g},{hi:g}]"
    else:
        lo, hi = float(window[0]), float(window[1])
        label = f"[{lo:g},{hi:g}]"
    if hi <= lo:
        hi = lo + 1.0
    a = (arr.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), label


def resolve_device(pref: str = "auto") -> str:
    """``cuda`` when requested/available (needs torch), otherwise ``cpu``."""
    pref = (pref or "auto").lower()
    if pref in ("cpu", "none"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return pref if pref.startswith("cuda") else "cuda"
    except Exception:
        pass
    return "cpu"


# ------------------------------------------------------------- mask helpers
def largest_component(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labeled, n = ndimage.label(binary)
    if n <= 1:
        return binary.astype(bool)
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_fill_holes(binary)


def mask_summary(mask: np.ndarray, geom: dict[str, Any]) -> dict[str, Any]:
    spacing = geom.get("spacing") or [1.0] * mask.ndim
    vox_mm3 = float(np.prod(spacing[: mask.ndim]))
    labels = [int(v) for v in np.unique(mask) if v != 0]
    per = {}
    for lab in labels:
        b = mask == lab
        n = int(b.sum())
        coords = np.nonzero(b)
        bbox = list(reversed([int(c.min()) for c in coords])) + list(reversed([int(c.max()) for c in coords]))
        per[str(lab)] = {"voxels": n, "volume_ml": round(n * vox_mm3 / 1000, 3), "bbox_xyz": bbox}
    return {
        "labels_present": labels,
        "foreground_voxels": int((mask != 0).sum()),
        "per_label": per,
        "voxel_volume_mm3": vox_mm3,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
