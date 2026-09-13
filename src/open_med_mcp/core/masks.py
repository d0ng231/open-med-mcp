"""Label-map utilities: statistics, post-processing operations and prompt extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from open_med_mcp.core.image import MedicalImage, Plane


def _labels_in(mask: np.ndarray, labels: list[int] | None = None) -> list[int]:
    present = [int(v) for v in np.unique(mask) if v != 0]
    if labels:
        return [lab for lab in labels if lab in present]
    return present


def bbox_xyz(binary: np.ndarray) -> list[int] | None:
    """Bounding box ``[x0, y0, (z0,) x1, y1, (z1)]`` (inclusive, native index) of a boolean array."""
    if not binary.any():
        return None
    coords = np.nonzero(binary)
    mins = [int(c.min()) for c in coords]  # numpy axis order (z, y, x) or (y, x)
    maxs = [int(c.max()) for c in coords]
    return list(reversed(mins)) + list(reversed(maxs))


def mask_stats(
    mask: MedicalImage,
    image: MedicalImage | None = None,
    labels: list[int] | None = None,
    label_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Per-label voxel counts, volumes, bounding boxes, centroids and (optionally) intensity stats."""
    arr = mask.array
    vox_mm3 = mask.voxel_volume_mm3()
    per_label: list[dict[str, Any]] = []
    for lab in _labels_in(arr, labels):
        binary = arr == lab
        n = int(binary.sum())
        centroid = [float(c) for c in reversed(ndimage.center_of_mass(binary))]
        entry: dict[str, Any] = {
            "label": lab,
            "name": (label_names or {}).get(lab),
            "voxels": n,
            "volume_mm3": round(n * vox_mm3, 2),
            "volume_ml": round(n * vox_mm3 / 1000.0, 3),
            "bbox_xyz": bbox_xyz(binary),
            "centroid_xyz": [round(c, 2) for c in centroid],
            "components": int(ndimage.label(binary)[1]),
        }
        if not mask.is_2d:
            entry["slice_range"] = {
                p: _slice_range(binary, mask, p) for p in ("axial", "coronal", "sagittal")
            }
        if image is not None and image.shape_zyx == mask.shape_zyx:
            vals = image.scalar_array[binary].astype(np.float64)
            if vals.size:
                entry["intensity"] = {
                    "mean": round(float(vals.mean()), 3),
                    "std": round(float(vals.std()), 3),
                    "min": round(float(vals.min()), 3),
                    "max": round(float(vals.max()), 3),
                    "median": round(float(np.median(vals)), 3),
                }
        per_label.append(entry)
    total = int((arr != 0).sum())
    return {
        "shape_zyx": list(mask.shape_zyx),
        "spacing_xyz_mm": [round(float(s), 4) for s in mask.spacing],
        "voxel_volume_mm3": round(vox_mm3, 5),
        "n_labels": len(per_label),
        "foreground_voxels": total,
        "foreground_volume_ml": round(total * vox_mm3 / 1000.0, 3),
        "labels": per_label,
    }


def _slice_range(binary: np.ndarray, mask: MedicalImage, plane: Plane) -> list[int] | None:
    axis = mask.numpy_axis_for_plane(plane)
    other = tuple(i for i in range(binary.ndim) if i != axis)
    has = np.any(binary, axis=other)
    idx = np.nonzero(has)[0]
    if idx.size == 0:
        return None
    return [int(idx.min()), int(idx.max())]


# ----------------------------------------------------------------- operations
def _struct(ndim: int, radius: int) -> np.ndarray:
    return ndimage.iterate_structure(ndimage.generate_binary_structure(ndim, 1), max(int(radius), 1))


def postprocess(
    mask: np.ndarray, ops: list[dict[str, Any] | str], spacing: tuple[float, ...] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Apply a list of operations. Each op is ``{"op": name, ...params}`` or just ``"name"``.

    Supported ops: ``largest_component`` (per label), ``fill_holes``, ``remove_small`` (``min_voxels`` or
    ``min_mm3``), ``open``/``close``/``dilate``/``erode`` (``radius`` in voxels), ``keep_labels``
    (``labels``), ``binarize``, ``relabel`` (``mapping``), ``keep_slices`` (``axis``, ``start``, ``stop``).
    """
    out = np.asarray(mask).copy()
    log: list[str] = []
    for op in ops:
        spec = {"op": op} if isinstance(op, str) else dict(op)
        name = str(spec.get("op", "")).lower()
        if name == "largest_component":
            new = np.zeros_like(out)
            for lab in _labels_in(out):
                labeled, n = ndimage.label(out == lab)
                if n == 0:
                    continue
                sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
                keep = int(np.argmax(sizes)) + 1
                new[labeled == keep] = lab
            log.append("largest_component")
            out = new
        elif name == "fill_holes":
            new = np.zeros_like(out)
            for lab in _labels_in(out):
                new[ndimage.binary_fill_holes(out == lab)] = lab
            out = new
            log.append("fill_holes")
        elif name == "remove_small":
            min_vox = int(spec.get("min_voxels", 0))
            if "min_mm3" in spec and spacing:
                min_vox = max(min_vox, int(float(spec["min_mm3"]) / float(np.prod(spacing))))
            new = np.zeros_like(out)
            for lab in _labels_in(out):
                labeled, n = ndimage.label(out == lab)
                sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1)) if n else []
                for i, s in enumerate(sizes, start=1):
                    if s >= min_vox:
                        new[labeled == i] = lab
            out = new
            log.append(f"remove_small(min_voxels={min_vox})")
        elif name in ("open", "close", "dilate", "erode"):
            radius = int(spec.get("radius", 1))
            st = _struct(out.ndim, radius)
            fn = {
                "open": ndimage.binary_opening,
                "close": ndimage.binary_closing,
                "dilate": ndimage.binary_dilation,
                "erode": ndimage.binary_erosion,
            }[name]
            new = np.zeros_like(out)
            for lab in _labels_in(out):
                new[fn(out == lab, structure=st)] = lab
            out = new
            log.append(f"{name}(radius={radius})")
        elif name == "keep_labels":
            keep = {int(v) for v in spec.get("labels", [])}
            out = np.where(np.isin(out, list(keep)), out, 0)
            log.append(f"keep_labels({sorted(keep)})")
        elif name == "binarize":
            out = (out != 0).astype(np.uint8)
            log.append("binarize")
        elif name == "relabel":
            mapping = {int(k): int(v) for k, v in dict(spec.get("mapping", {})).items()}
            new = out.copy()
            for src, dst in mapping.items():
                new[out == src] = dst
            out = new
            log.append(f"relabel({mapping})")
        elif name == "keep_slices":
            axis = int(spec.get("axis", 0))
            start, stop = int(spec.get("start", 0)), int(spec.get("stop", out.shape[axis]))
            sl = [slice(None)] * out.ndim
            new = np.zeros_like(out)
            sl[axis] = slice(start, stop)
            new[tuple(sl)] = out[tuple(sl)]
            out = new
            log.append(f"keep_slices(axis={axis}, {start}:{stop})")
        else:
            raise ValueError(f"unknown post-processing op {name!r}")
    return out, log


# --------------------------------------------------------------- prompts
def mask_to_prompts(
    mask: MedicalImage,
    plane: Plane = "axial",
    label: int | None = None,
    slice_index: int | str = "auto",
    margin: int = 2,
    n_points: int = 1,
) -> dict[str, Any]:
    """Derive box/point prompts from an existing mask (e.g. a coarse result or another model's output).

    Returns 3D prompts in native ``(x, y, z)`` voxel coordinates (2D ``(x, y)`` for 2D images).
    """
    arr = mask.array
    binary = (arr == label) if label else (arr != 0)
    if not binary.any():
        return {"boxes": [], "points": [], "slice": None, "note": "mask is empty"}
    if mask.is_2d:
        box = bbox_xyz(binary)
        assert box is not None
        x0, y0, x1, y1 = box
        cy, cx = ndimage.center_of_mass(binary)
        return {
            "plane": "axial",
            "slice": None,
            "boxes": [
                [
                    max(x0 - margin, 0),
                    max(y0 - margin, 0),
                    min(x1 + margin, arr.shape[1] - 1),
                    min(y1 + margin, arr.shape[0] - 1),
                ]
            ],
            "points": [[round(float(cx), 1), round(float(cy), 1)]],
            "bbox_3d": None,
        }
    axis = mask.numpy_axis_for_plane(plane)
    other = tuple(i for i in range(3) if i != axis)
    areas = binary.sum(axis=other)
    if slice_index == "auto":
        s = int(np.argmax(areas))
    else:
        s = int(slice_index)
    sl = np.take(binary, s, axis=axis)
    if not sl.any():
        return {"plane": plane, "slice": s, "boxes": [], "points": [], "note": "mask empty on this slice"}
    # in-plane numpy axes (row, col) -> native index axes
    row_np, col_np = other  # numpy axes of the 2D slice
    rows, cols = np.nonzero(sl)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    shape = binary.shape

    def np_to_xyz(np_axis: int) -> int:
        return 2 - np_axis

    lo = [0, 0, 0]
    hi = [0, 0, 0]
    lo[np_to_xyz(row_np)], hi[np_to_xyz(row_np)] = max(r0 - margin, 0), min(r1 + margin, shape[row_np] - 1)
    lo[np_to_xyz(col_np)], hi[np_to_xyz(col_np)] = max(c0 - margin, 0), min(c1 + margin, shape[col_np] - 1)
    lo[np_to_xyz(axis)] = hi[np_to_xyz(axis)] = s
    box3d = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
    # centre points
    points: list[list[float]] = []
    labeled, n = ndimage.label(sl)
    comps = sorted(range(1, n + 1), key=lambda i: -int((labeled == i).sum()))[: max(n_points, 1)]
    for comp in comps:
        cr, cc = ndimage.center_of_mass(labeled == comp)
        pt = [0.0, 0.0, 0.0]
        pt[np_to_xyz(row_np)] = round(float(cr), 1)
        pt[np_to_xyz(col_np)] = round(float(cc), 1)
        pt[np_to_xyz(axis)] = float(s)
        points.append(pt)
    return {
        "plane": plane,
        "slice": s,
        "boxes": [box3d],
        "points": points,
        "bbox_3d": bbox_xyz(binary),
        "slice_range": [int(np.nonzero(areas)[0].min()), int(np.nonzero(areas)[0].max())],
    }
