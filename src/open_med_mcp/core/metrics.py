"""Segmentation quality metrics between two label maps."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _surface(binary: np.ndarray) -> np.ndarray:
    if not binary.any():
        return binary
    eroded = ndimage.binary_erosion(binary, structure=ndimage.generate_binary_structure(binary.ndim, 1))
    return binary & ~eroded


def surface_distances(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric surface distances (mm) from surface(a)->b and surface(b)->a."""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return np.array([]), np.array([])
    samp = tuple(reversed(spacing))  # spacing is (x, y, z); arrays are (z, y, x)
    dist_to_b = ndimage.distance_transform_edt(~sb, sampling=samp)
    dist_to_a = ndimage.distance_transform_edt(~sa, sampling=samp)
    return dist_to_b[sa], dist_to_a[sb]


def compare_binary(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...], surface: bool = True
) -> dict[str, Any]:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    na, nb = int(a.sum()), int(b.sum())
    union = na + nb - inter
    vox = float(np.prod(spacing))
    out: dict[str, Any] = {
        "dice": round(2 * inter / (na + nb), 4) if (na + nb) else 1.0,
        "iou": round(inter / union, 4) if union else 1.0,
        "precision": round(inter / na, 4) if na else 0.0,  # a = prediction, b = reference
        "recall": round(inter / nb, 4) if nb else 0.0,
        "voxels_a": na,
        "voxels_b": nb,
        "volume_a_ml": round(na * vox / 1000, 3),
        "volume_b_ml": round(nb * vox / 1000, 3),
        "volume_diff_ml": round((na - nb) * vox / 1000, 3),
        "volume_diff_pct": round(100.0 * (na - nb) / nb, 2) if nb else None,
    }
    if surface and na and nb:
        d_ab, d_ba = surface_distances(a, b, spacing)
        alld = np.concatenate([d_ab, d_ba])
        out.update(
            {
                "hausdorff_mm": round(float(alld.max()), 3),
                "hd95_mm": round(float(np.percentile(alld, 95)), 3),
                "assd_mm": round(float(alld.mean()), 3),
            }
        )
    return out


def compare_masks(
    a: np.ndarray,
    b: np.ndarray,
    spacing: tuple[float, ...],
    labels: list[int] | None = None,
    surface: bool = True,
) -> dict[str, Any]:
    """Compare two label maps label by label (plus a binary foreground comparison)."""
    present = sorted({int(v) for v in np.unique(a)} | {int(v) for v in np.unique(b)})
    present = [v for v in present if v != 0]
    if labels:
        present = [v for v in present if v in labels]
    per_label = {str(lab): compare_binary(a == lab, b == lab, spacing, surface) for lab in present}
    result = {"foreground": compare_binary(a != 0, b != 0, spacing, surface), "per_label": per_label}
    if per_label:
        result["mean_dice"] = round(float(np.mean([v["dice"] for v in per_label.values()])), 4)
    return result
