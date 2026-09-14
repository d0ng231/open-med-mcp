"""Intensity windowing (window center / width) and conversion to 8-bit for display and models."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Common CT window presets ``name -> (center, width)`` in Hounsfield units.
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


def resolve_window(
    window: str | dict[str, Any] | list[float] | tuple[float, float] | None,
    arr: np.ndarray,
    modality: str = "",
) -> tuple[float, float, str]:
    """Return ``(lower, upper, label)`` for a window specification.

    ``window`` may be a preset name, ``{"center": c, "width": w}``, ``{"lower": a, "upper": b}``,
    ``[lower, upper]``, ``"auto"`` (robust percentiles) or ``None`` (auto, or soft-tissue for CT).
    """
    if window is None:
        window = "soft-tissue" if modality == "CT" else "auto"
    if isinstance(window, str):
        key = window.lower().replace("_", "-")
        if key == "auto":
            lo, hi = np.percentile(arr.astype(np.float32), [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(arr.min()), float(arr.max()) or 1.0
            return float(lo), float(hi), "auto"
        if key == "full":
            return float(arr.min()), float(arr.max()), "full"
        if key in CT_WINDOWS:
            c, w = CT_WINDOWS[key]
            return c - w / 2, c + w / 2, key
        raise ValueError(f"unknown window preset {window!r}; known: auto, full, {', '.join(CT_WINDOWS)}")
    if isinstance(window, dict):
        if "center" in window and "width" in window:
            c, w = float(window["center"]), float(window["width"])
            return c - w / 2, c + w / 2, f"C{c:g}/W{w:g}"
        if "lower" in window and "upper" in window:
            return (
                float(window["lower"]),
                float(window["upper"]),
                f"[{window['lower']:g}, {window['upper']:g}]",
            )
        raise ValueError("window dict needs center+width or lower+upper")
    lo, hi = float(window[0]), float(window[1])
    return lo, hi, f"[{lo:g}, {hi:g}]"


def apply_window(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Linearly map ``[lower, upper]`` to ``[0, 255]`` and return ``uint8``."""
    a = arr.astype(np.float32)
    if upper <= lower:
        upper = lower + 1.0
    a = (a - lower) / (upper - lower)
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint8(arr: np.ndarray, window: Any = None, modality: str = "") -> tuple[np.ndarray, str]:
    lo, hi, label = resolve_window(window, arr, modality)
    return apply_window(arr, lo, hi), label
