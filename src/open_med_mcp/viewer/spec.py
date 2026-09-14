"""Declarative view specification consumed by every renderer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from open_med_mcp.core.prompts import Prompt

Layout = Literal["single", "three-plane", "montage"]
MaskMode = Literal["fill", "contour", "both"]

#: Default overlay palette (label 1, 2, 3 ... cycle through it).
PALETTE: list[str] = [
    "#ff3b30",
    "#34c759",
    "#0a84ff",
    "#ffd60a",
    "#bf5af2",
    "#ff9f0a",
    "#64d2ff",
    "#ff375f",
    "#30d158",
    "#ac8e68",
]


class MaskLayer(BaseModel):
    path: str = Field(description="Label map with the geometry of the image")
    name: str | None = Field(default=None, description="Legend name (defaults to file name)")
    color: str | None = Field(
        default=None,
        description="Single colour for all labels of this layer (hex); default = per-label palette",
    )
    alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    mode: MaskMode = "both"
    labels: list[int] | None = Field(default=None, description="Only draw these label ids")
    linewidth: float = 1.4


class ViewSpec(BaseModel):
    """What to draw. Paths are resolved against the workspace by the tool layer."""

    image: str
    masks: list[MaskLayer] = Field(default_factory=list)
    plane: Literal["axial", "coronal", "sagittal"] = "axial"
    layout: Layout = "single"
    slices: list[int] | None = Field(
        default=None,
        description="Slice indices for single/montage; None = auto (largest mask area, or middle)",
    )
    n_slices: int = Field(default=9, ge=1, le=64, description="montage: number of slices")
    center_xyz: list[float] | None = Field(
        default=None,
        description="three-plane: crossing point in native voxel coords; None = mask centroid or volume centre",
    )
    window: Any = Field(
        default=None,
        description="Preset name (soft-tissue, lung, bone, brain ...), {center,width}, [lower, upper] or 'auto'",
    )
    grid: bool = Field(default=True, description="Draw tick labels/grid in native voxel coordinates")
    prompts: list[Prompt] = Field(default_factory=list, description="Prompts to draw (points/boxes)")
    title: str | None = None
    label_names: dict[int, str] | None = Field(default=None, description="Legend names per label id")
    max_px: int = Field(default=900, ge=200, le=4000)
    dpi: int = Field(default=110, ge=50, le=300)
    crop_to_mask: bool = Field(default=False, description="Zoom to the mask bounding box (with margin)")
    crop_margin: int = 12
    show_legend: bool = True
    show_title: bool = True
    background: str = "#000000"
    font_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Multiply all font sizes (use >1 for figures that will be downscaled)",
    )
    panel_inches: float | None = Field(
        default=None,
        description="Edge length of each (square) panel in inches; default depends on the layout",
    )
