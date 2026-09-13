"""Prompt schemas for promptable segmentation models (points, boxes) and coordinate normalisation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from open_med_mcp.core.image import MedicalImage, Plane


class Prompt(BaseModel):
    """A single visual prompt in **native voxel coordinates**.

    * ``point``: ``coords=[x, y]`` (2D image) or ``[x, y, z]`` (3D); ``label`` 1 = foreground, 0 = background.
    * ``box``: ``coords=[x0, y0, x1, y1]`` on ``slice`` (3D) or ``[x0, y0, z0, x1, y1, z1]`` with
      ``z0 == z1`` for a box on a single axial slice (or the plane's stacking axis).

    For 3D images a 2D prompt is placed on ``slice`` along the ``plane`` used for propagation.
    """

    type: Literal["point", "box"]
    coords: list[float] = Field(description="Native voxel coordinates, see docs/coordinates.md")
    label: int = Field(
        default=1, description="For points: 1 = foreground (include), 0 = background (exclude)"
    )
    slice: int | None = Field(default=None, description="Slice index (3D only) when 2D coordinates are given")
    object_id: int = Field(
        default=1, description="Group prompts that belong to the same object (multi-object)"
    )

    @model_validator(mode="after")
    def _check(self) -> Prompt:
        n = len(self.coords)
        if self.type == "point" and n not in (2, 3):
            raise ValueError("point coords must be [x, y] or [x, y, z]")
        if self.type == "box" and n not in (4, 6):
            raise ValueError("box coords must be [x0, y0, x1, y1] or [x0, y0, z0, x1, y1, z1]")
        return self


def normalize_prompts(
    prompts: list[Prompt] | list[dict[str, Any]], image: MedicalImage, plane: Plane = "axial"
) -> list[dict[str, Any]]:
    """Convert prompts to the adapter wire format.

    Output items: ``{"type", "label", "object_id", "slice", "coords": [c0, c1(, c2, c3)]}`` where ``slice``
    is the index along the plane's NumPy axis and ``coords`` are **in-plane (col, row)** pixel
    coordinates of the 2D slice as the adapter sees it (``slice[row, col]``).
    """
    items = [p if isinstance(p, Prompt) else Prompt.model_validate(p) for p in prompts]
    out: list[dict[str, Any]] = []
    if image.is_2d:
        for p in items:
            c = list(p.coords)
            if p.type == "point":
                x, y = c[0], c[1]
                out.append(
                    {
                        "type": "point",
                        "label": p.label,
                        "object_id": p.object_id,
                        "slice": 0,
                        "coords": [x, y],
                    }
                )
            else:
                if len(c) == 6:
                    c = [c[0], c[1], c[3], c[4]]
                out.append(
                    {
                        "type": "box",
                        "label": 1,
                        "object_id": p.object_id,
                        "slice": 0,
                        "coords": _sorted_box(c),
                    }
                )
        return out

    np_axis = image.numpy_axis_for_plane(plane)  # axis in (z, y, x)
    stack_xyz = 2 - np_axis  # 0=x 1=y 2=z
    inplane_np = [ax for ax in (0, 1, 2) if ax != np_axis]  # (row_axis, col_axis) in numpy order
    row_xyz, col_xyz = 2 - inplane_np[0], 2 - inplane_np[1]

    for p in items:
        c = list(p.coords)
        if p.type == "point":
            if len(c) == 2:
                if p.slice is None:
                    raise ValueError("2D point on a 3D image needs `slice`")
                xyz = [0.0, 0.0, 0.0]
                xyz[col_xyz], xyz[row_xyz], xyz[stack_xyz] = c[0], c[1], float(p.slice)
            else:
                xyz = c
            out.append(
                {
                    "type": "point",
                    "label": p.label,
                    "object_id": p.object_id,
                    "slice": int(round(xyz[stack_xyz])),
                    "coords": [xyz[col_xyz], xyz[row_xyz]],
                }
            )
        else:
            if len(c) == 4:
                if p.slice is None:
                    raise ValueError("2D box on a 3D image needs `slice`")
                s = int(p.slice)
                col0, row0, col1, row1 = c
            else:
                lo, hi = c[:3], c[3:]
                s = int(round((lo[stack_xyz] + hi[stack_xyz]) / 2))
                col0, row0, col1, row1 = lo[col_xyz], lo[row_xyz], hi[col_xyz], hi[row_xyz]
            out.append(
                {
                    "type": "box",
                    "label": 1,
                    "object_id": p.object_id,
                    "slice": s,
                    "coords": _sorted_box([col0, row0, col1, row1]),
                }
            )
    return out


def _sorted_box(c: list[float]) -> list[float]:
    x0, y0, x1, y1 = c
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
