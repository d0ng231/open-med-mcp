"""Matplotlib renderer producing PNG figures that an agent can look at."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from open_med_mcp.core.image import MedicalImage  # noqa: E402
from open_med_mcp.core.windowing import to_uint8  # noqa: E402
from open_med_mcp.viewer.registry import RenderResult, register_renderer  # noqa: E402
from open_med_mcp.viewer.slicing import (  # noqa: E402
    DisplaySlice,
    auto_slices,
    display_slice,
    mask_centroid_xyz,
)
from open_med_mcp.viewer.spec import PALETTE, MaskLayer, ViewSpec  # noqa: E402

MaskItem = tuple[MedicalImage, dict[str, Any]]  # (mask image, {"layer": MaskLayer, "labels": {id: name}})


def label_color(layer: MaskLayer, label: int, layer_index: int) -> str:
    if layer.color:
        return layer.color
    return PALETTE[(label - 1 + layer_index * 3) % len(PALETTE)]


def _draw_overlays(ax: plt.Axes, ds: DisplaySlice, masks: list[MaskItem], legend: dict[str, str]) -> None:
    for li, (mask_img, meta) in enumerate(masks):
        layer: MaskLayer = meta["layer"]
        names: dict[int, str] = meta.get("labels") or {}
        m2d = ds.apply_to_mask(mask_img.array)
        labels = [int(v) for v in np.unique(m2d) if v != 0]
        if layer.labels:
            labels = [v for v in labels if v in layer.labels]
        for lab in labels:
            color = label_color(layer, lab, li)
            binary = m2d == lab
            if layer.mode in ("fill", "both"):
                rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
                rgba[binary, :3] = to_rgb(color)
                rgba[binary, 3] = layer.alpha
                ax.imshow(rgba, interpolation="nearest", aspect=ds.aspect)
            if layer.mode in ("contour", "both"):
                try:
                    ax.contour(
                        binary.astype(np.float32), levels=[0.5], colors=[color], linewidths=layer.linewidth
                    )
                except Exception:
                    pass
            key = f"{layer.name or mask_img.path.name if mask_img.path else 'mask'}"
            legend_name = names.get(lab) or (key if len(labels) == 1 else f"{key}:{lab}")
            legend.setdefault(legend_name, color)


def _draw_prompts(ax: plt.Axes, ds: DisplaySlice, spec: ViewSpec, img: MedicalImage) -> None:
    for p in spec.prompts:
        coords = list(p.coords)
        if p.type == "point":
            if img.is_2d:
                xyz = [coords[0], coords[1], 0.0]
            elif len(coords) == 2:
                if p.slice is None:
                    continue
                xyz = [0.0, 0.0, 0.0]
                xyz[ds.col_axis_xyz], xyz[ds.row_axis_xyz], xyz[ds.stack_axis_xyz] = (
                    coords[0],
                    coords[1],
                    float(p.slice),
                )
            else:
                xyz = coords
            if not img.is_2d and int(round(xyz[ds.stack_axis_xyz])) != ds.index:
                continue
            c, r = ds.to_display(xyz)
            ax.plot(
                c,
                r,
                marker="P" if p.label else "X",
                markersize=9,
                markeredgecolor="black",
                markerfacecolor="#39ff14" if p.label else "#ff1744",
                linestyle="none",
            )
        else:
            if img.is_2d:
                lo = [coords[0], coords[1], 0.0]
                hi = [coords[2], coords[3], 0.0]
            elif len(coords) == 4:
                if p.slice is None:
                    continue
                lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                lo[ds.col_axis_xyz], lo[ds.row_axis_xyz], lo[ds.stack_axis_xyz] = (
                    coords[0],
                    coords[1],
                    float(p.slice),
                )
                hi[ds.col_axis_xyz], hi[ds.row_axis_xyz], hi[ds.stack_axis_xyz] = (
                    coords[2],
                    coords[3],
                    float(p.slice),
                )
            else:
                lo, hi = coords[:3], coords[3:]
            if not img.is_2d and not (
                min(lo[ds.stack_axis_xyz], hi[ds.stack_axis_xyz])
                <= ds.index
                <= max(lo[ds.stack_axis_xyz], hi[ds.stack_axis_xyz])
            ):
                continue
            c0, r0 = ds.to_display(lo)
            c1, r1 = ds.to_display(hi)
            x0, x1 = sorted((c0, c1))
            y0, y1 = sorted((r0, r1))
            ax.add_patch(
                patches.Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0 + 1,
                    y1 - y0 + 1,
                    fill=False,
                    edgecolor="#00e5ff",
                    linewidth=1.8,
                    linestyle="--",
                )
            )


def _panel(
    ax: plt.Axes,
    img: MedicalImage,
    ds: DisplaySlice,
    spec: ViewSpec,
    masks: list[MaskItem],
    legend: dict[str, str],
    wlabel: str,
    crop: tuple[int, int, int, int] | None,
) -> None:
    if img.is_rgb:
        base = ds.array[..., :3].astype(np.uint8)
        ax.imshow(base, interpolation="nearest", aspect=ds.aspect)
    else:
        ax.imshow(ds.array, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect=ds.aspect)
    _draw_overlays(ax, ds, masks, legend)
    _draw_prompts(ax, ds, spec, img)
    if spec.grid:
        xt, xl = ds.native_ticks("x")
        yt, yl = ds.native_ticks("y")
        ax.set_xticks(xt, xl)
        ax.set_yticks(yt, yl)
        ax.tick_params(labelsize=7, colors="#dddddd", length=2)
        ax.grid(True, color="#ffffff", alpha=0.18, linewidth=0.6)
        ax.set_xlabel(f"{'xyz'[ds.col_axis_xyz]} (voxel)", fontsize=8, color="#dddddd")
        ax.set_ylabel(f"{'xyz'[ds.row_axis_xyz]} (voxel)", fontsize=8, color="#dddddd")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#555555")
    if crop:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0 - 0.5, x1 + 0.5)
        ax.set_ylim(y1 + 0.5, y0 - 0.5)
    if spec.show_title:
        t = (
            f"{ds.plane} {'xyz'[ds.stack_axis_xyz]}={ds.index}/{ds.n_slices - 1}"
            if not img.is_2d
            else "image"
        )
        ax.set_title(f"{t}   window: {wlabel}", fontsize=9, color="white")


def _crop_box(ds: DisplaySlice, masks: list[MaskItem], margin: int) -> tuple[int, int, int, int] | None:
    union = None
    for mask_img, _ in masks:
        m2d = ds.apply_to_mask(mask_img.array) != 0
        union = m2d if union is None else (union | m2d)
    if union is None or not union.any():
        return None
    rows, cols = np.nonzero(union)
    h, w = union.shape
    return (
        max(int(cols.min()) - margin, 0),
        max(int(rows.min()) - margin, 0),
        min(int(cols.max()) + margin, w - 1),
        min(int(rows.max()) + margin, h - 1),
    )


@register_renderer("png")
class PngRenderer:
    name = "png"

    def render(self, image: MedicalImage, masks: list[MaskItem], spec: ViewSpec) -> RenderResult:
        modality = image.modality_guess()
        if image.is_rgb:
            windowed, wlabel = image.array, "rgb"
        else:
            windowed, wlabel = to_uint8(image.scalar_array, spec.window, modality)
        base = MedicalImage(
            windowed,
            image.spacing,
            image.origin,
            image.direction,
            image.path,
            image.format,
            dict(image.metadata),
        )
        mask_arrays = [m.array for m, _ in masks]
        panels: list[DisplaySlice] = []
        if spec.layout == "three-plane" and not image.is_2d:
            center = spec.center_xyz or mask_centroid_xyz(mask_arrays) or [s / 2 for s in image.size_xyz]
            for plane in ("axial", "coronal", "sagittal"):
                ax_xyz = image.index_axis_for_plane(plane)
                panels.append(display_slice(base, plane, int(round(center[ax_xyz]))))
        elif spec.layout == "montage" and not image.is_2d:
            idx = spec.slices or auto_slices(image, spec.plane, mask_arrays, n=spec.n_slices)
            panels = [display_slice(base, spec.plane, i) for i in idx]
        else:
            idx = spec.slices[:1] if spec.slices else auto_slices(image, spec.plane, mask_arrays, n=1)
            panels = [display_slice(base, spec.plane, idx[0])]

        n = len(panels)
        ncols = 3 if spec.layout == "three-plane" else int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        # figure size from the panel aspect so that the longest edge is ~max_px
        h0, w0 = panels[0].shape
        panel_w_in = 3.6
        panel_h_in = panel_w_in * (h0 * panels[0].aspect) / max(w0, 1)
        panel_h_in = float(np.clip(panel_h_in, 1.6, 6.0))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                panel_w_in * ncols + 0.4,
                panel_h_in * nrows + (0.6 if spec.show_title or spec.title else 0.2),
            ),
            dpi=spec.dpi,
            squeeze=False,
        )
        fig.patch.set_facecolor(spec.background)
        legend: dict[str, str] = {}
        for i, ax in enumerate(axes.flat):
            ax.set_facecolor(spec.background)
            if i >= n:
                ax.axis("off")
                continue
            crop = _crop_box(panels[i], masks, spec.crop_margin) if spec.crop_to_mask else None
            _panel(ax, image, panels[i], spec, masks, legend, wlabel, crop)
        if spec.title:
            fig.suptitle(spec.title, color="white", fontsize=10)
        if legend and spec.show_legend:
            handles = [patches.Patch(facecolor=c, edgecolor=c, label=k) for k, c in legend.items()][:14]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(len(handles), 5),
                fontsize=7,
                frameon=False,
                labelcolor="white",
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.tight_layout(rect=(0, 0.05 if legend and spec.show_legend else 0, 1, 0.96 if spec.title else 1))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=spec.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        data = _limit_size(buf.getvalue(), spec.max_px)
        info = {
            "layout": spec.layout,
            "plane": spec.plane,
            "window": wlabel,
            "panels": [
                {
                    "plane": p.plane,
                    "index": p.index,
                    "n_slices": p.n_slices,
                    "stack_axis": "xyz"[p.stack_axis_xyz],
                    "screen_x_axis": "xyz"[p.col_axis_xyz],
                    "screen_y_axis": "xyz"[p.row_axis_xyz],
                    "flip_x": p.flip_cols,
                    "flip_y": p.flip_rows,
                }
                for p in panels
            ],
            "legend": legend,
        }
        return RenderResult(data, "image/png", ".png", info)


def _limit_size(png: bytes, max_px: int) -> bytes:
    with PILImage.open(io.BytesIO(png)) as im:
        w, h = im.size
        if max(w, h) <= max_px:
            return png
        scale = max_px / max(w, h)
        im2 = im.resize((max(int(w * scale), 1), max(int(h * scale), 1)), PILImage.LANCZOS)
        out = io.BytesIO()
        im2.save(out, format="PNG", optimize=True)
        return out.getvalue()
