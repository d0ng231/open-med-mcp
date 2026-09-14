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
        if p.type == "text":
            continue
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


def _fs(spec: ViewSpec, base: float) -> float:
    return base * spec.font_scale


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
        ax.imshow(ds.array[..., :3].astype(np.uint8), interpolation="nearest", aspect=ds.aspect)
    else:
        ax.imshow(ds.array, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect=ds.aspect)
    _draw_overlays(ax, ds, masks, legend)
    _draw_prompts(ax, ds, spec, img)
    if spec.grid:
        xt, xl = ds.native_ticks("x")
        yt, yl = ds.native_ticks("y")
        ax.set_xticks(xt, xl)
        ax.set_yticks(yt, yl)
        ax.tick_params(labelsize=_fs(spec, 8), colors="#cfcfcf", length=2.5, pad=2)
        ax.grid(True, color="#ffffff", alpha=0.16, linewidth=0.6)
        ax.set_xlabel(f"{'xyz'[ds.col_axis_xyz]} (voxel)", fontsize=_fs(spec, 9), color="#cfcfcf", labelpad=2)
        ax.set_ylabel(f"{'xyz'[ds.row_axis_xyz]} (voxel)", fontsize=_fs(spec, 9), color="#cfcfcf", labelpad=2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#444444")
    # square, letter-boxed view: every panel gets the same size regardless of slice shape
    h, w = ds.shape
    if crop:
        x0, y0, x1, y1 = crop
        w_data, h_data = (x1 - x0 + 1), (y1 - y0 + 1)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    else:
        w_data, h_data = w, h
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    side = max(w_data, h_data * ds.aspect) * 1.02
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy + side / (2 * ds.aspect), cy - side / (2 * ds.aspect))
    if spec.show_title:
        t = (
            f"{ds.plane}  {'xyz'[ds.stack_axis_xyz]} = {ds.index} / {ds.n_slices - 1}"
            if not img.is_2d
            else "image"
        )
        if spec.layout != "montage":
            t = f"{t}    window: {wlabel}"
        ax.set_title(t, fontsize=_fs(spec, 10), color="white", pad=6)


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
        ncols = 3 if spec.layout == "three-plane" else (1 if n == 1 else min(4, int(np.ceil(np.sqrt(n)))))
        nrows = int(np.ceil(n / ncols))
        panel_in = spec.panel_inches or {"single": 5.2, "three-plane": 3.9, "montage": 3.0}[spec.layout]
        has_legend = bool(masks) and spec.show_legend
        title = spec.title
        if title is None and spec.layout == "montage" and spec.show_title:
            title = f"{spec.plane} montage    window: {wlabel}"
        fs = spec.font_scale
        # explicit layout (no tight_layout): square slots sized exactly, margins scaled with the fonts
        left = 0.62 * fs if spec.grid else 0.12
        right = 0.14
        wspace = (0.66 * fs if spec.grid else 0.18) if ncols > 1 else 0.0
        hspace = ((0.68 if spec.show_title else 0.2) + (0.5 if spec.grid else 0.0)) * fs if nrows > 1 else 0.0
        top = (0.5 * fs if title else 0.12) + (0.32 * fs if spec.show_title else 0.06)
        bottom = (0.52 * fs if spec.grid else 0.12) + (0.42 * fs if has_legend else 0.0)
        fig_w = left + ncols * panel_in + (ncols - 1) * wspace + right
        fig_h = bottom + nrows * panel_in + (nrows - 1) * hspace + top
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=spec.dpi, squeeze=False)
        fig.subplots_adjust(
            left=left / fig_w,
            right=1 - right / fig_w,
            bottom=bottom / fig_h,
            top=1 - top / fig_h,
            wspace=wspace / panel_in,
            hspace=hspace / panel_in,
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
        if title:
            fig.suptitle(title, color="white", fontsize=_fs(spec, 12), y=1 - 0.12 / fig_h, va="top")
        if legend and has_legend:
            handles = [patches.Patch(facecolor=c, edgecolor=c, label=k) for k, c in legend.items()][:14]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(len(handles), 4),
                fontsize=_fs(spec, 9.5),
                frameon=False,
                labelcolor="white",
                bbox_to_anchor=(0.5, 0.06 / fig_h),
                borderaxespad=0.0,
                handlelength=1.4,
                columnspacing=1.6,
            )
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
