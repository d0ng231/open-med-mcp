"""Self-contained HTML slice viewer (no CDN, no server): scroll through slices, toggle overlays,
click to read native voxel coordinates for prompts. Rendered slices are embedded as base64 images."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.colors import to_rgb
from PIL import Image as PILImage

from open_med_mcp.core.image import MedicalImage
from open_med_mcp.core.windowing import to_uint8
from open_med_mcp.viewer.png import MaskItem, label_color
from open_med_mcp.viewer.registry import RenderResult, register_renderer
from open_med_mcp.viewer.slicing import auto_slices, display_slice
from open_med_mcp.viewer.spec import ViewSpec

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))


def _b64(im: PILImage.Image, fmt: str = "PNG", **kw: Any) -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize(arr: np.ndarray, max_size: int, aspect: float, resample: int) -> PILImage.Image:
    im = PILImage.fromarray(arr)
    w, h = im.size
    h2 = h * aspect
    scale = min(1.0, max_size / max(w, h2))
    return im.resize((max(int(w * scale), 1), max(int(h2 * scale), 1)), resample)


@register_renderer("html")
class HtmlSliceViewer:
    name = "html"

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
        max_size = min(spec.max_px, 768)
        if image.is_2d:
            indices = [0]
        elif spec.slices:
            indices = sorted(set(spec.slices))
        else:
            n = image.n_slices(spec.plane)
            indices = (
                list(range(n))
                if n <= 400
                else auto_slices(image, spec.plane, [m.array for m, _ in masks], n=400)
            )
        frames: list[dict[str, Any]] = []
        legend: dict[str, str] = {}
        first = None
        for idx in indices:
            ds = display_slice(base, spec.plane, idx)
            first = first or ds
            im = _resize(
                ds.array if not image.is_rgb else ds.array[..., :3], max_size, ds.aspect, PILImage.BILINEAR
            )
            overlay = np.zeros((*ds.shape, 4), dtype=np.uint8)
            for li, (mask_img, meta) in enumerate(masks):
                layer = meta["layer"]
                names = meta.get("labels") or {}
                m2d = ds.apply_to_mask(mask_img.array)
                for lab in [int(v) for v in np.unique(m2d) if v != 0]:
                    if layer.labels and lab not in layer.labels:
                        continue
                    color = label_color(layer, lab, li)
                    rgb = tuple(int(255 * c) for c in to_rgb(color))
                    sel = m2d == lab
                    overlay[sel, :3] = rgb
                    overlay[sel, 3] = 255
                    key = layer.name or (mask_img.path.name if mask_img.path else "mask")
                    legend.setdefault(
                        names.get(lab) or (key if len(np.unique(m2d)) <= 2 else f"{key}:{lab}"), color
                    )
            ov = _resize(overlay, max_size, ds.aspect, PILImage.NEAREST) if masks else None
            frames.append(
                {
                    "index": idx,
                    "img": _b64(im.convert("L") if not image.is_rgb else im, "JPEG", quality=88),
                    "ov": _b64(ov, "PNG") if ov is not None else None,
                    "w": im.size[0],
                    "h": im.size[1],
                    "flip_x": ds.flip_cols,
                    "flip_y": ds.flip_rows,
                    "cols": ds.shape[1],
                    "rows": ds.shape[0],
                }
            )
        assert first is not None
        ctx = {
            "title": spec.title or (image.path.name if image.path else "open-med-mcp viewer"),
            "plane": spec.plane if not image.is_2d else "image",
            "window": wlabel,
            "frames_json": json.dumps(frames),
            "legend": legend,
            "meta": {
                "size_xyz": list(image.size_xyz),
                "spacing": [round(float(s), 3) for s in image.spacing],
                "orientation": image.orientation_code(),
                "modality": modality,
                "stack_axis": "xyz"[first.stack_axis_xyz],
                "col_axis": "xyz"[first.col_axis_xyz],
                "row_axis": "xyz"[first.row_axis_xyz],
            },
            "axes": {"stack": first.stack_axis_xyz, "col": first.col_axis_xyz, "row": first.row_axis_xyz},
            "default_alpha": masks[0][1]["layer"].alpha if masks else 0.4,
            "n_frames": len(frames),
        }
        html = _env().get_template("viewer.html.j2").render(**ctx)
        return RenderResult(
            html.encode("utf-8"),
            "text/html",
            ".html",
            {"frames": len(frames), "window": wlabel, "legend": legend},
        )
