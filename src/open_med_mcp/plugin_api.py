"""Public helpers for plug-ins and custom tools (stable across 0.x releases).

Example ``omm_plugins/my_tool.py``::

    from open_med_mcp.plugin_api import load_image_cached, load_mask_cached, resolve, result, tool_errors

    def register(server):
        @server.tool()
        @tool_errors
        def mask_volume_ml(mask: str) -> dict:
            \"\"\"Foreground volume of a mask in mL.\"\"\"
            m = load_mask_cached(resolve(mask))
            return {"volume_ml": float((m.array != 0).sum() * m.voxel_volume_mm3() / 1000)}
"""

from __future__ import annotations

from open_med_mcp.config import Settings, get_settings
from open_med_mcp.core.image import MedicalImage, load_mask, save_mask, write_labels_sidecar
from open_med_mcp.core.masks import mask_stats, postprocess
from open_med_mcp.core.metrics import compare_masks
from open_med_mcp.tools._common import (
    Window,
    error_result,
    image_block,
    labels_for,
    load_image_cached,
    load_mask_cached,
    resolve,
    resolve_output,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import RenderResult, get_renderer, register_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import display_path, new_run_dir, record_provenance


def render_preview(
    image: MedicalImage,
    mask_path: str | None = None,
    title: str | None = None,
    layout: str = "three-plane",
    window: Window = None,
) -> RenderResult:
    """Render the standard preview an agent sees (three planes with an optional mask)."""
    layers = []
    items = []
    if mask_path:
        p = resolve(mask_path)
        layer = MaskLayer(path=str(p), name=p.name)
        layers.append(layer)
        items.append((load_mask_cached(p, image), {"layer": layer, "labels": labels_for(p)}))
    spec = ViewSpec(
        image=str(image.path),
        masks=layers,
        layout=layout if not image.is_2d else "single",
        window=window,
        title=title,
        max_px=get_settings().preview_max_px,
    )  # type: ignore[arg-type]
    return get_renderer("png").render(image, items, spec)


__all__ = [
    "MaskLayer",
    "MedicalImage",
    "RenderResult",
    "Settings",
    "ViewSpec",
    "Window",
    "compare_masks",
    "display_path",
    "error_result",
    "get_renderer",
    "get_settings",
    "image_block",
    "labels_for",
    "load_image_cached",
    "load_mask",
    "load_mask_cached",
    "mask_stats",
    "new_run_dir",
    "postprocess",
    "record_provenance",
    "register_renderer",
    "render_preview",
    "resolve",
    "resolve_output",
    "result",
    "save_mask",
    "tool_errors",
    "write_labels_sidecar",
]
