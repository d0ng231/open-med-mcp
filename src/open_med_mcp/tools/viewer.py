"""Viewer tools: PNG views for the agent, HTML viewers and reports for humans."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import Plane
from open_med_mcp.core.prompts import Prompt
from open_med_mcp.tools._common import (
    Window,
    labels_for,
    load_image_cached,
    load_mask_cached,
    resolve,
    resolve_output,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.report import write_report as _write_report
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import display_path, record_provenance

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _mask_items(
    image: Any,
    masks: list[str] | None,
    mode: str,
    alpha: float,
    labels: list[int] | None,
    colors: list[str] | None,
) -> tuple[list[MaskLayer], list[tuple[Any, dict[str, Any]]]]:
    layers: list[MaskLayer] = []
    items = []
    for i, mpath in enumerate(masks or []):
        mp = resolve(mpath)
        m = load_mask_cached(mp, image)
        layer = MaskLayer(
            path=str(mp),
            name=mp.name,
            mode=mode,
            alpha=alpha,
            labels=labels,
            color=(colors[i] if colors and i < len(colors) else None),
        )
        layers.append(layer)
        items.append((m, {"layer": layer, "labels": labels_for(mp)}))
    return layers, items


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def render_view(
        image: str,
        masks: Annotated[
            list[str] | None, Field(description="Label maps to overlay (same geometry as the image)")
        ] = None,
        plane: Plane = "axial",
        layout: Annotated[
            Literal["single", "three-plane", "montage"],
            Field(description="single slice, three orthogonal planes, or a montage of several slices"),
        ] = "single",
        slices: Annotated[
            list[int] | None,
            Field(
                description="Slice indices (native, along `plane`); default: largest mask area or the middle"
            ),
        ] = None,
        n_slices: Annotated[
            int, Field(ge=1, le=36, description="montage: number of slices spread over the mask extent")
        ] = 9,
        window: Annotated[
            Window,
            Field(
                description="Preset (soft-tissue, lung, bone, brain, liver ...), {center,width}, [lower, upper] or 'auto'"
            ),
        ] = None,
        mode: Annotated[Literal["fill", "contour", "both"], Field(description="How to draw masks")] = "both",
        alpha: Annotated[float, Field(ge=0, le=1)] = 0.35,
        labels: Annotated[list[int] | None, Field(description="Only draw these label ids")] = None,
        colors: Annotated[
            list[str] | None, Field(description="Hex colour per mask file (default: palette per label)")
        ] = None,
        prompts: Annotated[
            list[Prompt] | None,
            Field(description="Draw these prompts (to verify placement before segmenting)"),
        ] = None,
        grid: Annotated[bool, Field(description="Tick labels in native voxel coordinates")] = True,
        crop_to_mask: Annotated[bool, Field(description="Zoom to the mask bounding box")] = False,
        center_xyz: Annotated[
            list[float] | None, Field(description="three-plane: crossing point (native voxels)")
        ] = None,
        title: str | None = None,
        max_px: Annotated[int, Field(ge=200, le=3000)] = 900,
        renderer: Annotated[
            str, Field(description="Renderer name: png (default) or html, or a plugin")
        ] = "png",
        output: Annotated[
            str | None, Field(description="Save the figure here (default: omm_outputs/views/...)")
        ] = None,
    ) -> CallToolResult:
        """Render the image (with masks and prompts) so you can look at it. Tick labels are native
        voxel coordinates. Use layout='three-plane' for orientation, 'montage' to check the whole
        stack, and crop_to_mask=True for detail."""
        p = resolve(image)
        img = load_image_cached(p)
        layers, items = _mask_items(img, masks, mode, alpha, labels, colors)
        spec = ViewSpec(
            image=str(p),
            masks=layers,
            plane=plane,
            layout=layout,
            slices=slices,
            n_slices=n_slices,
            window=window,
            prompts=prompts or [],
            grid=grid,
            crop_to_mask=crop_to_mask,
            center_xyz=center_xyz,
            title=title,
            max_px=max_px,
        )
        render = get_renderer(renderer).render(img, items, spec)
        settings = get_settings()
        stamp = datetime.now().strftime("%H%M%S")
        default = (
            settings.outputs_dir / "views" / f"{p.name.split('.')[0]}_{plane}_{layout}_{stamp}{render.suffix}"
        )
        out = resolve_output(output, default)
        render.save(out)
        payload = {"output": display_path(out), "renderer": renderer, **render.info}
        images = [render] if render.mime_type.startswith("image/") else []
        return result(payload, images)

    @server.tool()
    @tool_errors
    def export_viewer(
        image: str,
        masks: list[str] | None = None,
        output: Annotated[
            str | None,
            Field(description="HTML file to write (default omm_outputs/views/<image>_viewer.html)"),
        ] = None,
        plane: Plane = "axial",
        window: Window = None,
        alpha: float = 0.4,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Write a self-contained interactive HTML slice viewer (scroll, overlay toggle, click to
        read prompt coordinates) for a human to open in a browser."""
        p = resolve(image)
        img = load_image_cached(p)
        layers, items = _mask_items(img, masks, "fill", alpha, None, None)
        spec = ViewSpec(image=str(p), masks=layers, plane=plane, window=window, title=title, max_px=768)
        render = get_renderer("html").render(img, items, spec)
        out = resolve_output(
            output, get_settings().outputs_dir / "views" / f"{p.name.split('.')[0]}_viewer.html"
        )
        render.save(out)
        record_provenance(
            "export_viewer", {"image": display_path(p), "masks": masks}, {"output": display_path(out)}
        )
        return {
            "output": display_path(out),
            **render.info,
            "hint": "open the file in a browser; drag on the image to get a box prompt",
        }

    @server.tool()
    @tool_errors
    def write_report(
        title: str,
        sections: Annotated[
            list[dict[str, Any]],
            Field(
                description="[{'heading': str, 'text': markdown, 'figures': [png paths], 'table': [row dicts], 'columns': [...]}, ...]"
            ),
        ],
        output: Annotated[
            str | None,
            Field(
                description="Markdown file to write; an .html twin with embedded figures is written next to it"
            ),
        ] = None,
        metadata: Annotated[
            dict[str, Any] | None, Field(description="Key/value table at the top (image, model, run_dir ...)")
        ] = None,
    ) -> dict[str, Any]:
        """Assemble a Markdown + HTML report from text, tables and figures produced by other tools."""
        settings = get_settings()
        out = resolve_output(
            output,
            settings.outputs_dir
            / "reports"
            / f"{title.strip().lower().replace(' ', '_')[:40] or 'report'}.md",
        )
        secs = []
        for s in sections:
            s = dict(s)
            s["figures"] = [str(resolve(f)) for f in s.get("figures") or []]
            secs.append(s)
        paths = _write_report(title, secs, out, metadata)
        record_provenance("write_report", {"title": title}, paths)
        return {k: display_path(resolve(v)) for k, v in paths.items()}

    @server.tool(annotations=READ_ONLY)
    def list_renderers() -> dict[str, Any]:
        """Available viewer renderers (built-in png/html plus any installed plugins)."""
        return {"renderers": list_renderers()}
