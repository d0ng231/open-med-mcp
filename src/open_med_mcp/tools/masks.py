"""Mask tools: statistics, post-processing, comparison and prompt extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import Plane, save_mask, write_labels_sidecar
from open_med_mcp.core.masks import mask_stats as _mask_stats
from open_med_mcp.core.masks import mask_to_prompts as _mask_to_prompts
from open_med_mcp.core.masks import postprocess
from open_med_mcp.core.metrics import compare_masks as _compare_masks
from open_med_mcp.tools._common import (
    labels_for,
    load_image_cached,
    load_mask_cached,
    resolve,
    resolve_output,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import display_path, record_provenance

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def mask_stats(
        mask: Annotated[str, Field(description="Label map path")],
        image: Annotated[
            str | None,
            Field(description="Image with the same geometry, for intensity statistics inside each label"),
        ] = None,
        labels: Annotated[list[int] | None, Field(description="Restrict to these label ids")] = None,
    ) -> dict[str, Any]:
        """Per-label voxel counts, volumes (mL), bounding boxes, centroids, connected components,
        slice ranges and (with image) intensity statistics."""
        mp = resolve(mask)
        img = load_image_cached(resolve(image)) if image else None
        m = load_mask_cached(mp, img)
        stats = _mask_stats(m, img, labels, labels_for(mp))
        stats["mask"] = display_path(mp)
        return stats

    @server.tool()
    @tool_errors
    def postprocess_mask(
        mask: str,
        ops: Annotated[
            list[dict[str, Any] | str],
            Field(
                description="Operations in order, e.g. ['largest_component', 'fill_holes', {'op': 'remove_small', 'min_mm3': 100}, {'op': 'keep_labels', 'labels': [5]}, {'op': 'close', 'radius': 1}, 'binarize']"
            ),
        ],
        output: Annotated[
            str | None, Field(description="Destination path (default: next to the input with a _pp suffix)")
        ] = None,
        image: Annotated[str | None, Field(description="Image for a preview of the result")] = None,
        preview: bool = False,
    ) -> CallToolResult:
        """Clean up a label map with morphological/connected-component operations."""
        mp = resolve(mask)
        m = load_mask_cached(mp)
        new, log = postprocess(m.array, ops, m.spacing)
        name = mp.name
        for sfx in (".nii.gz", ".nii", ".png", ".nrrd", ".mha"):
            if name.endswith(sfx):
                name = name[: -len(sfx)] + "_pp" + sfx
                break
        out = resolve_output(output, mp.with_name(name))
        save_mask(new, m, out)
        labels = labels_for(mp)
        if labels:
            write_labels_sidecar(out, {k: v for k, v in labels.items() if (new == k).any()})
        payload: dict[str, Any] = {
            "output": display_path(out),
            "ops": log,
            "before_voxels": int((m.array != 0).sum()),
            "after_voxels": int((new != 0).sum()),
        }
        images = []
        if preview and image:
            img = load_image_cached(resolve(image))
            mm = load_mask_cached(out, img)
            layer = MaskLayer(path=str(out), name="postprocessed")
            spec = ViewSpec(
                image=str(img.path),
                masks=[layer],
                layout="three-plane" if not img.is_2d else "single",
                max_px=get_settings().preview_max_px,
            )
            images.append(get_renderer("png").render(img, [(mm, {"layer": layer, "labels": labels})], spec))
        record_provenance(
            "postprocess_mask", {"mask": display_path(mp), "ops": ops}, {"output": display_path(out)}
        )
        return result(payload, images)

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def compare_masks(
        mask_a: Annotated[str, Field(description="Prediction / first mask")],
        mask_b: Annotated[str, Field(description="Reference / second mask")],
        image: Annotated[str | None, Field(description="Image for the visual comparison")] = None,
        labels: list[int] | None = None,
        surface_metrics: Annotated[
            bool, Field(description="Compute Hausdorff/HD95/ASSD (slower on large masks)")
        ] = True,
        preview: bool = True,
    ) -> CallToolResult:
        """Dice, IoU, precision/recall, volume difference and surface distances between two label maps,
        with an optional contour overlay of both."""
        pa, pb = resolve(mask_a), resolve(mask_b)
        img = load_image_cached(resolve(image)) if image else None
        a = load_mask_cached(pa, img)
        b = load_mask_cached(pb, img)
        if a.shape_zyx != b.shape_zyx:
            raise ValueError(f"shapes differ: {a.shape_zyx} vs {b.shape_zyx}")
        metrics = _compare_masks(a.array, b.array, a.spacing, labels, surface_metrics)
        metrics.update({"mask_a": display_path(pa), "mask_b": display_path(pb)})
        images = []
        if preview and img is not None:
            la = MaskLayer(path=str(pa), name=f"A: {pa.name}", color="#ff3b30", mode="contour", alpha=0.0)
            lb = MaskLayer(path=str(pb), name=f"B: {pb.name}", color="#0a84ff", mode="contour", alpha=0.0)
            spec = ViewSpec(
                image=str(img.path),
                masks=[la, lb],
                layout="three-plane" if not img.is_2d else "single",
                title="A (red) vs B (blue)",
                max_px=get_settings().preview_max_px,
            )
            images.append(get_renderer("png").render(img, [(a, {"layer": la}), (b, {"layer": lb})], spec))
        return result(metrics, images)

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def mask_to_prompts(
        mask: str,
        plane: Plane = "axial",
        label: Annotated[
            int | None, Field(description="Use only this label (default: all foreground)")
        ] = None,
        slice_index: Annotated[
            int | str,
            Field(description="Slice to place the prompt on, or 'auto' = slice with the largest area"),
        ] = "auto",
        margin: Annotated[int, Field(description="Box margin in voxels")] = 2,
    ) -> dict[str, Any]:
        """Turn any mask (coarse threshold result, another model's output) into box/point prompts
        for a promptable model - the usual way to chain automatic and promptable models."""
        mp = resolve(mask)
        m = load_mask_cached(mp)
        out = _mask_to_prompts(m, plane, label, slice_index, margin)
        prompts = [{"type": "box", "coords": b} for b in out.get("boxes", [])] + [
            {"type": "point", "coords": p, "label": 1} for p in out.get("points", [])
        ]
        return {**out, "prompts": prompts, "mask": display_path(Path(mp))}
