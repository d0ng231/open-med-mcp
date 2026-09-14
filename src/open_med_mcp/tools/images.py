"""Workspace and image inspection tools."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import (
    DICOM_SUFFIXES,
    IMAGE2D_SUFFIXES,
    VOLUME_SUFFIXES,
    MedicalImage,
    detect_format,
    file_suffix,
)
from open_med_mcp.tools._common import Window, load_image_cached, resolve, resolve_output, result, tool_errors
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.spec import ViewSpec
from open_med_mcp.workspace import display_path, record_provenance

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _classify(p: Path) -> str:
    if p.is_dir():
        try:
            if any(
                f.is_file() and (file_suffix(f) in DICOM_SUFFIXES or detect_format(f) == "dicom")
                for f in list(p.iterdir())[:20]
            ):
                return "dicom-series"
        except OSError:
            pass
        return "dir"
    sfx = file_suffix(p)
    name = p.name.lower()
    if sfx in VOLUME_SUFFIXES or sfx in IMAGE2D_SUFFIXES:
        return "mask" if any(k in name for k in ("mask", "seg", "label")) else "image"
    if sfx in DICOM_SUFFIXES:
        return "dicom"
    if sfx in (".json", ".md", ".html", ".txt", ".csv", ".yaml", ".yml"):
        return "text"
    return "other"


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    def list_workspace(
        subdir: Annotated[
            str, Field(description="Sub-directory of the workspace to list ('' = workspace root)")
        ] = "",
        pattern: Annotated[str, Field(description="Glob pattern, e.g. '*.nii.gz'")] = "*",
        recursive: bool = False,
        max_items: Annotated[int, Field(ge=1, le=2000)] = 300,
    ) -> dict[str, Any]:
        """List images, masks, DICOM folders and other files in the workspace (paths are relative to it)."""
        settings = get_settings()
        root = resolve(subdir or ".")
        it = root.rglob("*") if recursive else root.iterdir()
        items = []
        truncated = False
        for p in it:
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if not fnmatch.fnmatch(p.name, pattern):
                continue
            try:
                size = round(p.stat().st_size / 1e6, 3) if p.is_file() else None
            except OSError:
                size = None
            items.append({"path": display_path(p), "type": _classify(p), "size_mb": size})
            if len(items) >= max_items:
                truncated = True
                break
        items.sort(key=lambda e: e["path"])
        return {
            "workspace": str(settings.workspace),
            "outputs_dir": display_path(settings.outputs_dir),
            "count": len(items),
            "truncated": truncated,
            "items": items,
        }

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def inspect_image(
        path: Annotated[
            str,
            Field(
                description="Image file (NIfTI, NRRD, MetaImage, PNG/JPEG/TIFF, DICOM file) or a DICOM series folder"
            ),
        ],
        preview: Annotated[bool, Field(description="Return a preview PNG of the middle slice")] = True,
        window: Annotated[
            Window, Field(description="Preview window: preset name, {center,width}, [lower, upper] or 'auto'")
        ] = None,
    ) -> CallToolResult:
        """Describe an image: geometry (size, spacing, orientation, planes), intensity statistics,
        modality guess and metadata. Call this first for every new image; all coordinates in later
        calls refer to the reported native voxel index space."""
        p = resolve(path)
        img = load_image_cached(p)
        info = img.describe()
        info["path"] = display_path(p)
        info["hints"] = _hints(img)
        images = []
        if preview:
            spec = ViewSpec(
                image=str(p),
                layout="three-plane" if not img.is_2d else "single",
                window=window,
                max_px=min(get_settings().preview_max_px, 800),
                show_legend=False,
            )
            images.append(get_renderer("png").render(img, [], spec))
        return result(info, images)

    @server.tool()
    @tool_errors
    def convert_image(
        path: Annotated[str, Field(description="Source image or DICOM series folder")],
        output: Annotated[
            str | None, Field(description="Destination file; default <omm_outputs>/converted/<name>.nii.gz")
        ] = None,
    ) -> dict[str, Any]:
        """Convert an image (e.g. a DICOM series folder) to NIfTI (or another format by extension)."""
        p = resolve(path)
        img = MedicalImage.load(p)
        settings = get_settings()
        default = settings.outputs_dir / "converted" / (p.stem.replace(".nii", "") + ".nii.gz")
        out = resolve_output(output, default)
        img.save(out)
        record_provenance("convert_image", {"path": display_path(p)}, {"output": display_path(out)})
        return {
            "output": display_path(out),
            "size_xyz": list(img.size_xyz),
            "spacing_xyz_mm": [float(s) for s in img.spacing],
            "format": img.format,
            "metadata": {k: v for k, v in img.metadata.items() if k != "channels_last"},
        }


def _hints(img: MedicalImage) -> list[str]:
    hints = []
    mod = img.modality_guess()
    if mod == "CT":
        hints.append(
            "CT in Hounsfield units: use window presets soft-tissue/lung/bone; totalsegmentator gives anatomy without prompts."
        )
    elif mod == "RGB":
        hints.append(
            "RGB photo/endoscopy/dermoscopy/histology: use `sam2` (or `medsam2`) with a box; window is ignored."
        )
    elif mod.startswith("MR"):
        hints.append(
            "Intensities are not calibrated: use window 'auto' and derive thresholds from the reported percentiles."
        )
    if mod == "XR/2D":
        hints.append(
            "2D radiograph: classify_image with torchxrayvision for findings; medsam2/sam2 with a box for 2D segmentation."
        )
    if img.ndim == 3:
        sp = img.spacing
        if max(sp) / min(sp) > 2.5:
            hints.append(
                f"Anisotropic voxels {[round(float(s), 2) for s in sp]} mm: propagate along the plane with the fewest slices for best results."
            )
        if img.metadata.get("collapsed_singleton_z"):
            hints.append("The file had a singleton third dimension and was treated as 2D.")
    return hints
