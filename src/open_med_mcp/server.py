"""MCP server factory."""

from __future__ import annotations

from mcp.server import MCPServer

from open_med_mcp import __version__
from open_med_mcp.config import Settings, get_settings, set_settings
from open_med_mcp.tools import register_all

INSTRUCTIONS = """open-med-mcp gives you local medical image analysis tools: inspect images, run
segmentation models (automatic like TotalSegmentator, promptable like MedSAM2/SAM 2, classical
thresholds), post-process and measure masks, render views you can look at, and write reports.

Workflow: get_conventions -> inspect_image -> list_guidelines/get_guideline -> list_models ->
segment/run_model -> render_view (look at it!) -> refine/postprocess_mask -> mask_stats/compare_masks
-> write_report. All coordinates are native voxel indices (x, y, z) as reported by inspect_image.
Never call a model result a clinical finding; report what was measured and how it was checked."""


def create_server(settings: Settings | None = None) -> MCPServer:
    if settings is not None:
        set_settings(settings)
    settings = get_settings()
    settings.ensure_dirs()
    server = MCPServer(
        "open-med-mcp",
        title="open-med-mcp",
        description="Open, modular medical image analysis tools for AI agents",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/d0ng231/open-med-mcp",
        version=__version__,
    )
    register_all(server)
    return server
