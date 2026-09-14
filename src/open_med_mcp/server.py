"""MCP server factory: logging (stderr + optional file), core tools, plug-ins."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Any

from mcp.server import MCPServer

from open_med_mcp import __version__
from open_med_mcp.config import Settings, get_settings, set_settings
from open_med_mcp.plugins import load_plugins, loaded_plugins
from open_med_mcp.tools import register_all

INSTRUCTIONS = """open-med-mcp gives you local medical image analysis tools: inspect images, run
segmentation / classification models (automatic like TotalSegmentator or lungmask, promptable like
MedSAM2 / SAM 2 with boxes or VoxTell with text, classical thresholds), pre-process (resample,
reorient, N4, register), post-process and measure masks, render views you can look at, and write
reports.

Workflow: get_conventions -> inspect_image -> list_guidelines/get_guideline -> list_models ->
segment/run_model -> render_view (look at it!) -> refine/postprocess_mask -> mask_stats/compare_masks
-> write_report. All coordinates are native voxel indices (x, y, z) as reported by inspect_image.
Long runs report progress; use wait=false + get_job when a run may exceed your tool timeout.
Never call a model result a clinical finding; report what was measured and how it was checked."""


def configure_logging(settings: Settings) -> None:
    """Log to stderr (never stdout: stdio transport uses it) and to ``<home>/logs/server.log``."""
    root = logging.getLogger("open_med_mcp")
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.propagate = False  # the SDK installs its own root handler; avoid duplicate lines
    if any(getattr(h, "_omm", False) for h in root.handlers):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    stderr._omm = True  # type: ignore[attr-defined]
    root.addHandler(stderr)
    log_file = settings.log_file
    if log_file is None:
        settings.ensure_dirs()
        log_file = str(settings.logs_dir / "server.log")
    if log_file:
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            fh._omm = True  # type: ignore[attr-defined]
            root.addHandler(fh)
        except OSError:
            root.warning("cannot open log file %s", log_file)


def create_server(settings: Settings | None = None, plugins: bool = True) -> MCPServer:
    if settings is not None:
        set_settings(settings)
    settings = get_settings()
    settings.ensure_dirs()
    configure_logging(settings)
    log = logging.getLogger("open_med_mcp.server")
    server = MCPServer(
        "open-med-mcp",
        title="open-med-mcp",
        description="Open, modular medical image analysis tools for AI agents",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/d0ng231/open-med-mcp",
        version=__version__,
        log_level=settings.log_level
        if settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        else "INFO",  # type: ignore[arg-type]
    )
    register_all(server)
    loaded: list[Any] = load_plugins(server, settings) if plugins else []

    @server.tool(annotations=None)
    def list_plugins() -> dict[str, Any]:
        """Plug-ins loaded into this server (workspace omm_plugins/, OMM_PLUGIN_DIRS, entry points)."""
        return {
            "plugins": [
                {"name": p.name, "source": p.source, "kind": p.kind, "tools": p.tools_added, "error": p.error}
                for p in loaded_plugins()
            ],
            "plugin_dirs": [
                str(d) for d in (settings.workspace / "omm_plugins", *settings.extra_plugin_dirs)
            ],
        }

    log.info(
        "open-med-mcp %s ready: workspace=%s runner=%s plugins=%d",
        __version__,
        settings.workspace,
        settings.runner,
        len(loaded),
    )
    return server
