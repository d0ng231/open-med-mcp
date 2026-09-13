"""Viewer: composable rendering of images + masks + prompts to PNG (for agents) and HTML (for humans)."""

from open_med_mcp.viewer.registry import get_renderer, list_renderers, register_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec

__all__ = ["ViewSpec", "MaskLayer", "get_renderer", "list_renderers", "register_renderer"]
