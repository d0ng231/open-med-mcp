"""MCP tool registration. Each module exposes ``register(server)``."""

from __future__ import annotations

from mcp.server import MCPServer

from open_med_mcp.tools import clinical, guidelines, images, masks, models, processing, viewer


def register_all(server: MCPServer) -> None:
    images.register(server)
    models.register(server)
    masks.register(server)
    processing.register(server)
    clinical.register(server)
    viewer.register(server)
    guidelines.register(server)
