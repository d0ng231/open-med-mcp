"""open-med-mcp: an open, modular MCP server for medical image analysis agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("open-med-mcp")
except PackageNotFoundError:  # pragma: no cover - editable/dev checkouts
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
