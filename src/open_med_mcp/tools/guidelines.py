"""Guideline tools, resources and prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.types import ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.guidelines.loader import get_library
from open_med_mcp.models.registry import get_registry

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
CONVENTIONS = Path(__file__).resolve().parent.parent / "conventions.md"


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    def list_guidelines(
        query: Annotated[str | None, Field(description="Free-text filter on name/title/summary/tags")] = None,
        tags: list[str] | None = None,
        modality: Annotated[str | None, Field(description="CT, MR, US, XR, RGB ...")] = None,
    ) -> dict[str, Any]:
        """List preset and user guidelines (step-by-step protocols for common tasks)."""
        lib = get_library(get_settings(), reload=True)
        return {"guidelines": [g.summary_dict() for g in lib.search(query, tags, modality)]}

    @server.tool(annotations=READ_ONLY)
    def get_guideline(name: Annotated[str, Field(description="Guideline name from list_guidelines")]) -> str:
        """Return the full Markdown text of a guideline. Follow it step by step and report deviations."""
        lib = get_library(get_settings(), reload=True)
        return lib.get(name).render()

    @server.tool(annotations=READ_ONLY)
    def get_conventions() -> str:
        """Coordinate, prompt, mask and unit conventions used by every tool (read once per session)."""
        return CONVENTIONS.read_text(encoding="utf-8")

    @server.resource(
        "guideline://{name}",
        name="guideline",
        description="Guideline Markdown by name",
        mime_type="text/markdown",
    )
    def guideline_resource(name: str) -> str:
        try:
            return get_library(get_settings(), reload=True).get(name).render()
        except KeyError as exc:
            raise ResourceNotFoundError(str(exc)) from None

    @server.resource(
        "omm://conventions",
        name="conventions",
        description="Coordinate and prompt conventions",
        mime_type="text/markdown",
    )
    def conventions_resource() -> str:
        return CONVENTIONS.read_text(encoding="utf-8")

    @server.resource(
        "model://{name}", name="model", description="Model manifest as JSON", mime_type="application/json"
    )
    def model_resource(name: str) -> str:
        try:
            return get_registry(get_settings()).get(name).model_dump_json(indent=2, exclude={"adapter_dir"})
        except KeyError as exc:
            raise ResourceNotFoundError(str(exc)) from None

    # one prompt per guideline so clients expose them as slash commands
    for g in get_library(get_settings()).all():
        _register_prompt(server, g.name, g.title, g.summary)


def _register_prompt(server: MCPServer, name: str, title: str, summary: str) -> None:
    def prompt_fn(
        image: Annotated[str, Field(description="Image path to work on (optional)")] = "",
        goal: Annotated[str, Field(description="What to segment / measure (optional)")] = "",
    ) -> str:
        text = get_library(get_settings(), reload=True).get(name).render()
        task = []
        if image:
            task.append(f"Image: `{image}`")
        if goal:
            task.append(f"Goal: {goal}")
        header = ("\n".join(task) + "\n\n") if task else ""
        return f"{header}Follow this guideline with the open-med-mcp tools. Start with `inspect_image` and `get_conventions` if you have not read them yet.\n\n{text}"

    prompt_fn.__name__ = name.replace("-", "_")
    prompt_fn.__doc__ = summary or title
    server.prompt(name=name, title=title, description=summary or title)(prompt_fn)
