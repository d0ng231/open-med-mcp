"""Renderer registry. Built-in: ``png`` (matplotlib) and ``html`` (self-contained slice viewer).

Third-party renderers register through the ``open_med_mcp.renderers`` entry-point group or with the
``@register_renderer("name")`` decorator (see docs/viewer.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

from open_med_mcp.core.image import MedicalImage
from open_med_mcp.viewer.spec import ViewSpec


@dataclass
class RenderResult:
    data: bytes
    mime_type: str
    suffix: str
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path


class Renderer(Protocol):
    name: str

    def render(
        self, image: MedicalImage, masks: list[tuple[MedicalImage, dict[str, Any]]], spec: ViewSpec
    ) -> RenderResult: ...


_RENDERERS: dict[str, Callable[[], Renderer]] = {}
_LOADED_ENTRY_POINTS = False


def register_renderer(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _RENDERERS[name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def _load_entry_points() -> None:
    global _LOADED_ENTRY_POINTS
    if _LOADED_ENTRY_POINTS:
        return
    _LOADED_ENTRY_POINTS = True
    try:
        eps = entry_points(group="open_med_mcp.renderers")
    except TypeError:  # pragma: no cover - py<3.10 API
        eps = entry_points().get("open_med_mcp.renderers", [])  # type: ignore[assignment]
    for ep in eps:
        try:
            obj = ep.load()
            _RENDERERS.setdefault(ep.name, obj)
        except Exception:
            continue


def list_renderers() -> list[str]:
    _ensure_builtins()
    _load_entry_points()
    return sorted(_RENDERERS)


def get_renderer(name: str = "png") -> Renderer:
    _ensure_builtins()
    _load_entry_points()
    try:
        return _RENDERERS[name]()
    except KeyError:
        raise KeyError(f"unknown renderer {name!r}; available: {', '.join(sorted(_RENDERERS))}") from None


def _ensure_builtins() -> None:
    if "png" not in _RENDERERS:
        from open_med_mcp.viewer import html, png  # noqa: F401  (registers on import)
