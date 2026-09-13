"""Shared helpers for tool modules: path handling, image cache, result packaging."""

from __future__ import annotations

import base64
import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, ImageContent, TextContent

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import MedicalImage, load_mask, read_labels_sidecar
from open_med_mcp.viewer.registry import RenderResult
from open_med_mcp.workspace import display_path, resolve_path

_CACHE: dict[tuple[str, float], MedicalImage] = {}
_CACHE_MAX = 6


def load_image_cached(path: Path) -> MedicalImage:
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    if key in _CACHE:
        return _CACHE[key]
    img = MedicalImage.load(path)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = img
    return img


def load_mask_cached(path: Path, like: MedicalImage | None = None) -> MedicalImage:
    key = (str(path) + "#mask", path.stat().st_mtime if path.exists() else 0.0)
    if key in _CACHE:
        m = _CACHE[key]
    else:
        m = load_mask(path, None)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = m
    if like is not None and m.shape_zyx != like.shape_zyx:
        raise ValueError(
            f"mask {display_path(path)} has shape {m.shape_zyx} but the image has {like.shape_zyx}"
        )
    return m


def labels_for(mask_path: Path, override: dict[int, str] | None = None) -> dict[int, str]:
    labels = read_labels_sidecar(mask_path) or {}
    if override:
        labels.update({int(k): v for k, v in override.items()})
    return labels


def resolve(path: str, must_exist: bool = True) -> Path:
    return resolve_path(path, get_settings(), must_exist=must_exist)


def resolve_output(path: str | None, default: Path) -> Path:
    if not path:
        default.parent.mkdir(parents=True, exist_ok=True)
        return default
    p = resolve_path(path, get_settings(), must_exist=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def json_text(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def image_block(render: RenderResult | bytes, mime_type: str = "image/png") -> ImageContent:
    data = render.data if isinstance(render, RenderResult) else render
    mt = render.mime_type if isinstance(render, RenderResult) else mime_type
    return ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mime_type=mt)


def result(
    data: dict[str, Any], images: list[RenderResult | bytes] | None = None, text: str | None = None
) -> CallToolResult:
    """Package a JSON payload (as text + structured content) with optional preview images."""
    content: list[Any] = [TextContent(type="text", text=text or json_text(data))]
    for im in images or []:
        content.append(image_block(im))
    return CallToolResult(content=content, structured_content=data)


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def tool_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn exceptions into ``is_error`` results with a readable message (the agent can retry)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return error_result(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")

    return wrapper
