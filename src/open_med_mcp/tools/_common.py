"""Shared helpers for tool modules (and plug-ins): path handling, a bounded thread-safe image cache,
result packaging with size-capped inline images, and error wrapping."""

from __future__ import annotations

import base64
import functools
import io
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, ImageContent, TextContent

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import MedicalImage, load_mask, read_labels_sidecar
from open_med_mcp.viewer.registry import RenderResult
from open_med_mcp.workspace import display_path, resolve_path

log = logging.getLogger("open_med_mcp.tools")

#: ``str | {"center","width"} | {"lower","upper"} | [lower, upper] | None`` - documented once here.
Window = str | dict[str, float] | list[float] | None

_CACHE: OrderedDict[tuple[str, float], MedicalImage] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple[str, float]) -> MedicalImage | None:
    with _CACHE_LOCK:
        img = _CACHE.get(key)
        if img is not None:
            _CACHE.move_to_end(key)
        return img


def _cache_put(key: tuple[str, float], img: MedicalImage) -> None:
    budget = get_settings().cache_mb * 1_000_000
    with _CACHE_LOCK:
        _CACHE[key] = img
        _CACHE.move_to_end(key)
        while len(_CACHE) > 1 and sum(v.array.nbytes for v in _CACHE.values()) > budget:
            _CACHE.popitem(last=False)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def load_image_cached(path: Path) -> MedicalImage:
    """Load an image (any supported format), reusing the last few loaded volumes."""
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    img = _cache_get(key)
    if img is None:
        img = MedicalImage.load(path)
        _cache_put(key, img)
    return img


def load_mask_cached(path: Path, like: MedicalImage | None = None) -> MedicalImage:
    key = (str(path) + "#mask", path.stat().st_mtime if path.exists() else 0.0)
    m = _cache_get(key)
    if m is None:
        m = load_mask(path, None)
        _cache_put(key, m)
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


def _shrink_png(data: bytes, max_bytes: int) -> bytes:
    """Downscale a PNG until it fits ``max_bytes`` (clients reject very large inline images)."""
    from PIL import Image as PILImage

    for _ in range(6):
        if len(data) <= max_bytes:
            return data
        with PILImage.open(io.BytesIO(data)) as im:
            w, h = im.size
            im2 = im.resize((max(int(w * 0.75), 64), max(int(h * 0.75), 64)), PILImage.LANCZOS)
            buf = io.BytesIO()
            im2.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
    return data


def image_block(render: RenderResult | bytes, mime_type: str = "image/png") -> ImageContent:
    data = render.data if isinstance(render, RenderResult) else render
    mt = render.mime_type if isinstance(render, RenderResult) else mime_type
    if mt == "image/png":
        data = _shrink_png(data, get_settings().max_image_bytes)
    return ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mime_type=mt)


def result(
    data: dict[str, Any], images: list[RenderResult | bytes] | None = None, text: str | None = None
) -> CallToolResult:
    """Package a JSON payload (as text + structured content) with optional preview images.

    Images are omitted when ``OMM_RETURN_IMAGES=0`` (text-only clients); the JSON always carries
    the saved file paths, so nothing is lost."""
    content: list[Any] = [TextContent(type="text", text=text or json_text(data))]
    if get_settings().return_images:
        for im in images or []:
            content.append(image_block(im))
    return CallToolResult(content=content, structured_content=data)


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def tool_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn exceptions into ``is_error`` results with a readable message (the agent can retry).

    Works for sync and async tool functions."""
    import inspect

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
                return error_result(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")

        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
            return error_result(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")

    return wrapper
