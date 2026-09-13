"""Workspace helpers: path resolution, run directories and a provenance log."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_med_mcp.config import Settings, get_settings

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class WorkspaceError(ValueError):
    """Raised for invalid paths (outside the workspace, missing files ...)."""


def resolve_path(
    path: str | os.PathLike[str], settings: Settings | None = None, must_exist: bool = True
) -> Path:
    """Resolve ``path`` relative to the workspace and validate it.

    Absolute paths are accepted. When ``allow_outside_workspace`` is false the resolved
    path must live below the workspace root.
    """
    settings = settings or get_settings()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = settings.workspace / p
    p = p.resolve()
    if not settings.allow_outside_workspace:
        try:
            p.relative_to(settings.workspace.resolve())
        except ValueError as exc:
            raise WorkspaceError(f"{p} is outside the workspace {settings.workspace}") from exc
    if must_exist and not p.exists():
        raise WorkspaceError(f"path does not exist: {p}")
    return p


def display_path(path: Path, settings: Settings | None = None) -> str:
    """Return ``path`` relative to the workspace when possible (shorter for the agent)."""
    settings = settings or get_settings()
    try:
        return str(Path(path).resolve().relative_to(settings.workspace.resolve()))
    except ValueError:
        return str(path)


def slugify(text: str, max_len: int = 40) -> str:
    slug = _SAFE.sub("-", text).strip("-")
    return (slug or "run")[:max_len]


def new_run_dir(kind: str, settings: Settings | None = None) -> Path:
    """Create ``<workspace>/omm_outputs/<timestamp>-<kind>-<id>/`` and return it."""
    settings = settings or get_settings()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{slugify(kind)}-{secrets.token_hex(2)}"
    d = settings.outputs_dir / run_id
    d.mkdir(parents=True, exist_ok=False)
    return d


def stage_file(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` (hard-link when possible to avoid duplicating large volumes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def provenance_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.outputs_dir / "provenance.jsonl"


def record_provenance(tool: str, inputs: dict[str, Any], outputs: dict[str, Any], **extra: Any) -> None:
    """Append one JSON line describing a tool invocation that produced artifacts."""
    settings = get_settings()
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "inputs": _jsonable(inputs),
        "outputs": _jsonable(outputs),
        **{k: _jsonable(v) for k, v in extra.items()},
    }
    path = provenance_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Timer:
    """Tiny context manager measuring wall-clock seconds."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = round(time.perf_counter() - self.start, 3)
