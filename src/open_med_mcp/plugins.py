"""Plug-in loading.

A plug-in is a Python module that defines ``register(server)`` and adds tools, resources, prompts
or viewer renderers to the running server. Sources, in order:

1. ``<workspace>/omm_plugins/*.py`` - drop-in files, no packaging needed
2. directories listed in ``OMM_PLUGIN_DIRS`` (``os.pathsep``-separated)
3. installed packages exposing the ``open_med_mcp.plugins`` entry-point group
   (each entry point resolves to a module or object with ``register``)

Failures are logged and never take the server down. ``open_med_mcp.plugin_api`` bundles the helpers
plug-ins usually need (path resolution, cached image loading, result packaging, previews).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from open_med_mcp.config import Settings, get_settings

log = logging.getLogger("open_med_mcp.plugins")


@dataclass
class LoadedPlugin:
    name: str
    source: str  # file path or entry point
    kind: str  # "file" | "entry_point"
    tools_added: list[str]
    error: str | None = None


_LOADED: list[LoadedPlugin] = []


def loaded_plugins() -> list[LoadedPlugin]:
    return list(_LOADED)


def plugin_dirs(settings: Settings | None = None) -> list[Path]:
    settings = settings or get_settings()
    return [settings.workspace / "omm_plugins", *settings.extra_plugin_dirs]


def _tool_names(server: Any) -> set[str]:
    try:
        return {t.name for t in server._tool_manager.list_tools()}  # noqa: SLF001 - SDK internal, best effort
    except Exception:
        return set()


def _call_register(server: Any, obj: Any, name: str, source: str, kind: str) -> LoadedPlugin:
    before = _tool_names(server)
    register = getattr(obj, "register", None)
    if register is None:
        return LoadedPlugin(name, source, kind, [], error="no register(server) function")
    try:
        register(server)
    except Exception as exc:  # noqa: BLE001
        log.exception("plug-in %s failed to register", name)
        return LoadedPlugin(name, source, kind, [], error=f"{type(exc).__name__}: {exc}")
    added = sorted(_tool_names(server) - before)
    log.info("loaded plug-in %s (%s): tools %s", name, source, added)
    return LoadedPlugin(name, source, kind, added)


def load_file_plugin(server: Any, path: Path) -> LoadedPlugin:
    name = f"omm_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return LoadedPlugin(path.stem, str(path), "file", [], error="cannot import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        log.exception("plug-in %s failed to import", path)
        return LoadedPlugin(path.stem, str(path), "file", [], error=f"{type(exc).__name__}: {exc}")
    return _call_register(server, module, path.stem, str(path), "file")


def load_plugins(server: Any, settings: Settings | None = None) -> list[LoadedPlugin]:
    """Load every plug-in source into ``server``. Safe to call once per server."""
    settings = settings or get_settings()
    _LOADED.clear()
    for d in plugin_dirs(settings):
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            _LOADED.append(load_file_plugin(server, path))
    try:
        eps = list(entry_points(group="open_med_mcp.plugins"))
    except TypeError:  # pragma: no cover
        eps = list(entry_points().get("open_med_mcp.plugins", []))  # type: ignore[union-attr]
    for ep in eps:
        try:
            obj = ep.load()
            if isinstance(obj, str):
                obj = importlib.import_module(obj)
        except Exception as exc:  # noqa: BLE001
            log.exception("plug-in entry point %s failed", ep.name)
            _LOADED.append(
                LoadedPlugin(ep.name, ep.value, "entry_point", [], error=f"{type(exc).__name__}: {exc}")
            )
            continue
        _LOADED.append(_call_register(server, obj, ep.name, ep.value, "entry_point"))
    return list(_LOADED)
