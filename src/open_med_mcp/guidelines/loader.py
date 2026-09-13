"""Discovery and parsing of guideline files.

A guideline is a Markdown file with YAML front matter::

    ---
    name: segmentation-3d-ct
    title: Segment a structure in a 3D CT
    tags: [segmentation, ct, 3d]
    modalities: [CT]
    tasks: [segmentation]
    models: [totalsegmentator, medsam2, classical]
    version: 1
    summary: One-line description shown in listings.
    ---
    # Protocol ...

Presets ship with the package; users add their own with ``OMM_GUIDELINE_DIRS`` or a
``omm_guidelines/`` folder in the workspace. Later sources override earlier ones by ``name``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from open_med_mcp.config import Settings, get_settings

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Guideline:
    name: str
    title: str
    body: str
    path: Path
    source: str  # "preset" | "user"
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    version: str = "1"
    meta: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "modalities": self.modalities,
            "tasks": self.tasks,
            "models": self.models,
            "source": self.source,
            "version": self.version,
        }

    def render(self) -> str:
        return f"# {self.title}\n\n{self.body.strip()}\n"


def parse_guideline(path: Path, source: str = "user") -> Guideline:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = text
    m = _FRONT.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = text[m.end() :]
    name = str(meta.get("name") or path.stem)
    title = str(meta.get("title") or _first_heading(body) or name)
    body = _strip_first_heading(body, title)

    def as_list(key: str) -> list[str]:
        v = meta.get(key) or []
        return [str(x) for x in (v if isinstance(v, list) else [v])]

    return Guideline(
        name=name,
        title=title,
        body=body,
        path=path,
        source=source,
        summary=str(meta.get("summary") or _first_paragraph(body)),
        tags=as_list("tags"),
        modalities=as_list("modalities"),
        tasks=as_list("tasks"),
        models=as_list("models"),
        version=str(meta.get("version", "1")),
        meta=meta,
    )


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _strip_first_heading(body: str, title: str) -> str:
    lines = body.lstrip().splitlines()
    if lines and lines[0].strip() == f"# {title}":
        return "\n".join(lines[1:]).lstrip("\n")
    return body


def _first_paragraph(body: str) -> str:
    para: list[str] = []
    for line in body.strip().splitlines():
        if not line.strip():
            if para:
                break
            continue
        if line.startswith("#"):
            continue
        para.append(line.strip())
    return " ".join(para)[:240]


class GuidelineLibrary:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._items: dict[str, Guideline] = {}
        self.reload()

    def reload(self) -> None:
        self._items.clear()
        sources: list[tuple[Path, str]] = [(PRESETS_DIR, "preset")]
        sources += [(d, "user") for d in self.settings.extra_guideline_dirs]
        sources.append((self.settings.workspace / "omm_guidelines", "user"))
        for directory, source in sources:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                if path.name.lower() == "readme.md":
                    continue
                try:
                    g = parse_guideline(path, source)
                except Exception:
                    continue
                self._items[g.name] = g

    def names(self) -> list[str]:
        return sorted(self._items)

    def all(self) -> list[Guideline]:
        return [self._items[n] for n in self.names()]

    def get(self, name: str) -> Guideline:
        if name in self._items:
            return self._items[name]
        # tolerant lookup: case/underscore-insensitive
        key = name.lower().replace("_", "-")
        for n, g in self._items.items():
            if n.lower().replace("_", "-") == key:
                return g
        raise KeyError(f"unknown guideline {name!r}; available: {', '.join(self.names())}")

    def search(
        self, query: str | None = None, tags: list[str] | None = None, modality: str | None = None
    ) -> list[Guideline]:
        out = []
        q = (query or "").lower()
        for g in self.all():
            if tags and not set(t.lower() for t in tags) & set(t.lower() for t in g.tags):
                continue
            if (
                modality
                and g.modalities
                and modality.lower() not in [m.lower() for m in g.modalities]
                and "any" not in [m.lower() for m in g.modalities]
            ):
                continue
            if q and q not in (g.name + " " + g.title + " " + g.summary + " " + " ".join(g.tags)).lower():
                continue
            out.append(g)
        return out


_library: GuidelineLibrary | None = None


def get_library(settings: Settings | None = None, reload: bool = False) -> GuidelineLibrary:
    global _library
    if _library is None or reload or (settings is not None and settings is not _library.settings):
        _library = GuidelineLibrary(settings)
    return _library
