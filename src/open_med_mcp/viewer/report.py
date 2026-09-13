"""Markdown + HTML reports assembling figures, tables and provenance."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

from open_med_mcp import __version__

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    cols = columns or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, dict)):
        return "`" + json.dumps(v) + "`"
    return str(v)


def build_markdown(
    title: str,
    sections: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> str:
    """``sections``: ``[{"heading": str, "text": markdown, "figures": [paths], "table": [rows], "columns": [...]}, ...]``."""
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by open-med-mcp {__version__}. Research use only - not for clinical decision making._",
        "",
    ]
    if metadata:
        lines += [
            "## Metadata",
            "",
            _table([{"key": k, "value": _fmt(v)} for k, v in metadata.items()], ["key", "value"]),
            "",
        ]
    for sec in sections:
        if sec.get("heading"):
            lines += [f"## {sec['heading']}", ""]
        if sec.get("text"):
            lines += [str(sec["text"]).strip(), ""]
        if sec.get("table"):
            lines += [_table(list(sec["table"]), sec.get("columns")), ""]
        for fig in sec.get("figures") or []:
            fp = Path(fig)
            rel = fp
            if out_dir is not None:
                try:
                    rel = fp.resolve().relative_to(out_dir.resolve())
                except ValueError:
                    rel = fp
            caption = sec.get("caption") or fp.stem.replace("_", " ")
            lines += [f"![{caption}]({rel.as_posix()})", "", f"*{caption}*", ""]
    return "\n".join(lines).rstrip() + "\n"


def markdown_to_html(md_text: str, title: str, embed_dir: Path | None = None) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    if embed_dir is not None:
        body = _embed_images(body, embed_dir)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    return env.get_template("report.html.j2").render(title=title, body=body, version=__version__)


def _embed_images(html: str, base_dir: Path) -> str:
    import re

    def repl(m: re.Match[str]) -> str:
        src = m.group(1)
        p = base_dir / src
        if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            return f'src="data:{mime};base64,{data}"'
        return m.group(0)

    return re.sub(r'src="([^"]+)"', repl, html)


def write_report(
    title: str, sections: list[dict[str, Any]], out_md: Path, metadata: dict[str, Any] | None = None
) -> dict[str, str]:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_text = build_markdown(title, sections, metadata, out_md.parent)
    out_md.write_text(md_text, encoding="utf-8")
    out_html = out_md.with_suffix(".html")
    out_html.write_text(markdown_to_html(md_text, title, out_md.parent), encoding="utf-8")
    return {"markdown": str(out_md), "html": str(out_html)}
