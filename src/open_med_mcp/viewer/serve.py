"""Optional interactive 3D viewer for humans: serves a NiiVue page plus the volumes over HTTP.

NiiVue (https://niivue.com) is loaded from a CDN in the browser; nothing is uploaded anywhere - the
page fetches the NIfTI files from this local server.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).resolve().parent / "templates"
NIIVUE_URL = "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/index.min.js"


def build_niivue_page(
    image: Path, masks: list[Path], serve_root: Path, title: str = "open-med-mcp 3D viewer"
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    rel = lambda p: Path(p).resolve().relative_to(serve_root.resolve()).as_posix()  # noqa: E731
    return env.get_template("niivue.html.j2").render(
        title=title, image=rel(image), masks=[rel(m) for m in masks], niivue_url=NIIVUE_URL
    )


def serve_directory(
    root: Path, port: int = 8765, index_html: str | None = None, block: bool = True
) -> tuple[str, threading.Thread | None]:
    root = Path(root).resolve()
    if index_html is not None:
        (root / "index.html").write_text(index_html, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/index.html"
    if block:
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
        return url, None
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    t.httpd = httpd  # type: ignore[attr-defined]
    return url, t
