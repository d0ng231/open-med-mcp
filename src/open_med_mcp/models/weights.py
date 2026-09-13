"""Download and verify model weights declared in a manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from open_med_mcp.config import Settings, get_settings
from open_med_mcp.models.manifest import ModelManifest, WeightSpec


def weights_dir_for(manifest: ModelManifest, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    d = settings.weights_dir / manifest.adapter
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_status(manifest: ModelManifest, settings: Settings | None = None) -> list[dict[str, object]]:
    d = weights_dir_for(manifest, settings)
    out = []
    for w in manifest.weights:
        p = d / w.file
        out.append(
            {
                "id": w.id,
                "file": str(p),
                "present": p.exists(),
                "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
                "url": w.url,
            }
        )
    return out


def missing_weights(
    manifest: ModelManifest, ids: list[str] | None = None, settings: Settings | None = None
) -> list[WeightSpec]:
    d = weights_dir_for(manifest, settings)
    wanted = [w for w in manifest.weights if not ids or w.id in ids]
    return [w for w in wanted if not (d / w.file).exists()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weight(
    spec: WeightSpec,
    dest_dir: Path,
    progress: Callable[[str, int, int | None], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    if not spec.url:
        raise RuntimeError(
            f"weight {spec.id!r} has no download URL; place {spec.file} in {dest_dir} manually"
        )
    dest = dest_dir / spec.file
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(follow_redirects=True, timeout=timeout) as client,
        client.stream("GET", spec.url) as resp,
    ):
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0) or None
        done = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(spec.id, done, total)
    if spec.sha256:
        got = sha256_of(tmp)
        if got.lower() != spec.sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"sha256 mismatch for {spec.file}: expected {spec.sha256}, got {got}")
    tmp.replace(dest)
    return dest


def ensure_weights(
    manifest: ModelManifest,
    ids: list[str] | None = None,
    settings: Settings | None = None,
    progress: Callable[[str, int, int | None], None] | None = None,
) -> list[Path]:
    """Download every missing weight (optionally restricted to ``ids``) and return their paths."""
    settings = settings or get_settings()
    d = weights_dir_for(manifest, settings)
    paths = []
    for w in missing_weights(manifest, ids, settings):
        paths.append(download_weight(w, d, progress))
    return paths
