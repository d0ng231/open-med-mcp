"""Discovery of model manifests (bundled zoo + user directories)."""

from __future__ import annotations

from pathlib import Path

from open_med_mcp.config import Settings, get_settings
from open_med_mcp.models.manifest import ModelManifest, load_manifest

ZOO_DIR = Path(__file__).resolve().parent.parent / "zoo"


def _manifest_files(adapter_dir: Path) -> list[Path]:
    return sorted(
        p for p in adapter_dir.glob("*.yaml") if p.parent == adapter_dir and not p.name.startswith("_")
    )


def _scan(root: Path) -> list[Path]:
    """Return manifest files under ``root`` (an adapter dir or a directory of adapter dirs)."""
    if not root.exists():
        return []
    if (root / "run.py").exists() or _manifest_files(root):
        return _manifest_files(root)
    files: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("_") and not child.name.startswith("."):
            files.extend(_manifest_files(child))
    return files


class ModelRegistry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._models: dict[str, ModelManifest] = {}
        self._errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._models.clear()
        self._errors.clear()
        roots = [ZOO_DIR, *self.settings.extra_model_dirs, self.settings.workspace / "omm_models"]
        for root in roots:
            for mf in _scan(root):
                try:
                    m = load_manifest(mf)
                except Exception as exc:  # keep the server alive on a broken user manifest
                    self._errors[str(mf)] = f"{type(exc).__name__}: {exc}"
                    continue
                self._models[m.name] = m  # later roots override bundled models on purpose

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def names(self) -> list[str]:
        return sorted(self._models)

    def all(self) -> list[ModelManifest]:
        return [self._models[n] for n in self.names()]

    def get(self, name: str) -> ModelManifest:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}; available: {', '.join(self.names())}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._models


_registry: ModelRegistry | None = None


def get_registry(settings: Settings | None = None, reload: bool = False) -> ModelRegistry:
    global _registry
    if _registry is None or reload or (settings is not None and settings is not _registry.settings):
        _registry = ModelRegistry(settings)
    return _registry
