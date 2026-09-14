"""Runtime configuration.

Every setting can be provided through an environment variable with the ``OMM_`` prefix
(``OMM_WORKSPACE``, ``OMM_RUNNER`` ...). Nothing is read from the network and nothing is
written outside of the workspace and ``OMM_HOME``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RunnerName = Literal["auto", "local", "docker", "apptainer"]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """All tunables of the server, resolved once at start-up."""

    #: Root directory that relative paths in tool calls are resolved against.
    workspace: Path = field(default_factory=lambda: _env_path("OMM_WORKSPACE", Path.cwd()))
    #: Per-user state (downloaded weights, Apptainer images, caches).
    home: Path = field(default_factory=lambda: _env_path("OMM_HOME", Path.home() / ".cache" / "open-med-mcp"))
    #: Directory (inside the workspace by default) where runs and artifacts are written.
    outputs_dirname: str = field(default_factory=lambda: os.environ.get("OMM_OUTPUTS_DIRNAME", "omm_outputs"))
    #: Which execution backend to use for models: auto | local | docker | apptainer.
    runner: RunnerName = field(default_factory=lambda: os.environ.get("OMM_RUNNER", "auto"))  # type: ignore[assignment]
    #: "auto" (use a GPU when one is visible), "none", or an explicit device string like "cuda:0".
    device: str = field(default_factory=lambda: os.environ.get("OMM_DEVICE", "auto"))
    #: Extra directories with model manifests (``name.yaml`` files or adapter folders).
    extra_model_dirs: list[Path] = field(
        default_factory=lambda: [
            Path(p).expanduser() for p in os.environ.get("OMM_MODEL_DIRS", "").split(os.pathsep) if p
        ]
    )
    #: Extra directories with custom guidelines (Markdown files with YAML front matter).
    extra_guideline_dirs: list[Path] = field(
        default_factory=lambda: [
            Path(p).expanduser() for p in os.environ.get("OMM_GUIDELINE_DIRS", "").split(os.pathsep) if p
        ]
    )
    #: Container image registry prefix used when a manifest does not pin a full image name.
    image_prefix: str = field(
        default_factory=lambda: os.environ.get("OMM_IMAGE_PREFIX", "ghcr.io/d0ng231/open-med-mcp")
    )
    #: Maximum number of seconds a single model run may take.
    run_timeout_s: int = field(default_factory=lambda: int(os.environ.get("OMM_RUN_TIMEOUT", "3600")))
    #: Allow tools to read/write paths outside of the workspace.
    allow_outside_workspace: bool = field(
        default_factory=lambda: _env_bool("OMM_ALLOW_OUTSIDE_WORKSPACE", True)
    )
    #: Automatically download missing weights before a run.
    auto_download_weights: bool = field(default_factory=lambda: _env_bool("OMM_AUTO_DOWNLOAD", True))
    #: Maximum edge length (pixels) of preview images returned to the agent.
    preview_max_px: int = field(default_factory=lambda: int(os.environ.get("OMM_PREVIEW_MAX_PX", "900")))
    #: Return inline images in tool results (set to false for text-only clients; files are still written).
    return_images: bool = field(default_factory=lambda: _env_bool("OMM_RETURN_IMAGES", True))
    #: Hard cap on the size of one inline image (bytes); larger renders are downscaled.
    max_image_bytes: int = field(
        default_factory=lambda: int(os.environ.get("OMM_MAX_IMAGE_BYTES", str(1_500_000)))
    )
    #: Budget of the in-memory image cache (MB).
    cache_mb: int = field(default_factory=lambda: int(os.environ.get("OMM_CACHE_MB", "1500")))
    #: Extra directories with plug-in modules (``*.py`` files exposing ``register(server)``).
    extra_plugin_dirs: list[Path] = field(
        default_factory=lambda: [
            Path(p).expanduser() for p in os.environ.get("OMM_PLUGIN_DIRS", "").split(os.pathsep) if p
        ]
    )
    #: Logging: level and optional file (default ``<home>/logs/server.log``; empty string disables the file).
    log_level: str = field(default_factory=lambda: os.environ.get("OMM_LOG_LEVEL", "INFO").upper())
    log_file: str | None = field(default_factory=lambda: os.environ.get("OMM_LOG_FILE"))

    # ------------------------------------------------------------------ paths
    @property
    def outputs_dir(self) -> Path:
        return self.workspace / self.outputs_dirname

    @property
    def weights_dir(self) -> Path:
        return self.home / "weights"

    @property
    def images_dir(self) -> Path:
        """Where Apptainer/Singularity ``.sif`` images are cached."""
        return self.home / "images"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.weights_dir, self.images_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def local_python_for(self, adapter: str) -> str | None:
        """Interpreter to use for the local runner of ``adapter`` (``OMM_ZOO_<ADAPTER>_PYTHON``)."""
        return os.environ.get(f"OMM_ZOO_{adapter.upper().replace('-', '_')}_PYTHON")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> Settings:
    global _settings
    _settings = settings
    return settings
