"""Model manifest schema (``<adapter>/<name>.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONTRACT_VERSION = 1


class WeightSpec(BaseModel):
    id: str
    file: str = Field(description="File name inside the adapter's weights directory")
    url: str | None = None
    sha256: str | None = None
    size_mb: float | None = None
    license: str | None = None
    note: str | None = None


class ParamSpec(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"] = "string"
    default: Any = None
    description: str = ""
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    items: Literal["string", "number", "integer", "boolean", "object"] | None = None

    def json_schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": self.type}
        if self.description:
            s["description"] = self.description
        if self.default is not None:
            s["default"] = self.default
        if self.choices:
            s["enum"] = self.choices
        if self.minimum is not None:
            s["minimum"] = self.minimum
        if self.maximum is not None:
            s["maximum"] = self.maximum
        if self.type == "array" and self.items:
            s["items"] = {"type": self.items}
        return s


class InputSpec(BaseModel):
    kind: Literal["image", "mask", "file"] = "image"
    required: bool = True
    description: str = ""


class OutputSpec(BaseModel):
    kind: Literal["mask", "image", "file", "json"] = "mask"
    description: str = ""
    file: str | None = Field(
        default=None,
        description="Wrapped mode: path (relative to the job dir) the tool writes, e.g. outputs/mask.nii.gz",
    )
    required: bool = True


class ContainerSpec(BaseModel):
    image: str | None = Field(
        default=None, description="Full image reference; defaults to <OMM_IMAGE_PREFIX>-<adapter>:<version>"
    )
    dockerfile: str = "Dockerfile"
    gpu: Literal["required", "preferred", "none"] = "preferred"
    mode: Literal["adapter", "wrapped"] = Field(
        default="adapter",
        description="adapter = our run.py inside the image; wrapped = an existing third-party image driven by `command`",
    )
    entrypoint: list[str] = Field(default_factory=lambda: ["python", "/app/run.py"])
    command: list[str] = Field(
        default_factory=list,
        description="wrapped mode: command template with {inputs.x}, {outputs.y}, {params.p}, {flag:p:--flag}, {opt:p:--opt} placeholders (container paths under /job)",
    )
    gpu_image: str | None = Field(
        default=None, description="wrapped mode: alternative image used when a GPU is available"
    )
    shm_size: str = "2g"


class LocalSpec(BaseModel):
    entrypoint: str = "run.py"
    requires: list[str] = Field(default_factory=list, description="Import names that must be available")
    extra: str | None = Field(
        default=None, description="pip extra of open-med-mcp that installs the requirements"
    )
    command: list[str] = Field(
        default_factory=list,
        description="wrapped mode: host command template (same placeholders, host paths); used when command[0] is on PATH",
    )


class ModelManifest(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    version: str = "0.1"
    contract_version: int = CONTRACT_VERSION
    adapter: str = Field(default="", description="Adapter directory name (defaults to the manifest's folder)")
    adapter_dir: Path | None = Field(default=None, exclude=True)
    category: Literal["segmentation", "classification", "preprocessing", "detection", "other"] = (
        "segmentation"
    )
    tasks: list[str] = Field(default_factory=lambda: ["segment"])
    default_task: str = ""
    task_descriptions: dict[str, str] = Field(default_factory=dict)
    modalities: list[str] = Field(default_factory=lambda: ["any"])
    dims: list[int] = Field(default_factory=lambda: [2, 3])
    prompt_types: list[str] = Field(
        default_factory=list, description="e.g. point, box, text; empty = automatic model"
    )
    inputs: dict[str, InputSpec] = Field(default_factory=lambda: {"image": InputSpec()})
    outputs: dict[str, OutputSpec] = Field(default_factory=lambda: {"mask": OutputSpec()})
    labels: dict[int, str] = Field(
        default_factory=dict, description="Static label names for models with a fixed label map"
    )
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    weights: list[WeightSpec] = Field(default_factory=list)
    container: ContainerSpec = Field(default_factory=ContainerSpec)
    local: LocalSpec = Field(default_factory=LocalSpec)
    runners: list[Literal["local", "docker", "apptainer"]] = Field(
        default_factory=lambda: ["local", "docker", "apptainer"]
    )
    license: str = ""
    homepage: str = ""
    citation: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _defaults(self) -> ModelManifest:
        if not self.display_name:
            self.display_name = self.name
        if not self.default_task:
            self.default_task = self.tasks[0] if self.tasks else "segment"
        if self.is_wrapped:
            if not self.container.image:
                raise ValueError(f"wrapped model {self.name!r} needs container.image")
            if not self.container.command and not self.local.command:
                raise ValueError(f"wrapped model {self.name!r} needs container.command or local.command")
            for key, spec in self.outputs.items():
                if not spec.file:
                    raise ValueError(f"wrapped model {self.name!r}: output {key!r} needs `file`")
        return self

    # ------------------------------------------------------------------
    @property
    def is_promptable(self) -> bool:
        return bool(self.prompt_types)

    @property
    def is_wrapped(self) -> bool:
        return self.container.mode == "wrapped"

    def default_params(self) -> dict[str, Any]:
        return {k: v.default for k, v in self.params.items() if v.default is not None}

    def params_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {k: v.json_schema() for k, v in self.params.items()},
            "additionalProperties": True,
        }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Merge with defaults and check enum/range constraints. Unknown keys are passed through."""
        merged = {**self.default_params(), **(params or {})}
        for key, spec in self.params.items():
            if key not in merged or merged[key] is None:
                continue
            val = merged[key]
            if spec.choices and val not in spec.choices:
                raise ValueError(f"param {key!r}={val!r} not in {spec.choices}")
            if spec.type in ("number", "integer") and isinstance(val, (int, float)):
                if spec.minimum is not None and val < spec.minimum:
                    raise ValueError(f"param {key!r}={val} < minimum {spec.minimum}")
                if spec.maximum is not None and val > spec.maximum:
                    raise ValueError(f"param {key!r}={val} > maximum {spec.maximum}")
        return merged

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "tasks": self.tasks,
            "modalities": self.modalities,
            "dims": self.dims,
            "prompt_types": self.prompt_types,
            "automatic": not self.is_promptable,
            "container_mode": self.container.mode,
            "tags": self.tags,
            "license": self.license,
        }

    def dockerfile_path(self) -> Path | None:
        if self.adapter_dir is None:
            return None
        p = self.adapter_dir / self.container.dockerfile
        return p if p.exists() else None

    def entrypoint_path(self) -> Path:
        assert self.adapter_dir is not None, "manifest is not bound to an adapter directory"
        return self.adapter_dir / self.local.entrypoint


def load_manifest(path: Path) -> ModelManifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"manifest {path} must be a mapping")
    raw.setdefault("name", path.stem)
    raw.setdefault("adapter", path.parent.name)
    m = ModelManifest.model_validate(raw)
    m.adapter_dir = path.parent.resolve()
    return m
