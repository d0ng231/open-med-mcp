"""Scaffolding for the three plug-in kinds: models (adapters), tool plug-ins and guidelines."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from open_med_mcp.models.registry import ZOO_DIR

_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,40}$")


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError(
            "name must be lowercase letters, digits, '-' or '_' (2-41 chars) and start with a letter"
        )
    return name


def new_model(root: Path, name: str, wrapped_image: str | None = None) -> Path:
    """Create ``<root>/<name>/`` from the adapter template (or a wrapped-image manifest)."""
    _check_name(name)
    dest = root / name
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    if wrapped_image:
        (dest / f"{name}.yaml").write_text(
            f"""# Wrapped third-party image: no run.py needed. Edit the command template and outputs.
name: {name}
display_name: {name}
description: Describe when an agent should use this model.
category: segmentation
tasks: [segment]
modalities: [CT]
dims: [3]
inputs:
  image: {{kind: image, required: true}}
outputs:
  mask: {{kind: mask, file: outputs/mask.nii.gz}}
labels: {{1: object}}
params:
  fast: {{type: boolean, default: false, description: "Example flag"}}
container:
  mode: wrapped
  image: {wrapped_image}
  gpu: preferred
  command: ["my-tool", "-i", "{{inputs.image}}", "-o", "{{outputs.mask}}", "{{flag:fast:--fast}}"]
local:
  command: ["my-tool", "-i", "{{inputs.image}}", "-o", "{{outputs.mask}}", "{{flag:fast:--fast}}"]
runners: [docker, apptainer, local]
license: see upstream
tags: [wrapped-image]
""",
            encoding="utf-8",
        )
        (dest / "README.md").write_text(
            f"# {name}\n\nWrapped image `{wrapped_image}`. See docs/models.md ('Wrapped images').\n",
            encoding="utf-8",
        )
        return dest
    template = ZOO_DIR / "_template"
    for src in template.iterdir():
        if src.name == "mymodel.yaml":
            text = (
                src.read_text(encoding="utf-8")
                .replace("name: mymodel", f"name: {name}")
                .replace("display_name: My Model", f"display_name: {name}")
            )
            (dest / f"{name}.yaml").write_text(text, encoding="utf-8")
        elif src.name == "Dockerfile":
            (dest / "Dockerfile").write_text(
                src.read_text(encoding="utf-8").replace("zoo/_template/run.py", f"zoo/{name}/run.py"),
                encoding="utf-8",
            )
        elif src.name == "run.py":
            (dest / "run.py").write_text(
                src.read_text(encoding="utf-8").replace('"adapter": "mymodel"', f'"adapter": "{name}"'),
                encoding="utf-8",
            )
        else:
            shutil.copy(src, dest / src.name)
    return dest


PLUGIN_TEMPLATE = '''"""open-med-mcp plug-in: {name}

Drop this file into <workspace>/omm_plugins/ (or a directory in OMM_PLUGIN_DIRS) and restart the
server. Every function decorated with @server.tool() becomes an MCP tool.
"""

from typing import Annotated

from pydantic import Field

from open_med_mcp.plugin_api import load_image_cached, load_mask_cached, render_preview, resolve, result, tool_errors


def register(server):
    @server.tool()
    @tool_errors
    def {func}(
        mask: Annotated[str, Field(description="Label map path")],
        image: Annotated[str | None, Field(description="Image for the preview")] = None,
    ):
        """Example tool: foreground volume of a mask (mL) with an optional preview."""
        m = load_mask_cached(resolve(mask))
        payload = {{"mask": mask, "volume_ml": round(float((m.array != 0).sum() * m.voxel_volume_mm3() / 1000), 3)}}
        images = []
        if image:
            images.append(render_preview(load_image_cached(resolve(image)), mask, title="{name}"))
        return result(payload, images)
'''


def new_plugin(root: Path, name: str) -> Path:
    _check_name(name)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{name.replace('-', '_')}.py"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(
        PLUGIN_TEMPLATE.format(name=name, func=name.replace("-", "_") + "_volume"), encoding="utf-8"
    )
    return dest


def new_guideline(root: Path, name: str, title: str | None = None) -> Path:
    _check_name(name)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{name}.md"
    if dest.exists():
        raise FileExistsError(dest)
    title = title or name.replace("-", " ").capitalize()
    dest.write_text(
        f"""---
name: {name}
title: {title}
summary: One sentence an agent reads in list_guidelines.
tags: [custom]
modalities: [any]
tasks: [segmentation]
version: 1
---
## 1. Inspect
`inspect_image(path)` - what to check before starting.

## 2. Segment
Which model, which parameters, and why.

## 3. Quality control
Concrete checks (`render_view`, `mask_stats`) and acceptance criteria.

## 4. Report
What the final report must contain.
""",
        encoding="utf-8",
    )
    return dest
