"""Wrapped mode: drive an existing third-party container image (or host CLI) with a command
template and turn its output files into a normal ``response.json``.

Placeholders in ``command`` templates:

* ``{inputs.image}``  -> path of the staged input (``/job/inputs/image.nii.gz`` in containers)
* ``{outputs.mask}``  -> path declared in ``outputs.mask.file``
* ``{params.name}``   -> parameter value (list values are expanded to several arguments)
* ``{flag:name:--flag}`` -> ``--flag`` when ``params.name`` is truthy, otherwise nothing
* ``{opt:name:--opt}``   -> ``--opt <value>`` when ``params.name`` is set, otherwise nothing
* ``{job}`` / ``{weights}`` -> job directory / weights directory
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from open_med_mcp.models.manifest import CONTRACT_VERSION, ModelManifest

_PLACEHOLDER = re.compile(r"^\{(inputs|outputs|params)\.([A-Za-z0-9_]+)\}$")
_FLAG = re.compile(r"^\{flag:([A-Za-z0-9_]+):(.+)\}$")
_OPT = re.compile(r"^\{opt:([A-Za-z0-9_]+):(.+)\}$")


def render_command(
    template: list[str], request: dict[str, Any], manifest: ModelManifest, root: str
) -> list[str]:
    """Expand a command template. ``root`` is the job directory as seen by the command."""
    params = request.get("params") or {}
    inputs = request.get("inputs") or {}
    out: list[str] = []
    for token in template:
        if token == "{job}":
            out.append(root)
            continue
        if token == "{weights}":
            out.append(str((request.get("resources") or {}).get("weights_dir", root)))
            continue
        m = _PLACEHOLDER.match(token)
        if m:
            kind, key = m.groups()
            if kind == "inputs":
                if key not in inputs:
                    raise ValueError(f"command needs input {key!r} which was not provided")
                out.append(f"{root}/{inputs[key]}")
            elif kind == "outputs":
                spec = manifest.outputs.get(key)
                if spec is None or not spec.file:
                    raise ValueError(f"command references unknown output {key!r}")
                out.append(f"{root}/{spec.file}")
            else:
                value = params.get(key)
                if value is None:
                    raise ValueError(f"command needs parameter {key!r}")
                out.extend([str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)])
            continue
        m = _FLAG.match(token)
        if m:
            key, flag = m.groups()
            if params.get(key):
                out.append(flag)
            continue
        m = _OPT.match(token)
        if m:
            key, opt = m.groups()
            value = params.get(key)
            if value is not None and value != "" and value is not False:
                out.extend([opt, str(value)])
            continue

        # inline substitution inside longer tokens, e.g. "--out={outputs.mask}"
        def sub(mm: re.Match[str]) -> str:
            kind, key = mm.group(1), mm.group(2)
            if kind == "inputs":
                return f"{root}/{inputs[key]}"
            if kind == "outputs":
                return f"{root}/{manifest.outputs[key].file}"
            return str(params.get(key, ""))

        out.append(re.sub(r"\{(inputs|outputs|params)\.([A-Za-z0-9_]+)\}", sub, token))
    return out


def finalize(
    run_dir: Path,
    manifest: ModelManifest,
    request: dict[str, Any],
    returncode: int,
    command: list[str],
    started: float,
) -> dict[str, Any]:
    """Write ``response.json`` for a wrapped run based on the files the tool produced."""
    outputs: dict[str, str] = {}
    missing: list[str] = []
    for key, spec in manifest.outputs.items():
        assert spec.file is not None
        p = run_dir / spec.file
        if p.exists():
            outputs[key] = spec.file
        elif spec.required:
            missing.append(spec.file)
    stats: dict[str, Any] = {}
    labels = {str(k): v for k, v in manifest.labels.items()}
    if returncode == 0 and not missing:
        status = "ok"
        error = None
        if "mask" in outputs:
            try:
                from open_med_mcp.core.image import load_mask

                m = load_mask(run_dir / outputs["mask"])
                present = [int(v) for v in np.unique(m.array) if v]
                vox = m.voxel_volume_mm3()
                stats = {
                    "labels_present": present,
                    "foreground_voxels": int((m.array != 0).sum()),
                    "per_label": {
                        str(lab): {
                            "voxels": int((m.array == lab).sum()),
                            "volume_ml": round(float((m.array == lab).sum() * vox / 1000), 3),
                        }
                        for lab in present
                    },
                    "voxel_volume_mm3": vox,
                }
                if not labels:
                    labels = {str(lab): f"label_{lab}" for lab in present}
            except Exception as exc:  # pragma: no cover - keep the run usable
                stats = {"warning": f"could not summarize mask: {exc}"}
    else:
        status = "error"
        log_tail = (
            (run_dir / "log.txt").read_text(errors="replace")[-2000:]
            if (run_dir / "log.txt").exists()
            else ""
        )
        msg = (
            f"command exited with code {returncode}"
            if returncode != 0
            else f"expected output(s) missing: {missing}"
        )
        error = {"type": "WrappedCommandError", "message": msg, "traceback": log_tail}
    resp = {
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "outputs": outputs,
        "labels": labels,
        "stats": stats,
        "model_info": {
            "adapter": manifest.adapter,
            "mode": "wrapped",
            "image": manifest.container.image,
            "command": command,
        },
        "warnings": [],
        "timing": {"total_s": round(time.time() - started, 3)},
    }
    if error:
        resp["error"] = error
    (run_dir / "response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")
    return resp
