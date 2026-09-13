"""The job-directory contract shared by every execution backend.

::

    <run_dir>/
      request.json      # written by the server
      inputs/           # staged inputs in canonical formats (.nii.gz / .png)
      outputs/          # written by the adapter
      response.json     # written by the adapter
      log.txt           # adapter log (stdout/stderr are appended by the runner)

The adapter only ever sees ``/job`` (containers) or the run directory (local runner).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from open_med_mcp.core.image import IMAGE2D_SUFFIXES, MedicalImage, file_suffix
from open_med_mcp.models.manifest import CONTRACT_VERSION, ModelManifest
from open_med_mcp.workspace import stage_file


class JobError(RuntimeError):
    pass


def stage_input(src: Path, dest_dir: Path, key: str) -> tuple[Path, dict[str, Any]]:
    """Stage one input file into ``dest_dir`` in a canonical format and return ``(path, info)``.

    * volumes (NIfTI/NRRD/MetaImage/DICOM series ...) -> ``<key>.nii.gz``
    * 2D integer images (PNG/JPEG/TIFF ...) -> ``<key>.png`` (kept lossless)
    * anything else (2D float, .npy) -> ``<key>.nii.gz``
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sfx = file_suffix(src) if src.is_file() else ""
    if src.is_file() and sfx == ".nii.gz":
        return stage_file(src, dest_dir / f"{key}.nii.gz"), {"converted": False}
    if src.is_file() and sfx == ".png":
        return stage_file(src, dest_dir / f"{key}.png"), {"converted": False}
    img = MedicalImage.load(src)
    if img.is_2d and (img.is_rgb or img.array.dtype in (np.uint8, np.uint16)):
        out = dest_dir / f"{key}.png"
    else:
        out = dest_dir / f"{key}.nii.gz"
    img.save(out)
    return out, {"converted": True, "from_format": img.format, "original": str(src)}


def prepare_job(
    run_dir: Path,
    manifest: ModelManifest,
    task: str,
    inputs: dict[str, Path],
    params: dict[str, Any],
    device: str,
    weights_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage inputs and write ``request.json``. Returns the request dict."""
    if task not in manifest.tasks:
        raise JobError(f"model {manifest.name!r} does not support task {task!r}; tasks: {manifest.tasks}")
    for key, spec in manifest.inputs.items():
        if spec.required and key not in inputs:
            raise JobError(f"missing required input {key!r} for model {manifest.name!r}")
    staged: dict[str, str] = {}
    staging_info: dict[str, Any] = {}
    for key, src in inputs.items():
        path, info = stage_input(Path(src), run_dir / "inputs", key)
        staged[key] = str(path.relative_to(run_dir))
        staging_info[key] = info
    (run_dir / "outputs").mkdir(exist_ok=True)
    request = {
        "contract_version": CONTRACT_VERSION,
        "model": manifest.name,
        "adapter": manifest.adapter,
        "task": task,
        "inputs": staged,
        "params": manifest.validate_params(params),
        "output_dir": "outputs",
        "resources": {"device": device, "weights_dir": str(weights_dir)},
        "staging": staging_info,
        **(extra or {}),
    }
    (run_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request


def read_response(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "response.json"
    if not p.exists():
        log = (
            (run_dir / "log.txt").read_text(errors="replace")[-4000:]
            if (run_dir / "log.txt").exists()
            else ""
        )
        raise JobError(f"adapter produced no response.json in {run_dir}\n--- log tail ---\n{log}")
    resp = json.loads(p.read_text(encoding="utf-8"))
    if resp.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise JobError(f"unsupported contract version {resp.get('contract_version')}")
    # resolve output paths
    outputs: dict[str, str] = {}
    for key, rel in (resp.get("outputs") or {}).items():
        outputs[key] = str((run_dir / rel).resolve())
    resp["outputs"] = outputs
    if resp.get("status") != "ok":
        err = resp.get("error") or {}
        raise JobError(
            f"model run failed: {err.get('type', 'Error')}: {err.get('message', '')}\n{err.get('traceback', '')[-3000:]}"
        )
    return resp


def output_suffix_for(img: MedicalImage) -> str:
    return (
        ".png" if img.is_2d and file_suffix(img.path or Path("x.nii.gz")) in IMAGE2D_SUFFIXES else ".nii.gz"
    )
