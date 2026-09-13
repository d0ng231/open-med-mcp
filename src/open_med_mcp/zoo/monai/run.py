"""MONAI bundle adapter: download a Model Zoo bundle and run its inference config on one image."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def _bundle_dir(job: omm_job.Job) -> Path:
    from monai.bundle import download

    name = str(job.param("bundle", "spleen_ct_segmentation"))
    local = Path(name)
    if local.is_dir() and (local / "configs").is_dir():
        return local.resolve()
    root = job.weights_dir
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if not (target / "configs").is_dir():
        job.log(f"downloading bundle {name} into {root}")
        download(name=name, version=job.param("version"), bundle_dir=str(root), progress=False)
    if not (target / "configs").is_dir():
        raise FileNotFoundError(f"bundle {name} not found under {root} after download")
    return target


def _labels(bundle: Path) -> dict[int, str]:
    meta = bundle / "configs" / "metadata.json"
    if not meta.exists():
        return {}
    try:
        outputs = (
            json.loads(meta.read_text(encoding="utf-8")).get("network_data_format", {}).get("outputs", {})
        )
        pred = outputs.get("pred") or next(iter(outputs.values()), {})
        return {int(k): str(v) for k, v in (pred.get("channel_def") or {}).items() if int(k) != 0}
    except Exception:
        return {}


def run(job: omm_job.Job) -> None:
    bundle = _bundle_dir(job)
    if job.task == "download":
        job.finish(
            {}, stats={"bundle_dir": str(bundle)}, model_info={"adapter": "monai", "bundle": bundle.name}
        )
        return
    import torch
    from monai.bundle import run as bundle_run

    device = omm_job.resolve_device(job.device_pref)
    image = job.input_path("image")
    config = bundle / str(job.param("config", "configs/inference.json"))
    if not config.exists():
        raise FileNotFoundError(f"config {config} not found in bundle")
    # stage the image with a clean name so the bundle's output naming is predictable
    staged_dir = job.dir / "tmp"
    staged_dir.mkdir(exist_ok=True)
    staged = staged_dir / "image.nii.gz"
    shutil.copyfile(image, staged)
    out_dir = job.dir / "bundle_out"
    out_dir.mkdir(exist_ok=True)
    overrides = dict(job.param("overrides", {}) or {})
    overrides.setdefault("device", f"$torch.device('{device}')")
    cfg = json.loads(config.read_text(encoding="utf-8")) if config.suffix == ".json" else {}
    if device == "cpu" and "checkpointloader" in cfg:
        # zoo checkpoints are usually saved from CUDA; map them to the CPU explicitly
        overrides.setdefault("checkpointloader#map_location", "$torch.device('cpu')")
    job.log(f"bundle={bundle.name} config={config.name} device={device} torch={torch.__version__}")
    meta_file = bundle / "configs" / "metadata.json"
    logging_file = bundle / "configs" / "logging.conf"
    try:
        bundle_run(
            run_id="run",
            meta_file=str(meta_file) if meta_file.exists() else None,
            config_file=str(config),
            logging_file=str(logging_file) if logging_file.exists() else None,
            bundle_root=str(bundle),
            datalist=[str(staged)],
            output_dir=str(out_dir),
            **overrides,
        )
    except Exception as exc:  # surface the real cause hidden behind ConfigExpression errors
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        raise RuntimeError(f"{type(cause).__name__}: {cause}") from exc
    job.mark("inference")
    produced = sorted(out_dir.rglob("*.nii*"))
    if not produced:
        raise RuntimeError(f"bundle wrote no NIfTI output under {out_dir}")
    out = job.output_path("mask.nii.gz")
    shutil.copyfile(produced[0], out)
    shutil.rmtree(staged_dir, ignore_errors=True)
    mask, geom = omm_job.load_image(out)
    if not np.issubdtype(mask.dtype, np.integer):
        mask = np.rint(mask).astype(np.uint8)
        omm_job.save_mask(mask, geom, out)
    present = [int(v) for v in np.unique(mask) if v]
    labels = {k: v for k, v in _labels(bundle).items() if k in present} or {v: f"label_{v}" for v in present}
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={
            "adapter": "monai",
            "bundle": bundle.name,
            "device": device,
            "produced": str(produced[0].relative_to(job.dir)),
        },
    )


if __name__ == "__main__":
    omm_job.main(run)
