"""nnU-Net v2 adapter: predict with any trained model folder; install model zips."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def _setup_env(job: omm_job.Job) -> Path:
    root = job.weights_dir / "nnunet"
    for sub in ("raw", "preprocessed", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("nnUNet_raw", str(root / "raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(root / "preprocessed"))
    os.environ.setdefault("nnUNet_results", str(root / "results"))
    return root / "results"


def _resolve_model_dir(job: omm_job.Job, results: Path) -> Path:
    model_dir = job.param("model_dir")
    if model_dir:
        p = Path(str(model_dir))
        candidates = [p, results / p, job.weights_dir / p]
    else:
        dataset = job.param("dataset")
        if not dataset:
            raise ValueError("give `model_dir` or `dataset` (+ configuration/trainer/plans)")
        folder = f"{job.param('trainer', 'nnUNetTrainer')}__{job.param('plans', 'nnUNetPlans')}__{job.param('configuration', '3d_fullres')}"
        candidates = [results / str(dataset) / folder]
    for c in candidates:
        if (c / "plans.json").exists() and (c / "dataset.json").exists():
            return c
    raise FileNotFoundError(
        f"no nnU-Net model folder with plans.json/dataset.json among {[str(c) for c in candidates]}"
    )


def _labels_from_dataset_json(model_dir: Path) -> dict[int, str]:
    ds = json.loads((model_dir / "dataset.json").read_text(encoding="utf-8"))
    labels = ds.get("labels", {})
    out: dict[int, str] = {}
    for name, value in labels.items():
        ids = value if isinstance(value, (list, tuple)) else [value]
        for v in ids:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv != 0 and iv not in out:
                out[iv] = str(name)
    return out


def run(job: omm_job.Job) -> None:
    results = _setup_env(job)
    task = job.task or "predict"
    if task == "install":
        from nnunetv2.model_sharing.model_import import install_model_from_zip_file

        zip_path = job.param("zip_path")
        if not zip_path and job.param("zip_url"):
            import urllib.request

            zip_path = str(job.dir / "model.zip")
            job.log(f"downloading {job.param('zip_url')}")
            urllib.request.urlretrieve(str(job.param("zip_url")), zip_path)
        if not zip_path:
            raise ValueError("install needs zip_path or zip_url")
        install_model_from_zip_file(str(zip_path))
        installed = sorted(p.name for p in results.iterdir() if p.is_dir())
        job.finish(
            {},
            stats={"installed_datasets": installed, "results_dir": str(results)},
            model_info={"adapter": "nnunet", "task": "install"},
        )
        return

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = omm_job.resolve_device(job.device_pref)
    model_dir = _resolve_model_dir(job, results)
    folds_param = job.param("folds")
    if folds_param:
        folds = tuple(int(f) for f in folds_param)
    else:
        found = sorted(
            int(p.name.split("_")[1]) for p in model_dir.glob("fold_*") if p.name.split("_")[1].isdigit()
        )
        folds = tuple(found) if found else ("all",)
    job.log(f"model_dir={model_dir} folds={folds} device={device}")
    predictor = nnUNetPredictor(
        tile_step_size=float(job.param("tile_step_size", 0.5)),
        use_gaussian=True,
        use_mirroring=bool(job.param("tta", False)),
        perform_everything_on_device=device.startswith("cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=folds, checkpoint_name=str(job.param("checkpoint", "checkpoint_final.pth"))
    )
    job.mark("model_loaded")
    channels = [job.input_path("image")]
    i = 1
    while f"image_{i}" in job.inputs:
        channels.append(job.input_path(f"image_{i}"))
        i += 1
    out_prefix = job.output_path("mask")
    predictor.predict_from_files(
        [[str(c) for c in channels]],
        [str(out_prefix)],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    job.mark("predicted")
    ds = json.loads((model_dir / "dataset.json").read_text(encoding="utf-8"))
    ending = ds.get("file_ending", ".nii.gz")
    out = out_prefix.with_name("mask" + ending)
    if not out.exists():
        raise RuntimeError(f"nnU-Net did not write {out}")
    mask, geom = omm_job.load_image(out)
    present = [int(v) for v in np.unique(mask) if v]
    labels = {k: v for k, v in _labels_from_dataset_json(model_dir).items() if k in present}
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={"adapter": "nnunet", "model_dir": str(model_dir), "folds": list(folds), "device": device},
    )


if __name__ == "__main__":
    omm_job.main(run)
