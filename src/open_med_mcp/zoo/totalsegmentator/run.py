"""TotalSegmentator adapter: automatic multi-organ segmentation of CT/MR volumes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def _prepare_env(job: omm_job.Job) -> Path:
    home = job.weights_dir / "totalsegmentator"
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOTALSEG_HOME_DIR", str(home))
    os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", str(home / "nnunet" / "results"))
    return home


def run(job: omm_job.Job) -> None:
    home = _prepare_env(job)
    task = job.task or "total"
    device = omm_job.resolve_device(job.device_pref)
    job.log(f"task={task} device={device} TOTALSEG_HOME_DIR={home}")

    if task == "download":
        from totalsegmentator.config import setup_nnunet
        from totalsegmentator.libs import download_pretrained_weights
        from totalsegmentator.python_api import get_task_id  # type: ignore[attr-defined]

        setup_nnunet()
        tasks = list(job.param("download_tasks", ["total"]) or ["total"])
        fetched = []
        for t in tasks:
            for task_id in _task_ids(t, bool(job.param("fast", False)), get_task_id):
                download_pretrained_weights(task_id)
                fetched.append(task_id)
        job.finish({}, stats={"downloaded_task_ids": fetched}, model_info={"adapter": "totalsegmentator"})
        return

    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    image = job.input_path("image")
    fast = bool(job.param("fast", False))
    fastest = bool(job.param("fastest", False))
    ts_task = task
    if task == "total_fast":
        ts_task, fast = "total", True
    roi_subset = job.param("roi_subset") or None
    out = job.output_path("mask.nii.gz")
    nthr = int(job.param("nr_threads", 4) or 4)
    job.log(f"running TotalSegmentator task={ts_task} fast={fast} fastest={fastest} roi_subset={roi_subset}")
    totalsegmentator(
        str(image),
        str(out),
        ml=True,
        task=ts_task,
        fast=fast,
        fastest=fastest,
        roi_subset=list(roi_subset) if roi_subset else None,
        robust_crop=bool(job.param("robust_crop", False)),
        device="gpu" if device.startswith("cuda") else "cpu",
        quiet=True,
        verbose=False,
        nr_thr_resamp=nthr,
        nr_thr_saving=nthr,
        skip_saving=False,
    )
    job.mark("segmented")
    if not out.exists():
        raise RuntimeError("TotalSegmentator did not write the multi-label output")
    mask, geom = omm_job.load_image(out)
    cmap_key = ts_task if ts_task in class_map else ("total" if ts_task == "total" else ts_task)
    labels_all = class_map.get(cmap_key, {})
    present = [int(v) for v in np.unique(mask) if v]
    labels = {int(k): str(v) for k, v in labels_all.items() if int(k) in present} or {
        int(v): f"label_{int(v)}" for v in present
    }
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={"adapter": "totalsegmentator", "task": ts_task, "fast": fast, "device": device},
    )


def _task_ids(task: str, fast: bool, get_task_id) -> list[int]:
    try:
        ids = get_task_id(task, fast)  # newer API
    except TypeError:
        ids = get_task_id(task)
    return list(ids) if isinstance(ids, (list, tuple)) else [int(ids)]


if __name__ == "__main__":
    omm_job.main(run)
