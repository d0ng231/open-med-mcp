"""lungmask adapter: lungs (R231) or lobes (LTRCLobes) in chest CT + LAA% emphysema index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402

LUNG_LABELS = {1: "right lung", 2: "left lung"}
LOBE_LABELS = {
    1: "left upper lobe",
    2: "left lower lobe",
    3: "right upper lobe",
    4: "right middle lobe",
    5: "right lower lobe",
}


def run(job: omm_job.Job) -> None:
    os.environ.setdefault("TORCH_HOME", str(job.weights_dir))  # weights land in <weights>/hub/checkpoints
    import SimpleITK as sitk
    from lungmask import LMInferer

    device = omm_job.resolve_device(job.device_pref)
    task = job.task or "lungs"
    model = str(job.param("model", "R231"))
    if task == "lobes":
        model = "LTRCLobes"
    image_path = job.input_path("image")
    img = sitk.ReadImage(str(image_path))
    job.log(f"task={task} model={model} device={device} size={img.GetSize()}")
    inferer = LMInferer(
        modelname=model,
        fillmodel="R231" if model == "LTRCLobes" else None,
        force_cpu=(device == "cpu"),
        batch_size=int(job.param("batch_size", 20) or 20),
        volume_postprocessing=bool(job.param("postprocess", True)),
        tqdm_disable=True,
    )
    job.mark("model_loaded")
    seg = inferer.apply(img).astype(np.uint8)  # numpy [z, y, x] in the input's array order
    job.mark("segmented")
    arr, geom = omm_job.load_image(image_path)
    if seg.shape != arr.shape:
        raise RuntimeError(f"unexpected output shape {seg.shape} vs image {arr.shape}")
    out = omm_job.save_mask(seg, geom, job.output_path("mask.nii.gz"))
    labels = LOBE_LABELS if model == "LTRCLobes" else LUNG_LABELS
    stats = omm_job.mask_summary(seg, geom)
    # emphysema index: fraction of lung voxels below the LAA threshold (only meaningful in HU)
    thr = float(job.param("laa_threshold_hu", -950))
    data = omm_job.scalar(arr, geom)
    laa = {}
    for lab in stats["labels_present"]:
        sel = seg == lab
        n = int(sel.sum())
        if n:
            laa[str(lab)] = {
                "laa_percent": round(100.0 * float((data[sel] < thr).sum()) / n, 2),
                "mean_hu": round(float(data[sel].mean()), 1),
            }
    stats["low_attenuation_area"] = {"threshold_hu": thr, "per_label": laa}
    job.finish(
        {"mask": out},
        stats=stats,
        labels={k: v for k, v in labels.items() if k in stats["labels_present"]},
        model_info={"adapter": "lungmask", "model": model, "device": device},
    )


if __name__ == "__main__":
    omm_job.main(run)
