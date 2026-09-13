"""Template adapter. Replace `segment()` with your model; keep the contract handling as is."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def segment(volume: np.ndarray, threshold: float, weights_dir: Path, device: str) -> np.ndarray:
    # ckpt = weights_dir / "mymodel.pt"      # weights declared in the manifest are found here
    # model = load_model(ckpt, device)
    prob = (volume - volume.min()) / max(float(volume.max() - volume.min()), 1e-6)
    return (prob > threshold).astype(np.uint8)


def run(job: omm_job.Job) -> None:
    image, geom = omm_job.load_image(job.input_path("image"))
    device = omm_job.resolve_device(job.device_pref)
    job.log(f"task={job.task} device={device} shape={image.shape}")
    mask = segment(
        omm_job.scalar(image, geom).astype(np.float32),
        float(job.param("threshold", 0.5)),
        job.weights_dir,
        device,
    )
    out = omm_job.save_mask(mask, geom, job.output_path("mask.nii.gz" if not geom["is_2d"] else "mask.png"))
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels={1: "object"},
        model_info={"adapter": "mymodel"},
    )


if __name__ == "__main__":
    omm_job.main(run)
