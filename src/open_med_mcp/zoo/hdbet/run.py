"""HD-BET adapter: MRI brain extraction (mask + skull-stripped image)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def run(job: omm_job.Job) -> None:
    # HD-BET stores its parameters under ~/hd-bet_params; point HOME at the weights directory
    job.weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(job.weights_dir)
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

    device = omm_job.resolve_device(job.device_pref)
    image = job.input_path("image")
    job.log(f"device={device} weights={job.weights_dir / 'hd-bet_params'}")
    maybe_download_parameters()
    job.mark("weights_ready")
    predictor = get_hdbet_predictor(
        use_tta=bool(job.param("tta", False)), device=torch.device(device), verbose=False
    )
    job.mark("model_loaded")
    brain_out = job.output_path("brain.nii.gz")
    save_brain = bool(job.param("save_brain", True))
    hdbet_predict(
        str(image), str(brain_out), predictor, keep_brain_mask=True, compute_brain_extracted_image=save_brain
    )
    job.mark("predicted")
    produced_mask = brain_out.with_name("brain_bet.nii.gz")
    mask_out = job.output_path("mask.nii.gz")
    if not produced_mask.exists():
        raise RuntimeError("HD-BET did not write the brain mask")
    shutil.move(str(produced_mask), str(mask_out))
    mask, geom = omm_job.load_image(mask_out)
    outputs = {"mask": mask_out}
    if save_brain and brain_out.exists():
        outputs["brain"] = brain_out
    job.finish(
        outputs,
        stats=omm_job.mask_summary(mask, geom),
        labels={1: "brain"},
        model_info={"adapter": "hdbet", "device": device, "tta": bool(job.param("tta", False))},
    )


if __name__ == "__main__":
    omm_job.main(run)
