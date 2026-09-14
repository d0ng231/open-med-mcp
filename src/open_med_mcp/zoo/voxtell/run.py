"""VoxTell adapter: free-text promptable 3D segmentation (Rokuss et al., CVPR 2026)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402

HF_REPO = "mrokuss/VoxTell"


def _model_dir(job: omm_job.Job, variant: str) -> Path:
    """Checkpoint directory; downloaded from Hugging Face on first use."""
    target = job.weights_dir / variant
    if (target / "plans.json").exists() and (target / "fold_0" / "checkpoint_final.pth").exists():
        return target
    from huggingface_hub import snapshot_download

    job.log(f"downloading {HF_REPO}/{variant} into {job.weights_dir} (about 1.7 GB)")
    snapshot_download(
        repo_id=HF_REPO, allow_patterns=[f"{variant}/*", "*.json"], local_dir=str(job.weights_dir)
    )
    if not (target / "plans.json").exists():
        raise FileNotFoundError(f"checkpoint {variant} not found after download in {job.weights_dir}")
    return target


def _prompts(job: omm_job.Job) -> list[str]:
    texts: list[str] = [str(t).strip() for t in (job.param("text", []) or []) if str(t).strip()]
    for p in job.param("prompts", []) or []:
        if isinstance(p, dict) and p.get("type") == "text" and str(p.get("text", "")).strip():
            texts.append(str(p["text"]).strip())
    # keep order, drop duplicates
    seen: set[str] = set()
    out = []
    for t in texts:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def run(job: omm_job.Job) -> None:
    # text encoder + checkpoints live in the weights directory so containers can reuse them
    os.environ.setdefault("HF_HOME", str(job.weights_dir / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    variant = str(job.param("variant", "voxtell_v1.1"))
    model_dir = _model_dir(job, variant)
    if job.task == "download":
        from huggingface_hub import snapshot_download

        enc = str(job.param("text_encoder", "Qwen/Qwen3-Embedding-4B"))
        job.log(f"downloading text encoder {enc}")
        snapshot_download(repo_id=enc)
        job.finish(
            {},
            stats={"model_dir": str(model_dir), "text_encoder": enc},
            model_info={"adapter": "voxtell", "task": "download"},
        )
        return

    texts = _prompts(job)
    if not texts:
        raise ValueError("VoxTell needs at least one text prompt (params.text or prompts of type 'text')")
    import torch
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor import VoxTellPredictor

    device = omm_job.resolve_device(job.device_pref)
    if device == "cpu":
        job.warn("VoxTell on CPU is very slow; a GPU is strongly recommended")
    image = job.input_path("image")
    if not str(image).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("VoxTell needs a NIfTI input with orientation metadata")
    job.log(f"variant={variant} device={device} prompts={texts}")
    rw = NibabelIOWithReorient()
    img, props = rw.read_images([str(image)])
    job.mark("image_loaded")
    predictor = VoxTellPredictor(
        model_dir=str(model_dir),
        device=torch.device(device),
        text_encoding_model=str(job.param("text_encoder", "Qwen/Qwen3-Embedding-4B")),
    )
    job.mark("model_loaded")
    seg = predictor.predict_single_image(img, texts)  # (n_prompts, x, y, z) in the reader's orientation
    job.mark("predicted")
    seg = np.asarray(seg)
    combined = np.zeros(seg.shape[1:], dtype=np.uint8)
    per_prompt: dict[str, int] = {}
    for i in range(seg.shape[0]):
        fg = seg[i] > 0
        per_prompt[texts[i]] = int(fg.sum())
        combined[fg] = i + 1
    out = job.output_path("mask.nii.gz")
    rw.write_seg(combined, str(out), props)  # restores the original orientation / affine
    if not out.exists():
        raise RuntimeError("VoxTell did not write the segmentation")
    mask, geom = omm_job.load_image(out)
    labels = {i + 1: texts[i] for i in range(len(texts))}
    stats = omm_job.mask_summary(mask, geom)
    stats["voxels_per_prompt"] = per_prompt
    empty = [t for t, n in per_prompt.items() if n == 0]
    if empty:
        job.warn(f"empty result for prompt(s): {empty} - try a more specific or more common phrasing")
    job.finish(
        {"mask": out},
        stats=stats,
        labels={k: v for k, v in labels.items() if k in stats["labels_present"]},
        model_info={"adapter": "voxtell", "variant": variant, "device": device, "prompts": texts},
    )


if __name__ == "__main__":
    omm_job.main(run)
