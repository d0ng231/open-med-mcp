"""TorchXRayVision adapter: chest X-ray multi-label pathology probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def _prep(job: omm_job.Job, size: int):
    import torchvision
    import torchxrayvision as xrv

    arr, geom = omm_job.load_image(job.input_path("image"))
    img = omm_job.scalar(arr, geom).astype(np.float32)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D radiograph, got shape {img.shape}")
    maxval = 255.0 if arr.dtype == np.uint8 else float(max(img.max(), 1.0))
    norm = xrv.datasets.normalize(img, maxval)[None, ...]
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)]
    )
    return arr, geom, transform(norm)


def segment_anatomy(job: omm_job.Job, device: str) -> None:
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    cache = job.weights_dir / "models_data"
    cache.mkdir(parents=True, exist_ok=True)
    arr, geom, img = _prep(job, 512)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(cache)).to(device).eval()
    job.mark("model_loaded")
    with torch.inference_mode():
        out = model(torch.from_numpy(img).unsqueeze(0).to(device))[0].float().cpu().numpy()  # (14, 512, 512)
    targets = [str(t) for t in model.targets]
    probs = 1.0 / (1.0 + np.exp(-out))  # PSPNet returns logits; sigmoid -> per-structure probability
    h, w = arr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2

    resized = (
        np.stack(
            [
                np.asarray(
                    Image.fromarray((probs[i] * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)
                )
                for i in range(len(targets))
            ]
        ).astype(np.float32)
        / 255.0
    )
    fg = resized.max(axis=0) > 0.5
    winner = resized.argmax(axis=0) + 1  # each pixel -> the structure with the highest probability
    label_map = np.zeros((h, w), dtype=np.uint8)
    crop = np.zeros((side, side), dtype=np.uint8)
    crop[fg] = winner[fg]
    label_map[y0 : y0 + side, x0 : x0 + side] = crop
    out_path = omm_job.save_mask(label_map, geom, job.output_path("mask.png"))
    labels = {i + 1: targets[i] for i in range(len(targets)) if (label_map == i + 1).any()}
    job.finish(
        {"mask": out_path},
        stats=omm_job.mask_summary(label_map, geom),
        labels=labels,
        model_info={
            "adapter": "torchxrayvision",
            "task": "segment",
            "model": "chestx_det PSPNet",
            "device": device,
        },
    )


def estimate_age(job: omm_job.Job, device: str) -> None:
    import torch
    import torchxrayvision as xrv

    cache = job.weights_dir / "models_data"
    cache.mkdir(parents=True, exist_ok=True)
    xrv.utils.get_cache_dir = lambda: str(cache) + "/"  # the age model has no cache_dir argument
    _, _, img = _prep(job, 512)
    model = xrv.baseline_models.riken.AgeModel().to(device).eval()
    job.mark("model_loaded")
    with torch.inference_mode():
        age = float(model(torch.from_numpy(img).unsqueeze(0).to(device))[0].float().cpu().numpy().ravel()[0])
    result = {
        "estimated_age_years": round(age, 1),
        "model": "RIKEN xray-age (Ieki et al., 2022)",
        "note": "biological age estimate from the radiograph; a large gap to the chronological age has been associated with cardiovascular risk in the original study",
    }
    out_path = job.output_path("predictions.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    job.finish(
        {"predictions": out_path},
        stats=result,
        model_info={"adapter": "torchxrayvision", "task": "age", "device": device},
    )


def run(job: omm_job.Job) -> None:
    import torch
    import torchvision
    import torchxrayvision as xrv

    if job.task == "segment":
        segment_anatomy(job, omm_job.resolve_device(job.device_pref))
        return
    if job.task == "age":
        estimate_age(job, omm_job.resolve_device(job.device_pref))
        return

    device = omm_job.resolve_device(job.device_pref)
    weights = str(job.param("weights", "densenet121-res224-all"))
    top_k = int(job.param("top_k", 5) or 5)
    arr, geom = omm_job.load_image(job.input_path("image"))
    img = omm_job.scalar(arr, geom).astype(np.float32)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D radiograph, got shape {img.shape}")
    maxval = 255.0 if arr.dtype == np.uint8 else float(max(img.max(), 1.0))
    img = xrv.datasets.normalize(img, maxval)  # -> roughly [-1024, 1024]
    img = img[None, ...]
    size = 512 if "res512" in weights else 224
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)]
    )
    img = transform(img)
    cache = job.weights_dir / "models_data"
    cache.mkdir(parents=True, exist_ok=True)
    job.log(f"weights={weights} device={device} input={arr.shape} -> {img.shape}")
    model = (
        xrv.models.ResNet(weights=weights, cache_dir=str(cache))
        if weights.startswith("resnet")
        else xrv.models.DenseNet(weights=weights, cache_dir=str(cache))
    )
    model = model.to(device).eval()
    job.mark("model_loaded")
    with torch.inference_mode():
        out = model(torch.from_numpy(img).unsqueeze(0).to(device))[0].float().cpu().numpy()
    probs = {}
    for name, value in zip(model.pathologies, out, strict=False):
        if name and not np.isnan(value):
            probs[str(name)] = round(float(value), 4)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    results = {
        "weights": weights,
        "input_resolution": size,
        "probabilities": probs,
        "top": [{"finding": k, "probability": v} for k, v in ranked[:top_k]],
        "note": "Calibrated model probabilities for research use; not a diagnosis. Check image orientation/quality first.",
    }
    out_path = job.output_path("predictions.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    job.finish(
        {"predictions": out_path},
        stats={"n_findings": len(probs), "top": results["top"]},
        model_info={"adapter": "torchxrayvision", "weights": weights, "device": device},
    )


if __name__ == "__main__":
    omm_job.main(run)
