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


def run(job: omm_job.Job) -> None:
    import torch
    import torchvision
    import torchxrayvision as xrv

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
