"""Vision-language adapter: one image + prompt -> text, through Hugging Face transformers."""

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


def _to_pil(job: omm_job.Job, image_path: Path):
    from PIL import Image

    arr, geom = omm_job.load_image(image_path)
    if geom["is_2d"]:
        if geom.get("channels_last"):
            return Image.fromarray(arr[..., :3].astype(np.uint8)), {"kind": "2d"}
        u8, wlabel = omm_job.to_uint8(arr, job.param("window"), omm_job.guess_modality(arr, geom))
        return Image.fromarray(u8).convert("RGB"), {"kind": "2d", "window": wlabel}
    plane = str(job.param("plane", "axial"))
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}.get(
        plane, 0
    )  # [z, y, x] array order for standard volumes
    n = arr.shape[axis]
    s = int(job.param("slice")) if job.param("slice") is not None else n // 2
    s = max(0, min(n - 1, s))
    modality = omm_job.guess_modality(arr, geom)
    sl = np.take(arr, s, axis=axis)
    u8, wlabel = omm_job.to_uint8(sl, job.param("window"), modality)
    if axis != 0:
        u8 = u8[::-1]  # superior at the top for coronal / sagittal views of [z, y, x] volumes
    return Image.fromarray(u8).convert("RGB"), {
        "kind": "3d-slice",
        "plane": plane,
        "slice": s,
        "window": wlabel,
        "note": "raw array orientation; use render_view for radiological convention",
    }


def run(job: omm_job.Job) -> None:
    os.environ.setdefault("HF_HOME", str(job.weights_dir / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    model_id = str(job.param("model_id", "google/medgemma-4b-it"))
    if job.task == "download":
        from huggingface_hub import snapshot_download

        job.log(f"downloading {model_id}")
        path = snapshot_download(repo_id=model_id)
        job.finish({}, stats={"model_id": model_id, "path": path}, model_info={"adapter": "vlm"})
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = omm_job.resolve_device(job.device_pref)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        str(job.param("dtype", "bfloat16"))
    ]
    if device == "cpu":
        dtype = torch.float32
        job.warn("running a VLM on CPU is slow; expect minutes")
    image, shown_info = _to_pil(job, job.input_path("image"))
    shown = job.output_path("shown.png")
    image.save(shown)
    prompt = str(job.param("prompt", "Describe the key findings in this image."))
    system_prompt = str(job.param("system_prompt", "") or "")
    job.log(f"model={model_id} device={device} dtype={dtype} image={image.size} {shown_info}")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device if device != "cpu" else None
        )
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            raise RuntimeError(
                f"cannot download {model_id}: it is gated. Accept the licence on https://huggingface.co/{model_id}, run `hf auth login`, or choose an open model such as Qwen/Qwen2.5-VL-3B-Instruct via model_id"
            ) from exc
        raise
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    job.mark("model_loaded")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    )
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    n_in = int(inputs["input_ids"].shape[-1])
    temperature = float(job.param("temperature", 0.0) or 0.0)
    gen_kwargs = {
        "max_new_tokens": int(job.param("max_new_tokens", 400) or 400),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    answer = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
    job.mark("generated")
    result = {
        "model_id": model_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "answer": answer,
        "shown": shown_info,
        "generated_tokens": int(out.shape[-1] - n_in),
        "note": "model-generated text for research use; verify against the image",
    }
    out_path = job.output_path("answer.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    job.finish(
        {"answer": out_path, "shown": shown},
        stats={"generated_tokens": result["generated_tokens"], "answer_preview": answer[:200]},
        model_info={"adapter": "vlm", "model_id": model_id, "device": device},
    )


if __name__ == "__main__":
    omm_job.main(run)
