"""Use open-med-mcp from Python (no agent): run a model, post-process, measure, render.

python examples/python_api.py sample/example_ct_sm.nii.gz
"""

from __future__ import annotations

import sys
from pathlib import Path

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import MedicalImage, load_mask, save_mask
from open_med_mcp.core.masks import mask_stats, mask_to_prompts, postprocess
from open_med_mcp.core.prompts import normalize_prompts
from open_med_mcp.models.job import prepare_job, read_response
from open_med_mcp.models.registry import get_registry
from open_med_mcp.models.runners import resolve_device, select_runner
from open_med_mcp.models.weights import ensure_weights, weights_dir_for
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import new_run_dir


def run_model(name: str, task: str, image: Path, params: dict) -> dict:
    settings = get_settings()
    settings.ensure_dirs()
    manifest = get_registry(settings).get(name)
    if manifest.weights:
        ensure_weights(manifest, [params.get("variant")] if params.get("variant") else None, settings)
    run_dir = new_run_dir(name, settings)
    prepare_job(
        run_dir,
        manifest,
        task,
        {"image": image},
        params,
        resolve_device(settings),
        weights_dir_for(manifest, settings),
    )
    select_runner(manifest, settings).run(
        manifest, run_dir, weights_dir_for(manifest, settings), resolve_device(settings)
    )
    return read_response(run_dir)


def main(image_path: str) -> None:
    image = Path(image_path).resolve()
    img = MedicalImage.load(image)
    print("image:", img.describe()["size_xyz"], img.spacing, img.orientation_code(), img.modality_guess())

    # 1. coarse mask with the classical adapter: bright bone
    resp = run_model("classical", "threshold", image, {"lower": 250, "upper": 3000, "keep": "largest"})
    mask_path = Path(resp["outputs"]["mask"])
    mask = load_mask(mask_path, img)
    clean, log = postprocess(mask.array, ["fill_holes"], mask.spacing)
    clean_path = save_mask(clean, img, mask_path.with_name("bone_clean.nii.gz"))
    print(
        "post-processing:", log, mask_stats(load_mask(clean_path, img), img)["labels"][0]["volume_ml"], "mL"
    )

    # 2. prompts for a promptable model (medsam2) derived from the coarse mask
    prompts = mask_to_prompts(load_mask(clean_path, img), "axial")
    wire = normalize_prompts([{"type": "box", "coords": prompts["boxes"][0]}], img, "axial")
    print("box prompt (native xyz):", prompts["boxes"][0], "-> wire:", wire)
    try:
        resp = run_model(
            "medsam2",
            "segment",
            image,
            {"prompts": wire, "axis": img.numpy_axis_for_plane("axial"), "window": "bone"},
        )
        print("medsam2 mask:", resp["outputs"]["mask"], resp["timing"])
    except RuntimeError as exc:
        print("medsam2 not runnable here:", str(exc).splitlines()[0])

    # 3. render a three-plane figure
    layer = MaskLayer(path=str(clean_path), name="bone")
    spec = ViewSpec(
        image=str(image), masks=[layer], layout="three-plane", window="bone", title="classical bone mask"
    )
    out = get_renderer("png").render(
        img, [(load_mask(clean_path, img), {"layer": layer, "labels": {1: "bone"}})], spec
    )
    dest = out.save(image.with_name("bone_three_plane.png"))
    print("figure:", dest)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample/example_ct_sm.nii.gz")
