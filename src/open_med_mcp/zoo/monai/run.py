"""MONAI bundle adapter: download a Model Zoo bundle and run its inference config on one image."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def _bundle_dir(job: omm_job.Job) -> Path:
    from monai.bundle import download

    name = str(job.param("bundle", "spleen_ct_segmentation"))
    local = Path(name)
    if local.is_dir() and (local / "configs").is_dir():
        return local.resolve()
    root = job.weights_dir
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if not (target / "configs").is_dir():
        job.log(f"downloading bundle {name} into {root}")
        download(name=name, version=job.param("version"), bundle_dir=str(root), progress=False)
    if not (target / "configs").is_dir():
        raise FileNotFoundError(f"bundle {name} not found under {root} after download")
    return target


def _labels(bundle: Path) -> dict[int, str]:
    meta = bundle / "configs" / "metadata.json"
    if not meta.exists():
        return {}
    try:
        outputs = (
            json.loads(meta.read_text(encoding="utf-8")).get("network_data_format", {}).get("outputs", {})
        )
        pred = outputs.get("pred") or next(iter(outputs.values()), {})
        return {int(k): str(v) for k, v in (pred.get("channel_def") or {}).items() if int(k) != 0}
    except Exception:
        return {}


def run(job: omm_job.Job) -> None:
    bundle = _bundle_dir(job)
    if job.task == "download":
        job.finish(
            {}, stats={"bundle_dir": str(bundle)}, model_info={"adapter": "monai", "bundle": bundle.name}
        )
        return
    import torch
    from monai.bundle import run as bundle_run

    device = omm_job.resolve_device(job.device_pref)
    image = job.input_path("image")
    if job.task == "detect":
        _detect(job, bundle, image, device, torch.__version__)
        return
    config = bundle / str(job.param("config", "configs/inference.json"))
    if not config.exists():
        raise FileNotFoundError(f"config {config} not found in bundle")
    # stage the image with a clean name so the bundle's output naming is predictable
    staged_dir = job.dir / "tmp"
    staged_dir.mkdir(exist_ok=True)
    staged = staged_dir / "image.nii.gz"
    shutil.copyfile(image, staged)
    out_dir = job.dir / "bundle_out"
    out_dir.mkdir(exist_ok=True)
    overrides = dict(job.param("overrides", {}) or {})
    overrides.setdefault("device", f"$torch.device('{device}')")
    cfg = json.loads(config.read_text(encoding="utf-8")) if config.suffix == ".json" else {}
    if device == "cpu" and "checkpointloader" in cfg:
        # zoo checkpoints are usually saved from CUDA; map them to the CPU explicitly
        overrides.setdefault("checkpointloader#map_location", "$torch.device('cpu')")
    job.log(f"bundle={bundle.name} config={config.name} device={device} torch={torch.__version__}")
    meta_file = bundle / "configs" / "metadata.json"
    logging_file = bundle / "configs" / "logging.conf"
    try:
        bundle_run(
            run_id="run",
            meta_file=str(meta_file) if meta_file.exists() else None,
            config_file=str(config),
            logging_file=str(logging_file) if logging_file.exists() else None,
            bundle_root=str(bundle),
            datalist=[str(staged)],
            output_dir=str(out_dir),
            **overrides,
        )
    except Exception as exc:  # surface the real cause hidden behind ConfigExpression errors
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        raise RuntimeError(f"{type(cause).__name__}: {cause}") from exc
    job.mark("inference")
    produced = sorted(out_dir.rglob("*.nii*"))
    if not produced:
        raise RuntimeError(f"bundle wrote no NIfTI output under {out_dir}")
    out = job.output_path("mask.nii.gz")
    shutil.copyfile(produced[0], out)
    shutil.rmtree(staged_dir, ignore_errors=True)
    mask, geom = omm_job.load_image(out)
    if not np.issubdtype(mask.dtype, np.integer):
        mask = np.rint(mask).astype(np.uint8)
        omm_job.save_mask(mask, geom, out)
    present = [int(v) for v in np.unique(mask) if v]
    labels = {k: v for k, v in _labels(bundle).items() if k in present} or {v: f"label_{v}" for v in present}
    job.finish(
        {"mask": out},
        stats=omm_job.mask_summary(mask, geom),
        labels=labels,
        model_info={
            "adapter": "monai",
            "bundle": bundle.name,
            "device": device,
            "produced": str(produced[0].relative_to(job.dir)),
        },
    )


def _run_bundle(
    bundle: Path,
    config: Path,
    meta_file: Path,
    logging_file: Path,
    staged: Path,
    out_dir: Path,
    overrides: dict,
) -> None:
    """Call monai.bundle.run with the standard overrides (datalist / output_dir come from ``overrides``)."""
    import sys

    from monai.bundle import run as bundle_run

    # bundles that ship a custom ``scripts`` package (e.g. detection RetinaNetInferer) must import it
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))

    kwargs = dict(overrides)
    kwargs.setdefault("bundle_root", str(bundle))
    kwargs.setdefault("output_dir", str(out_dir))
    if "datalist" not in kwargs and "test_datalist" not in kwargs:
        kwargs["datalist"] = [str(staged)]
    bundle_run(
        run_id="run",
        meta_file=str(meta_file) if meta_file.exists() else None,
        config_file=str(config),
        logging_file=str(logging_file) if logging_file.exists() else None,
        **kwargs,
    )


def _detect(job: omm_job.Job, bundle: Path, image: Path, device: str, torch_version: str) -> None:
    """Object detection by calling the bundle's detector network directly (with its sliding-window
    inferer), bypassing the bundle's evaluation harness so a single arbitrary image can be used."""
    import sys

    import numpy as np
    import SimpleITK as sitk
    import torch
    from monai.bundle import ConfigParser

    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    staged_dir = job.dir / "tmp"
    staged_dir.mkdir(exist_ok=True)
    staged = staged_dir / "image.nii.gz"
    shutil.copyfile(image, staged)
    config = bundle / str(job.param("config", "configs/inference.json"))
    parser = ConfigParser()
    parser.read_config(str(config))
    meta = bundle / "configs" / "metadata.json"
    if meta.exists():
        parser.read_meta(str(meta))
    parser.update(
        pairs={"bundle_root": str(bundle), "whether_raw_luna16": True, "device": f"$torch.device('{device}')"}
    )
    job.log(f"bundle={bundle.name} task=detect device={device} torch={torch_version}")
    # build + configure the detector (detector_ops sets target keys, box selector, sliding-window inferer)
    parser.get_parsed_content("detector_ops")
    detector = parser.get_parsed_content("detector")
    network = parser.get_parsed_content("network")
    ckpt = torch.load(
        str(bundle / "models" / "model.pt"), map_location=torch.device(device), weights_only=False
    )
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    network.load_state_dict(state)
    detector.eval()
    job.mark("model_loaded")
    preprocessing = parser.get_parsed_content("preprocessing")
    data = preprocessing({"image": str(staged)})
    img_t = data["image"]
    affine = np.asarray(
        img_t.affine if hasattr(img_t, "affine") else data["image_meta_dict"]["affine"], dtype=float
    )
    with torch.inference_mode():
        dets = detector([img_t.to(device)], use_inferer=True)
    job.mark("inference")
    det = dets[0]
    box_key = getattr(detector, "target_box_key", "box")
    boxes = det[box_key].detach().cpu().numpy() if box_key in det else det.get("box").detach().cpu().numpy()
    scores = det.get("label_scores", det.get("score"))
    scores = scores.detach().cpu().numpy() if scores is not None else np.zeros(len(boxes))
    labels = det.get("label")
    labels = labels.detach().cpu().numpy() if labels is not None else np.zeros(len(boxes), dtype=int)
    src = sitk.ReadImage(str(image))
    size = src.GetSize()
    thr = float(job.param("score_threshold", 0.1) or 0.0)
    detections = []
    for box, score, lab in zip(boxes, scores, labels, strict=False):
        if float(score) < thr:
            continue
        # box is xyzxyz in the resampled/RAS index space; map both corners to world (RAS) then to
        # the original image's voxel grid (RAS -> LPS for SimpleITK)
        corners_idx = np.array([[box[0], box[1], box[2], 1.0], [box[3], box[4], box[5], 1.0]])
        world_ras = (affine @ corners_idx.T).T[:, :3]
        vox = []
        for wr in world_ras:
            ci = src.TransformPhysicalPointToContinuousIndex((-float(wr[0]), -float(wr[1]), float(wr[2])))
            vox.append(ci)
        lo = [min(vox[0][i], vox[1][i]) for i in range(3)]
        hi = [max(vox[0][i], vox[1][i]) for i in range(3)]
        box_xyz = [max(int(round(lo[i])), 0) for i in range(3)] + [
            min(int(round(hi[i])), size[i] - 1) for i in range(3)
        ]
        size_mm = [round(abs(float(world_ras[1][i] - world_ras[0][i])), 1) for i in range(3)]
        detections.append(
            {
                "score": round(float(score), 4),
                "label": int(lab),
                "box_xyz": box_xyz,
                "center_xyz": [round((box_xyz[i] + box_xyz[i + 3]) / 2, 1) for i in range(3)],
                "size_mm": size_mm,
            }
        )
    detections.sort(key=lambda d: -d["score"])
    detections = detections[: int(job.param("max_detections", 50) or 50)]
    result = {
        "bundle": bundle.name,
        "n_detections": len(detections),
        "score_threshold": thr,
        "detections": detections,
        "note": "boxes are inclusive native voxel coordinates [x0, y0, z0, x1, y1, z1]; scores are model confidences",
    }
    out = job.output_path("detections.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.rmtree(staged_dir, ignore_errors=True)
    job.finish(
        {"detections": out},
        stats={"n_detections": len(detections), "top_scores": [d["score"] for d in detections[:5]]},
        model_info={"adapter": "monai", "bundle": bundle.name, "task": "detect", "device": device},
    )


if __name__ == "__main__":
    omm_job.main(run)
