"""pyradiomics adapter: standard radiomic features per label."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_sdk"))
import omm_job  # noqa: E402


def run(job: omm_job.Job) -> None:
    import logging

    import SimpleITK as sitk
    from radiomics import featureextractor, logger

    logger.setLevel(logging.ERROR)
    image_path, mask_path = job.input_path("image"), job.input_path("mask")
    img = sitk.ReadImage(str(image_path))
    msk = sitk.ReadImage(str(mask_path))
    if img.GetDimension() == 2:
        msk = sitk.Cast(msk, sitk.sitkUInt8)
    settings: dict = {
        "minimumROIDimensions": 2,
        "minimumROISize": int(job.param("minimum_roi_voxels", 10) or 10),
        "geometryTolerance": 1e-3,
        "correctMask": True,
    }
    if job.param("bin_count"):
        settings["binCount"] = int(job.param("bin_count"))
    else:
        settings["binWidth"] = float(job.param("bin_width", 25) or 25)
    if job.param("resample_spacing"):
        settings["resampledPixelSpacing"] = [float(v) for v in job.param("resample_spacing")]
        settings["interpolator"] = "sitkBSpline"
    if job.param("normalize", False):
        settings["normalize"] = True
    settings["force2D"] = bool(job.param("force_2d", False))
    if job.param("log_sigma"):
        settings["sigma"] = [float(v) for v in job.param("log_sigma")]
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    classes = list(
        job.param("feature_classes", ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]) or []
    )
    if img.GetDimension() == 2 or settings["force2D"]:
        classes = ["shape2D" if c == "shape" else c for c in classes]
    for c in classes:
        extractor.enableFeatureClassByName(c)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    for f in job.param("filters", []) or []:
        extractor.enableImageTypeByName(str(f))
    arr = sitk.GetArrayFromImage(msk)
    present = [int(v) for v in np.unique(arr) if v]
    wanted = [int(v) for v in (job.param("labels") or present) if int(v) in present]
    job.log(
        f"labels={wanted} classes={classes} settings={ {k: v for k, v in settings.items() if k in ('binWidth', 'binCount', 'resampledPixelSpacing', 'normalize', 'force2D')} }"
    )
    results: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, str]] = {}
    for lab in wanted:
        try:
            feats = extractor.execute(img, msk, label=lab)
        except Exception as exc:  # e.g. ROI too small
            job.warn(f"label {lab}: {exc}")
            continue
        vals: dict[str, float] = {}
        diag: dict[str, str] = {}
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                diag[k] = str(v)
            else:
                try:
                    vals[k] = float(np.asarray(v).ravel()[0])
                except Exception:
                    vals[k] = float("nan")
        results[str(lab)] = vals
        diagnostics[str(lab)] = diag
    if not results:
        raise RuntimeError("no features extracted (all labels failed - see warnings)")
    out_json = job.output_path("features.json")
    out_json.write_text(
        json.dumps(
            {
                "settings": {
                    k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in settings.items()
                },
                "feature_classes": classes,
                "filters": list(job.param("filters", []) or []),
                "labels": results,
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_csv = job.output_path("features.csv")
    names = sorted({k for v in results.values() for k in v})
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", *names])
        for lab, vals in results.items():
            w.writerow([lab, *[vals.get(n, "") for n in names]])
    job.finish(
        {"features": out_json, "features_csv": out_csv},
        stats={"n_labels": len(results), "n_features": len(names)},
        model_info={"adapter": "radiomics", "pyradiomics": __import__("radiomics").__version__},
    )


if __name__ == "__main__":
    omm_job.main(run)
