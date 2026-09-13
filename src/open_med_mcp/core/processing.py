"""Image processing primitives built on SimpleITK / scikit-image: resampling, re-orientation,
cropping, N4 bias correction, registration, mask algebra, shape/intensity features and meshes."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, stats

from open_med_mcp.core.image import MedicalImage

Interp = Literal["linear", "nearest", "bspline"]
_INTERP = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor, "bspline": sitk.sitkBSpline}


def _to_sitk(img: MedicalImage) -> sitk.Image:
    return img.to_sitk()


def resample(
    img: MedicalImage,
    spacing: list[float] | tuple[float, ...] | None = None,
    reference: MedicalImage | None = None,
    factor: float | None = None,
    interpolation: Interp = "linear",
    is_mask: bool = False,
    default_value: float | None = None,
) -> MedicalImage:
    """Resample to a new voxel spacing, to the grid of ``reference`` or by an isotropic ``factor``."""
    src = _to_sitk(img)
    interp = sitk.sitkNearestNeighbor if is_mask else _INTERP[interpolation]
    if default_value is None:
        default_value = 0.0 if is_mask else float(np.min(img.scalar_array)) if not img.is_rgb else 0.0
    if reference is not None:
        ref = _to_sitk(reference)
        out = sitk.Resample(src, ref, sitk.Transform(), interp, default_value, src.GetPixelID())
    else:
        old_sp = np.asarray(src.GetSpacing(), dtype=float)
        old_sz = np.asarray(src.GetSize(), dtype=float)
        if spacing is not None:
            new_sp = np.asarray(spacing, dtype=float)
            if new_sp.size == 1:
                new_sp = np.repeat(new_sp, old_sp.size)
        elif factor is not None:
            new_sp = old_sp / float(factor)
        else:
            raise ValueError("give spacing, reference or factor")
        new_sz = [int(max(1, round(s * o / n))) for s, o, n in zip(old_sz, old_sp, new_sp, strict=False)]
        out = sitk.Resample(
            src,
            new_sz,
            sitk.Transform(),
            interp,
            src.GetOrigin(),
            [float(v) for v in new_sp],
            src.GetDirection(),
            default_value,
            src.GetPixelID(),
        )
    res = MedicalImage.from_sitk(out, path=None, fmt=img.format)
    res.metadata.update({k: v for k, v in img.metadata.items() if k != "collapsed_singleton_z"})
    return res


def reorient(img: MedicalImage, orientation: str = "RAS") -> MedicalImage:
    """Re-orient a 3D image's axes to the requested code (e.g. ``RAS``, ``LPS``, ``LPI``)."""
    if img.is_2d:
        raise ValueError("reorientation only applies to 3D images")
    out = sitk.DICOMOrient(_to_sitk(img), orientation.upper())
    res = MedicalImage.from_sitk(out, path=None, fmt=img.format)
    res.metadata.update(img.metadata)
    return res


def crop(img: MedicalImage, box_xyz: list[int], margin: int = 0) -> tuple[MedicalImage, list[int]]:
    """Crop to an inclusive native-index box ``[x0, y0, (z0,) x1, y1, (z1)]`` plus ``margin`` voxels.

    Returns the cropped image (origin updated) and the box actually used."""
    src = _to_sitk(img)
    n = src.GetDimension()
    lo = [int(v) for v in box_xyz[:n]]
    hi = [int(v) for v in box_xyz[n:]]
    size = src.GetSize()
    lo = [max(0, v - margin) for v in lo]
    hi = [min(size[i] - 1, hi[i] + margin) for i in range(n)]
    extent = [hi[i] - lo[i] + 1 for i in range(n)]
    if any(e <= 0 for e in extent):
        raise ValueError(f"empty crop box {box_xyz}")
    out = sitk.RegionOfInterest(src, extent, lo)
    res = MedicalImage.from_sitk(out, path=None, fmt=img.format)
    res.metadata.update(img.metadata)
    res.metadata["crop_box_xyz"] = lo + hi
    return res, lo + hi


def n4_bias_correction(
    img: MedicalImage, mask: MedicalImage | None = None, shrink: int = 4, iterations: list[int] | None = None
) -> tuple[MedicalImage, MedicalImage]:
    """N4 bias field correction (MRI). Returns ``(corrected, bias_field)``."""
    src = sitk.Cast(_to_sitk(img), sitk.sitkFloat32)
    if mask is not None:
        m = sitk.Cast(mask.to_sitk() != 0, sitk.sitkUInt8)
    else:
        m = sitk.OtsuThreshold(src, 0, 1, 200)
    work, work_mask = src, m
    if shrink > 1:
        work = sitk.Shrink(src, [shrink] * src.GetDimension())
        work_mask = sitk.Shrink(m, [shrink] * src.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations or [50, 50, 30, 20]))
    corrector.Execute(work, work_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(src)
    corrected = src / sitk.Exp(log_bias)
    out = MedicalImage.from_sitk(corrected, path=None, fmt=img.format)
    out.metadata.update(img.metadata)
    return out, MedicalImage.from_sitk(sitk.Exp(log_bias), path=None, fmt=img.format)


# --------------------------------------------------------------- registration
TransformKind = Literal["rigid", "affine", "bspline"]


def register(
    fixed: MedicalImage,
    moving: MedicalImage,
    transform: TransformKind = "rigid",
    metric: Literal["mattes", "correlation", "meansquares"] = "mattes",
    sampling: float = 0.2,
    iterations: int = 200,
    initial: Literal["geometry", "moments"] = "geometry",
    bspline_grid_mm: float = 50.0,
    shrink_factors: list[int] | None = None,
    fixed_mask: MedicalImage | None = None,
) -> tuple[sitk.Transform, dict[str, Any]]:
    """Intensity-based registration with SimpleITK (multi-resolution, regular-step gradient descent).

    ``rigid``/``affine`` return a global transform; ``bspline`` runs rigid first and then a
    deformable B-spline refinement and returns a composite transform.
    """
    f = sitk.Cast(fixed.to_sitk(), sitk.sitkFloat32)
    m = sitk.Cast(moving.to_sitk(), sitk.sitkFloat32)
    dim = f.GetDimension()
    if dim == 3:
        init_tx: sitk.Transform = (
            sitk.Euler3DTransform() if transform != "affine" else sitk.AffineTransform(3)
        )
    else:
        init_tx = sitk.Euler2DTransform() if transform != "affine" else sitk.AffineTransform(2)
    centering = (
        sitk.CenteredTransformInitializerFilter.GEOMETRY
        if initial == "geometry"
        else sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    init_tx = sitk.CenteredTransformInitializer(f, m, init_tx, centering)
    shrink = shrink_factors or ([4, 2, 1] if dim == 3 else [2, 1])
    smooth = [max(s - 1, 0) for s in shrink]

    def new_method() -> sitk.ImageRegistrationMethod:
        r = sitk.ImageRegistrationMethod()
        if metric == "mattes":
            r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif metric == "correlation":
            r.SetMetricAsCorrelation()
        else:
            r.SetMetricAsMeanSquares()
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(float(sampling), seed=0)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetShrinkFactorsPerLevel(shrink)
        r.SetSmoothingSigmasPerLevel(smooth)
        r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        if fixed_mask is not None:
            r.SetMetricFixedMask(sitk.Cast(fixed_mask.to_sitk() != 0, sitk.sitkUInt8))
        return r

    reg = new_method()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=int(iterations),
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init_tx, inPlace=False)
    global_tx = reg.Execute(f, m)
    info: dict[str, Any] = {
        "transform": transform,
        "metric": metric,
        "global": {
            "final_metric": float(reg.GetMetricValue()),
            "iterations": int(reg.GetOptimizerIteration()),
            "stop": reg.GetOptimizerStopConditionDescription(),
        },
    }
    if transform != "bspline":
        return global_tx, info
    # deformable refinement on top of the global transform
    mesh = [
        max(int(round(sz * sp / bspline_grid_mm)), 1)
        for sz, sp in zip(f.GetSize(), f.GetSpacing(), strict=False)
    ]
    bspline = sitk.BSplineTransformInitializer(f, mesh, order=3)
    reg2 = new_method()
    reg2.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5, numberOfIterations=max(int(iterations) // 4, 20)
    )
    reg2.SetInitialTransform(bspline, inPlace=True)
    reg2.SetMovingInitialTransform(global_tx)
    reg2.Execute(f, m)
    composite = sitk.CompositeTransform([global_tx, bspline])
    info["bspline"] = {"final_metric": float(reg2.GetMetricValue()), "grid_mm": bspline_grid_mm, "mesh": mesh}
    return composite, info


def apply_transform(
    moving: MedicalImage,
    reference: MedicalImage,
    tx: sitk.Transform,
    is_mask: bool = False,
    interpolation: Interp = "linear",
    default_value: float | None = None,
) -> MedicalImage:
    src = moving.to_sitk()
    ref = reference.to_sitk()
    if default_value is None:
        default_value = 0.0 if is_mask else float(np.min(moving.scalar_array))
    out = sitk.Resample(
        src,
        ref,
        tx,
        sitk.sitkNearestNeighbor if is_mask else _INTERP[interpolation],
        default_value,
        src.GetPixelID(),
    )
    res = MedicalImage.from_sitk(out, path=None, fmt=moving.format)
    res.metadata.update(moving.metadata)
    return res


def write_transform(tx: sitk.Transform, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(path))
    return path


def read_transform(path: Path) -> sitk.Transform:
    return sitk.ReadTransform(str(path))


# ------------------------------------------------------------------- masks
def combine(
    masks: list[np.ndarray], mode: Literal["union", "intersection", "subtract", "label"] = "union"
) -> np.ndarray:
    """Combine label maps. ``union``/``intersection``/``subtract`` work on foreground (result is 0/1);
    ``label`` stacks them so mask *i* becomes label *i+1* (later masks overwrite earlier ones)."""
    if not masks:
        raise ValueError("no masks")
    shapes = {m.shape for m in masks}
    if len(shapes) != 1:
        raise ValueError(f"masks have different shapes: {shapes}")
    fg = [m != 0 for m in masks]
    if mode == "union":
        return np.logical_or.reduce(fg).astype(np.uint8)
    if mode == "intersection":
        return np.logical_and.reduce(fg).astype(np.uint8)
    if mode == "subtract":
        out = fg[0].copy()
        for f in fg[1:]:
            out &= ~f
        return out.astype(np.uint8)
    out = np.zeros(masks[0].shape, dtype=np.uint8)
    for i, f in enumerate(fg, start=1):
        out[f] = i
    return out


def surface_area_mm2(binary: np.ndarray, spacing: tuple[float, ...]) -> float | None:
    """Surface area from a marching-cubes mesh (3D) or contour length (2D)."""
    try:
        from skimage import measure
    except ImportError:  # pragma: no cover
        return None
    if binary.ndim == 3:
        if binary.sum() < 2:
            return None
        padded = np.pad(binary, 1)
        verts, faces, _, _ = measure.marching_cubes(
            padded.astype(np.float32), level=0.5, spacing=tuple(reversed(spacing[:3]))
        )
        return float(measure.mesh_surface_area(verts, faces))
    contours = measure.find_contours(np.pad(binary, 1).astype(np.float32), 0.5)
    sp = np.asarray(list(reversed(spacing[:2])))
    return float(sum(np.sum(np.linalg.norm(np.diff(c, axis=0) * sp, axis=1)) for c in contours))


def mask_features(
    mask: np.ndarray, spacing: tuple[float, ...], image: np.ndarray | None = None, label: int | None = None
) -> dict[str, Any]:
    """Shape and first-order intensity features of one label (radiomics-style, dependency free)."""
    binary = (mask == label) if label else (mask != 0)
    n = int(binary.sum())
    if n == 0:
        return {"voxels": 0}
    sp = np.asarray(spacing[: binary.ndim], dtype=float)
    vox_vol = float(np.prod(sp))
    coords = np.argwhere(binary)[:, ::-1].astype(float) * sp  # native xyz in mm
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T) if n > 1 else np.zeros((binary.ndim, binary.ndim))
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 0, None)
    volume = n * vox_vol
    feats: dict[str, Any] = {
        "voxels": n,
        "volume_mm3": round(volume, 2),
        "volume_ml": round(volume / 1000, 4),
        "centroid_mm": [round(float(c), 2) for c in centroid],
        "bbox_size_mm": [round(float(v), 2) for v in (coords.max(axis=0) - coords.min(axis=0) + sp)],
        "max_extent_mm": round(float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0) + sp)), 2),
        "pca_axes_mm": [round(4 * float(np.sqrt(e)), 2) for e in eig],
        "elongation": round(float(np.sqrt(eig[1] / eig[0])), 4) if eig[0] > 0 and len(eig) > 1 else None,
        "flatness": round(float(np.sqrt(eig[-1] / eig[0])), 4) if eig[0] > 0 and len(eig) > 2 else None,
        "components": int(ndimage.label(binary)[1]),
    }
    area = surface_area_mm2(binary, tuple(spacing))
    if area:
        feats["surface_area_mm2"] = round(area, 2)
        if binary.ndim == 3:
            feats["sphericity"] = round(float((np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3) / area), 4)
            feats["surface_to_volume"] = round(float(area / volume), 5)
        else:
            feats["circularity"] = round(float(4 * np.pi * volume / area**2), 4)
    # equivalent diameter
    feats["equivalent_diameter_mm"] = round(
        float((6 * volume / np.pi) ** (1 / 3)) if binary.ndim == 3 else float(2 * np.sqrt(volume / np.pi)), 2
    )
    if image is not None:
        vals = image[binary].astype(np.float64)
        hist, _ = np.histogram(vals, bins=64)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        q = np.percentile(vals, [10, 25, 50, 75, 90])
        feats["intensity"] = {
            "mean": round(float(vals.mean()), 3),
            "std": round(float(vals.std()), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "p10": round(float(q[0]), 3),
            "p25": round(float(q[1]), 3),
            "median": round(float(q[2]), 3),
            "p75": round(float(q[3]), 3),
            "p90": round(float(q[4]), 3),
            "iqr": round(float(q[3] - q[1]), 3),
            "skewness": round(float(stats.skew(vals)), 4) if vals.size > 2 else None,
            "kurtosis": round(float(stats.kurtosis(vals)), 4) if vals.size > 3 else None,
            "entropy_bits": round(float(-(p * np.log2(p)).sum()), 4),
            "energy": round(float((vals**2).sum()), 1),
        }
    return feats


def mask_to_mesh(
    mask: MedicalImage, label: int | None = None, step: int = 1, smooth_iterations: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes surface in **physical (LPS, mm) coordinates**. Returns ``(vertices, faces)``."""
    from skimage import measure

    if mask.is_2d:
        raise ValueError("meshes need a 3D mask")
    binary = (mask.array == label) if label else (mask.array != 0)
    if binary.sum() < 2:
        raise ValueError("mask is empty")
    padded = np.pad(binary, 1).astype(np.float32)
    if smooth_iterations > 0:
        padded = ndimage.gaussian_filter(padded, sigma=0.7 * smooth_iterations)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, step_size=max(int(step), 1))
    idx = verts[:, ::-1] - 1.0  # (z,y,x) -> (x,y,z) index, undo padding
    d = mask.direction_matrix
    phys = np.asarray(mask.origin)[:3] + (idx * np.asarray(mask.spacing)[:3]) @ d.T
    return phys.astype(np.float32), faces.astype(np.int64)


def write_mesh(verts: np.ndarray, faces: np.ndarray, path: Path) -> Path:
    """Write a binary STL (``.stl``) or Wavefront OBJ (``.obj``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".obj":
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# generated by open-med-mcp (LPS mm)\n")
            for v in verts:
                fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for f in faces:
                fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
        return path
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 0, normals / np.maximum(norms, 1e-12), 0)
    with path.open("wb") as fh:
        fh.write(b"open-med-mcp binary STL".ljust(80, b" "))
        fh.write(struct.pack("<I", len(faces)))
        for nrm, t in zip(normals, tri, strict=False):
            fh.write(struct.pack("<12fH", *nrm.tolist(), *t[0].tolist(), *t[1].tolist(), *t[2].tolist(), 0))
    return path


# ------------------------------------------------------------------- DICOM
_TAGS = {
    "0008|0060": "modality",
    "0008|103e": "series_description",
    "0008|0020": "study_date",
    "0018|0050": "slice_thickness",
    "0008|0070": "manufacturer",
    "0018|1030": "protocol_name",
    "0020|0011": "series_number",
    "0010|0010": "patient_name",
    "0010|0020": "patient_id",
}


def list_dicom_series(folder: Path, recursive: bool = True) -> list[dict[str, Any]]:
    """Enumerate DICOM series below ``folder`` with counts, geometry and key tags."""
    reader = sitk.ImageSeriesReader()
    out: list[dict[str, Any]] = []
    dirs = [folder] + ([p for p in folder.rglob("*") if p.is_dir()] if recursive else [])
    seen: set[str] = set()
    for d in dirs:
        try:
            ids = reader.GetGDCMSeriesIDs(str(d))
        except Exception:
            continue
        for sid in ids:
            if sid in seen:
                continue
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if not files:
                continue
            seen.add(sid)
            entry: dict[str, Any] = {"series_uid": sid, "directory": str(d), "n_files": len(files)}
            try:
                fr = sitk.ImageFileReader()
                fr.SetFileName(files[0])
                fr.LoadPrivateTagsOff()
                fr.ReadImageInformation()
                for tag, name in _TAGS.items():
                    if fr.HasMetaDataKey(tag):
                        entry[name] = fr.GetMetaData(tag).strip()
                entry["size_xy"] = list(fr.GetSize()[:2])
                entry["spacing_xy_mm"] = [round(float(s), 4) for s in fr.GetSpacing()[:2]]
            except Exception as exc:  # unreadable header
                entry["error"] = str(exc)
            out.append(entry)
    out.sort(key=lambda e: (-e["n_files"], e.get("series_number", "")))
    return out
