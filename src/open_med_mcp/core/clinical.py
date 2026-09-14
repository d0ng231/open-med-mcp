"""Computable clinical criteria used by radiologists: RECIST 1.1 measurements and response,
Fleischner 2017 pulmonary-nodule recommendations, ACR TI-RADS 2017, Agatston coronary calcium
score, cardiothoracic ratio, future liver remnant and the Mayo ADPKD imaging classification.

Every function returns a dict with the numbers, the category and the reference it implements.
They are decision-support helpers, not a substitute for the full published documents.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from open_med_mcp.core.image import MedicalImage, Plane

# ----------------------------------------------------------------------------- RECIST 1.1
RECIST_REF = "Eisenhauer et al., New response evaluation criteria in solid tumours: revised RECIST guideline (version 1.1), Eur J Cancer 45:228-247 (2009)"


def _caliper(points_mm: np.ndarray) -> tuple[float, int, int]:
    """Longest distance between any two of the 2D points (mm); returns (length, i, j)."""
    if len(points_mm) < 2:
        return 0.0, 0, 0
    if len(points_mm) >= 4:
        try:
            hull = ConvexHull(points_mm)
            pts = points_mm[hull.vertices]
            idx = hull.vertices
        except Exception:  # degenerate (collinear) sets
            pts, idx = points_mm, np.arange(len(points_mm))
    else:
        pts, idx = points_mm, np.arange(len(points_mm))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(d[i, j]), int(idx[i]), int(idx[j])


def measure_lesion(
    mask: MedicalImage, label: int | None = None, plane: Plane = "axial", image: MedicalImage | None = None
) -> dict[str, Any]:
    """RECIST-style measurements of one label: longest in-plane diameter across all slices of the
    plane, its perpendicular short axis on the same slice, the slice index, the endpoints (native
    voxel coordinates) and the volume."""
    binary = (mask.array == label) if label else (mask.array != 0)
    if not binary.any():
        return {"label": label, "voxels": 0, "note": "empty mask"}
    if mask.is_2d:
        axis = None
        slices = [0]
        sp_row, sp_col = float(mask.spacing[1]), float(mask.spacing[0])
        row_xyz, col_xyz = 1, 0
    else:
        axis = mask.numpy_axis_for_plane(plane)
        other = [a for a in (0, 1, 2) if a != axis]
        row_xyz, col_xyz = 2 - other[0], 2 - other[1]
        sp_row, sp_col = float(mask.spacing[row_xyz]), float(mask.spacing[col_xyz])
        slices = [int(s) for s in np.nonzero(binary.sum(axis=tuple(other)))[0]]
    best: dict[str, Any] = {"long_axis_mm": 0.0}
    for s in slices:
        sl = binary if axis is None else np.take(binary, s, axis=axis)
        # boundary voxels (use outer edge points: voxel centres +- half spacing along the caliper)
        edge = sl & ~ndimage.binary_erosion(sl)
        rows, cols = np.nonzero(edge)
        if rows.size == 0:
            continue
        pts = np.stack([cols * sp_col, rows * sp_row], axis=1)  # (x, y) in mm
        length, i, j = _caliper(pts)
        length += 0.5 * (sp_col + sp_row)  # account for voxel extent at both ends
        if length > best["long_axis_mm"]:
            direction = pts[j] - pts[i]
            norm = np.linalg.norm(direction) or 1.0
            perp = np.array([-direction[1], direction[0]]) / norm
            proj = pts @ perp
            short = float(proj.max() - proj.min()) + 0.5 * (sp_col + sp_row)
            p_i, p_j = (int(cols[i]), int(rows[i])), (int(cols[j]), int(rows[j]))

            def to_xyz(col: int, row: int, s: int = s) -> list[int]:
                if axis is None:
                    return [col, row]
                out = [0, 0, 0]
                out[col_xyz], out[row_xyz], out[2 - axis] = col, row, s
                return out

            best = {
                "long_axis_mm": round(length, 2),
                "short_axis_mm": round(short, 2),
                "slice": s,
                "plane": plane if axis is not None else "image",
                "long_axis_endpoints_xyz": [to_xyz(*p_i), to_xyz(*p_j)],
            }
    vox = mask.voxel_volume_mm3()
    n = int(binary.sum())
    out: dict[str, Any] = {
        "label": label,
        "voxels": n,
        "volume_ml": round(n * vox / 1000, 3),
        **best,
        "reference": RECIST_REF,
    }
    out["measurable_target_lesion"] = bool(best.get("long_axis_mm", 0) >= 10.0)
    out["note"] = (
        "RECIST 1.1: non-nodal target lesions need a longest diameter >= 10 mm (CT slice thickness <= 5 mm); lymph nodes are measured by short axis (>= 15 mm target, 10-14 mm non-target)"
    )
    if image is not None and image.shape_zyx == mask.shape_zyx:
        vals = image.scalar_array[binary]
        out["intensity"] = {"mean": round(float(vals.mean()), 2), "std": round(float(vals.std()), 2)}
    return out


def recist_response(
    baseline_sld_mm: float,
    current_sld_mm: float,
    nadir_sld_mm: float | None = None,
    new_lesions: bool = False,
    non_target_progression: bool = False,
    all_nodes_short_axis_below_10mm: bool = True,
) -> dict[str, Any]:
    """RECIST 1.1 target-lesion response from the sum of longest diameters (SLD)."""
    nadir = min(
        baseline_sld_mm, nadir_sld_mm if nadir_sld_mm is not None else baseline_sld_mm, current_sld_mm
    )
    pct_vs_baseline = (current_sld_mm - baseline_sld_mm) / baseline_sld_mm * 100 if baseline_sld_mm else 0.0
    ref_nadir = nadir_sld_mm if nadir_sld_mm is not None else baseline_sld_mm
    pct_vs_nadir = (current_sld_mm - ref_nadir) / ref_nadir * 100 if ref_nadir else 0.0
    abs_vs_nadir = current_sld_mm - ref_nadir
    if new_lesions or non_target_progression or (pct_vs_nadir >= 20 and abs_vs_nadir >= 5):
        cat, why = (
            "PD",
            "new lesions / unequivocal non-target progression, or SLD increase >= 20 % vs nadir with an absolute increase >= 5 mm",
        )
    elif current_sld_mm == 0 and all_nodes_short_axis_below_10mm:
        cat, why = "CR", "all target lesions disappeared (nodes < 10 mm short axis)"
    elif pct_vs_baseline <= -30:
        cat, why = "PR", "SLD decrease >= 30 % vs baseline"
    else:
        cat, why = "SD", "neither PR nor PD criteria met"
    return {
        "response": cat,
        "reason": why,
        "baseline_sld_mm": baseline_sld_mm,
        "current_sld_mm": current_sld_mm,
        "nadir_sld_mm": nadir,
        "change_vs_baseline_pct": round(pct_vs_baseline, 1),
        "change_vs_nadir_pct": round(pct_vs_nadir, 1),
        "reference": RECIST_REF,
    }


# ----------------------------------------------------------------------------- Fleischner 2017
FLEISCHNER_REF = "MacMahon et al., Guidelines for Management of Incidental Pulmonary Nodules Detected on CT Images: From the Fleischner Society 2017, Radiology 284:228-243 (2017)"


def fleischner_recommendation(
    nodule_type: Literal["solid", "part-solid", "ground-glass"],
    size_mm: float,
    multiple: bool = False,
    high_risk: bool = False,
) -> dict[str, Any]:
    """Follow-up recommendation for incidental pulmonary nodules in adults >= 35 years (not for
    lung-cancer screening, immunocompromised patients or known cancer). ``size_mm`` is the average
    of long and short axis (rounded to the nearest mm); ``high_risk`` per the guideline's risk factors."""
    t, s = nodule_type, float(size_mm)
    if t == "solid":
        if not multiple:
            if s < 6:
                rec = "optional CT at 12 months" if high_risk else "no routine follow-up"
            elif s <= 8:
                rec = (
                    "CT at 6-12 months, then CT at 18-24 months"
                    if high_risk
                    else "CT at 6-12 months, then consider CT at 18-24 months"
                )
            else:
                rec = "consider CT at 3 months, PET/CT, or tissue sampling"
        else:
            if s < 6:
                rec = "optional CT at 12 months" if high_risk else "no routine follow-up"
            else:
                rec = (
                    "CT at 3-6 months, then CT at 18-24 months"
                    if high_risk
                    else "CT at 3-6 months, then consider CT at 18-24 months"
                )
    else:  # subsolid
        if not multiple:
            if s < 6:
                rec = "no routine follow-up"
            elif t == "ground-glass":
                rec = "CT at 6-12 months to confirm persistence, then CT every 2 years until 5 years"
            else:
                rec = "CT at 3-6 months to confirm persistence; if persistent and solid component < 6 mm, annual CT for 5 years"
        else:
            rec = (
                "CT at 3-6 months; if stable, consider CT at 2 and 4 years"
                if s < 6
                else "CT at 3-6 months; subsequent management based on the most suspicious nodule"
            )
    return {
        "nodule_type": t,
        "size_mm": s,
        "multiple": multiple,
        "high_risk": high_risk,
        "recommendation": rec,
        "scope": "incidental nodules, adults >= 35 years; not for screening (use Lung-RADS), immunocompromised patients or patients with known cancer",
        "reference": FLEISCHNER_REF,
    }


# ----------------------------------------------------------------------------- ACR TI-RADS 2017
TIRADS_REF = "Tessler et al., ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White Paper of the ACR TI-RADS Committee, J Am Coll Radiol 14:587-595 (2017)"
_TIRADS_POINTS = {
    "composition": {"cystic": 0, "spongiform": 0, "mixed": 1, "solid": 2, "almost completely solid": 2},
    "echogenicity": {"anechoic": 0, "hyperechoic": 1, "isoechoic": 1, "hypoechoic": 2, "very hypoechoic": 3},
    "shape": {"wider-than-tall": 0, "taller-than-wide": 3},
    "margin": {"smooth": 0, "ill-defined": 0, "lobulated": 2, "irregular": 2, "extra-thyroidal extension": 3},
    "echogenic_foci": {
        "none": 0,
        "comet-tail": 0,
        "macrocalcifications": 1,
        "peripheral": 2,
        "rim": 2,
        "punctate": 3,
    },
}


def tirads_score(
    composition: str,
    echogenicity: str,
    shape: str,
    margin: str,
    echogenic_foci: list[str] | None = None,
    max_diameter_mm: float | None = None,
) -> dict[str, Any]:
    """ACR TI-RADS points, level (TR1-TR5) and the size-based FNA / follow-up recommendation."""
    foci = [f.lower() for f in (echogenic_foci or ["none"])]
    pts = {
        "composition": _TIRADS_POINTS["composition"][composition.lower()],
        "echogenicity": _TIRADS_POINTS["echogenicity"][echogenicity.lower()],
        "shape": _TIRADS_POINTS["shape"][shape.lower()],
        "margin": _TIRADS_POINTS["margin"][margin.lower()],
        "echogenic_foci": sum(_TIRADS_POINTS["echogenic_foci"][f] for f in foci),
    }
    total = sum(pts.values())
    if composition.lower() in ("cystic", "spongiform") and total == 0:
        level = "TR1"
    elif total <= 2:
        level = "TR2" if total == 2 else "TR1"
    elif total == 3:
        level = "TR3"
    elif total <= 6:
        level = "TR4"
    else:
        level = "TR5"
    fna = {"TR1": None, "TR2": None, "TR3": 25.0, "TR4": 15.0, "TR5": 10.0}[level]
    follow = {"TR1": None, "TR2": None, "TR3": 15.0, "TR4": 10.0, "TR5": 5.0}[level]
    rec = "no FNA or follow-up"
    if fna is not None:
        if max_diameter_mm is None:
            rec = f"FNA if >= {fna:.0f} mm; follow-up if >= {follow:.0f} mm"
        elif max_diameter_mm >= fna:
            rec = f"FNA recommended (>= {fna:.0f} mm)"
        elif max_diameter_mm >= follow:
            rec = f"follow-up ultrasound recommended (>= {follow:.0f} mm, < {fna:.0f} mm)"
        else:
            rec = f"no FNA or follow-up (< {follow:.0f} mm)"
    return {
        "points": pts,
        "total_points": total,
        "level": level,
        "max_diameter_mm": max_diameter_mm,
        "recommendation": rec,
        "reference": TIRADS_REF,
    }


# ----------------------------------------------------------------------------- Agatston
AGATSTON_REF = "Agatston et al., Quantification of coronary artery calcium using ultrafast computed tomography, J Am Coll Cardiol 15:827-832 (1990)"


def agatston_score(
    image: MedicalImage,
    mask: MedicalImage,
    threshold_hu: float = 130.0,
    min_area_mm2: float = 1.0,
    labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Agatston score of calcified plaque inside ``mask`` (e.g. coronary arteries or the heart on a
    non-contrast, ECG-gated CT): per axial slice, connected components >= ``threshold_hu`` with area
    >= ``min_area_mm2`` contribute area x density factor (1-4); scores are scaled to the 3 mm slice
    convention (thickness / 3)."""
    if image.is_2d or image.shape_zyx != mask.shape_zyx:
        raise ValueError("Agatston scoring needs a 3D image and a mask with the same geometry")
    axis = image.numpy_axis_for_plane("axial")
    others = [a for a in (0, 1, 2) if a != axis]
    sp = image.spacing
    in_plane = [float(sp[2 - a]) for a in others]
    area_vox = in_plane[0] * in_plane[1]
    thickness = float(sp[2 - axis])
    scale = thickness / 3.0
    data = image.scalar_array
    total = 0.0
    volume_mm3 = 0.0
    per_label: dict[int, float] = {}
    n_lesions = 0
    for lab in [int(v) for v in np.unique(mask.array) if v]:
        region = mask.array == lab
        score_lab = 0.0
        for s in range(image.shape_zyx[axis]):
            sl = np.take(region, s, axis=axis)
            if not sl.any():
                continue
            calc = np.take(data, s, axis=axis) >= threshold_hu
            comp, n = ndimage.label(sl & calc)
            for i in range(1, n + 1):
                c = comp == i
                area = float(c.sum()) * area_vox
                if area < min_area_mm2:
                    continue
                peak = float(np.take(data, s, axis=axis)[c].max())
                factor = 1 if peak < 200 else 2 if peak < 300 else 3 if peak < 400 else 4
                score_lab += area * factor * scale
                volume_mm3 += area * thickness
                n_lesions += 1
        per_label[lab] = round(score_lab, 1)
        total += score_lab
    if total == 0:
        cat = "0 (no identifiable plaque)"
    elif total <= 10:
        cat = "1-10 (minimal)"
    elif total <= 100:
        cat = "11-100 (mild)"
    elif total <= 400:
        cat = "101-400 (moderate)"
    else:
        cat = "> 400 (severe)"
    return {
        "agatston_total": round(total, 1),
        "per_label": {str(k): {"name": (labels or {}).get(k), "agatston": v} for k, v in per_label.items()},
        "calcium_volume_mm3": round(volume_mm3, 1),
        "n_lesions": n_lesions,
        "category": cat,
        "threshold_hu": threshold_hu,
        "slice_thickness_mm": thickness,
        "note": "valid for non-contrast ECG-gated CT; contrast-enhanced scans and motion artefacts invalidate the score; age/sex percentiles (e.g. MESA) are needed for risk interpretation",
        "reference": AGATSTON_REF,
    }


# ----------------------------------------------------------------------------- chest X-ray CTR
def cardiothoracic_ratio(
    mask: MedicalImage, heart_labels: list[int], thorax_labels: list[int]
) -> dict[str, Any]:
    """Cardiothoracic ratio on a frontal radiograph: widest horizontal heart extent divided by the
    widest horizontal extent of the thorax (here: the lungs' outer boundaries)."""
    arr = mask.array
    heart = np.isin(arr, heart_labels)
    thorax = np.isin(arr, thorax_labels)
    if not heart.any() or not thorax.any():
        raise ValueError("heart or thorax labels are empty in the mask")

    def width(b: np.ndarray) -> tuple[float, int, int, int]:
        cols = np.nonzero(b.any(axis=0))[0]
        best_row = int(np.argmax(b.sum(axis=1)))
        return float(cols.max() - cols.min() + 1), int(cols.min()), int(cols.max()), best_row

    hw, hx0, hx1, hrow = width(heart)
    tw, tx0, tx1, trow = width(thorax)
    sx = float(mask.spacing[0])
    ratio = hw / tw
    return {
        "ctr": round(ratio, 3),
        "heart_width_px": hw,
        "thorax_width_px": tw,
        "heart_width_mm": round(hw * sx, 1) if sx != 1.0 else None,
        "thorax_width_mm": round(tw * sx, 1) if sx != 1.0 else None,
        "heart_extent_x": [hx0, hx1],
        "thorax_extent_x": [tx0, tx1],
        "enlarged": bool(ratio > 0.5),
        "note": "CTR > 0.5 on a PA erect radiograph suggests cardiomegaly; AP / supine films overestimate the ratio. Thorax width here uses the lung outer margins (an approximation of the inner rib margins).",
        "reference": "Danzer, The cardiothoracic ratio: an index of cardiac enlargement, Am J Med Sci 157:513 (1919); Dimopoulos et al., Int J Cardiol 167:e59 (2013) (CTR in adults)",
    }


# ----------------------------------------------------------------------------- liver / kidney
def future_liver_remnant(
    total_liver_ml: float,
    remnant_ml: float,
    tumor_ml: float = 0.0,
    liver_condition: Literal["healthy", "chemotherapy", "cirrhosis"] = "healthy",
    body_weight_kg: float | None = None,
) -> dict[str, Any]:
    """Standardized future liver remnant as a fraction of the functional (tumor-free) liver, with the
    commonly used minimum thresholds (about 20 % healthy, 30 % after chemotherapy, 40 % cirrhosis)."""
    functional = total_liver_ml - tumor_ml
    flr_pct = remnant_ml / functional * 100 if functional > 0 else 0.0
    threshold = {"healthy": 20.0, "chemotherapy": 30.0, "cirrhosis": 40.0}[liver_condition]
    out: dict[str, Any] = {
        "functional_liver_ml": round(functional, 1),
        "remnant_ml": remnant_ml,
        "flr_percent": round(flr_pct, 1),
        "threshold_percent": threshold,
        "adequate": bool(flr_pct >= threshold),
        "reference": "Clavien et al., Strategies for safer liver surgery and partial liver transplantation, N Engl J Med 356:1545-1559 (2007); Vauthey et al., Surgery 127:512 (2000) (standardized FLR)",
    }
    if body_weight_kg:
        out["remnant_to_body_weight_ratio_pct"] = round(remnant_ml / (body_weight_kg * 1000) * 100, 2)
        out["rlbw_note"] = (
            "remnant liver to body weight ratio < 0.5 % is associated with liver failure (Truant et al., J Am Coll Surg 2007)"
        )
    return out


def mayo_adpkd_class(total_kidney_volume_ml: float, height_m: float, age_years: float) -> dict[str, Any]:
    """Mayo Imaging Classification of typical ADPKD (class 1A-1E) from height-adjusted total kidney
    volume and age (Irazabal et al., J Am Soc Nephrol 26:160-172, 2015)."""
    ht_tkv = total_kidney_volume_ml / height_m
    if age_years <= 20:
        return {
            "htTKV_ml_per_m": round(ht_tkv, 1),
            "class": None,
            "note": "classification is defined for ages > 20",
            "reference": "Irazabal et al., JASN 26:160-172 (2015)",
        }
    rate = ((ht_tkv / 150.0) ** (1.0 / (age_years - 20.0)) - 1.0) * 100.0
    if rate < 1.5:
        cls = "1A"
    elif rate < 3.0:
        cls = "1B"
    elif rate < 4.5:
        cls = "1C"
    elif rate < 6.0:
        cls = "1D"
    else:
        cls = "1E"
    return {
        "htTKV_ml_per_m": round(ht_tkv, 1),
        "estimated_growth_rate_pct_per_year": round(rate, 2),
        "class": cls,
        "note": "classes 1C-1E (>= 3 %/year) indicate rapidly progressive disease; requires typical (class 1) bilateral diffuse cystic disease",
        "reference": "Irazabal et al., Imaging classification of autosomal dominant polycystic kidney disease, J Am Soc Nephrol 26:160-172 (2015)",
    }
