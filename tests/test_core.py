import numpy as np
import pytest

from open_med_mcp.core.image import MedicalImage
from open_med_mcp.core.masks import bbox_xyz, mask_stats, mask_to_prompts, postprocess
from open_med_mcp.core.metrics import compare_binary, compare_masks
from open_med_mcp.core.prompts import Prompt, normalize_prompts
from open_med_mcp.core.windowing import CT_WINDOWS, resolve_window, to_uint8


def test_windows():
    arr = np.linspace(-1200, 1500, 1000).astype(np.float32)
    lo, hi, label = resolve_window("lung", arr, "CT")
    assert (lo, hi) == (-1350.0, 150.0) and label == "lung"
    lo, hi, _ = resolve_window({"center": 0, "width": 100}, arr)
    assert (lo, hi) == (-50.0, 50.0)
    lo, hi, _ = resolve_window([10, 20], arr)
    assert (lo, hi) == (10.0, 20.0)
    lo, hi, _ = resolve_window(None, arr, "CT")
    assert (lo, hi) == (-160.0, 240.0)
    u8, _ = to_uint8(arr, "auto")
    assert u8.dtype == np.uint8 and u8.min() == 0 and u8.max() == 255
    with pytest.raises(ValueError):
        resolve_window("nope", arr)
    assert "bone" in CT_WINDOWS


def test_mask_stats_and_bbox(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    mask = img.with_array(ct_volume["ball"].astype(np.uint8))
    st = mask_stats(mask, img)
    lab = st["labels"][0]
    assert lab["voxels"] == int(ct_volume["ball"].sum())
    assert abs(lab["volume_ml"] - lab["voxels"] * 2.5 / 1000) < 1e-3
    assert lab["components"] == 1
    assert lab["intensity"]["mean"] > 250
    assert lab["slice_range"]["axial"][0] <= lab["slice_range"]["axial"][1]
    assert bbox_xyz(ct_volume["ball"]) == lab["bbox_xyz"]
    assert len(lab["bbox_xyz"]) == 6


def test_postprocess_ops():
    m = np.zeros((10, 20, 20), dtype=np.uint8)
    m[2:8, 2:10, 2:10] = 1
    m[5, 5, 5] = 0  # hole
    m[1, 15, 15] = 1  # small island
    m[3:6, 12:18, 12:18] = 2
    out, log = postprocess(
        m, ["fill_holes", {"op": "remove_small", "min_voxels": 5}, {"op": "keep_labels", "labels": [1]}]
    )
    assert out[5, 5, 5] == 1 and out[1, 15, 15] == 0 and (out == 2).sum() == 0
    assert log == ["fill_holes", "remove_small(min_voxels=5)", "keep_labels([1])"]
    out, _ = postprocess(m, ["largest_component"])
    assert (out == 1).sum() == 6 * 8 * 8 - 1
    out, _ = postprocess(m, [{"op": "relabel", "mapping": {2: 7}}, "binarize"])
    assert set(np.unique(out)) == {0, 1}
    out, _ = postprocess(m, [{"op": "keep_slices", "axis": 0, "start": 0, "stop": 3}])
    assert out[3:].sum() == 0 and out[:3].sum() > 0
    out, _ = postprocess(m, [{"op": "dilate", "radius": 1}])
    assert out.sum() > m.sum()
    with pytest.raises(ValueError):
        postprocess(m, ["unknown"])


def test_mask_to_prompts_3d_and_2d(ct_volume, blob_png):
    img = MedicalImage.load(ct_volume["path"])
    mask = img.with_array(ct_volume["ball"].astype(np.uint8))
    pr = mask_to_prompts(mask, "axial", margin=0)
    box = pr["boxes"][0]
    assert box[2] == box[5] == pr["slice"]
    zs = np.nonzero(ct_volume["ball"].any(axis=(1, 2)))[0]
    assert pr["slice_range"] == [int(zs.min()), int(zs.max())]
    # the box must contain the mask on that slice
    sl = ct_volume["ball"][pr["slice"]]
    ys, xs = np.nonzero(sl)
    assert box[0] <= xs.min() and box[3] >= xs.max() and box[1] <= ys.min() and box[4] >= ys.max()
    pt = pr["points"][0]
    assert sl[int(round(pt[1])), int(round(pt[0]))]
    img2 = MedicalImage.load(blob_png["path"])
    pr2 = mask_to_prompts(img2.with_array(blob_png["disk"].astype(np.uint8)), margin=1)
    assert len(pr2["boxes"][0]) == 4 and pr2["slice"] is None


def test_normalize_prompts(ct_volume, blob_png):
    img = MedicalImage.load(ct_volume["path"])
    wire = normalize_prompts(
        [
            Prompt(type="box", coords=[5, 6, 7, 20, 22, 7]),
            {"type": "point", "coords": [3, 4], "slice": 9, "label": 0},
        ],
        img,
        "axial",
    )
    assert wire[0] == {
        "type": "box",
        "label": 1,
        "object_id": 1,
        "slice": 7,
        "coords": [5.0, 6.0, 20.0, 22.0],
    }
    assert wire[1]["slice"] == 9 and wire[1]["label"] == 0 and wire[1]["coords"] == [3.0, 4.0]
    # coronal: stack along y; in-plane (row=z, col=x)
    wire = normalize_prompts([{"type": "point", "coords": [10, 11, 12]}], img, "coronal")
    assert wire[0]["slice"] == 11 and wire[0]["coords"] == [10.0, 12.0]
    with pytest.raises(ValueError):
        normalize_prompts([{"type": "box", "coords": [1, 2, 3, 4]}], img)  # needs slice
    with pytest.raises(ValueError):
        Prompt(type="point", coords=[1])
    img2 = MedicalImage.load(blob_png["path"])
    wire = normalize_prompts([{"type": "box", "coords": [1, 2, 3, 4]}], img2)
    assert wire[0]["slice"] == 0 and wire[0]["coords"] == [1.0, 2.0, 3.0, 4.0]


def test_metrics():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[2:6, 2:6, 2:6] = True
    b = np.roll(a, 1, axis=2)
    r = compare_binary(a, a, (1.0, 1.0, 1.0))
    assert r["dice"] == 1.0 and r["hd95_mm"] == 0.0
    r = compare_binary(a, b, (1.0, 1.0, 1.0))
    assert 0.7 < r["dice"] < 0.8 and r["hausdorff_mm"] == 1.0 and r["volume_diff_ml"] == 0.0
    lab_a = a.astype(np.uint8)
    lab_b = b.astype(np.uint8) * 1
    lab_b[8, 8, 8] = 2
    r = compare_masks(lab_a, lab_b, (1.0, 1.0, 1.0))
    assert set(r["per_label"]) == {"1", "2"} and r["per_label"]["2"]["dice"] == 0.0
    assert "mean_dice" in r
