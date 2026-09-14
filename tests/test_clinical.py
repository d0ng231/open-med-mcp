import numpy as np
import pytest
from mcp import Client
from mcp.types import ImageContent

from open_med_mcp.core import clinical
from open_med_mcp.core.image import MedicalImage
from open_med_mcp.server import create_server


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_measure_lesion_ball(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    mask = img.with_array(ct_volume["ball"].astype(np.uint8))
    r = clinical.measure_lesion(mask, 1, "axial", img)
    # the synthetic ball has radius 5 voxels in-plane (1 mm) -> diameter about 10-11 mm
    assert 9.5 <= r["long_axis_mm"] <= 12.5 and 9.0 <= r["short_axis_mm"] <= 12.5
    assert r["measurable_target_lesion"] is True and r["intensity"]["mean"] > 250
    assert len(r["long_axis_endpoints_xyz"]) == 2 and len(r["long_axis_endpoints_xyz"][0]) == 3
    assert (
        r["slice"] == int(np.argmax(ct_volume["ball"].sum(axis=(1, 2))))
        or ct_volume["ball"][r["slice"]].sum() > 0
    )
    empty = clinical.measure_lesion(img.with_array(np.zeros(img.shape_zyx, np.uint8)), None)
    assert empty["voxels"] == 0


def test_recist_rules():
    assert clinical.recist_response(100, 65)["response"] == "PR"
    assert clinical.recist_response(100, 90)["response"] == "SD"
    assert clinical.recist_response(100, 0)["response"] == "CR"
    assert (
        clinical.recist_response(100, 8, all_nodes_short_axis_below_10mm=False)["response"] == "PR"
    )  # residual nodes: not CR
    pd = clinical.recist_response(100, 61, nadir_sld_mm=50)
    assert pd["response"] == "PD" and pd["change_vs_nadir_pct"] == 22.0
    assert clinical.recist_response(20, 23, nadir_sld_mm=20)["response"] == "SD"  # +15 %, < 5 mm
    assert clinical.recist_response(100, 90, new_lesions=True)["response"] == "PD"


def test_fleischner_table():
    f = clinical.fleischner_recommendation
    assert f("solid", 4)["recommendation"] == "no routine follow-up"
    assert f("solid", 4, high_risk=True)["recommendation"] == "optional CT at 12 months"
    assert f("solid", 7)["recommendation"].startswith("CT at 6-12 months")
    assert f("solid", 9, high_risk=True)["recommendation"].startswith("consider CT at 3 months")
    assert f("solid", 7, multiple=True)["recommendation"].startswith("CT at 3-6 months")
    assert f("ground-glass", 8)["recommendation"].startswith("CT at 6-12 months")
    assert f("part-solid", 8)["recommendation"].startswith("CT at 3-6 months")
    assert f("part-solid", 5)["recommendation"] == "no routine follow-up"
    assert "most suspicious" in f("ground-glass", 7, multiple=True)["recommendation"]


def test_tirads_points():
    r = clinical.tirads_score("solid", "hypoechoic", "wider-than-tall", "smooth", ["punctate"], 12)
    assert r["total_points"] == 7 and r["level"] == "TR5" and "FNA" in r["recommendation"]
    r = clinical.tirads_score("spongiform", "anechoic", "wider-than-tall", "smooth", None, 30)
    assert r["level"] == "TR1"
    r = clinical.tirads_score("mixed", "isoechoic", "wider-than-tall", "smooth", ["macrocalcifications"], 20)
    assert r["total_points"] == 3 and r["level"] == "TR3" and "follow-up" in r["recommendation"]
    r = clinical.tirads_score("solid", "isoechoic", "wider-than-tall", "lobulated", None, 9)
    assert r["level"] == "TR4" and r["recommendation"].startswith("no FNA")


def test_agatston_synthetic(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    arr = img.array.copy().astype(np.float32)
    # a 3x3 in-plane plaque (1 mm voxels -> 9 mm2) of 350 HU on two slices (thickness 2.5 mm), away from the test ball
    arr[4:6, 8:11, 10:13] = 350.0
    img2 = MedicalImage(arr, img.spacing, img.origin, img.direction, img.path, img.format)
    mask = np.zeros(img.shape_zyx, np.uint8)
    mask[2:8, 5:14, 5:18] = 1
    r = clinical.agatston_score(img2, img.with_array(mask), labels={1: "LAD"})
    expected = 2 * (9.0 * 3 * (2.5 / 3.0))  # two slices x area x factor 3 (300-399 HU) x thickness scaling
    assert abs(r["agatston_total"] - expected) < 0.5 and r["n_lesions"] == 2
    assert r["per_label"]["1"]["name"] == "LAD" and r["category"].startswith("11-100")
    assert clinical.agatston_score(img, img.with_array(mask))["agatston_total"] == 0


def test_ctr_flr_mayo():
    arr = np.zeros((100, 200), np.uint8)
    arr[20:80, 20:90] = 1  # right lung
    arr[20:80, 110:180] = 2  # left lung
    arr[40:70, 60:140] = 3  # heart, 80 px wide over a 160 px thorax
    m = MedicalImage(arr, (1.0, 1.0), (0.0, 0.0), (1.0, 0.0, 0.0, 1.0))
    r = clinical.cardiothoracic_ratio(m, [3], [1, 2])
    assert r["ctr"] == 0.5 and r["heart_width_px"] == 80 and r["thorax_width_px"] == 160
    f = clinical.future_liver_remnant(
        1500, 400, tumor_ml=100, liver_condition="chemotherapy", body_weight_kg=80
    )
    assert (
        f["flr_percent"] == 28.6 and f["adequate"] is False and f["remnant_to_body_weight_ratio_pct"] == 0.5
    )
    k = clinical.mayo_adpkd_class(1500, 1.7, 40)
    assert k["htTKV_ml_per_m"] == 882.4 and k["class"] == "1E"
    assert clinical.mayo_adpkd_class(300, 1.7, 40)["class"] == "1A"
    assert clinical.mayo_adpkd_class(300, 1.7, 18)["class"] is None


@pytest.mark.anyio
async def test_clinical_tools_via_server(isolated_settings, ct_volume):
    async with Client(create_server(isolated_settings, plugins=False), raise_exceptions=True) as c:
        r = await c.call_tool("measure_lesion", {"mask": "ball_mask.nii.gz", "image": "ct.nii.gz"})
        assert not r.is_error, r.content[0].text
        sc = r.structured_content
        assert sc["lesions"][0]["measurable_target_lesion"] and sc["sum_of_longest_diameters_mm"] > 9
        assert any(isinstance(x, ImageContent) for x in r.content)
        r = await c.call_tool("recist_response", {"baseline_sld_mm": 50, "current_sld_mm": 30})
        assert r.structured_content["response"] == "PR"
        r = await c.call_tool(
            "fleischner_recommendation", {"nodule_type": "solid", "size_mm": 7, "high_risk": True}
        )
        assert "18-24 months" in r.structured_content["recommendation"]
        r = await c.call_tool(
            "tirads_score",
            {
                "composition": "solid",
                "echogenicity": "very hypoechoic",
                "shape": "taller-than-wide",
                "margin": "irregular",
                "echogenic_foci": ["punctate"],
                "max_diameter_mm": 11,
            },
        )
        assert r.structured_content["level"] == "TR5"
        r = await c.call_tool("future_liver_remnant", {"total_liver_ml": 1600, "remnant_ml": 500})
        assert r.structured_content["adequate"] is True
        r = await c.call_tool(
            "mayo_adpkd_class", {"total_kidney_volume_ml": 900, "height_m": 1.75, "age_years": 45}
        )
        assert r.structured_content["class"] in ("1A", "1B", "1C", "1D", "1E")
        r = await c.call_tool("agatston_score", {"image": "ct.nii.gz", "mask": "ball_mask.nii.gz"})
        assert not r.is_error and "agatston_total" in r.structured_content
        r = await c.call_tool("segment", {"image": "ct.nii.gz", "model": "torchxrayvision"})
        assert (
            r.is_error is False or "does not produce a mask" not in r.content[0].text
        )  # xrv has a segment task
        r = await c.call_tool("segment", {"image": "ct.nii.gz", "model": "radiomics"})
        assert r.is_error and "does not produce a mask" in r.content[0].text
        r = await c.call_tool("detect", {"image": "ct.nii.gz", "model": "classical"})
        assert r.is_error and "no detect task" in r.content[0].text
        names = {
            g["name"]
            for g in (await c.call_tool("list_guidelines", {"tags": ["clinical"]})).structured_content[
                "guidelines"
            ]
        }
        assert {
            "recist-1-1",
            "fleischner-2017",
            "lung-rads-2022",
            "li-rads-2018",
            "acr-ti-rads-2017",
            "coronary-calcium-agatston",
        } <= names
