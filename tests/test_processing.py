import numpy as np
import pytest
import SimpleITK as sitk
from mcp import Client
from mcp.types import ImageContent

from open_med_mcp.core import processing as proc
from open_med_mcp.core.image import MedicalImage, load_mask
from open_med_mcp.server import create_server
from tests.conftest import make_volume


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(isolated_settings):
    async with Client(create_server(isolated_settings), raise_exceptions=True) as c:
        yield c


def test_resample_reorient_crop(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    r = proc.resample(img, spacing=[2.0, 2.0, 2.0])
    assert r.size_xyz == (24, 20, 30) and r.spacing == (2.0, 2.0, 2.0)
    back = proc.resample(r, reference=img)
    assert back.size_xyz == img.size_xyz
    mask = img.with_array(ct_volume["ball"].astype(np.uint8))
    rm = proc.resample(mask, factor=0.5, is_mask=True)
    assert set(np.unique(rm.array)) <= {0, 1}
    ro = proc.reorient(img, "LPS")
    assert ro.orientation_code() == "LPS" and ro.size_xyz == img.size_xyz
    cropped, box = proc.crop(img, [10, 10, 5, 20, 22, 9], margin=1)
    assert cropped.size_xyz == (13, 15, 7) and box == [9, 9, 4, 21, 23, 10]
    # physical position of the crop corner equals the original voxel's position
    assert np.allclose(cropped.index_to_physical((0, 0, 0)), img.index_to_physical((9, 9, 4)))


def test_n4_and_registration(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    corrected, field = proc.n4_bias_correction(img, shrink=4, iterations=[5, 5])
    assert corrected.size_xyz == img.size_xyz and field.array.min() > 0
    # registration: recover a known translation
    src = img.to_sitk()
    tx = sitk.TranslationTransform(3, (3.0, -2.0, 2.5))
    moved = sitk.Resample(src, src, tx, sitk.sitkLinear, -1000.0)
    moving = MedicalImage.from_sitk(moved)
    est, info = proc.register(img, moving, "rigid", metric="meansquares", iterations=100, sampling=0.5)
    assert info["global"]["iterations"] > 0
    warped = proc.apply_transform(moving, img, est)
    diff_before = np.abs(moving.array.astype(float) - img.array).mean()
    diff_after = np.abs(warped.array.astype(float) - img.array).mean()
    assert diff_after < diff_before * 0.6


def test_combine_features_mesh(ct_volume, tmp_path):
    img = MedicalImage.load(ct_volume["path"])
    ball = ct_volume["ball"]
    body = ct_volume["body"]
    assert proc.combine([ball, body], "union").sum() == np.logical_or(ball, body).sum()
    assert proc.combine([body, ball], "subtract").sum() == body.sum() - ball.sum()
    lab = proc.combine([body, ball], "label")  # later masks overwrite earlier ones
    assert set(np.unique(lab)) == {0, 1, 2}
    with pytest.raises(ValueError):
        proc.combine([ball, ball[:-1]], "union")
    f = proc.mask_features(ball.astype(np.uint8), img.spacing, img.array)
    assert f["voxels"] == int(ball.sum()) and 0.7 < f["sphericity"] <= 1.05 and f["components"] == 1
    assert f["intensity"]["mean"] > 250 and "entropy_bits" in f["intensity"]
    verts, faces = proc.mask_to_mesh(img.with_array(ball.astype(np.uint8)), step=1, smooth_iterations=1)
    assert len(verts) > 50 and faces.max() < len(verts)
    stl = proc.write_mesh(verts, faces, tmp_path / "ball.stl")
    assert stl.stat().st_size == 84 + 50 * len(faces)
    obj = proc.write_mesh(verts, faces, tmp_path / "ball.obj")
    assert obj.read_text().startswith("# generated")


def test_list_dicom_series(isolated_settings):
    img, _, _ = make_volume(shape_zyx=(4, 16, 16))
    d = isolated_settings.workspace / "dcm"
    d.mkdir()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for i in range(img.GetDepth()):
        sl = img[:, :, i]
        sl.SetMetaData("0020|000e", "1.2.3.4.5")
        sl.SetMetaData("0008|0060", "MR")
        sl.SetMetaData("0008|103e", "T1 test")
        sl.SetMetaData(
            "0020|0032", "\\".join(f"{v:.3f}" for v in img.TransformIndexToPhysicalPoint((0, 0, i)))
        )
        sl.SetMetaData("0020|0037", "\\".join(f"{v:.3f}" for v in img.GetDirection()[:6]))
        sl.SetMetaData("0008|0018", f"1.2.3.4.5.{i}")
        w.SetFileName(str(d / f"{i}.dcm"))
        w.Execute(sl)
    series = proc.list_dicom_series(d)
    assert (
        len(series) == 1
        and series[0]["n_files"] == 4
        and series[0]["modality"] == "MR"
        and series[0]["series_description"] == "T1 test"
    )


@pytest.mark.anyio
async def test_processing_tools_via_server(client: Client, ct_volume):
    r = await client.call_tool("resample_image", {"path": "ct.nii.gz", "spacing": [2, 2, 2]})
    assert not r.is_error and r.structured_content["size_xyz"] == [24, 20, 30]
    r = await client.call_tool(
        "resample_image",
        {"path": "ball_mask.nii.gz", "reference": r.structured_content["output"], "is_mask": True},
    )
    assert r.structured_content["size_xyz"] == [24, 20, 30]
    r = await client.call_tool("reorient_image", {"path": "ct.nii.gz", "orientation": "LPS"})
    assert r.structured_content["orientation"] == "LPS"
    r = await client.call_tool(
        "crop_image",
        {"path": "ct.nii.gz", "mask": "ball_mask.nii.gz", "margin": 2, "also": ["ball_mask.nii.gz"]},
    )
    assert not r.is_error and r.structured_content["cropped_masks"]
    cropped = r.structured_content["output"]
    r = await client.call_tool("n4_bias_correction", {"path": cropped, "shrink": 2})
    assert not r.is_error and r.structured_content["output"].endswith("_n4.nii.gz")
    r = await client.call_tool(
        "combine_masks",
        {"masks": ["ball_mask.nii.gz", "ball_mask.nii.gz"], "mode": "label", "names": ["a", "b"]},
    )
    assert r.structured_content["labels"] == {"1": "a", "2": "b"}
    r = await client.call_tool("mask_features", {"mask": "ball_mask.nii.gz", "image": "ct.nii.gz"})
    assert "sphericity" in r.structured_content["features"]["1"]
    r = await client.call_tool("mask_to_mesh", {"mask": "ball_mask.nii.gz"})
    assert r.structured_content["output"].endswith(".stl") and r.structured_content["faces"] > 0
    r = await client.call_tool(
        "register_images",
        {
            "fixed": "ct.nii.gz",
            "moving": "ct.nii.gz",
            "transform": "rigid",
            "iterations": 20,
            "moving_masks": ["ball_mask.nii.gz"],
        },
    )
    assert not r.is_error, r.content[0].text
    sc = r.structured_content
    assert (
        sc["transform_file"].endswith(".tfm")
        and sc["warped_masks"]
        and any(isinstance(c, ImageContent) for c in r.content)
    )
    r = await client.call_tool(
        "apply_transform",
        {
            "transform_file": sc["transform_file"],
            "moving": "ball_mask.nii.gz",
            "reference": "ct.nii.gz",
            "is_mask": True,
        },
    )
    assert not r.is_error and load_mask(r.structured_content["output"]).array.sum() > 0
    r = await client.call_tool("list_dicom_series", {"folder": "."})
    assert r.structured_content["n_series"] == 0
