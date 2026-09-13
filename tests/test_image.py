import numpy as np
import SimpleITK as sitk

from open_med_mcp.core.image import (
    MedicalImage,
    load_mask,
    read_labels_sidecar,
    save_mask,
    write_labels_sidecar,
)
from tests.conftest import make_volume


def test_load_describe_3d(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    assert img.ndim == 3
    assert img.size_xyz == (48, 40, 24)
    assert img.orientation_code() == "RAS"
    d = img.describe()
    assert d["modality_guess"] == "CT"
    assert d["planes"]["axial"]["index_axis"] == 2
    assert d["planes"]["axial"]["n_slices"] == 24
    assert d["planes"]["sagittal"]["n_slices"] == 48
    assert abs(d["voxel_volume_mm3"] - 2.5) < 1e-6
    assert d["intensity"]["min"] < -900


def test_plane_axis_from_direction():
    # permuted axes: index axis 0 runs along P, axis 1 along S, axis 2 along L
    img, _, _ = make_volume(direction=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    m = MedicalImage.from_sitk(img)
    # axial plane stacks along the physical S axis, which is index axis 0 here
    assert m.index_axis_for_plane("axial") == 1
    assert m.numpy_axis_for_plane("axial") == 1
    assert m.index_axis_for_plane("sagittal") == 2
    assert m.index_axis_for_plane("coronal") == 0


def test_mask_roundtrip_geometry(ct_volume, tmp_path):
    img = MedicalImage.load(ct_volume["path"])
    mask = ct_volume["ball"].astype(np.uint8)
    out = save_mask(mask, img, tmp_path / "m.nii.gz")
    back = load_mask(out, img)
    assert back.shape_zyx == img.shape_zyx
    assert back.spacing == img.spacing
    assert back.direction == img.direction
    assert int(back.array.sum()) == int(mask.sum())
    write_labels_sidecar(out, {1: "ball"})
    assert read_labels_sidecar(out) == {1: "ball"}


def test_2d_png(blob_png):
    img = MedicalImage.load(blob_png["path"])
    assert img.is_2d and not img.is_rgb
    assert img.size_xyz == (80, 64)
    assert img.numpy_axis_for_plane("axial") == 0
    assert img.modality_guess() == "XR/2D"
    m = load_mask(blob_png["mask"], img)
    assert int(m.array.sum()) == int(blob_png["disk"].sum())


def test_rgb_png(isolated_settings):
    from PIL import Image

    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[8:24, 10:30] = (200, 50, 50)
    p = isolated_settings.workspace / "rgb.png"
    Image.fromarray(arr).save(p)
    img = MedicalImage.load(p)
    assert img.is_rgb and img.is_2d and img.ndim == 2
    assert img.scalar_array.shape == (32, 48)
    assert img.modality_guess() == "RGB"


def test_dicom_series(isolated_settings):
    img, _, _ = make_volume(shape_zyx=(6, 20, 24))
    d = isolated_settings.workspace / "dicom"
    d.mkdir()
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for i in range(img.GetDepth()):
        sl = img[:, :, i]
        pos = img.TransformIndexToPhysicalPoint((0, 0, i))
        sl.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125.1")
        sl.SetMetaData("0020|000d", "1.2.826.0.1.3680043.2.1125.2")
        sl.SetMetaData("0008|0060", "CT")
        sl.SetMetaData("0020|0032", "\\".join(f"{v:.6f}" for v in pos))
        sl.SetMetaData("0020|0037", "\\".join(f"{v:.6f}" for v in img.GetDirection()[:6]))
        sl.SetMetaData("0020|0013", str(i + 1))
        sl.SetMetaData("0008|0018", f"1.2.826.0.1.3680043.2.1125.3.{i}")
        writer.SetFileName(str(d / f"slice{i:03d}.dcm"))
        writer.Execute(sl)
    loaded = MedicalImage.load(d)
    assert loaded.format == "dicom-series"
    assert loaded.size_xyz == (24, 20, 6)
    assert loaded.metadata.get("modality") == "CT"
