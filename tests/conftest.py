from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

from open_med_mcp.config import Settings, set_settings

RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Every test gets its own workspace + home so nothing leaks between tests."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("OMM_WORKSPACE", str(ws))
    monkeypatch.setenv("OMM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMM_RUNNER", "local")
    monkeypatch.setenv("OMM_DEVICE", "cpu")
    monkeypatch.delenv("OMM_MODEL_DIRS", raising=False)
    monkeypatch.delenv("OMM_GUIDELINE_DIRS", raising=False)
    monkeypatch.chdir(ws)
    settings = set_settings(Settings())
    from open_med_mcp.guidelines.loader import get_library
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.tools import _common

    _common._CACHE.clear()
    get_registry(settings, reload=True)
    get_library(settings, reload=True)
    return settings


def make_volume(shape_zyx=(24, 40, 48), spacing=(1.0, 1.0, 2.5), direction=RAS_DIRECTION):
    """Synthetic 'CT': air background, a soft-tissue ellipsoid body and a bright ball inside."""
    z, y, x = np.indices(shape_zyx).astype(np.float32)
    cz, cy, cx = (s / 2 for s in shape_zyx)
    body = (((x - cx) / 20) ** 2 + ((y - cy) / 16) ** 2 + ((z - cz) / 10) ** 2) <= 1.0
    ball = (((x - cx - 6) / 5) ** 2 + ((y - cy + 3) / 5) ** 2 + ((z - cz) / 3) ** 2) <= 1.0
    vol = np.full(shape_zyx, -1000.0, dtype=np.float32)
    vol[body] = 40.0
    vol[ball] = 300.0
    rng = np.random.default_rng(0)
    vol += rng.normal(0, 5, shape_zyx).astype(np.float32)
    img = sitk.GetImageFromArray(vol.astype(np.int16))
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    img.SetOrigin((10.0, -20.0, 5.0))
    return img, body, ball


@pytest.fixture
def ct_volume(isolated_settings: Settings) -> dict:
    img, body, ball = make_volume()
    path = isolated_settings.workspace / "ct.nii.gz"
    sitk.WriteImage(img, str(path), True)
    ball_mask = sitk.GetImageFromArray(ball.astype(np.uint8))
    ball_mask.CopyInformation(img)
    mpath = isolated_settings.workspace / "ball_mask.nii.gz"
    sitk.WriteImage(ball_mask, str(mpath), True)
    return {"path": path, "mask": mpath, "body": body, "ball": ball, "shape_zyx": body.shape}


@pytest.fixture
def blob_png(isolated_settings: Settings) -> dict:
    y, x = np.indices((64, 80))
    disk = ((x - 50) ** 2 + (y - 30) ** 2) <= 12**2
    arr = np.where(disk, 220, 30).astype(np.uint8)
    path = isolated_settings.workspace / "blob.png"
    Image.fromarray(arr).save(path)
    mpath = isolated_settings.workspace / "blob_mask.png"
    Image.fromarray(disk.astype(np.uint8)).save(mpath)
    return {"path": path, "mask": mpath, "disk": disk}


def slurm_or_gpu_available() -> bool:
    return os.environ.get("OMM_TEST_GPU") == "1"
