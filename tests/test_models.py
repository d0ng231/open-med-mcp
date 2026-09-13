import json
from pathlib import Path

import numpy as np
import pytest

from open_med_mcp.core.image import MedicalImage, load_mask
from open_med_mcp.models import containers
from open_med_mcp.models.job import JobError, prepare_job, read_response, stage_input
from open_med_mcp.models.manifest import ModelManifest, load_manifest
from open_med_mcp.models.registry import ZOO_DIR, get_registry
from open_med_mcp.models.runners import LocalRunner, runner_report, select_runner
from open_med_mcp.models.weights import missing_weights, weights_dir_for, weights_status
from open_med_mcp.workspace import new_run_dir


def test_registry_bundled(isolated_settings):
    reg = get_registry(isolated_settings, reload=True)
    assert {"classical", "sam2", "medsam2", "totalsegmentator"} <= set(reg.names())
    assert reg.errors == {}
    m = reg.get("medsam2")
    assert m.adapter == "sam2" and m.is_promptable and m.default_params()["variant"] == "medsam2_latest"
    assert m.entrypoint_path().exists() and m.dockerfile_path() is not None
    assert "variant" in m.params_json_schema()["properties"]
    with pytest.raises(ValueError):
        m.validate_params({"variant": "nope"})
    with pytest.raises(KeyError):
        reg.get("missing-model")


def test_user_model_dir(isolated_settings, monkeypatch):
    d = isolated_settings.workspace / "omm_models" / "mymodel"
    d.mkdir(parents=True)
    src = ZOO_DIR / "_template"
    for f in ("mymodel.yaml", "run.py", "Dockerfile"):
        (d / f).write_text((src / f).read_text())
    reg = get_registry(isolated_settings, reload=True)
    assert "mymodel" in reg.names()
    m = reg.get("mymodel")
    assert m.adapter_dir == d.resolve()
    # broken manifest is reported, not fatal
    (isolated_settings.workspace / "omm_models" / "broken").mkdir()
    (isolated_settings.workspace / "omm_models" / "broken" / "broken.yaml").write_text("- not a mapping")
    reg.reload()
    assert any("broken" in k for k in reg.errors)


def test_manifest_defaults(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("description: d\n")
    m = load_manifest(p)
    assert (
        m.name == "x" and m.adapter == tmp_path.name and m.display_name == "x" and m.default_task == "segment"
    )
    assert isinstance(m, ModelManifest)


def test_classical_local_run(ct_volume, isolated_settings):
    reg = get_registry(isolated_settings)
    m = reg.get("classical")
    rep = runner_report(m, isolated_settings)
    assert rep["local"]["available"]
    run_dir = new_run_dir("t", isolated_settings)
    req = prepare_job(
        run_dir,
        m,
        "threshold",
        {"image": ct_volume["path"]},
        {"lower": 200, "upper": 1000, "keep": "largest"},
        "cpu",
        weights_dir_for(m, isolated_settings),
    )
    assert req["inputs"]["image"] == "inputs/image.nii.gz" and req["params"]["keep"] == "largest"
    runner = select_runner(m, isolated_settings)
    assert isinstance(runner, LocalRunner)
    out = runner.run(m, run_dir, weights_dir_for(m, isolated_settings), "cpu")
    assert out.returncode == 0, out.stderr_tail
    resp = read_response(run_dir)
    img = MedicalImage.load(ct_volume["path"])
    mask = load_mask(resp["outputs"]["mask"], img)
    ball = ct_volume["ball"]
    inter = np.logical_and(mask.array > 0, ball).sum()
    dice = 2 * inter / (mask.array.astype(bool).sum() + ball.sum())
    assert dice > 0.9
    assert resp["labels"] == {"1": "object"} and (run_dir / "log.txt").exists()


def test_classical_tasks(ct_volume, isolated_settings):
    m = get_registry(isolated_settings).get("classical")
    runner = LocalRunner(isolated_settings)
    wd = weights_dir_for(m, isolated_settings)
    img = MedicalImage.load(ct_volume["path"])
    ball = ct_volume["ball"]
    zc, yc, xc = (int(v) for v in np.argwhere(ball).mean(axis=0))
    cases = {
        "region_grow": {"seeds": [[xc, yc, zc]], "tolerance": 60},
        "otsu": {"keep": "largest"},
        "multi_threshold": {"ranges": [[-1100, -500, 1], [-100, 150, 2], [200, 2000, 3]]},
    }
    for task, params in cases.items():
        rd = new_run_dir(task, isolated_settings)
        prepare_job(rd, m, task, {"image": ct_volume["path"]}, params, "cpu", wd)
        assert runner.run(m, rd, wd, "cpu").returncode == 0
        resp = read_response(rd)
        arr = load_mask(resp["outputs"]["mask"], img).array
        assert arr.any()
        if task == "multi_threshold":
            assert {1, 2, 3} <= {int(v) for v in np.unique(arr)} <= {0, 1, 2, 3}
        if task == "region_grow":
            assert arr[zc, yc, xc] == 1 and arr.sum() <= ball.sum() * 1.3
    # a seed outside the range yields a warning and an empty mask
    rd = new_run_dir("warn", isolated_settings)
    prepare_job(
        rd,
        m,
        "region_grow",
        {"image": ct_volume["path"]},
        {"seeds": [[0, 0, 0]], "lower": 200, "upper": 400},
        "cpu",
        wd,
    )
    runner.run(m, rd, wd, "cpu")
    resp = read_response(rd)
    assert resp["warnings"] and resp["stats"]["foreground_voxels"] == 0


def test_job_errors(ct_volume, isolated_settings):
    m = get_registry(isolated_settings).get("classical")
    rd = new_run_dir("bad", isolated_settings)
    with pytest.raises(JobError):
        prepare_job(
            rd,
            m,
            "no-such-task",
            {"image": ct_volume["path"]},
            {},
            "cpu",
            weights_dir_for(m, isolated_settings),
        )
    with pytest.raises(JobError):
        prepare_job(rd, m, "threshold", {}, {}, "cpu", weights_dir_for(m, isolated_settings))
    # adapter failure is reported through response.json
    prepare_job(
        rd,
        m,
        "multi_threshold",
        {"image": ct_volume["path"]},
        {},
        "cpu",
        weights_dir_for(m, isolated_settings),
    )
    out = LocalRunner(isolated_settings).run(m, rd, weights_dir_for(m, isolated_settings), "cpu")
    assert out.returncode != 0
    with pytest.raises(JobError, match="needs `ranges"):
        read_response(rd)


def test_stage_input_2d_and_convert(blob_png, tmp_path):
    p, info = stage_input(blob_png["path"], tmp_path / "in", "image")
    assert p.name == "image.png" and info["converted"] is False
    from PIL import Image

    Image.open(blob_png["path"]).save(tmp_path / "x.tif")
    p, info = stage_input(tmp_path / "x.tif", tmp_path / "in2", "image")
    assert p.name == "image.png" and info["converted"] is True


def test_weights_helpers(isolated_settings):
    m = get_registry(isolated_settings).get("sam2")
    st = weights_status(m, isolated_settings)
    assert len(st) == len(m.weights) and not any(s["present"] for s in st)
    assert [w.id for w in missing_weights(m, ["sam2.1_hiera_tiny"], isolated_settings)] == [
        "sam2.1_hiera_tiny"
    ]
    (weights_dir_for(m, isolated_settings) / "sam2.1_hiera_tiny.pt").write_bytes(b"x")
    assert missing_weights(m, ["sam2.1_hiera_tiny"], isolated_settings) == []


def test_dockerfile_to_def_all_adapters(isolated_settings):
    for m in get_registry(isolated_settings).all():
        df = m.dockerfile_path()
        if m.is_wrapped:
            assert df is None and m.container.image, m.name  # wrapped models use the upstream image
            continue
        assert df is not None, m.name
        text = containers.dockerfile_to_def(df, containers.build_context(m))
        assert text.startswith("Bootstrap: docker\nFrom: ")
        assert "%post" in text and "%files" in text and "/app/run.py" in text and "/app/omm_job.py" in text
        for line in text.splitlines():
            if line.startswith("    /") and " /app" in line:
                assert Path(line.split()[0]).exists(), line
    assert containers.image_name(get_registry(isolated_settings).get("medsam2"), isolated_settings).endswith(
        "-sam2:0.1"
    )


def test_runner_selection_errors(isolated_settings, monkeypatch):
    m = get_registry(isolated_settings).get("sam2")
    monkeypatch.setenv("OMM_ZOO_SAM2_PYTHON", "/nonexistent/python")
    rep = runner_report(m, isolated_settings)
    assert not rep["local"]["available"]
    monkeypatch.setattr(containers, "docker_usable", lambda: (False, "no docker"))
    monkeypatch.setattr(containers, "apptainer_cmd", lambda: None)
    isolated_settings.runner = "auto"
    with pytest.raises(RuntimeError, match="no execution backend"):
        select_runner(m, isolated_settings)


def test_local_runner_prefers_configured_python(isolated_settings, monkeypatch):
    import sys

    m = get_registry(isolated_settings).get("classical")
    monkeypatch.setenv("OMM_ZOO_CLASSICAL_PYTHON", sys.executable)
    assert LocalRunner(isolated_settings).python_for(m) == sys.executable
    assert json.loads(json.dumps(m.summary()))["automatic"] is True
