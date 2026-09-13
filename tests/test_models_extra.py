"""Wrapped-image mode, classification outputs and batch runs, tested with tiny fake adapters."""

import json
import sys
import textwrap

import pytest
from mcp import Client
from mcp.types import ImageContent

from open_med_mcp.models import wrapped
from open_med_mcp.models.registry import get_registry
from open_med_mcp.models.runners import LocalRunner, runner_report, select_runner
from open_med_mcp.server import create_server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake_models(isolated_settings):
    root = isolated_settings.workspace / "omm_models"
    # 1. a wrapped model that "segments" by copying the input (cp is the host command)
    d = root / "copytool"
    d.mkdir(parents=True)
    (d / "copytool.yaml").write_text(
        textwrap.dedent(
            """
            name: copytool
            category: preprocessing
            tasks: [copy]
            outputs:
              mask: {kind: mask, file: outputs/mask.nii.gz}
              extra: {kind: file, file: outputs/never.txt, required: false}
            labels: {1: copied}
            params:
              verbose: {type: boolean, default: false}
              level: {type: integer}
            container:
              mode: wrapped
              image: example/copytool:1.0
              gpu: none
              command: ["cp", "{flag:verbose:-v}", "{inputs.image}", "{outputs.mask}"]
            local:
              command: ["cp", "{flag:verbose:-v}", "{inputs.image}", "{outputs.mask}"]
            runners: [local, docker, apptainer]
            """
        )
    )
    # 2. a classification adapter writing predictions.json
    c = root / "fakeclf"
    c.mkdir()
    (c / "fakeclf.yaml").write_text(
        textwrap.dedent(
            """
            name: fakeclf
            category: classification
            tasks: [classify]
            outputs:
              predictions: {kind: json}
            local: {entrypoint: run.py, requires: [json]}
            runners: [local]
            """
        )
    )
    (c / "run.py").write_text(
        textwrap.dedent(
            """
            import json, sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            sys.path.insert(0, "%s")
            import omm_job
            def run(job):
                out = job.output_path("predictions.json")
                out.write_text(json.dumps({"probabilities": {"finding_a": 0.9, "finding_b": 0.2}, "top": [{"finding": "finding_a", "probability": 0.9}]}))
                job.finish({"predictions": out}, stats={"n_findings": 2})
            if __name__ == "__main__":
                omm_job.main(run)
            """
            % str(get_registry(isolated_settings).get("classical").adapter_dir.parent / "_sdk")
        )
    )
    return get_registry(isolated_settings, reload=True)


def test_render_command():
    reg_like = get_registry()
    m = reg_like.get("synthstrip")
    request = {
        "inputs": {"image": "inputs/image.nii.gz"},
        "params": {"no_csf": True, "border": 2, "threads": None},
        "resources": {"weights_dir": "/w"},
    }
    cmd = wrapped.render_command(m.container.command, request, m, "/job")
    assert cmd == [
        "mri_synthstrip",
        "-i",
        "/job/inputs/image.nii.gz",
        "-o",
        "/job/outputs/brain.nii.gz",
        "-m",
        "/job/outputs/mask.nii.gz",
        "--no-csf",
        "-b",
        "2",
    ]
    request["params"] = {"no_csf": False, "threads": 4}
    cmd = wrapped.render_command(m.container.command, request, m, "/job")
    assert "--no-csf" not in cmd and cmd[-2:] == ["-t", "4"]
    with pytest.raises(ValueError):
        wrapped.render_command(["x", "{inputs.missing}"], request, m, "/job")


def test_wrapped_local_run(fake_models, ct_volume, isolated_settings):
    m = fake_models.get("copytool")
    assert m.is_wrapped
    rep = runner_report(m, isolated_settings)
    assert rep["local"]["available"] and "cp" in rep["local"]["detail"]
    from open_med_mcp.models.job import prepare_job, read_response
    from open_med_mcp.models.weights import weights_dir_for
    from open_med_mcp.workspace import new_run_dir

    rd = new_run_dir("copy", isolated_settings)
    prepare_job(
        rd,
        m,
        "copy",
        {"image": ct_volume["mask"]},
        {"verbose": True},
        "cpu",
        weights_dir_for(m, isolated_settings),
    )
    runner = select_runner(m, isolated_settings)
    assert isinstance(runner, LocalRunner)
    out = runner.run(m, rd, weights_dir_for(m, isolated_settings), "cpu")
    assert out.returncode == 0 and out.command[:2] == ["cp", "-v"]
    resp = read_response(rd)
    assert (
        resp["status"] == "ok"
        and resp["labels"] == {"1": "copied"}
        and resp["stats"]["foreground_voxels"] > 0
    )
    assert "extra" not in resp["outputs"] and resp["model_info"]["mode"] == "wrapped"
    # docker/apptainer command rendering for wrapped models
    from open_med_mcp.models.runners import ApptainerRunner, DockerRunner

    dcmd = DockerRunner(isolated_settings).command(m, rd, weights_dir_for(m, isolated_settings), "cpu")
    assert (
        "--entrypoint" in dcmd
        and dcmd[dcmd.index("--entrypoint") + 1] == "cp"
        and dcmd[-1] == "/job/outputs/mask.nii.gz"
    )
    acmd = ApptainerRunner(isolated_settings).command(m, rd, weights_dir_for(m, isolated_settings), "cpu")
    assert acmd[-4:] == ["cp", "-v", "/job/inputs/image.nii.gz", "/job/outputs/mask.nii.gz"]


def test_wrapped_failure_is_reported(fake_models, ct_volume, isolated_settings):
    m = fake_models.get("copytool")
    from open_med_mcp.models.job import JobError, prepare_job, read_response
    from open_med_mcp.models.weights import weights_dir_for
    from open_med_mcp.workspace import new_run_dir

    rd = new_run_dir("copyfail", isolated_settings)
    prepare_job(rd, m, "copy", {"image": ct_volume["mask"]}, {}, "cpu", weights_dir_for(m, isolated_settings))
    (rd / "inputs" / "image.nii.gz").unlink()  # make cp fail
    LocalRunner(isolated_settings).run(m, rd, weights_dir_for(m, isolated_settings), "cpu")
    with pytest.raises(JobError, match="exited with code"):
        read_response(rd)


@pytest.mark.anyio
async def test_classify_and_batch_tools(fake_models, ct_volume, blob_png, isolated_settings, monkeypatch):
    monkeypatch.setenv("OMM_ZOO_FAKECLF_PYTHON", sys.executable)
    async with Client(create_server(isolated_settings), raise_exceptions=True) as c:
        r = await c.call_tool("list_models", {"category": "classification"})
        assert "fakeclf" in {m["name"] for m in r.structured_content["models"]}
        r = await c.call_tool("classify_image", {"image": "blob.png", "model": "fakeclf"})
        assert not r.is_error, r.content[0].text
        assert r.structured_content["results"]["predictions"]["probabilities"]["finding_a"] == 0.9
        assert any(isinstance(x, ImageContent) for x in r.content)
        r = await c.call_tool("segment", {"image": "blob.png", "model": "fakeclf"})
        assert r.is_error and "classification model" in r.content[0].text
        r = await c.call_tool("run_model", {"model": "copytool", "inputs": {"image": "ball_mask.nii.gz"}})
        assert (
            not r.is_error
            and r.structured_content["labels"] == {"1": "copied"}
            and r.structured_content["category"] == "preprocessing"
        )
        # batch over two copies of the CT with the classical model
        import shutil

        shutil.copy(ct_volume["path"], isolated_settings.workspace / "ct2.nii.gz")
        r = await c.call_tool(
            "run_batch",
            {
                "model": "classical",
                "pattern": "ct*.nii.gz",
                "task": "threshold",
                "params": {"lower": 200, "upper": 1000},
                "output_dir": "masks",
            },
        )
        assert not r.is_error, r.content[0].text
        sc = r.structured_content
        assert sc["n_cases"] == 2 and sc["n_ok"] == 2 and sc["csv"].endswith(".csv")
        assert (isolated_settings.workspace / "masks").exists()
        rows = json.loads((isolated_settings.workspace / sc["json"]).read_text())
        assert rows[0]["foreground_ml"] > 0 and rows[0]["mask"].startswith("masks/")
