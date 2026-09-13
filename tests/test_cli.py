from typer.testing import CliRunner

from open_med_mcp import __version__
from open_med_mcp.cli import app

runner = CliRunner()


def test_version_and_lists(isolated_settings):
    assert runner.invoke(app, ["--version"]).stdout.strip() == __version__
    r = runner.invoke(app, ["models", "list"])
    assert r.exit_code == 0 and "medsam2" in r.stdout
    r = runner.invoke(app, ["guidelines", "list"])
    assert r.exit_code == 0 and "qc-checklist" in r.stdout
    r = runner.invoke(app, ["guidelines", "show", "qc-checklist"])
    assert "Visual" in r.stdout
    r = runner.invoke(app, ["models", "def", "classical"])
    assert r.exit_code == 0 and r.stdout.startswith("Bootstrap: docker")
    r = runner.invoke(app, ["models", "check", "classical"])
    assert r.exit_code == 0 and "local" in r.stdout
    r = runner.invoke(app, ["client-config", "codex"])
    assert "mcp_servers.open_med_mcp" in r.stdout
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0 and "classical" in r.stdout


def test_cli_run_and_view(ct_volume, isolated_settings):
    r = runner.invoke(
        app,
        [
            "run",
            "classical",
            "--image",
            str(ct_volume["path"]),
            "--task",
            "threshold",
            "--params",
            '{"lower": 200, "upper": 1000}',
            "--output",
            str(isolated_settings.workspace / "out.nii.gz"),
        ],
    )
    assert r.exit_code == 0, r.stdout + str(r.exception)
    assert (isolated_settings.workspace / "out.nii.gz").exists()
    r = runner.invoke(
        app,
        [
            "view",
            str(ct_volume["path"]),
            "--mask",
            str(isolated_settings.workspace / "out.nii.gz"),
            "--layout",
            "three-plane",
        ],
    )
    assert r.exit_code == 0 and "wrote" in r.stdout
    r = runner.invoke(app, ["view", str(ct_volume["path"]), "--html"])
    assert r.exit_code == 0
