"""MCP scenario tests: real stdio subprocess, streamable HTTP, schema validity, concurrency,
background jobs, plug-ins and scaffolding."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import jsonschema
import pytest
from mcp import Client, StdioServerParameters
from mcp.types import ImageContent, TextContent

from open_med_mcp.server import create_server

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _env(isolated_settings) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMM_")}
    env.update(
        {
            "OMM_WORKSPACE": str(isolated_settings.workspace),
            "OMM_HOME": str(isolated_settings.home),
            "OMM_RUNNER": "local",
            "OMM_DEVICE": "cpu",
            "OMM_LOG_LEVEL": "WARNING",
            "PYTHONPATH": str(REPO / "src"),
        }
    )
    return env


@pytest.mark.anyio
async def test_stdio_subprocess_end_to_end(isolated_settings, ct_volume):
    """The CLI server over stdio, exactly as Claude Code / Codex launch it."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "open_med_mcp", "serve", "--workspace", str(isolated_settings.workspace)],
        env=_env(isolated_settings),
    )
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        assert {"inspect_image", "segment", "get_job", "list_plugins"} <= {t.name for t in tools}
        assert "open-med-mcp" in (c.instructions or "")
        r = await c.call_tool("inspect_image", {"path": "ct.nii.gz"})
        assert not r.is_error and isinstance(r.content[1], ImageContent)
        # an error must come back as a tool error, not kill the server
        r = await c.call_tool("inspect_image", {"path": "missing.nii.gz"})
        assert r.is_error
        r = await c.call_tool(
            "segment",
            {
                "image": "ct.nii.gz",
                "model": "classical",
                "task": "threshold",
                "params": {"lower": 200, "upper": 1000},
            },
        )
        assert not r.is_error, r.content[0].text
        mask = r.structured_content["outputs"]["mask"]
        # concurrent calls (tools run in worker threads; rendering is serialised by a lock)
        calls = [
            c.call_tool("render_view", {"image": "ct.nii.gz", "masks": [mask], "layout": lay})
            for lay in ("single", "three-plane", "montage", "single", "three-plane")
        ]
        calls += [c.call_tool("mask_stats", {"mask": mask, "image": "ct.nii.gz"}) for _ in range(3)]
        results = await asyncio.gather(*calls)
        assert all(not x.is_error for x in results), [x.content[0].text for x in results if x.is_error][:2]
        # background job
        r = await c.call_tool(
            "run_model",
            {"model": "classical", "inputs": {"image": "ct.nii.gz"}, "task": "otsu", "wait": False},
        )
        job_id = r.structured_content["job_id"]
        for _ in range(60):
            g = await c.call_tool("get_job", {"job_id": job_id})
            if (
                g.structured_content.get("job", {}).get("status") == "done"
                or g.structured_content.get("status") == "done"
            ):
                break
            await asyncio.sleep(0.5)
        assert g.structured_content["job"]["status"] == "done" and g.structured_content["outputs"][
            "mask"
        ].endswith(".nii.gz")
        r = await c.call_tool("list_jobs", {})
        assert r.structured_content["jobs"][0]["status"] == "done"
        # prompts + resources over the real transport
        p = await c.get_prompt("segmentation-3d-ct", {"image": "ct.nii.gz"})
        assert "ct.nii.gz" in p.messages[0].content.text
        res = await c.read_resource("omm://conventions")
        assert "Native voxel" in res.contents[0].text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.anyio
async def test_streamable_http_transport(isolated_settings, ct_volume):
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "open_med_mcp",
            "serve",
            "--transport",
            "streamable-http",
            "--port",
            str(port),
            "--workspace",
            str(isolated_settings.workspace),
        ],
        env=_env(isolated_settings),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        async with Client(f"http://127.0.0.1:{port}/mcp") as c:
            assert "inspect_image" in {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("mask_stats", {"mask": "ball_mask.nii.gz"})
            assert not r.is_error and r.structured_content["n_labels"] == 1
            # over HTTP, paths outside the workspace are refused by default
            r = await c.call_tool("inspect_image", {"path": str(Path("/etc/hostname")), "preview": False})
            assert r.is_error and "outside" in r.content[0].text
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.anyio
async def test_tool_schemas_are_valid_and_descriptive(isolated_settings):
    async with Client(create_server(isolated_settings), raise_exceptions=True) as c:
        tools = (await c.list_tools()).tools
        assert len(tools) >= 30
        for t in tools:
            assert re.fullmatch(r"[a-z][a-z0-9_]{1,63}", t.name), t.name
            assert t.description and len(t.description) > 20, t.name
            schema = t.input_schema
            jsonschema.Draft202012Validator.check_schema(schema)
            for pname, ps in schema.get("properties", {}).items():
                assert pname != "ctx", f"{t.name}: Context leaked into the schema"
                assert ps, f"{t.name}.{pname} has an empty schema"
                assert any(k in ps for k in ("type", "anyOf", "$ref", "enum", "oneOf", "allOf")), (
                    f"{t.name}.{pname}: {ps}"
                )
        prompts = (await c.list_prompts()).prompts
        assert all(p.description for p in prompts)


def test_no_stdout_writes_during_tool_calls(isolated_settings, ct_volume, capsys):
    """stdio transport: nothing but the protocol may be written to stdout."""

    async def run() -> None:
        async with Client(create_server(isolated_settings), raise_exceptions=True) as c:
            await c.call_tool("inspect_image", {"path": "ct.nii.gz"})
            await c.call_tool("segment", {"image": "ct.nii.gz", "model": "classical", "task": "otsu"})
            await c.call_tool("list_models", {"check_backends": True})

    asyncio.run(run())
    assert capsys.readouterr().out == ""


@pytest.mark.anyio
async def test_text_only_client_mode(isolated_settings, ct_volume, monkeypatch):
    monkeypatch.setenv("OMM_RETURN_IMAGES", "0")
    from open_med_mcp.config import Settings, set_settings

    s = set_settings(Settings())
    async with Client(create_server(s), raise_exceptions=True) as c:
        r = await c.call_tool("render_view", {"image": "ct.nii.gz"})
        assert (
            not r.is_error
            and all(isinstance(x, TextContent) for x in r.content)
            and r.structured_content["output"].endswith(".png")
        )


@pytest.mark.anyio
async def test_large_inline_images_are_capped(isolated_settings, ct_volume, monkeypatch):
    monkeypatch.setenv("OMM_MAX_IMAGE_BYTES", "60000")
    from open_med_mcp.config import Settings, set_settings

    s = set_settings(Settings())
    async with Client(create_server(s), raise_exceptions=True) as c:
        r = await c.call_tool(
            "render_view", {"image": "ct.nii.gz", "layout": "montage", "n_slices": 16, "max_px": 2000}
        )
        img = next(x for x in r.content if isinstance(x, ImageContent))
        assert len(img.data) * 3 // 4 <= 60000 * 1.05


@pytest.mark.anyio
async def test_workspace_plugin_and_scaffold(isolated_settings, ct_volume):
    from open_med_mcp.scaffold import new_guideline, new_model, new_plugin

    pdir = isolated_settings.workspace / "omm_plugins"
    pdir.mkdir()
    (pdir / "lesion_count.py").write_text((REPO / "examples/plugins/lesion_count.py").read_text())
    (pdir / "broken.py").write_text("def register(server):\n    raise RuntimeError('boom')\n")
    (pdir / "_ignored.py").write_text("raise SystemExit('must not be imported')\n")
    scaffolded = new_plugin(pdir, "my-tool")
    assert scaffolded.name == "my_tool.py"
    async with Client(create_server(isolated_settings), raise_exceptions=True) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"count_lesions", "my_tool_volume"} <= names
        r = await c.call_tool("list_plugins", {})
        plugins = {p["name"]: p for p in r.structured_content["plugins"]}
        assert plugins["lesion_count"]["tools"] == ["count_lesions"] and plugins["broken"][
            "error"
        ].startswith("RuntimeError")
        r = await c.call_tool("count_lesions", {"mask": "ball_mask.nii.gz", "image": "ct.nii.gz"})
        assert (
            not r.is_error
            and r.structured_content["n_components"] == 1
            and any(isinstance(x, ImageContent) for x in r.content)
        )
        r = await c.call_tool("my_tool_volume", {"mask": "ball_mask.nii.gz"})
        assert r.structured_content["volume_ml"] > 0
    # model + guideline scaffolds are picked up by the registries
    new_model(isolated_settings.workspace / "omm_models", "demo-model")
    new_model(isolated_settings.workspace / "omm_models", "wrapped-demo", wrapped_image="example/tool:1")
    new_guideline(isolated_settings.workspace / "omm_guidelines", "my-protocol", "My protocol")
    from open_med_mcp.guidelines.loader import get_library
    from open_med_mcp.models.registry import get_registry

    reg = get_registry(isolated_settings, reload=True)
    assert {"demo-model", "wrapped-demo"} <= set(reg.names()) and reg.get("wrapped-demo").is_wrapped
    assert get_library(isolated_settings, reload=True).get("my-protocol").title == "My protocol"
    with pytest.raises(ValueError):
        new_plugin(pdir, "Bad Name")


def test_cli_client_configs_parse(isolated_settings):
    import tomllib
    from typer.testing import CliRunner

    from open_med_mcp.cli import app

    runner = CliRunner()
    out = runner.invoke(app, ["client-config", "claude-desktop"]).stdout
    json.loads(out[: out.rindex("}") + 1])
    out = runner.invoke(app, ["client-config", "codex"]).stdout
    tomllib.loads(out[out.index("[mcp_servers") :])
    r = runner.invoke(app, ["install", "codex", "--dry-run"])
    assert r.exit_code == 0 and "codex mcp add open-med-mcp" in r.stdout
    r = runner.invoke(app, ["new", "plugin", "extra-tool"])
    assert r.exit_code == 0 and (isolated_settings.workspace / "omm_plugins" / "extra_tool.py").exists()
    r = runner.invoke(app, ["plugins", "list"])
    assert r.exit_code == 0 and "extra_tool" in r.stdout
