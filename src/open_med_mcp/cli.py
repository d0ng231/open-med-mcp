"""Command line interface: ``open-med-mcp serve|models|guidelines|view|run|doctor|client-config``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from open_med_mcp import __version__

app = typer.Typer(
    help="open-med-mcp - medical image analysis tools for AI agents (MCP server + CLI)",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help="Model zoo: list, inspect, download weights, build/pull containers", no_args_is_help=True
)
guidelines_app = typer.Typer(help="Guideline library", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(guidelines_app, name="guidelines")
console = Console()


def _settings(workspace: Optional[Path] = None):
    from open_med_mcp.config import Settings, set_settings

    if workspace is not None:
        os.environ["OMM_WORKSPACE"] = str(workspace.resolve())
    return set_settings(Settings())


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Print version and exit", callback=_version_callback, is_eager=True
    ),
) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="stdio | streamable-http | sse"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
    workspace: Optional[Path] = typer.Option(
        None, help="Workspace root (default: current directory or $OMM_WORKSPACE)"
    ),
    log_level: Optional[str] = typer.Option(
        None, help="DEBUG | INFO | WARNING | ERROR (default $OMM_LOG_LEVEL or INFO)"
    ),
    no_plugins: bool = typer.Option(False, help="Do not load plug-ins"),
) -> None:
    """Start the MCP server (stdio by default - this is what Claude Code / Codex / Claude Desktop launch).

    Nothing is ever written to stdout except the MCP stream; logs go to stderr and
    $OMM_HOME/logs/server.log. Over HTTP/SSE, file access is confined to the workspace unless
    OMM_ALLOW_OUTSIDE_WORKSPACE=1 is set explicitly."""
    from open_med_mcp.server import create_server

    if log_level:
        os.environ["OMM_LOG_LEVEL"] = log_level.upper()
    if transport != "stdio" and "OMM_ALLOW_OUTSIDE_WORKSPACE" not in os.environ:
        os.environ["OMM_ALLOW_OUTSIDE_WORKSPACE"] = "0"
    settings = _settings(workspace)
    server = create_server(settings, plugins=not no_plugins)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=transport, host=host, port=port)


@app.command()
def doctor(workspace: Optional[Path] = typer.Option(None)) -> None:
    """Check the environment: Python, GPU, Docker/Apptainer, models, weights, guidelines."""
    from open_med_mcp.guidelines.loader import get_library
    from open_med_mcp.models import containers
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.models.runners import host_has_gpu, resolve_device, runner_report
    from open_med_mcp.models.weights import weights_status

    s = _settings(workspace)
    rprint(f"[bold]open-med-mcp[/bold] {__version__}  python {sys.version.split()[0]}  ({sys.executable})")
    rprint(
        f"workspace: {s.workspace}\nhome:      {s.home}\nrunner:    {s.runner}   device: {resolve_device(s)}   nvidia-smi: {'yes' if host_has_gpu() else 'no'}"
    )
    ok, msg = containers.docker_usable()
    rprint(f"docker:    {'ok' if ok else 'unavailable'} ({msg})")
    rprint(f"apptainer: {containers.apptainer_cmd() or 'not found'}")
    reg = get_registry(s, reload=True)
    table = Table(title="models")
    for col in ("name", "local", "docker", "apptainer", "weights"):
        table.add_column(col)
    for m in reg.all():
        rep = runner_report(m, s)
        ws = weights_status(m, s)
        wtxt = f"{sum(1 for w in ws if w['present'])}/{len(ws)}" if ws else "-"
        table.add_row(
            m.name,
            *[
                ("[green]yes[/green]" if rep[r]["available"] else f"[red]no[/red] {rep[r]['detail'][:50]}")
                for r in ("local", "docker", "apptainer")
            ],
            wtxt,
        )
    console.print(table)
    if reg.errors:
        rprint("[red]manifest errors:[/red]", reg.errors)
    rprint(f"guidelines: {', '.join(get_library(s, reload=True).names())}")


@models_app.command("list")
def models_list(workspace: Optional[Path] = typer.Option(None)) -> None:
    """List models."""
    from open_med_mcp.models.registry import get_registry

    s = _settings(workspace)
    table = Table()
    for col in ("name", "tasks", "modalities", "prompts", "description"):
        table.add_column(col)
    for m in get_registry(s, reload=True).all():
        table.add_row(
            m.name,
            ", ".join(m.tasks[:4]) + (" ..." if len(m.tasks) > 4 else ""),
            ", ".join(m.modalities),
            ", ".join(m.prompt_types) or "automatic",
            m.description[:90],
        )
    console.print(table)


@models_app.command("info")
def models_info(name: str) -> None:
    """Show a manifest, its parameters and backend availability."""
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.models.runners import runner_report
    from open_med_mcp.models.weights import weights_status

    s = _settings()
    m = get_registry(s).get(name)
    data = m.model_dump(mode="json", exclude={"adapter_dir"})
    data["backends"] = runner_report(m, s)
    data["weights_status"] = weights_status(m, s)
    typer.echo(json.dumps(data, indent=2))


@models_app.command("check")
def models_check(name: str) -> None:
    """Which backends can run the model right now?"""
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.models.runners import runner_report

    s = _settings()
    for k, v in runner_report(get_registry(s).get(name), s).items():
        rprint(f"{k:10s} {'[green]ok[/green]' if v['available'] else '[red]no[/red]'}  {v['detail']}")


@models_app.command("download")
def models_download(
    name: str,
    weights: Optional[list[str]] = typer.Option(None, "--weights", "-w", help="Weight ids (default: all)"),
) -> None:
    """Download weights declared in the manifest (or run the adapter's own download task)."""
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.models.weights import ensure_weights, weights_dir_for

    s = _settings()
    s.ensure_dirs()
    m = get_registry(s).get(name)
    if not m.weights:
        if "download" in m.tasks:
            rprint(f"running the adapter's download task into {weights_dir_for(m, s)}")
            _run_model(m, "download", {}, {}, s)
        else:
            rprint("this model has no downloadable weights (they are fetched on first run)")
        return

    def progress(wid: str, done: int, total: int | None) -> None:
        pct = f"{100 * done / total:5.1f}%" if total else f"{done / 1e6:.0f} MB"
        print(f"\r  {wid}: {pct}", end="", flush=True)

    for p in ensure_weights(m, weights, s, progress):
        print(f"\n  -> {p}")
    rprint(f"[green]weights ready in {weights_dir_for(m, s)}[/green]")


@models_app.command("build")
def models_build(
    name: str,
    engine: str = typer.Option("docker", help="docker | apptainer"),
    tag: Optional[str] = typer.Option(None),
    fakeroot: bool = typer.Option(False, help="apptainer: use --fakeroot"),
) -> None:
    """Build the model's container image from its Dockerfile (Apptainer builds from a converted definition)."""
    from open_med_mcp.models import containers
    from open_med_mcp.models.registry import get_registry

    s = _settings()
    m = get_registry(s).get(name)
    if engine == "docker":
        cmd = containers.build_docker(m, tag)
    else:
        cmd, sif = containers.build_apptainer(m, s, fakeroot=fakeroot)
        rprint(f"definition: {sif.with_suffix('.def')}")
    rprint("[dim]$ " + " ".join(cmd) + "[/dim]")
    raise typer.Exit(subprocess.call(cmd))


@models_app.command("pull")
def models_pull(name: str, engine: str = typer.Option("docker", help="docker | apptainer")) -> None:
    """Pull the pre-built image from the registry (GHCR)."""
    from open_med_mcp.models import containers
    from open_med_mcp.models.registry import get_registry

    s = _settings()
    m = get_registry(s).get(name)
    if engine == "docker":
        cmd = ["docker", "pull", containers.image_name(m, s)]
    else:
        cmd, _ = containers.pull_apptainer(m, s)
    rprint("[dim]$ " + " ".join(cmd) + "[/dim]")
    raise typer.Exit(subprocess.call(cmd))


@models_app.command("def")
def models_def(name: str) -> None:
    """Print the Apptainer definition generated from the model's Dockerfile."""
    from open_med_mcp.models import containers
    from open_med_mcp.models.registry import get_registry

    m = get_registry(_settings()).get(name)
    df = m.dockerfile_path()
    if df is None:
        raise typer.BadParameter("model has no Dockerfile")
    typer.echo(containers.dockerfile_to_def(df, containers.build_context(m)))


def _run_model(m, task: str, inputs: dict, params: dict, s):
    from open_med_mcp.models.job import prepare_job, read_response
    from open_med_mcp.models.runners import resolve_device, select_runner
    from open_med_mcp.models.weights import weights_dir_for
    from open_med_mcp.workspace import new_run_dir

    run_dir = new_run_dir(m.name, s)
    prepare_job(run_dir, m, task, inputs, params, resolve_device(s), weights_dir_for(m, s))
    runner = select_runner(m, s)
    rprint(f"[dim]runner={runner.name} run_dir={run_dir}[/dim]")
    out = runner.run(m, run_dir, weights_dir_for(m, s), resolve_device(s))
    if out.returncode != 0:
        rprint(f"[red]exit code {out.returncode}[/red]\n{out.stderr_tail}")
    return read_response(run_dir), run_dir


@app.command()
def run(
    model: str,
    image: Optional[Path] = typer.Option(None, help="Input image"),
    task: Optional[str] = typer.Option(None),
    params: str = typer.Option("{}", help="JSON parameters"),
    prompts: Optional[str] = typer.Option(None, help="JSON list of prompts (promptable models)"),
    plane: str = typer.Option("axial"),
    output: Optional[Path] = typer.Option(None, help="Copy the mask here"),
    workspace: Optional[Path] = typer.Option(None),
) -> None:
    """Run a model from the command line (same contract as the MCP tools)."""
    from open_med_mcp.core.image import MedicalImage
    from open_med_mcp.core.prompts import normalize_prompts
    from open_med_mcp.models.registry import get_registry
    from open_med_mcp.workspace import stage_file

    s = _settings(workspace)
    s.ensure_dirs()
    m = get_registry(s).get(model)
    p = json.loads(params)
    inputs = {}
    if image is not None:
        inputs["image"] = image.resolve()
        if prompts:
            img = MedicalImage.load(image)
            p["prompts"] = normalize_prompts(json.loads(prompts), img, plane)  # type: ignore[arg-type]
            p["axis"] = img.numpy_axis_for_plane(plane)  # type: ignore[arg-type]
    resp, run_dir = _run_model(m, task or m.default_task, inputs, p, s)
    if output and "mask" in resp["outputs"]:
        stage_file(Path(resp["outputs"]["mask"]), output.resolve())
        resp["outputs"]["mask"] = str(output.resolve())
    typer.echo(
        json.dumps(
            {k: resp[k] for k in ("status", "outputs", "labels", "stats", "timing", "warnings") if k in resp},
            indent=2,
        )
    )


@app.command()
def view(
    image: Path,
    mask: Optional[list[Path]] = typer.Option(None, "--mask", "-m"),
    plane: str = typer.Option("axial"),
    layout: str = typer.Option("single", help="single | three-plane | montage"),
    window: Optional[str] = typer.Option(None),
    html: bool = typer.Option(False, help="Write the interactive HTML viewer instead of a PNG"),
    out: Optional[Path] = typer.Option(None),
) -> None:
    """Render an image with masks to PNG (or the interactive HTML viewer)."""
    from open_med_mcp.core.image import MedicalImage, load_mask, read_labels_sidecar
    from open_med_mcp.viewer.registry import get_renderer
    from open_med_mcp.viewer.spec import MaskLayer, ViewSpec

    _settings()
    img = MedicalImage.load(image)
    items, layers = [], []
    for mp in mask or []:
        layer = MaskLayer(path=str(mp), name=mp.name)
        layers.append(layer)
        items.append((load_mask(mp, img), {"layer": layer, "labels": read_labels_sidecar(mp) or {}}))
    spec = ViewSpec(image=str(image), masks=layers, plane=plane, layout=layout, window=window)  # type: ignore[arg-type]
    res = get_renderer("html" if html else "png").render(img, items, spec)
    dest = out or image.with_name(
        image.name.split(".")[0] + ("_viewer.html" if html else f"_{plane}_{layout}.png")
    )
    res.save(dest)
    rprint(f"wrote {dest}")


@app.command("serve-viewer")
def serve_viewer(
    image: Path,
    mask: Optional[list[Path]] = typer.Option(None, "--mask", "-m"),
    port: int = typer.Option(8766),
) -> None:
    """Serve an interactive 3D NiiVue viewer for the image (+ masks) on localhost."""
    from open_med_mcp.viewer.serve import build_niivue_page, serve_directory

    root = image.resolve().parent
    files = [image.resolve(), *[m.resolve() for m in mask or []]]
    for f in files:
        if root not in f.parents and f.parent != root:
            raise typer.BadParameter("image and masks must live in the same directory (served as web root)")
    html = build_niivue_page(files[0], files[1:], root)
    rprint(f"serving {root} on http://127.0.0.1:{port}/index.html (Ctrl-C to stop)")
    serve_directory(root, port, html, block=True)


@guidelines_app.command("list")
def guidelines_list() -> None:
    """List guidelines."""
    from open_med_mcp.guidelines.loader import get_library

    table = Table()
    for col in ("name", "title", "modalities", "source"):
        table.add_column(col)
    for g in get_library(_settings(), reload=True).all():
        table.add_row(g.name, g.title, ", ".join(g.modalities), g.source)
    console.print(table)


@guidelines_app.command("show")
def guidelines_show(name: str) -> None:
    """Print a guideline."""
    from open_med_mcp.guidelines.loader import get_library

    typer.echo(get_library(_settings(), reload=True).get(name).render())


plugins_app = typer.Typer(
    help="Plug-ins (workspace omm_plugins/, OMM_PLUGIN_DIRS, entry points)", no_args_is_help=True
)
app.add_typer(plugins_app, name="plugins")
new_app = typer.Typer(help="Scaffold a new model adapter, tool plug-in or guideline", no_args_is_help=True)
app.add_typer(new_app, name="new")


@plugins_app.command("list")
def plugins_list(workspace: Optional[Path] = typer.Option(None)) -> None:
    """Load plug-ins the way the server does and report what they add."""
    from open_med_mcp.server import create_server

    s = _settings(workspace)
    create_server(s)
    from open_med_mcp.plugins import loaded_plugins, plugin_dirs

    rprint("plug-in dirs:", ", ".join(str(d) for d in plugin_dirs(s)))
    table = Table()
    for col in ("name", "kind", "tools", "source", "error"):
        table.add_column(col)
    for p in loaded_plugins():
        table.add_row(p.name, p.kind, ", ".join(p.tools_added), p.source, p.error or "")
    console.print(table)


@new_app.command("model")
def new_model_cmd(
    name: str,
    directory: Optional[Path] = typer.Option(
        None, "--dir", help="Parent directory (default: <workspace>/omm_models)"
    ),
    wrapped_image: Optional[str] = typer.Option(
        None, "--wrapped-image", help="Wrap an existing container image instead of writing run.py"
    ),
) -> None:
    """Create a model adapter folder from the template (manifest, run.py, Dockerfile, README)."""
    from open_med_mcp.scaffold import new_model

    s = _settings()
    dest = new_model(directory or (s.workspace / "omm_models"), name, wrapped_image)
    rprint(
        f"[green]created[/green] {dest}\nnext: edit {dest / (name + '.yaml')}, then `open-med-mcp models check {name}` and `open-med-mcp run {name} --image ...`"
    )


@new_app.command("plugin")
def new_plugin_cmd(
    name: str,
    directory: Optional[Path] = typer.Option(None, "--dir", help="Default: <workspace>/omm_plugins"),
) -> None:
    """Create a tool plug-in file (register(server) with an example tool)."""
    from open_med_mcp.scaffold import new_plugin

    s = _settings()
    dest = new_plugin(directory or (s.workspace / "omm_plugins"), name)
    rprint(
        f"[green]created[/green] {dest}\nrestart the server (or run `open-med-mcp plugins list`) to load it"
    )


@new_app.command("guideline")
def new_guideline_cmd(
    name: str,
    title: Optional[str] = typer.Option(None),
    directory: Optional[Path] = typer.Option(None, "--dir", help="Default: <workspace>/omm_guidelines"),
) -> None:
    """Create a guideline skeleton (Markdown with YAML front matter)."""
    from open_med_mcp.scaffold import new_guideline

    s = _settings()
    dest = new_guideline(directory or (s.workspace / "omm_guidelines"), name, title)
    rprint(f"[green]created[/green] {dest}")


@app.command()
def install(
    client: str = typer.Argument("claude-code", help="claude-code | codex"),
    workspace: Optional[Path] = typer.Option(None),
    scope: str = typer.Option("local", help="claude-code: local | project | user"),
    dry_run: bool = typer.Option(False, help="Only print the command"),
) -> None:
    """Register this server with a client CLI (runs `claude mcp add` / `codex mcp add`)."""
    exe = shutil.which("open-med-mcp") or f"{sys.executable} -m open_med_mcp"
    ws = str((workspace or Path.cwd()).resolve())
    if client == "claude-code":
        cmd = [
            "claude",
            "mcp",
            "add",
            "--scope",
            scope,
            "open-med-mcp",
            "-e",
            f"OMM_WORKSPACE={ws}",
            "--",
            *exe.split(),
            "serve",
        ]
    elif client == "codex":
        cmd = [
            "codex",
            "mcp",
            "add",
            "open-med-mcp",
            "--env",
            f"OMM_WORKSPACE={ws}",
            "--",
            *exe.split(),
            "serve",
        ]
    else:
        raise typer.BadParameter("client must be claude-code or codex (use client-config for others)")
    rprint("[dim]$ " + " ".join(cmd) + "[/dim]")
    if dry_run:
        return
    if shutil.which(cmd[0]) is None:
        raise typer.BadParameter(f"{cmd[0]} is not installed or not on PATH")
    raise typer.Exit(subprocess.call(cmd))


@app.command("client-config")
def client_config(
    client: str = typer.Argument(
        "claude-code", help="claude-code | claude-desktop | codex | cursor | gemini"
    ),
    workspace: Optional[Path] = typer.Option(None),
) -> None:
    """Print the MCP client configuration snippet for this installation."""
    exe = shutil.which("open-med-mcp") or f"{sys.executable} -m open_med_mcp"
    ws = str((workspace or Path.cwd()).resolve())
    if client == "claude-code":
        typer.echo(f"claude mcp add open-med-mcp -e OMM_WORKSPACE={ws} -- {exe} serve")
    elif client == "codex":
        typer.echo(f"codex mcp add open-med-mcp --env OMM_WORKSPACE={ws} -- {exe} serve")
        typer.echo(
            '\n# or in ~/.codex/config.toml:\n[mcp_servers.open_med_mcp]\ncommand = "'
            + exe.split()[0]
            + '"\nargs = ['
            + ", ".join(json.dumps(a) for a in exe.split()[1:] + ["serve"])
            + "]\n[mcp_servers.open_med_mcp.env]\nOMM_WORKSPACE = "
            + json.dumps(ws)
        )
    else:
        cfg = {
            "mcpServers": {
                "open-med-mcp": {
                    "command": exe.split()[0],
                    "args": exe.split()[1:] + ["serve"],
                    "env": {"OMM_WORKSPACE": ws},
                }
            }
        }
        typer.echo(json.dumps(cfg, indent=2))
        typer.echo(
            {
                "claude-desktop": "# -> claude_desktop_config.json",
                "cursor": "# -> .cursor/mcp.json",
                "gemini": "# -> ~/.gemini/settings.json",
            }.get(client, "")
        )


if __name__ == "__main__":
    app()
