"""Container helpers: image naming, Dockerfile -> Apptainer definition conversion, build/pull."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

from open_med_mcp.config import Settings, get_settings
from open_med_mcp.models.manifest import ModelManifest

REPO_ROOT = Path(__file__).resolve().parents[3]


def image_name(manifest: ModelManifest, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if manifest.container.image:
        return manifest.container.image
    return f"{settings.image_prefix}-{manifest.adapter}:{manifest.version}"


def sif_path(manifest: ModelManifest, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if manifest.is_wrapped and manifest.container.image:
        last = manifest.container.image.split("/")[-1]
        tag = last.rsplit(":", 1)[-1] if ":" in last else "latest"
        return settings.images_dir / f"{manifest.name}-{tag}.sif"
    return settings.images_dir / f"{manifest.adapter}-{manifest.version}.sif"


def build_context(manifest: ModelManifest) -> Path:
    """The docker build context is the repository root when running from a checkout, otherwise
    the installed package root (which still contains ``zoo/``)."""
    pkg_root = Path(__file__).resolve().parents[1]  # .../open_med_mcp
    if (REPO_ROOT / "pyproject.toml").exists() and (REPO_ROOT / "src" / "open_med_mcp").exists():
        return REPO_ROOT
    return pkg_root.parent


def _rel_to_context(path: Path, context: Path) -> str:
    try:
        return str(path.relative_to(context))
    except ValueError:
        return str(path)


def _join_continuations(text: str) -> list[str]:
    lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        lines.append(buf.strip())
        buf = ""
    if buf.strip():
        lines.append(buf.strip())
    return lines


def dockerfile_to_def(dockerfile: Path, context: Path) -> str:
    """Convert the subset of Dockerfile syntax used by the zoo into an Apptainer definition file.

    Supported instructions: FROM, ARG, ENV, WORKDIR, COPY, RUN, ENTRYPOINT, CMD, LABEL.
    Paths are resolved to absolute paths so the definition can be built from anywhere.
    """
    base = ""
    env: list[str] = []
    files: list[str] = []
    post: list[str] = []
    labels: list[str] = []
    entrypoint: list[str] = []
    cmd: list[str] = []
    workdir = "/"
    for line in _join_continuations(dockerfile.read_text(encoding="utf-8")):
        m = re.match(r"^(\w+)\s+(.*)$", line)
        if not m:
            continue
        instr, rest = m.group(1).upper(), m.group(2).strip()
        if instr == "FROM":
            base = rest.split()[0]
        elif instr == "ARG":
            name, _, default = rest.partition("=")
            post.append(f"export {name.strip()}=${{{name.strip()}:-{default.strip()}}}")
        elif instr == "ENV":
            pairs = re.findall(r'(\w+)=("[^"]*"|\'[^\']*\'|\S+)', rest)
            if not pairs:
                key, _, val = rest.partition(" ")
                pairs = [(key, val)]
            for key, val in pairs:
                env.append(f"export {key}={val}")
                post.append(f"export {key}={val}")
        elif instr == "WORKDIR":
            workdir = rest
            post.append(f"mkdir -p {shlex.quote(rest)} && cd {shlex.quote(rest)}")
        elif instr == "COPY":
            parts = shlex.split(rest)
            parts = [p for p in parts if not p.startswith("--")]
            *srcs, dest = parts
            for src in srcs:
                src_abs = (context / src).resolve()
                if dest.endswith("/"):
                    files.append(
                        f"{src_abs} {dest}{Path(src).name}"
                        if src_abs.is_file()
                        else f"{src_abs} {dest.rstrip('/')}"
                    )
                else:
                    files.append(f"{src_abs} {dest}")
        elif instr == "RUN":
            post.append(rest)
        elif instr == "ENTRYPOINT":
            entrypoint = _parse_exec_form(rest)
        elif instr == "CMD":
            cmd = _parse_exec_form(rest)
        elif instr == "LABEL":
            labels.append(rest.replace("=", " ", 1))
    if not base:
        raise ValueError(f"no FROM instruction in {dockerfile}")
    run_cmd = " ".join(shlex.quote(x) for x in entrypoint) or "/bin/sh -c"
    run_default = " ".join(shlex.quote(x) for x in cmd)
    out = [
        "Bootstrap: docker",
        f"From: {base}",
        "",
        "%labels",
        "    org.open-med-mcp.generated-from " + str(dockerfile),
        *[f"    {lab}" for lab in labels],
        "",
        "%files",
        *[f"    {f}" for f in files],
        "",
        "%environment",
        *[f"    {e}" for e in env],
        "",
        "%post",
        "    set -e",
        *[f"    {p}" for p in post],
        "",
        "%runscript",
        f"    cd {shlex.quote(workdir)}",
        f'    if [ "$#" -gt 0 ]; then exec {run_cmd} "$@"; else exec {run_cmd} {run_default}; fi',
        "",
    ]
    return "\n".join(out)


def _parse_exec_form(rest: str) -> list[str]:
    rest = rest.strip()
    if rest.startswith("["):
        import json

        return [str(x) for x in json.loads(rest)]
    return shlex.split(rest)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def docker_usable() -> tuple[bool, str]:
    if not have("docker"):
        return False, "docker executable not found"
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20
        )
    except Exception as exc:  # pragma: no cover
        return False, f"docker info failed: {exc}"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip().splitlines()[-1] if (
            r.stderr or r.stdout
        ).strip() else "docker daemon not reachable"
    return True, f"docker server {r.stdout.strip()}"


def apptainer_cmd() -> str | None:
    for name in ("apptainer", "singularity"):
        if have(name):
            return name
    return None


def docker_image_exists(image: str) -> bool:
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    return r.returncode == 0


def build_docker(
    manifest: ModelManifest, tag: str | None = None, extra_args: list[str] | None = None
) -> list[str]:
    df = manifest.dockerfile_path()
    if df is None:
        raise FileNotFoundError(f"adapter {manifest.adapter!r} has no Dockerfile")
    ctx = build_context(manifest)
    cmd = ["docker", "build", "-f", str(df), "-t", tag or image_name(manifest), *(extra_args or []), str(ctx)]
    return cmd


def build_apptainer(
    manifest: ModelManifest, settings: Settings | None = None, out: Path | None = None, fakeroot: bool = False
) -> tuple[list[str], Path]:
    settings = settings or get_settings()
    df = manifest.dockerfile_path()
    if df is None:
        raise FileNotFoundError(f"adapter {manifest.adapter!r} has no Dockerfile")
    exe = apptainer_cmd()
    if exe is None:
        raise RuntimeError("apptainer/singularity executable not found")
    sif = out or sif_path(manifest, settings)
    sif.parent.mkdir(parents=True, exist_ok=True)
    def_path = sif.with_suffix(".def")
    def_path.write_text(dockerfile_to_def(df, build_context(manifest)))
    cmd = [exe, "build", *(["--fakeroot"] if fakeroot else []), "--force", str(sif), str(def_path)]
    return cmd, sif


def pull_apptainer(manifest: ModelManifest, settings: Settings | None = None) -> tuple[list[str], Path]:
    settings = settings or get_settings()
    exe = apptainer_cmd()
    if exe is None:
        raise RuntimeError("apptainer/singularity executable not found")
    sif = sif_path(manifest, settings)
    sif.parent.mkdir(parents=True, exist_ok=True)
    return [exe, "pull", "--force", str(sif), f"docker://{image_name(manifest, settings)}"], sif
