"""Execution backends. Adapter models run ``run.py --job <dir>``; wrapped models run a rendered
command line inside a third-party image (or on the host). All of them use the job-directory contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from open_med_mcp.config import Settings, get_settings
from open_med_mcp.models import containers, wrapped
from open_med_mcp.models.manifest import ModelManifest

SDK_DIR = Path(__file__).resolve().parent.parent / "zoo" / "_sdk"


def host_has_gpu() -> bool:
    if os.environ.get("OMM_DEVICE", "").lower() in ("none", "cpu"):
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def resolve_device(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    dev = settings.device
    if dev == "auto":
        return "cuda" if host_has_gpu() else "cpu"
    if dev in ("none", "cpu"):
        return "cpu"
    return dev


@dataclass
class RunOutcome:
    returncode: int
    command: list[str]
    runner: str
    stdout_tail: str
    stderr_tail: str


def _request(job_dir: Path) -> dict:
    return json.loads((job_dir / "request.json").read_text(encoding="utf-8"))


class Runner:
    name = "base"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def available(self, manifest: ModelManifest) -> tuple[bool, str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def command(
        self, manifest: ModelManifest, job_dir: Path, weights_dir: Path, device: str
    ) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def env(self, manifest: ModelManifest, weights_dir: Path, device: str) -> dict[str, str]:
        return dict(os.environ)

    def run(
        self,
        manifest: ModelManifest,
        job_dir: Path,
        weights_dir: Path,
        device: str,
        timeout: int | None = None,
    ) -> RunOutcome:
        cmd = self.command(manifest, job_dir, weights_dir, device)
        started = time.time()
        log_path = job_dir / "log.txt"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(job_dir),
                env=self.env(manifest, weights_dir, device),
                capture_output=True,
                text=True,
                timeout=timeout or self.settings.run_timeout_s,
            )
            log.write(proc.stdout)
            log.write(proc.stderr)
        if manifest.is_wrapped:
            wrapped.finalize(job_dir, manifest, _request(job_dir), proc.returncode, cmd, started)
        return RunOutcome(proc.returncode, cmd, self.name, proc.stdout[-3000:], proc.stderr[-3000:])


class LocalRunner(Runner):
    """Runs the adapter's ``run.py`` with a Python interpreter on this machine (or a host CLI for wrapped models)."""

    name = "local"

    def python_for(self, manifest: ModelManifest) -> str:
        return self.settings.local_python_for(manifest.adapter) or sys.executable

    def available(self, manifest: ModelManifest) -> tuple[bool, str]:
        if "local" not in manifest.runners:
            return False, "manifest disables the local runner"
        if manifest.is_wrapped:
            if not manifest.local.command:
                return False, "wrapped model has no host command (use the container)"
            exe = shutil.which(manifest.local.command[0])
            return (
                (True, f"host command {exe}") if exe else (False, f"{manifest.local.command[0]} not on PATH")
            )
        if not manifest.entrypoint_path().exists():
            return False, f"entrypoint {manifest.entrypoint_path()} not found"
        py = self.python_for(manifest)
        if not manifest.local.requires:
            return True, f"python={py}"
        code = "import importlib,sys\n" + "\n".join(
            f"importlib.import_module({m!r})" for m in manifest.local.requires
        )
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=180)
        except Exception as exc:
            return False, f"cannot run {py}: {exc}"
        if r.returncode != 0:
            missing = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "import failed"
            hint = (
                f" (pip install 'open-med-mcp[{manifest.local.extra}]' or set OMM_ZOO_{manifest.adapter.upper().replace('-', '_')}_PYTHON)"
                if manifest.local.extra
                else ""
            )
            return False, f"{missing}{hint}"
        return True, f"python={py}"

    def command(self, manifest: ModelManifest, job_dir: Path, weights_dir: Path, device: str) -> list[str]:
        if manifest.is_wrapped:
            return wrapped.render_command(manifest.local.command, _request(job_dir), manifest, str(job_dir))
        return [self.python_for(manifest), str(manifest.entrypoint_path()), "--job", str(job_dir)]

    def env(self, manifest: ModelManifest, weights_dir: Path, device: str) -> dict[str, str]:
        env = dict(os.environ)
        env["OMM_WEIGHTS_DIR"] = str(weights_dir)
        env["OMM_DEVICE"] = device
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SDK_DIR), env.get("PYTHONPATH", "")) if p)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


class DockerRunner(Runner):
    name = "docker"

    def available(self, manifest: ModelManifest) -> tuple[bool, str]:
        if "docker" not in manifest.runners:
            return False, "manifest disables docker"
        ok, msg = containers.docker_usable()
        if not ok:
            return False, msg
        image = containers.image_name(manifest, self.settings)
        if not containers.docker_image_exists(image):
            if manifest.is_wrapped:
                return True, f"image {image} will be pulled on first use"
            return (
                False,
                f"image {image} not present locally (run: open-med-mcp models pull {manifest.name} / build)",
            )
        return True, f"image {image}"

    def command(self, manifest: ModelManifest, job_dir: Path, weights_dir: Path, device: str) -> list[str]:
        image = containers.image_name(manifest, self.settings)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{job_dir}:/job",
            "-v", f"{weights_dir}:/weights",
            "-e", "OMM_WEIGHTS_DIR=/weights",
            "-e", f"OMM_DEVICE={device}",
            "-e", "HOME=/tmp",
            "--shm-size", manifest.container.shm_size,
        ]  # fmt: skip
        if hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        use_gpu = device.startswith("cuda") and manifest.container.gpu != "none"
        if use_gpu:
            cmd += ["--gpus", "all"]
        if manifest.is_wrapped:
            if use_gpu and manifest.container.gpu_image:
                image = manifest.container.gpu_image
            rendered = wrapped.render_command(manifest.container.command, _request(job_dir), manifest, "/job")
            return cmd + ["--entrypoint", rendered[0], image, *rendered[1:]]
        return cmd + [image, *manifest.container.entrypoint, "--job", "/job"]

    def run(
        self,
        manifest: ModelManifest,
        job_dir: Path,
        weights_dir: Path,
        device: str,
        timeout: int | None = None,
    ) -> RunOutcome:
        outcome = super().run(manifest, job_dir, weights_dir, device, timeout)
        if (
            outcome.returncode != 0
            and "--gpus" in outcome.command
            and "could not select device driver" in outcome.stderr_tail
        ):
            with (job_dir / "log.txt").open("a", encoding="utf-8") as log:
                log.write("\n[open-med-mcp] docker has no GPU support; retrying on CPU\n")
            outcome = super().run(manifest, job_dir, weights_dir, "cpu", timeout)
        return outcome


class ApptainerRunner(Runner):
    name = "apptainer"

    def image_ref(self, manifest: ModelManifest) -> str:
        sif = containers.sif_path(manifest, self.settings)
        if sif.exists():
            return str(sif)
        return f"docker://{containers.image_name(manifest, self.settings)}"

    def available(self, manifest: ModelManifest) -> tuple[bool, str]:
        if "apptainer" not in manifest.runners:
            return False, "manifest disables apptainer"
        exe = containers.apptainer_cmd()
        if exe is None:
            return False, "apptainer/singularity not found"
        ref = self.image_ref(manifest)
        if ref.startswith("docker://"):
            return (
                True,
                f"{exe} will pull {ref} on first use (or: open-med-mcp models pull {manifest.name} --engine apptainer)",
            )
        return True, f"{exe} image {ref}"

    def command(self, manifest: ModelManifest, job_dir: Path, weights_dir: Path, device: str) -> list[str]:
        exe = containers.apptainer_cmd() or "apptainer"
        cmd = [
            exe, "exec", "--cleanenv", "--containall", "--writable-tmpfs",
            "-B", f"{job_dir}:/job",
            "-B", f"{weights_dir}:/weights",
            "--env", "OMM_WEIGHTS_DIR=/weights",
            "--env", f"OMM_DEVICE={device}",
            "--pwd", "/job",
        ]  # fmt: skip
        use_gpu = device.startswith("cuda") and manifest.container.gpu != "none"
        if use_gpu:
            cmd.append("--nv")
        ref = self.image_ref(manifest)
        if manifest.is_wrapped:
            if use_gpu and manifest.container.gpu_image and ref.startswith("docker://"):
                ref = f"docker://{manifest.container.gpu_image}"
            rendered = wrapped.render_command(manifest.container.command, _request(job_dir), manifest, "/job")
            return cmd + [ref, *rendered]
        return cmd + [ref, *manifest.container.entrypoint, "--job", "/job"]


RUNNERS: dict[str, type[Runner]] = {
    "local": LocalRunner,
    "docker": DockerRunner,
    "apptainer": ApptainerRunner,
}


def runner_report(manifest: ModelManifest, settings: Settings | None = None) -> dict[str, dict[str, object]]:
    settings = settings or get_settings()
    report = {}
    for name, cls in RUNNERS.items():
        ok, msg = cls(settings).available(manifest)
        report[name] = {"available": ok, "detail": msg}
    return report


def select_runner(
    manifest: ModelManifest, settings: Settings | None = None, prefer: str | None = None
) -> Runner:
    """Pick the first usable runner.

    ``auto`` order: local (if its Python can import the model / the host CLI exists) -> docker ->
    apptainer. An explicit ``OMM_RUNNER``/``prefer`` value is honoured or raises with the reason.
    """
    settings = settings or get_settings()
    choice = prefer or settings.runner
    if choice != "auto":
        runner = RUNNERS[choice](settings)
        ok, msg = runner.available(manifest)
        if not ok:
            raise RuntimeError(f"runner {choice!r} is not usable for model {manifest.name!r}: {msg}")
        return runner
    reasons = []
    for name in ("local", "docker", "apptainer"):
        if name not in manifest.runners:
            continue
        runner = RUNNERS[name](settings)
        ok, msg = runner.available(manifest)
        if ok:
            return runner
        reasons.append(f"{name}: {msg}")
    raise RuntimeError(
        f"no execution backend available for model {manifest.name!r}:\n  "
        + "\n  ".join(reasons)
        + "\nOptions: pip install the model extra for the local runner, build/pull the container "
        "(open-med-mcp models build|pull <model>), or set OMM_RUNNER explicitly."
    )
