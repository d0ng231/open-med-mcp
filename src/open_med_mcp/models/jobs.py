"""Background job manager for long model runs (``run_model(..., wait=False)`` + ``get_job``).

Jobs run in daemon threads inside the server process; their state is also on disk (the run
directory), so a job can be inspected after the server restarts (``response.json`` present = done).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult

log = logging.getLogger("open_med_mcp.jobs")


@dataclass
class Job:
    id: str
    description: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    state: dict[str, Any] = field(default_factory=dict)  # shared with the worker: {"run_dir": Path}
    result: CallToolResult | None = None
    error: str | None = None
    thread: threading.Thread | None = None

    @property
    def status(self) -> str:
        if self.finished is None:
            return "running"
        return "error" if self.error else "done"

    def log_tail(self, n: int = 12) -> str:
        rd = self.state.get("run_dir")
        if rd and (Path(rd) / "log.txt").exists():
            lines = (Path(rd) / "log.txt").read_text(errors="replace").splitlines()
            return "\n".join(lines[-n:])
        return ""

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "description": self.description,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1),
            "run_dir": str(self.state.get("run_dir") or ""),
            "log_tail": self.log_tail(),
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def submit(self, description: str, fn: Callable[[dict[str, Any]], CallToolResult]) -> Job:
        with self._lock:
            self._counter += 1
            job = Job(id=f"job-{int(time.time())}-{self._counter}", description=description)
            self._jobs[job.id] = job

        def worker() -> None:
            try:
                job.result = fn(job.state)
                if job.result.is_error:
                    job.error = "".join(getattr(c, "text", "") for c in job.result.content)[:2000]
            except Exception as exc:  # noqa: BLE001
                job.error = f"{type(exc).__name__}: {exc}"
                log.exception("job %s failed", job.id)
            finally:
                job.finished = time.time()

        job.thread = threading.Thread(target=worker, name=job.id, daemon=True)
        job.thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
