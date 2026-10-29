"""
Analysis jobs for the web GUI.

Every analysis runs in its own process (``spawn``), so that it can be stopped at any time:
on request (cancel), when it exceeds the time limit, or when the server shuts down. The
child process streams its console output and progress events back through a pipe and
finally sends the JSON result (:mod:`prodigy.analysis.result`).
"""
from __future__ import annotations

import contextlib
import multiprocessing
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MAX_LOG = 2_000_000

RUNNING_STATES = ("starting", "running")


class BusyError(Exception):
    """Too many analyses are running at the same time."""


# ---------------------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------------------

class _Sender:
    def __init__(self, conn):
        self._conn = conn
        self._buffer: List[str] = []
        self._size = 0
        self._last_log = time.monotonic()
        self._last_progress = 0.0
        self._pending_progress: Optional[Dict[str, Any]] = None

    def send(self, kind: str, payload: Any) -> None:
        self._conn.send((kind, payload))

    def log(self, text: str) -> None:
        text = _ANSI.sub("", text)
        if not text:
            return
        self._buffer.append(text)
        self._size += len(text)
        if self._size > 8192 or time.monotonic() - self._last_log > 0.25:
            self.flush()

    def progress(self, event: Dict[str, Any]) -> None:
        now = time.monotonic()
        self._pending_progress = event
        if now - self._last_progress > 0.15 or (event.get("total") and event.get("done") == event.get("total")):
            self.send("progress", event)
            self._pending_progress = None
            self._last_progress = now

    def flush(self) -> None:
        if self._buffer:
            self.send("log", "".join(self._buffer))
            self._buffer.clear()
            self._size = 0
        if self._pending_progress is not None:
            self.send("progress", self._pending_progress)
            self._pending_progress = None
        self._last_log = time.monotonic()


def _limit_memory(megabytes: Optional[int]) -> None:
    if not megabytes:
        return
    try:
        import resource  # pylint: disable=import-outside-toplevel
        limit = int(megabytes) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # pylint: disable=broad-except
        pass


def _worker(request: Dict[str, Any], conn, memory_limit: Optional[int]) -> None:
    """Entry point of the analysis process."""
    _limit_memory(memory_limit)
    sender = _Sender(conn)
    try:
        sender.send("state", "running")
        from prodigy.web.requests import run_analysis  # pylint: disable=import-outside-toplevel
        result = run_analysis(request, output=sender.log, progress=sender.progress)
        sender.flush()
        sender.send("result", result)
    except BaseException as exc:  # pylint: disable=broad-except
        try:
            sender.flush()
            sender.send("result", {"status": "error",
                                   "error": {"type": type(exc).__name__,
                                             "message": str(exc) or type(exc).__name__,
                                             "traceback": traceback.format_exc()}})
        except Exception:  # pylint: disable=broad-except
            pass
    finally:
        try:
            conn.close()
        except Exception:  # pylint: disable=broad-except
            pass


# ---------------------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------------------

@contextlib.contextmanager
def _importable_main():
    """``spawn`` re-imports the parent's main module in the child. If the server was started
    from a source that cannot be re-imported (stdin, a notebook), hide it for the moment the
    child starts; the worker does not need it."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if path is None or os.path.isfile(path):
        yield
        return
    del main.__file__
    try:
        yield
    finally:
        main.__file__ = path


@dataclass
class Job:
    id: str  # pylint: disable=invalid-name
    process: Any
    conn: Any
    time_limit: float
    summary: Dict[str, Any]
    created: float = field(default_factory=time.time)
    started: float = field(default_factory=time.monotonic)
    finished: Optional[float] = None
    state: str = "starting"
    log: str = ""
    log_dropped: int = 0
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    @property
    def elapsed(self) -> float:
        end = self.finished if self.finished is not None else time.monotonic()
        return end - self.started

    @property
    def running(self) -> bool:
        return self.state in RUNNING_STATES

    def to_json(self, log_offset: int = 0) -> Dict[str, Any]:
        start = max(0, log_offset - self.log_dropped)
        return {
            "id": self.id,
            "state": self.state,
            "elapsed": round(self.elapsed, 3),
            "time_limit": self.time_limit,
            "log": self.log[start:],
            "log_offset": self.log_dropped + len(self.log),
            "progress": list(self.progress.values()),
            "result": self.result,
            "summary": self.summary,
        }


class JobManager:
    """Starts, watches and stops analysis processes."""

    def __init__(self, *, max_jobs: int = 2, time_limit: float = 120.0, memory_limit: Optional[int] = None,
                 keep_finished: int = 32):
        self.max_jobs = max_jobs
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.keep_finished = keep_finished
        self._context = multiprocessing.get_context("spawn")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="prodigy-jobs", daemon=True)
        self._watcher.start()

    # -- public ------------------------------------------------------------------------

    def submit(self, request: Dict[str, Any], time_limit: Optional[float] = None) -> Job:
        limit = self.time_limit if time_limit is None else min(float(time_limit), self.time_limit)
        with self._lock:
            if sum(1 for job in self._jobs.values() if job.running) >= self.max_jobs:
                raise BusyError(f"{self.max_jobs} analyses are already running")
            receiver, sender = self._context.Pipe(duplex=False)
            process = self._context.Process(target=_worker, args=(request, sender, self.memory_limit),
                                            name="prodigy-analysis", daemon=True)
            with _importable_main():
                process.start()
            sender.close()
            job = Job(id=uuid.uuid4().hex[:12], process=process, conn=receiver, time_limit=limit,
                      summary={"engine": request.get("engine"), "decisions": request.get("decisions")})
            self._jobs[job.id] = job
            self._prune()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._drain(job)
            return job

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.running:
                self._drain(job)
                if job.running:
                    self._stop_process(job, "cancelled")
            return job

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for job in self._jobs.values():
                if job.running:
                    self._stop_process(job, "cancelled")

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.running)

    # -- internals ---------------------------------------------------------------------

    def _watch(self) -> None:
        while not self._stop.wait(0.2):
            with self._lock:
                for job in list(self._jobs.values()):
                    if not job.running:
                        continue
                    self._drain(job)
                    if job.running and job.elapsed > job.time_limit:
                        self._stop_process(job, "timeout")

    def _drain(self, job: Job) -> None:
        if job.conn is None:
            return
        try:
            while job.conn.poll():
                kind, payload = job.conn.recv()
                if kind == "log":
                    self._append_log(job, payload)
                elif kind == "progress":
                    key = f"{payload.get('loop')}:{payload.get('strategy')}"
                    job.progress[key] = payload
                elif kind == "state":
                    if job.running:
                        job.state = payload
                elif kind == "result":
                    job.result = payload
                    job.state = "done"
                    self._finish(job)
                    return
        except (EOFError, OSError):
            pass
        if job.running and not job.process.is_alive():
            # the process ended without a result (crash, memory limit, ...)
            job.state = "crashed"
            job.result = {"status": "error", "error": {
                "type": "crash",
                "message": f"The analysis process stopped unexpectedly (exit code {job.process.exitcode})."}}
            self._finish(job)

    @staticmethod
    def _append_log(job: Job, text: str) -> None:
        job.log += text
        if len(job.log) > _MAX_LOG:
            cut = len(job.log) - _MAX_LOG
            job.log = job.log[cut:]
            job.log_dropped += cut

    def _finish(self, job: Job) -> None:
        job.finished = time.monotonic()
        try:
            job.conn.close()
        except Exception:  # pylint: disable=broad-except
            pass
        job.conn = None
        job.process.join(timeout=1)

    def _stop_process(self, job: Job, state: str) -> None:
        job.process.terminate()
        job.process.join(timeout=1.5)
        if job.process.is_alive():
            job.process.kill()
            job.process.join(timeout=1)
        job.state = state
        message = ("The analysis was stopped after the time limit of "
                   f"{job.time_limit:g} s." if state == "timeout" else "The analysis was cancelled.")
        job.result = {"status": state, "error": {"type": state, "message": message}}
        job.finished = time.monotonic()
        try:
            job.conn.close()
        except Exception:  # pylint: disable=broad-except
            pass
        job.conn = None

    def _prune(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if not job.running]
        for job_id in finished[:-self.keep_finished] if len(finished) > self.keep_finished else []:
            del self._jobs[job_id]
