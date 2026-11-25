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
from typing import Any, Dict, Iterator, List, Optional

from prodigy.web.grid_cache import GridCache

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MAX_LOG = 2_000_000

RUNNING_STATES = ("starting", "running")
STREAM_KINDS = ("coefficients", "grid")
"""Progress events with streamed coefficients (:mod:`prodigy.analysis.streaming`)."""


AUXILIARY_KINDS = ("joint", "suggest")
"""Jobs the GUI starts in the background (grids of the joint view, suggestions of a language model). They
have a limit of their own (``max_auxiliary``) and never take a slot of the analyses and checks that the
user starts (``max_jobs``)."""
PREEMPTIBLE_KINDS = ("joint",)
"""A new job of such a kind replaces the oldest running one when the limit of auxiliary jobs is reached: the
joint view asks for a new grid when the user moves on, and the old one is no longer wanted."""
STALE_SECONDS = 20.0
"""An auxiliary job that nobody has asked about for this long is cancelled (the page was closed or reloaded)."""

_KIND_NAMES = {"analysis": "analysis", "equivalence": "equivalence check", "joint": "joint grid",
               "suggest": "suggestion of the language model"}


def describe_job(summary: Dict[str, Any]) -> str:
    """``"analysis of a.ppy"``, ``"joint grid"``, ... (for messages)."""
    name = _KIND_NAMES.get(str(summary.get("kind")), "analysis")
    file_name = summary.get("file_name")
    return f"{name} of {file_name}" if file_name and summary.get("kind") in ("analysis", "equivalence") else name


def describe_running(running: List[Dict[str, Any]]) -> str:
    """"``An analysis of a.ppy and a joint grid are running`` for the listings of running jobs."""
    parts = [job.get("description") or describe_job(job) for job in running]
    parts = [("an " if part[0] in "aeiou" else "a ") + part for part in parts]
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{text[0].upper()}{text[1:]} {'is' if len(parts) == 1 else 'are'} running"


class BusyError(Exception):
    """Too many jobs of one group are running at the same time; ``running`` lists them
    (:meth:`Job.listing`)."""

    def __init__(self, message: str, running: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.running = running or []


# ---------------------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------------------

class _Sender:
    def __init__(self, conn: Any) -> None:
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
        if event.get("kind") in STREAM_KINDS:
            # batches of streamed coefficients (prodigy.analysis.streaming): each one counts, none is dropped
            self.send("progress", event)
            return
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


def _worker(request: Dict[str, Any], conn: Any, memory_limit: Optional[int], kind: str = "analysis") -> None:
    """Entry point of the analysis process; ``kind`` "suggest" asks a language model for a
    candidate instead (:func:`prodigy.web.requests.run_suggestion`), ``kind`` "joint" computes the
    coefficient grid of two variables of a finished analysis (:func:`prodigy.web.requests.run_joint`), ``kind``
    "equivalence" compares two programs (:func:`prodigy.web.requests.run_equivalence`)."""
    _limit_memory(memory_limit)
    sender = _Sender(conn)
    try:
        sender.send("state", "running")
        # pylint: disable=import-outside-toplevel
        from prodigy.web.requests import run_analysis, run_equivalence, run_joint, run_suggestion
        run = {"suggest": run_suggestion, "joint": run_joint, "equivalence": run_equivalence}.get(kind, run_analysis)
        result = run(request, output=sender.log, progress=sender.progress)
        sender.flush()
        sender.send("result", result)
    except BaseException as exc:  # pylint: disable=broad-except
        try:
            sender.flush()
            # (a subclass of NotImplementedError is reported as NotImplementedError, as in the result)
            sender.send("result", {"status": "error",
                                   "error": {"type": "NotImplementedError" if isinstance(exc, NotImplementedError)
                                             else type(exc).__name__,
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

def merge_stream(stream: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Adds a batch of streamed coefficients (:mod:`prodigy.analysis.streaming`) to the state ``stream``
    of a job: ``marginals`` (variable -> ``{"rows", "done", ...}``) with ``variables``, ``limit``, ``bound``
    and ``residual_mass`` of an analysis, or the ``grid`` of a grid job with ``cells[r][c]`` (None while
    pending)."""
    if event.get("kind") == "coefficients":
        if event.get("event") == "start":
            stream.update({key: event.get(key) for key in ("variables", "limit", "bound", "residual_mass",
                                                           "residual_mass_float")})
            stream["marginals"] = {}
            return
        marginal = stream.setdefault("marginals", {}).setdefault(str(event.get("variable")),
                                                                 {"rows": [], "done": False})
        marginal["rows"].extend(event.get("rows") or [])
        marginal.update({key: value for key, value in event.items()
                         if key not in ("kind", "variable", "rows")})
    elif event.get("kind") == "grid":
        header = event.get("header")
        if isinstance(header, dict):
            stream["grid"] = {**header, "cells": [[None] * len(header.get("x_values") or [])
                                                  for _ in header.get("y_values") or []]}
        grid = stream.get("grid")
        if grid is None:
            return
        for r, c, cell in event.get("cells") or []:
            if 0 <= r < len(grid["cells"]) and 0 <= c < len(grid["cells"][r]):
                grid["cells"][r][c] = cell


@contextlib.contextmanager
def _importable_main() -> Iterator[None]:
    """``spawn`` re-imports the parent's main module in the child. If the server was started
    from a source that cannot be re-imported (stdin, a notebook), hide it for the moment the
    child starts; the worker does not need it."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if path is None or os.path.isfile(path):
        yield
        return
    assert main is not None  # a missing main module has no path, handled above
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
    started: float = field(default_factory=time.monotonic)
    finished: Optional[float] = None
    state: str = "starting"
    log: str = ""
    log_dropped: int = 0
    progress: Dict[str, Any] = field(default_factory=dict)
    stream: Dict[str, Any] = field(default_factory=dict)
    """The coefficients streamed so far (:func:`merge_stream`); while the job runs, the GUI draws them."""
    result: Optional[Dict[str, Any]] = None
    touched: float = field(default_factory=time.monotonic)
    """When a client last asked about the job."""
    grid_request: Optional[Dict[str, Any]] = None
    """The request of a grid job, whose grid goes to the :class:`GridCache` when it is done."""

    @property
    def auxiliary(self) -> bool:
        return self.summary.get("kind") in AUXILIARY_KINDS

    def listing(self) -> Dict[str, Any]:
        """What the GUI shows about a running job: id, kind, a description and the time so far."""
        return {"id": self.id, "kind": self.summary.get("kind"), "description": describe_job(self.summary),
                "elapsed": round(self.elapsed, 1), "auxiliary": self.auxiliary}

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
            # the final result replaces the streamed coefficients
            "stream": (self.stream or None) if self.running else None,
            "result": self.result,
            "summary": self.summary,
        }


class JobManager:
    """Starts, watches and stops analysis processes. Analyses and checks (started by the user) count against
    ``max_jobs``; the background jobs of the GUI (:data:`AUXILIARY_KINDS`) against ``max_auxiliary``."""

    def __init__(self, *, max_jobs: int = 2, time_limit: float = 120.0, memory_limit: Optional[int] = None,
                 keep_finished: int = 32, max_auxiliary: int = 2):
        self.max_jobs = max_jobs
        self.max_auxiliary = max_auxiliary
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.keep_finished = keep_finished
        self._context = multiprocessing.get_context("spawn")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.grids = GridCache()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="prodigy-jobs", daemon=True)
        self._watcher.start()

    # -- public ------------------------------------------------------------------------

    def submit(self, request: Dict[str, Any], time_limit: Optional[float] = None, kind: str = "analysis") -> Job:
        limit = self.time_limit if time_limit is None else min(float(time_limit), self.time_limit)
        grid_request = request if kind == "joint" else None
        if kind == "joint":
            # a window shown before needs no process; another one gets the cells computed before (#469)
            grid = self.grids.grid(request)
            if grid is not None:
                return self._finished(kind, request, limit, {"status": "ok", "joint": grid})
            request = {**request, "known_cells": self.grids.known_cells(request),
                       "known_header": self.grids.header(request)}
        with self._lock:
            self._make_room(kind)
            receiver, sender = self._context.Pipe(duplex=False)
            process = self._context.Process(target=_worker, args=(request, sender, self.memory_limit, kind),
                                            name="prodigy-analysis", daemon=True)
            with _importable_main():
                process.start()
            sender.close()
            job = Job(id=uuid.uuid4().hex[:12], process=process, conn=receiver, time_limit=limit,
                      summary={"kind": kind, "engine": request.get("engine"), "file_name": request.get("file_name"),
                               "decisions": request.get("decisions")},
                      grid_request=grid_request)
            self._jobs[job.id] = job
            self._prune()
            return job

    def _finished(self, kind: str, request: Dict[str, Any], limit: float, result: Dict[str, Any]) -> Job:
        """A job that is done when it is created (a grid from the cache)."""
        now = time.monotonic()
        job = Job(id=uuid.uuid4().hex[:12], process=None, conn=None, time_limit=limit,
                  summary={"kind": kind, "engine": request.get("engine"), "file_name": None, "decisions": None,
                           "cached": True},
                  state="done", result=result, started=now, finished=now)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.touched = time.monotonic()
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

    def running(self, auxiliary: Optional[bool] = None) -> List[Dict[str, Any]]:
        """The running jobs (oldest first, :meth:`Job.listing`); ``auxiliary`` selects one group."""
        with self._lock:
            return [job.listing() for job in self._jobs.values()
                    if job.running and (auxiliary is None or job.auxiliary == auxiliary)]

    # -- internals ---------------------------------------------------------------------

    def _make_room(self, kind: str) -> None:
        """Raises :class:`BusyError` unless a job of ``kind`` may start; a new job of a preemptible kind
        replaces the oldest one that is running."""
        auxiliary = kind in AUXILIARY_KINDS
        group = [job for job in self._jobs.values() if job.running and job.auxiliary == auxiliary]
        limit = self.max_auxiliary if auxiliary else self.max_jobs
        while len(group) >= limit:
            same = [job for job in group if job.summary.get("kind") == kind]
            if kind not in PREEMPTIBLE_KINDS or not same:
                listing = [job.listing() for job in group]
                raise BusyError(f"{describe_running(listing)} (at most {limit} at a time)", listing)
            self._stop_process(same[0], "cancelled")
            group.remove(same[0])

    def _watch(self) -> None:
        while not self._stop.wait(0.2):
            with self._lock:
                for job in list(self._jobs.values()):
                    if not job.running:
                        continue
                    self._drain(job)
                    if job.running and job.elapsed > job.time_limit:
                        self._stop_process(job, "timeout")
                    elif job.running and job.auxiliary and time.monotonic() - job.touched > STALE_SECONDS:
                        self._stop_process(job, "cancelled")  # nobody waits for it any more

    def _drain(self, job: Job) -> None:
        if job.conn is None:
            return
        try:
            while job.conn.poll():
                kind, payload = job.conn.recv()
                if kind == "log":
                    self._append_log(job, payload)
                elif kind == "progress" and payload.get("kind") in STREAM_KINDS:
                    merge_stream(job.stream, payload)
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
                    if job.grid_request is not None and isinstance(payload, dict) and payload.get("status") == "ok":
                        self.grids.store(job.grid_request, payload["joint"])
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
        what = {"suggest": "The request to the language model", "joint": "The computation of the grid",
                "equivalence": "The check"}.get(
            str(job.summary.get("kind")), "The analysis")
        message = (f"{what} was stopped after the time limit of {job.time_limit:g} s." if state == "timeout"
                   else f"{what} was cancelled.")
        job.result = {"status": state, "error": {"type": state, "message": message}}
        job.finished = time.monotonic()
        try:
            job.conn.close()
        except Exception:  # pylint: disable=broad-except
            pass
        job.conn = None

    def _prune(self) -> None:
        # grids go first: the analyses they are computed from must stay available
        finished = sorted((job_id for job_id, job in self._jobs.items() if not job.running),
                          key=lambda job_id: self._jobs[job_id].summary.get("kind") != "joint")
        for job_id in finished[:-self.keep_finished] if len(finished) > self.keep_finished else []:
            del self._jobs[job_id]
