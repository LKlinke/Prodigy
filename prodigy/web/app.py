"""
The Prodigy web GUI (``prodigy web``).

A small Flask application: one page (``templates/app.html`` with the ES modules in
``static/app``) and a JSON API:

================================  ==================================================================
``GET  /api/meta``                version, engines, loop strategies with their parameters
``POST /api/check``               diagnostics (syntax/type errors with positions) and outline;
                                  with ``input_distribution`` also whether that initial
                                  distribution is accepted
``POST /api/loop-program``        a loop of a program as a program of its own (``loop``: its id), for
                                  "Compare this loop" (:func:`prodigy.web.compare.loop_program`)
``POST /api/analyze``             starts an analysis in its own process, returns a job id
``GET /``, ``/analyze``, …        the page; ``/compare`` opens it in Compare mode (#154)
``POST /api/equivalence``         starts a job that compares two programs (``other_source``, see
                                  :func:`prodigy.web.compare.run_equivalence`), returns a job id
``GET  /api/jobs/<id>``           state, new console output, progress, result when done
``GET  /api/jobs``                 the running jobs (analyses and checks, background jobs)
``DELETE /api/jobs/<id>``         cancels the analysis
``POST /api/jobs/<id>/joint``     starts a job for the exact coefficient grid of two variables of a
                                  finished analysis (``x``, ``y``, optional windows), see
                                  :mod:`prodigy.analysis.joint_grid`
``POST /api/suggest``             asks a language model for a loop invariant candidate (a job)
``GET  /api/examples``            the example library (``pgfexamples``)
``GET  /api/examples/<id>``       source of an example and a suggested setup
``GET  /assets/<token>/<path>``   a static file under a token of the version and the files, so that
                                  an upgrade loads every ES module anew (:mod:`prodigy.web.assets`)
================================  ==================================================================

Programs are in the ``.ppy`` syntax. A request with ``syntax`` ``pgcl`` or a ``file_name`` ending in
``.pgcl`` is rejected with :data:`prodigy.language.parser.PGCL_REMOVED` (the syntax was removed in
v0.6.0).

The server binds to localhost by default. Requests to the API must be JSON and come from
the page itself (Host and Origin are checked), which keeps other web sites from using a
local server. Analyses parse expressions with SymPy, which evaluates code; do not expose
the server publicly without sandboxing (see ``docs/gui.md``).
"""
from __future__ import annotations

import importlib.util
import os
import threading
import webbrowser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, make_response, render_template, request

from prodigy.language.parser import PgclRemovedError, check_syntax
from prodigy.web.assets import register_assets
from prodigy.web.examples import ExampleLibrary, default_examples_dir
from prodigy.web.jobs import BusyError, JobManager
from prodigy.web.compare import loop_program, validate_equivalence
from prodigy.web.requests import (ENGINES, MAX_COEFFICIENTS, ai_status, check_input_distribution_bounded,
                                  joint_request, check_request_syntax, validate_request, validate_suggestion)

# Longest initial distribution that /api/check examines (longer ones are checked by the analysis).
MAX_INPUT_DISTRIBUTION = 500
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _engine_info(default_engine: str) -> List[Dict[str, Any]]:
    engines = []
    for name, label in ENGINES.items():
        available, reason = True, None
        # whether the bindings are installed, without loading them into the server (only the engine imports them)
        if name == "ginac" and importlib.util.find_spec("prodigy_ginac") is None:
            available = False
            reason = "Not installed: install prodigy-ppl-ginac (the extra ginac) to use GiNaC."
        engines.append({"name": name, "label": label, "available": available, "reason": reason,
                        "default": name == default_engine})
    return engines


def _permitted_hosts(host: str, allowed_hosts: Sequence[str]) -> Optional[Set[str]]:
    """The host names under which the API answers; None (any) if bound to a public address without
    ``--allowed-host``."""
    if host in _LOOPBACK or allowed_hosts:
        permitted_hosts = set(_LOOPBACK) | {h.lower() for h in allowed_hosts}
        if host not in ("0.0.0.0", "::"):
            permitted_hosts.add(host.lower())
        return permitted_hosts
    return None


def _guard(app: Flask, permitted_hosts: Optional[Set[str]]) -> None:
    """Requests to the API come from the page itself (Host and Origin are checked) and are JSON; its errors
    are JSON as well."""

    @app.before_request
    def _guard_api() -> None:
        if not request.path.startswith("/api/"):
            return None
        if permitted_hosts is not None:
            hostname = request.host.rsplit(":", 1)[0] if not request.host.startswith("[") \
                else request.host.split("]")[0] + "]"
            if hostname.lower() not in permitted_hosts:
                abort(403, description="Unknown host.")
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            abort(403, description="Cross-origin requests are not allowed.")
        if request.method == "POST" and not request.is_json:
            abort(415, description="Send JSON.")
        return None

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(415)
    def _error(error: Any) -> Any:
        if request.path.startswith("/api/"):
            return jsonify({"error": getattr(error, "description", str(error))}), error.code
        return error


def _page_routes(app: Flask, jobs: JobManager, library: ExampleLibrary, default_engine: str) -> None:
    """The page and the server's metadata."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.loop_strategies import loop_strategies
    from prodigy.analysis.result import prodigy_version
    from prodigy.analysis.tools import (MAX_AXIS, SELECTABLE_POSITIVITY_HEURISTICS, SynthesisStrategies,
                                        TemplateHeuristics)

    version = prodigy_version()
    strategies_json = [s.to_json() for s in loop_strategies()]
    # scripts and styles under /assets/<token>/: a new version or file gives every module a new URL
    register_assets(app, version)

    @app.get("/")
    @app.get("/analyze")
    @app.get("/compare")
    def index() -> Any:
        """The page; the path selects the mode (Analyze or Compare), which the page keeps in the URL."""
        response = make_response(render_template("app.html", version=version))
        # revalidated on every load, so that a reload always names the current asset token
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/meta")
    def meta() -> Any:
        return jsonify({
            "version": version,
            "engines": _engine_info(default_engine),
            "strategies": strategies_json,
            "synthesis_strategies": [s.name.lower() for s in SynthesisStrategies],
            "template_heuristics": [name.lower() for name in TemplateHeuristics.__members__],
            "positivity_heuristics": list(SELECTABLE_POSITIVITY_HEURISTICS),
            "limits": {"time_limit": jobs.limits.time_limit, "max_jobs": jobs.limits.max_jobs,
                       "max_background_jobs": jobs.limits.max_auxiliary,
                       "max_coefficients": MAX_COEFFICIENTS, "max_joint_axis": MAX_AXIS},
            "examples": library.root is not None,
            "ai": ai_status(),
        })


def _program_routes(app: Flask, library: ExampleLibrary) -> None:
    """The checks of a program while it is edited, a loop as a program, and the examples."""
    from prodigy.web.diagnostics import check_source  # pylint: disable=import-outside-toplevel

    @app.post("/api/check")
    def check() -> Any:
        payload = request.get_json(silent=True) or {}
        source = payload.get("source")
        if not isinstance(source, str):
            abort(400, description="'source' is missing.")
        try:
            check_request_syntax(payload)
        except PgclRemovedError as exc:
            # shown in the editor like any other error of the program
            return jsonify({"ok": False, "outline": None, "syntax": "pgcl", "diagnostics": [
                {"severity": "error", "line": 1, "column": 1, "end_line": 1, "end_column": 1,
                 "message": str(exc)}]})
        except ValueError as exc:
            abort(400, description=str(exc))
        result = {**check_source(source), "syntax": "ppy"}
        text = payload.get("input_distribution")
        if isinstance(text, str) and text.strip() and len(text) <= MAX_INPUT_DISTRIBUTION:
            engine = payload.get("engine")
            result["input_distribution"] = check_input_distribution_bounded(
                source, text, engine if isinstance(engine, str) else "sympy")
        return jsonify(result)

    @app.post("/api/loop-program")
    def isolate() -> Any:
        """A loop as a program of its own, for Compare mode (#399)."""
        payload = request.get_json(silent=True) or {}
        source, loop = payload.get("source"), payload.get("loop")
        if not isinstance(source, str) or not isinstance(loop, int) or isinstance(loop, bool):
            abort(400, description="'source' and the loop id 'loop' are needed.")
        try:
            check_request_syntax(payload)
            return jsonify(loop_program(source, loop))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

    @app.get("/api/examples")
    def examples() -> Any:
        return jsonify({"examples": library.list()})

    @app.get("/api/examples/<path:example_id>")
    def example(example_id: str) -> Any:
        try:
            check_syntax(request.args.get("syntax"))
            return jsonify(library.load(example_id))
        except (FileNotFoundError, OSError):
            abort(404, description="Unknown example.")
        except ValueError as exc:
            abort(400, description=str(exc))


_ANALYSIS_KEYS = ("source", "syntax", "file_name", "input_distribution", "engine", "normalize", "simplification",
                  "decisions", "coefficients", "record_steps", "watches", "step_iterations", "synthesis_strategy",
                  "template_heuristic", "positivity_heuristic", "solver")
"""The fields of ``POST /api/analyze`` that go to the job."""
_EQUIVALENCE_KEYS = ("source", "syntax", "file_name", "engine", "normalize", "simplification", "decisions",
                     "synthesis_strategy", "template_heuristic", "positivity_heuristic", "solver",
                     "other_source", "other_syntax", "other_decisions")
"""The fields of ``POST /api/equivalence`` that go to the job."""
_SUGGESTION_KEYS = ("source", "syntax", "file_name", "input_distribution", "engine", "normalize", "simplification",
                    "decisions", "synthesis_strategy", "template_heuristic", "positivity_heuristic", "solver", "loop",
                    "form")
"""The fields of ``POST /api/suggest`` that go to the job."""


def _submit(jobs: JobManager, job_request: Dict[str, Any], time_limit: Any = None, kind: str = "analysis") -> Any:
    """202 with the id and time limit of the new job, or 429 with what is running (``running``, to cancel with
    ``DELETE /api/jobs/<id>``)."""
    try:
        job = jobs.submit(job_request, time_limit=time_limit, kind=kind)
    except BusyError as exc:
        return jsonify({"error": f"{exc}. Wait for one to finish or cancel it.", "running": exc.running}), 429
    return jsonify({"job": job.id, "time_limit": job.time_limit}), 202


def _start_routes(app: Flask, jobs: JobManager) -> None:
    """The requests that start a job: an analysis, a comparison, a suggestion of a language model."""

    @app.post("/api/analyze")
    def analyze() -> Any:
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        problems = validate_request(payload)
        if problems:
            return jsonify({"error": " ".join(problems), "problems": problems}), 400
        return _submit(jobs, {key: payload.get(key) for key in _ANALYSIS_KEYS if key in payload},
                       payload.get("time_limit"))

    @app.post("/api/equivalence")
    def equivalence() -> Any:
        """Starts a job that checks whether two programs are equivalent for every input: the program, its
        loop decisions and options, plus ``other_source``, ``other_syntax`` and ``other_decisions`` for the
        second program. There is no initial distribution: the check covers every input."""
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        problems = validate_equivalence(payload)
        if problems:
            return jsonify({"error": " ".join(problems), "problems": problems}), 400
        equivalence_request = {key: payload.get(key) for key in _EQUIVALENCE_KEYS if key in payload}
        # the witness search gets a part of the job's time (compare.WITNESS_SECONDS)
        equivalence_request["time_limit"] = jobs.limits.time_limit if payload.get("time_limit") is None \
            else min(float(payload["time_limit"]), jobs.limits.time_limit)
        return _submit(jobs, equivalence_request, payload.get("time_limit"), kind="equivalence")

    @app.post("/api/suggest")
    def suggest() -> Any:
        """Starts a job that asks the language model for one candidate for a loop; the API key
        stays in the job process (see :func:`prodigy.web.requests.run_suggestion`)."""
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        problems = validate_suggestion(payload)
        if problems:
            return jsonify({"error": " ".join(problems), "problems": problems}), 400
        status = ai_status()
        if not status["available"]:
            return jsonify({"error": status["reason"]}), 503
        return _submit(jobs, {key: payload.get(key) for key in _SUGGESTION_KEYS if key in payload},
                       payload.get("time_limit"), kind="suggest")


def _job_routes(app: Flask, jobs: JobManager) -> None:
    """The requests about jobs: list, state, cancel, and the grid job of a finished analysis."""

    @app.get("/api/jobs")
    def job_list() -> Any:
        """The running jobs: ``jobs`` (analyses and checks) and ``background`` (grids, suggestions)."""
        return jsonify({"jobs": jobs.running(auxiliary=False), "background": jobs.running(auxiliary=True)})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        job = jobs.get(job_id)
        if job is None:
            abort(404, description="Unknown analysis.")
        try:
            offset = int(request.args.get("log_offset", 0))
        except ValueError:
            offset = 0
        return jsonify(job.to_json(offset))

    @app.delete("/api/jobs/<job_id>")
    def job_cancel(job_id: str) -> Any:
        job = jobs.cancel(job_id)
        if job is None:
            abort(404, description="Unknown analysis.")
        return jsonify(job.to_json(0))

    @app.post("/api/jobs/<job_id>/joint")
    def job_joint(job_id: str) -> Any:
        """Starts a job for the coefficient grid of two variables of a finished analysis; the
        distribution comes from the result the server keeps, never from the client."""
        job = jobs.get(job_id)
        if job is None or job.summary.get("kind") != "analysis":
            abort(404, description="Unknown analysis; analyze the program again.")
        if job.running or job.result is None:
            return jsonify({"error": "The analysis has not finished yet."}), 409
        payload = request.get_json(silent=True) or {}
        try:
            grid_request = joint_request(job.result, payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return _submit(jobs, grid_request, kind="joint")


# keyword-only, one parameter per option of `prodigy web` (serve passes them on)
def create_app(*, time_limit: float = 120.0, max_jobs: int = 2,  # pylint: disable=too-many-arguments
               max_background_jobs: int = 2, memory_limit: Optional[int] = None,
               examples_dir: Optional[str] = None, default_engine: str = "sympy",
               host: str = "127.0.0.1", allowed_hosts: Sequence[str] = ()) -> Flask:
    """The Flask application of the GUI with its job manager (``app.extensions["prodigy_jobs"]``) and example
    library (``app.extensions["prodigy_examples"]``)."""
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    provider = getattr(app, "json", None)
    if provider is not None and hasattr(provider, "sort_keys"):
        provider.sort_keys = False  # Flask >= 2.2: keep the result's key order
    else:
        app.config["JSON_SORT_KEYS"] = False

    jobs = JobManager(max_jobs=max_jobs, max_auxiliary=max_background_jobs, time_limit=time_limit,
                      memory_limit=memory_limit)
    library = ExampleLibrary(examples_dir or default_examples_dir())
    app.extensions["prodigy_jobs"] = jobs
    app.extensions["prodigy_examples"] = library

    _guard(app, _permitted_hosts(host, allowed_hosts))
    _page_routes(app, jobs, library, default_engine)
    _program_routes(app, library)
    _start_routes(app, jobs)
    _job_routes(app, jobs)
    return app


# keyword-only, one parameter per option of `prodigy web`
def serve(*, host: str = "127.0.0.1", port: int = 8080,  # pylint: disable=too-many-arguments
          open_browser: bool = True, time_limit: float = 120.0,
          max_jobs: int = 2, max_background_jobs: int = 2, memory_limit: Optional[int] = None,
          examples_dir: Optional[str] = None,
          default_engine: str = "sympy", allowed_hosts: Iterable[str] = ()) -> None:
    """Runs the GUI server until interrupted."""
    app = create_app(time_limit=time_limit, max_jobs=max_jobs, max_background_jobs=max_background_jobs,
                     memory_limit=memory_limit, examples_dir=examples_dir, default_engine=default_engine, host=host,
                     allowed_hosts=tuple(allowed_hosts))
    shown_host = "localhost" if host in _LOOPBACK else host
    url = f"http://{shown_host}:{port}/"
    if open_browser and not os.environ.get("PRODIGY_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Prodigy GUI: {url}  (stop with Ctrl+C)")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)
    finally:
        app.extensions["prodigy_jobs"].shutdown()
