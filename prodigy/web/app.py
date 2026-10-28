"""
The Prodigy web GUI (``prodigy web``).

A small Flask application: one page (``templates/app.html`` with the ES modules in
``static/app``) and a JSON API:

==========================  ==========================================================
``GET  /api/meta``          version, engines, loop strategies with their parameters
``POST /api/check``         diagnostics (syntax/type errors with positions) and outline
``POST /api/analyze``       starts an analysis in its own process, returns a job id
``GET  /api/jobs/<id>``     state, new console output, progress, result when done
``DELETE /api/jobs/<id>``   cancels the analysis
``GET  /api/examples``      the example library (``pgfexamples``)
``GET  /api/examples/<id>`` source of an example and a suggested setup
==========================  ==========================================================

The server binds to localhost by default. Requests to the API must be JSON and come from
the page itself (Host and Origin are checked), which keeps other web sites from using a
local server. Analyses parse expressions with SymPy, which evaluates code; do not expose
the server publicly without sandboxing (see ``docs/gui.md``).
"""
from __future__ import annotations

import os
import threading
import webbrowser
from typing import Any, Dict, Iterable, Optional, Sequence
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, render_template, request

from prodigy.web.examples import ExampleLibrary, default_examples_dir
from prodigy.web.jobs import BusyError, JobManager
from prodigy.web.requests import ENGINES, MAX_COEFFICIENTS, validate_request

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _engine_info(default_engine: str):
    engines = []
    for name, label in ENGINES.items():
        available = True
        if name == "ginac":
            try:
                import pygin  # type: ignore  # pylint: disable=import-outside-toplevel,unused-import
            except ImportError:
                available = False
        engines.append({"name": name, "label": label, "available": available,
                        "default": name == default_engine})
    return engines


def create_app(*, time_limit: float = 120.0, max_jobs: int = 2, memory_limit: Optional[int] = None,
               examples_dir: Optional[str] = None, default_engine: str = "sympy",
               host: str = "127.0.0.1", allowed_hosts: Sequence[str] = ()) -> Flask:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
    from prodigy.analysis.loop_strategies import loop_strategies
    from prodigy.analysis.result import prodigy_version
    from prodigy.web.diagnostics import check_source

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    app.json_sort_keys = False  # type: ignore[attr-defined]

    jobs = JobManager(max_jobs=max_jobs, time_limit=time_limit, memory_limit=memory_limit)
    library = ExampleLibrary(examples_dir or default_examples_dir())
    app.extensions["prodigy_jobs"] = jobs
    app.extensions["prodigy_examples"] = library

    if host in _LOOPBACK or allowed_hosts:
        permitted_hosts = set(_LOOPBACK) | {h.lower() for h in allowed_hosts}
        if host not in ("0.0.0.0", "::"):
            permitted_hosts.add(host.lower())
    else:
        permitted_hosts = None  # bound to a public address without --allowed-host

    version = prodigy_version()
    strategies_json = [s.to_json() for s in loop_strategies()]

    @app.before_request
    def _guard():
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
    def _error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": getattr(error, "description", str(error))}), error.code
        return error

    @app.get("/")
    def index():
        return render_template("app.html", version=version)

    @app.get("/api/meta")
    def meta():
        return jsonify({
            "version": version,
            "engines": _engine_info(default_engine),
            "strategies": strategies_json,
            "synthesis_strategies": [s.name.lower() for s in SynthesisStrategies],
            "solvers": ["sympy", "z3"],
            "limits": {"time_limit": jobs.time_limit, "max_jobs": jobs.max_jobs,
                       "max_coefficients": MAX_COEFFICIENTS},
            "examples": library.root is not None,
        })

    @app.post("/api/check")
    def check():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source")
        if not isinstance(source, str):
            abort(400, description="'source' is missing.")
        return jsonify(check_source(source))

    @app.post("/api/analyze")
    def analyze():
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        problems = validate_request(payload)
        if problems:
            return jsonify({"error": " ".join(problems), "problems": problems}), 400
        analysis_request = {key: payload.get(key) for key in (
            "source", "input_distribution", "engine", "normalize", "simplification", "decisions",
            "coefficients", "record_steps", "synthesis_strategy", "solver") if key in payload}
        try:
            job = jobs.submit(analysis_request, time_limit=payload.get("time_limit"))
        except BusyError as exc:
            return jsonify({"error": f"{exc}. Wait for one to finish or cancel it."}), 429
        return jsonify({"job": job.id, "time_limit": job.time_limit}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            abort(404, description="Unknown analysis.")
        try:
            offset = int(request.args.get("log_offset", 0))
        except ValueError:
            offset = 0
        return jsonify(job.to_json(offset))

    @app.delete("/api/jobs/<job_id>")
    def job_cancel(job_id: str):
        job = jobs.cancel(job_id)
        if job is None:
            abort(404, description="Unknown analysis.")
        return jsonify(job.to_json(0))

    @app.get("/api/examples")
    def examples():
        return jsonify({"examples": library.list()})

    @app.get("/api/examples/<path:example_id>")
    def example(example_id: str):
        try:
            return jsonify(library.load(example_id))
        except (FileNotFoundError, OSError):
            abort(404, description="Unknown example.")

    return app


def serve(*, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True, time_limit: float = 120.0,
          max_jobs: int = 2, memory_limit: Optional[int] = None, examples_dir: Optional[str] = None,
          default_engine: str = "sympy", allowed_hosts: Iterable[str] = ()) -> None:
    """Runs the GUI server until interrupted."""
    app = create_app(time_limit=time_limit, max_jobs=max_jobs, memory_limit=memory_limit,
                     examples_dir=examples_dir, default_engine=default_engine, host=host,
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
