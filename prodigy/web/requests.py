"""
Analysis requests of the web GUI (JSON in, JSON out).

A request looks like::

    {
      "source": "nat x; ...",
      "syntax": "pgcl" | "ppy",              # optional, else from "file_name", else pgcl
      "file_name": "dice.ppy",               # optional, only its extension is used
      "input_distribution": "x^3",          # optional, default: all variables 0
      "engine": "sympy" | "ginac" | "symengine",
      "normalize": true,                     # condition on observations
      "simplification": true,
      "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 50}}, "*": ...},
      "coefficients": 30,                    # coefficients per marginal in the result
      "record_steps": false,                 # record the step trace (prodigy.analysis.trace)
      "watches": ["Pr[x == 3]", "Ex[x]"],    # with record_steps: queries after every top-level step
      "step_iterations": 40,                 # with record_steps: iterates kept per loop
      "synthesis_strategy": "default",       # for EVT invariant synthesis
      "template_heuristic": "default",       # only with synthesis_strategy "from_heuristics"
      "positivity_heuristic": "default",     # only with synthesis_strategy "from_heuristics"
      "solver": "sympy" | "z3"
    }

Generating functions (``input_distribution`` and the ``expression`` parameters of the loop
strategies) are untrusted text for SymPy's parser: :func:`validate_request` admits only
expressions in the program's syntax (:func:`prodigy.analysis.ai_candidates.admissible_expression`),
and :func:`prepare_request` turns them into the engines' spelling (``x**2`` -> ``x^2``).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

ENGINES = {
    "sympy": "SymPy",
    "ginac": "GiNaC",
    "symengine": "SymEngine",
}

MAX_COEFFICIENTS = 200
MAX_WATCHES = 8
MAX_STEP_ITERATIONS = 200


def request_syntax(request: Dict[str, Any]) -> str:
    """The syntax of the request's program: ``syntax``, else the extension of ``file_name``, else
    pGCL. Raises ``ValueError`` for an unknown syntax."""
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.parser import select_syntax
    syntax = request.get("syntax")
    file_name = request.get("file_name")
    return select_syntax(syntax if isinstance(syntax, str) else None,
                         file_name if isinstance(file_name, str) else None)


def _expressions(request: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    """(label, container, key) of every generating function in the request:
    ``container[key]`` is a non-empty string."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.loop_strategies import get_loop_strategy
    found: List[Tuple[str, Dict[str, Any], str]] = []
    if isinstance(request.get("input_distribution"), str) and request["input_distribution"].strip():
        found.append(("Initial distribution", request, "input_distribution"))
    decisions = request.get("decisions")
    for key, decision in (decisions.items() if isinstance(decisions, dict) else ()):
        if not isinstance(decision, dict) or not isinstance(decision.get("params"), dict):
            continue
        try:
            strategy = get_loop_strategy(str(decision.get("strategy")))
        except KeyError:
            continue
        params = decision["params"]
        for param in strategy.params:
            value = params.get(param.name)
            if param.kind == "expression" and isinstance(value, str) and value.strip():
                where = "default decision" if str(key) == "*" else f"loop {key}"
                found.append((f"{where}: {param.label}", params, param.name))
    return found


def prepare_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a valid request with its generating functions in the engines' spelling
    (``x**2`` -> ``x^2``, which GiNaC needs)."""
    prepared = copy.deepcopy(request)
    for _, container, key in _expressions(prepared):
        container[key] = container[key].replace("**", "^")
    return prepared


def validate_request(request: Dict[str, Any]) -> List[str]:
    """Problems that can be found without running the analysis."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.evtinvariants.heuristics.strategies import SELECTABLE_POSITIVITY_HEURISTICS
    from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
    from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
    from prodigy.analysis.ai_candidates import MATH_FUNCTIONS, CandidateError, admissible_expression
    from prodigy.analysis.session import parse_decisions, validate_decisions
    from prodigy.analysis.solver.solver_type import SolverType

    problems = []
    if not isinstance(request.get("source"), str) or not request["source"].strip():
        problems.append("The program is empty.")
    try:
        syntax = request_syntax(request)
    except ValueError as exc:
        problems.append(str(exc))
        syntax = "pgcl"
    for label, container, key in _expressions(request):
        if container is not request:
            continue  # the generating functions of the decisions are checked by validate_decisions
        try:
            admissible_expression(container[key], syntax=syntax, functions=MATH_FUNCTIONS)
        except CandidateError as exc:
            problems.append(f"{label}: {exc}")
    if request.get("engine", "sympy") not in ENGINES:
        problems.append(f"Unknown engine {request.get('engine')!r}.")
    try:
        coefficients = int(request.get("coefficients", 30))
        if not 0 <= coefficients <= MAX_COEFFICIENTS:
            problems.append(f"Coefficients must be between 0 and {MAX_COEFFICIENTS}.")
    except (TypeError, ValueError):
        problems.append("Coefficients must be a number.")
    if str(request.get("synthesis_strategy", "default")).upper() not in SynthesisStrategies.__members__:
        problems.append(f"Unknown synthesis strategy {request.get('synthesis_strategy')!r}.")
    if str(request.get("template_heuristic", "default")).upper() not in TemplateHeuristics.__members__:
        problems.append(f"Unknown template heuristic {request.get('template_heuristic')!r}.")
    if str(request.get("positivity_heuristic", "default")).lower() not in SELECTABLE_POSITIVITY_HEURISTICS:
        problems.append(f"Unknown positivity heuristic {request.get('positivity_heuristic')!r}.")
    if str(request.get("solver", "sympy")).upper() not in SolverType.__members__:
        problems.append(f"Unknown solver {request.get('solver')!r}.")
    try:
        decisions = parse_decisions(request.get("decisions") or {})
        problems.extend(validate_decisions(decisions, syntax=syntax))
    except ValueError as exc:
        problems.append(str(exc))
    problems.extend(_trace_problems(request, syntax))
    return problems


def _trace_problems(request: Dict[str, Any], syntax: str) -> List[str]:
    """Problems with ``watches`` and ``step_iterations``; a watch is parsed by the program's
    expression parser (:func:`prodigy.analysis.trace.parse_watch`), never by SymPy."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.trace import parse_watch
    problems = []
    watches = request.get("watches")
    if watches is not None:
        if not isinstance(watches, list) or not all(isinstance(w, str) for w in watches):
            problems.append("'watches' must be a list of queries such as \"Pr[x == 3]\".")
        elif len(watches) > MAX_WATCHES:
            problems.append(f"At most {MAX_WATCHES} watches.")
        else:
            for watch in watches:
                try:
                    parse_watch(watch, syntax)
                except ValueError as exc:
                    problems.append(f"Watch: {exc}")
    iterations = request.get("step_iterations")
    if iterations is not None and (isinstance(iterations, bool) or not isinstance(iterations, int)
                                   or not 1 <= iterations <= MAX_STEP_ITERATIONS):
        problems.append(f"'step_iterations' must be a whole number between 1 and {MAX_STEP_ITERATIONS}.")
    return problems


def make_config(request: Dict[str, Any]):
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
    from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
    from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
    from prodigy.analysis.solver.solver_type import SolverType
    return ForwardAnalysisConfig(
        engine=ForwardAnalysisConfig.Engine[str(request.get("engine", "sympy")).upper()],
        use_simplification=bool(request.get("simplification", True)),
        normalize=bool(request.get("normalize", True)),
        strategy=SynthesisStrategies[str(request.get("synthesis_strategy", "default")).upper()],
        templ_heuristic=TemplateHeuristics[str(request.get("template_heuristic", "default")).upper()],
        positivity_heuristic=PositivityHeuristics[str(request.get("positivity_heuristic", "default")).upper()],
        solver_type=SolverType[str(request.get("solver", "sympy")).upper()],
    )


def check_input_distribution(source: str, syntax: str, text: str, engine: str = "sympy") -> Dict[str, Any]:
    """Whether the initial distribution ``text`` is accepted for the program ``source`` (the check
    :func:`run_analysis` makes after the job started, #307): ``{"status": "ok"}``,
    ``{"status": "refused", "message": ..., "proof": ...}`` or ``{"status": "skipped"}`` if the
    program does not parse (its diagnostics say why) or the engine is not available."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.input_distribution import InputDistributionError, input_distribution_from_text
    from prodigy.pgcl import parse_pgcl
    try:
        program = parse_pgcl(source, syntax)
        config = make_config({"engine": engine})
        with config.activate():
            input_distribution_from_text(config, text.strip().replace("**", "^"), program, syntax)
    except InputDistributionError as exc:
        return {"status": "refused", "message": str(exc), "proof": exc.details.get("proof")}
    except Exception:  # pylint: disable=broad-except
        return {"status": "skipped"}
    return {"status": "ok"}


INPUT_CHECK_SECONDS = 8.0
"""How long /api/check waits for the check of an initial distribution before it gives up."""
_INPUT_CHECK_CONTEXT = "spawn"  # tests use "fork", which carries their fakes into the process


def _input_check_worker(connection, source: str, syntax: str, text: str, engine: str) -> None:
    try:
        connection.send(check_input_distribution(source, syntax, text, engine))
    finally:
        connection.close()


def check_input_distribution_bounded(source: str, syntax: str, text: str, engine: str = "sympy",
                                     timeout: Optional[float] = None) -> Dict[str, Any]:
    """:func:`check_input_distribution` in a process of its own that is killed after ``timeout``
    seconds (:data:`INPUT_CHECK_SECONDS`), so that a hard input never blocks the server (the exact
    positivity checks of SymPy cannot be interrupted). Then ``{"status": "timeout"}``: the check is
    made when the analysis runs."""
    # pylint: disable=import-outside-toplevel
    import multiprocessing
    context: Any = multiprocessing.get_context(_INPUT_CHECK_CONTEXT)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_input_check_worker, args=(sender, source, syntax, text, engine), daemon=True)
    process.start()
    sender.close()
    try:
        if receiver.poll(INPUT_CHECK_SECONDS if timeout is None else timeout):
            return receiver.recv()
        return {"status": "timeout"}
    except (EOFError, OSError):
        return {"status": "skipped"}
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(1)


def run_analysis(request: Dict[str, Any], *, output: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Runs one analysis request and returns the JSON result (never raises for analysis errors)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl import parse_pgcl

    from prodigy.analysis.session import AnalysisSession
    from prodigy.analysis.trace import TraceLimits, parse_watch

    problems = validate_request(request)
    if problems:
        return {"status": "error", "error": {"type": "request", "message": " ".join(problems),
                                             "problems": problems}}
    request = prepare_request(request)
    source, syntax = request["source"], request_syntax(request)
    try:
        program = parse_pgcl(source, syntax)
    except Exception as exc:  # pylint: disable=broad-except
        return {"status": "error", "error": {"type": "syntax", "message": str(exc).split("\n", maxsplit=1)[0]}}
    input_distribution = request.get("input_distribution")
    if isinstance(input_distribution, str) and not input_distribution.strip():
        input_distribution = None
    record_steps = bool(request.get("record_steps", False))
    watches = list(request.get("watches") or []) if record_steps else []
    names = list(program.variables) + list(program.parameters)
    for watch in watches:
        try:
            parse_watch(watch, syntax, names)
        except ValueError as exc:
            return {"status": "error", "error": {"type": "request", "message": f"Watch: {exc}",
                                                 "problems": [f"Watch: {exc}"]}}
    session = AnalysisSession(
        program,
        source=source,
        syntax=syntax,
        config=make_config(request),
        decisions=request.get("decisions") or {},
        input_distribution=input_distribution,
        allow_files=False,
        capture_output=True,
        output_callback=output,
        progress_callback=progress,
        record_steps=record_steps,
        coefficient_limit=int(request.get("coefficients", 30)),
        trace_limits=TraceLimits(iterates=int(request.get("step_iterations") or 40)),
        watches=watches,
    )
    return session.run().to_json()


# ---------------------------------------------------------------------------------------
# Coefficient grids of two variables (POST /api/jobs/<id>/joint)
# ---------------------------------------------------------------------------------------

def _index(payload: Dict[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{key}' must be a whole number ≥ 0.")
    return value


def _step_snapshot(result: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The distribution of the step trace at a position of the GUI's timeline (``step``, optionally
    ``iterate`` of that step's loop), or None for the output distribution. Raises ``ValueError``."""
    step_index, iterate = _index(payload, "step"), _index(payload, "iterate")
    if step_index is None:
        if iterate is not None:
            raise ValueError("'iterate' needs 'step'.")
        return None
    steps = result.get("steps") or []
    if step_index >= len(steps):
        raise ValueError("The analysis has no such step; record the steps and analyze again.")
    step = steps[step_index]
    if iterate is not None:
        iterates = (step.get("loop") or {}).get("iterates") or []
        if iterate >= len(iterates):
            raise ValueError("The loop of this step has no such iterate.")
        snapshot = iterates[iterate].get("distribution")
    else:
        snapshot = step.get("after") or step.get("before")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("closed_form"), str):
        raise ValueError("No distribution at this step.")
    if snapshot.get("truncated"):
        raise ValueError("The closed form at this step is too large; the trace keeps only its beginning.")
    return snapshot


def joint_request(result: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """The request of a grid job for two variables of a finished analysis ``result``.

    Only the pair, the windows and the position come from the client (``payload``: ``x``, ``y``,
    optional ``x_window`` and ``y_window`` as ``[from, to]``, optional ``step`` and ``iterate`` for
    a distribution of the step trace instead of the output distribution); the distribution, the
    engine and the guarantee come from the result that the server keeps. Raises ``ValueError``
    with a message for the client."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.joint_grid import check_window
    distribution = result.get("distribution") if isinstance(result, dict) else None
    if result.get("status") != "ok" or not isinstance(distribution, dict):
        raise ValueError("The analysis has no output distribution.")
    variables = list(distribution.get("variables") or [])
    x, y = payload.get("x"), payload.get("y")
    if not isinstance(x, str) or not isinstance(y, str) or x not in variables or y not in variables:
        raise ValueError(f"'x' and 'y' must be variables of the program ({', '.join(variables)}).")
    if x == y:
        raise ValueError("Choose two different variables.")
    snapshot = _step_snapshot(result, payload)
    if snapshot is not None:
        source = {"closed_form": snapshot["closed_form"], "bound": snapshot.get("bound") or "unknown",
                  "residual_mass": snapshot.get("residual_mass")}
    else:
        bounds = result.get("bounds") or {}
        source = {"closed_form": distribution["closed_form"], "bound": bounds.get("kind", "unknown"),
                  "residual_mass": bounds.get("residual_mass")}
    return {
        **source,
        "variables": variables,
        "engine": (result.get("input") or {}).get("engine", "sympy"),
        "x": x,
        "y": y,
        "x_window": check_window(payload.get("x_window"), x),
        "y_window": check_window(payload.get("y_window"), y),
    }


def run_joint(request: Dict[str, Any], *, output: Optional[Callable[[str], None]] = None,
              progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Computes the coefficient grid of a :func:`joint_request` (in a job process)."""
    # pylint: disable=import-outside-toplevel,unused-argument
    from prodigy.analysis.joint_grid import GridError, joint_grid
    from prodigy.util.integers import unlimited_int_digits
    config = make_config({"engine": request["engine"]})
    with config.activate(), unlimited_int_digits():
        try:
            dist = config.factory.from_expr(request["closed_form"], *request["variables"])
            grid = joint_grid(dist, request["x"], request["y"], x_window=request.get("x_window"),
                              y_window=request.get("y_window"), kind=request["bound"],
                              residual=request.get("residual_mass") or 0)
        except GridError as exc:
            return _error("request", str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            return _error(type(exc).__name__, f"The grid could not be computed: {str(exc) or type(exc).__name__}")
    return {"status": "ok", "joint": grid}


# ---------------------------------------------------------------------------------------
# Suggestions of a language model (POST /api/suggest)
# ---------------------------------------------------------------------------------------

def ai_status() -> Dict[str, Any]:
    """Whether a language model is configured, for the GUI; never contains the API key."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import FORMS, AIConfigurationError, get_provider
    try:
        provider = get_provider()
    except (AIConfigurationError, ValueError) as exc:
        return {"available": False, "reason": str(exc), "forms": list(FORMS)}
    return {"available": True, "provider": provider.name, "model": provider.model, "forms": list(FORMS)}


def validate_suggestion(request: Dict[str, Any]) -> List[str]:
    """Problems with a suggestion request: an analysis request plus ``loop`` and ``form``."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import FORMS
    problems = []
    loop = request.get("loop")
    if isinstance(loop, bool) or not isinstance(loop, (int, str)) or not str(loop).isdigit():
        problems.append("'loop' must be the id of a loop.")
    if request.get("form") not in FORMS:
        problems.append(f"'form' must be one of {', '.join(FORMS)}.")
    if problems:
        return problems
    return validate_request(suggestion_analysis_request(request))


def suggestion_analysis_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """The analysis that leads up to the loop: the loop itself gets the strategy ``form``
    without its candidate (it never runs)."""
    decisions = dict(request.get("decisions") or {})
    decisions[str(request["loop"])] = {"strategy": request["form"]}
    return {**request, "decisions": decisions}


class _LoopReached(BaseException):
    """Stops the analysis at the loop to suggest a candidate for. A ``BaseException``, so
    that handlers which turn failures of a strategy into results do not catch it."""

    def __init__(self, task):
        super().__init__("loop reached")
        self.task = task


def _stopping_session(target: int, **kwargs):
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import AnalysisSession

    class StoppingSession(AnalysisSession):
        """Analyses the program up to the first time loop ``target`` is reached outside of
        another strategy's checks."""

        def run_loop(self, task):
            if task.loop is not None and task.loop.id == target and not self.silent:
                raise _LoopReached(task)
            return super().run_loop(task)

    return StoppingSession(**kwargs)


def _error(kind: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "error", "error": {"type": kind, "message": message, **extra}}


def run_suggestion(request: Dict[str, Any], *, output: Optional[Callable[[str], None]] = None,
                   progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Asks the configured language model for one candidate for loop ``request["loop"]`` (an
    EVT invariant or sub-invariant, ``request["form"]``).

    The analysis runs up to the loop to find the distribution entering it. The model's answer is
    untrusted: :func:`~prodigy.analysis.ai_candidates.candidate_from_answer` admits only
    expressions over the program's variables and parameters, in the program's syntax. The candidate
    is a proposal; it is
    checked when the user runs the analysis with it."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    from prodigy.analysis import ai_candidates
    from prodigy.analysis.analyzer import compute_discrete_distribution
    from prodigy.analysis.input_distribution import input_distribution_from_text
    from prodigy.pgcl import parse_pgcl

    problems = validate_suggestion(request)
    if problems:
        return _error("request", " ".join(problems), problems=problems)
    try:
        provider = ai_candidates.get_provider()
    except (ai_candidates.AIConfigurationError, ValueError) as exc:
        return _error("configuration", str(exc))
    analysis = prepare_request(suggestion_analysis_request(request))
    syntax = request_syntax(analysis)
    try:
        program = parse_pgcl(analysis["source"], syntax)
    except Exception as exc:  # pylint: disable=broad-except
        return _error("syntax", str(exc).split("\n", maxsplit=1)[0])
    loop_id, form = int(request["loop"]), request["form"]
    session = _stopping_session(
        loop_id, program=program, source=analysis["source"], syntax=syntax, config=make_config(analysis),
        decisions=analysis["decisions"], allow_files=False, capture_output=True,
        output_callback=output, progress_callback=progress)
    if loop_id not in {loop.id for loop in session.loops}:
        return _error("request", f"The program has no loop {loop_id}.")
    input_distribution = analysis.get("input_distribution")
    task = None
    with session.activate(), session._output():  # pylint: disable=protected-access
        try:
            variables = list(program.variables.keys())
            if isinstance(input_distribution, str) and input_distribution.strip():
                dist = input_distribution_from_text(session.config, input_distribution, program, syntax)
            else:
                dist = session.config.factory.one(*variables)
            compute_discrete_distribution(program, dist, session.config)
        except _LoopReached as reached:
            task = reached.task
        except Exception as exc:  # pylint: disable=broad-except
            return _error(type(exc).__name__, f"The distribution entering loop {loop_id} could not be "
                                              f"computed: {str(exc) or type(exc).__name__}")
        if task is None:
            return _error("unreachable", f"The analysis never reaches loop {loop_id}, so there is nothing "
                                         "to suggest.")
        names = [str(v) for v in program.variables.keys()]
        parameters = [str(p) for p in program.parameters.keys()]
        entering = str(task.distribution)
        print(f"Asking {provider.model} for an {form} candidate for loop {loop_id} (entered with {entering}).")
        try:
            answer = provider.ask(ai_candidates.SYSTEM_PROMPT, ai_candidates.prompt(
                form, str(program), str(task.instruction), names, parameters, entering, []))
        except ai_candidates.AIRequestError as exc:
            return _error("ai_request", str(exc))
    suggestion = {"loop": loop_id, "form": form, "param": ai_candidates.TARGET_PARAMS[form], "provider": provider.name,
                  "model": provider.model, "entering": entering, "answer": answer.strip()[-2000:]}
    try:
        suggestion["candidate"] = ai_candidates.candidate_from_answer(answer, names + parameters, syntax)
    except ai_candidates.CandidateError as exc:
        reason = str(exc)
        if not answer.strip() and provider.diagnostics:
            reason += f" The model stopped with {provider.diagnostics.get('stop_reason')}."
        return _error("candidate", f"The answer of {provider.model} contains no usable candidate: {reason}",
                      suggestion=suggestion)
    return {"status": "ok", "suggestion": suggestion}
