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
expressions in the program's syntax (:func:`prodigy.pgcl.expression_allowlist.admissible_expression`),
and :func:`prepare_request` turns them into the engines' spelling (``x**2`` -> ``x^2``).
"""
from __future__ import annotations

import contextlib
import copy
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.equivalence.verdict import CheckOutcome
    from prodigy.analysis.session import AnalysisSession
    from prodigy.pgcl import Program

ENGINES = {
    "sympy": "SymPy",
    "ginac": "GiNaC",
    "symengine": "SymEngine (deprecated)",
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
    from prodigy.pgcl.expression_allowlist import MATH_FUNCTIONS, InadmissibleExpression, admissible_expression
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
        except InadmissibleExpression as exc:
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


def make_config(request: Dict[str, Any]) -> ForwardAnalysisConfig:
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


def _input_check_worker(connection: Any, source: str, syntax: str, text: str, engine: str) -> None:
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
        # Analyze mode: parameter synthesis is a comparison (Compare mode, #154)
        parameter_synthesis="refuse",
        # the plots build up while the coefficients are computed (#405)
        stream_coefficients=progress is not None,
    )
    return session.run().to_json()


# ---------------------------------------------------------------------------------------
# Comparing two programs (POST /api/equivalence)
# ---------------------------------------------------------------------------------------

def loop_program(source: str, syntax: str, loop_id: int) -> Dict[str, Any]:
    """A loop of a program as a program of its own, for "Compare this loop" (#399): ``program`` (the
    program itself if the loop is all it does, else its declarations and the loop, see
    :func:`prodigy.analysis.session.isolate_loop`), ``loop`` (its id there), ``declarations`` (to start
    program B from) and ``whole``. Raises ``ValueError`` with a message for the person otherwise."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import LoopRegistry, isolate_loop
    from prodigy.pgcl import parse_pgcl

    try:
        program = parse_pgcl(source, syntax)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"The program does not parse: {str(exc).split(chr(10), maxsplit=1)[0]}") from exc
    registry = LoopRegistry(program)
    info = next((loop for loop in registry if loop.id == loop_id), None)
    if info is None:
        raise ValueError(f"The program has no loop {loop_id}.")
    if info.function is not None:
        raise ValueError(f"Loop {loop_id} is inside the function {info.function}; only loops of the program "
                         "itself can be compared.")
    instruction = registry.instruction(loop_id)
    assert instruction is not None
    isolated = isolate_loop(program, instruction, syntax, source)
    if isolated is None:
        raise ValueError(f"Loop {loop_id} cannot be written as a program of its own in this syntax.")
    return {"program": isolated.program, "loop": isolated.loop, "declarations": isolated.declarations,
            "whole": isolated.whole}


def other_program_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """The request of the second program of an equivalence request: ``other_source`` in ``other_syntax``
    (default: the syntax of the first program) with ``other_decisions``, and the options of the first."""
    other = {key: value for key, value in request.items()
             if key not in ("source", "syntax", "file_name", "decisions", "record_steps", "watches")}
    other["source"] = request.get("other_source")
    other["syntax"] = request.get("other_syntax") or request_syntax(request)
    other["decisions"] = request.get("other_decisions") or {}
    return {key: value for key, value in other.items() if not key.startswith("other_")}


def validate_equivalence(request: Dict[str, Any]) -> List[str]:
    """Problems with an equivalence request: an analysis request (the first program, without an initial
    distribution: the check covers every input) plus ``other_source``, optional ``other_syntax`` and
    ``other_decisions`` (the second program)."""
    first = {key: value for key, value in request.items() if not key.startswith("other_")}
    problems = validate_request(first)
    if isinstance(request.get("input_distribution"), str) and request["input_distribution"].strip():
        problems.append("A comparison checks every input at once; it takes no initial distribution.")
    if not isinstance(request.get("other_source"), str) or not request["other_source"].strip():
        return problems + ["Program B is empty."]
    try:
        other = other_program_request(request)
    except ValueError as exc:
        return problems + [str(exc)]
    return problems + [f"Program B: {problem}" for problem in validate_request(other)
                       if problem not in problems]


EQUIVALENCE_SCHEMA = "prodigy/equivalence-result@1"
"""The schema of the result of a comparison (:func:`run_equivalence`)."""

WITNESS_SECONDS = 20.0
"""How long a comparison looks for a witness (at most a third of the job's time limit)."""

WITNESS_UNROLLING = 30
"""Iterations of each loop when the witness search unrolls the loops (lower bounds with residual mass)."""


class WitnessTimeout(Exception):
    """The witness search ran out of its time (:data:`WITNESS_SECONDS`)."""


@contextlib.contextmanager
def _time_budget(seconds: float) -> Iterator[None]:
    """Raises :class:`WitnessTimeout` in the block after ``seconds`` (SIGALRM, so only in the main thread of
    a process; elsewhere there is no limit). A timer that was running (e.g. pytest-timeout's) is restored with
    the time it has left. Native code (GiNaC) is interrupted only when it returns to Python."""
    # pylint: disable=import-outside-toplevel
    import signal
    import threading
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(_signum: int, _frame: Any) -> None:
        raise WitnessTimeout()

    previous = signal.signal(signal.SIGALRM, expire)
    start = time.monotonic()
    remaining, interval = signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, max(remaining - (time.monotonic() - start), 0.001), interval)


WitnessInput = Dict[str, int]
"""The input state of a witness search: variable values, absent variables are 0."""


def _witness_input(check: "CheckOutcome") -> WitnessInput:
    """Where to look for a witness: the counterexample of the check, else the input on which an invariant was
    refuted, else every variable 0."""
    if check.counterexample:
        return dict(check.counterexample)
    details = (check.error or {}).get("details") or {}
    found = details.get("counterexample")
    if isinstance(found, dict) and all(isinstance(v, int) and not isinstance(v, bool) for v in found.values()):
        return {str(k): v for k, v in found.items()}
    return {}


def _search_witness(program: "Program", other: "Program", session: "AnalysisSession", state: WitnessInput,
                    decisions: Tuple[Dict[str, Any], Dict[str, Any]], seconds: float
                    ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """A witness on the input ``state`` (:func:`~prodigy.analysis.equivalence.verdict.find_witness`): both
    programs are analysed on it, first with the strategies of the request (``decisions`` of A and B), then
    with every loop unrolled (:data:`WITNESS_UNROLLING`; a lower bound with its residual mass is still
    sound for a witness). Returns the witness or None, and whether the search ran out of time."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.equivalence.verdict import analyse_on_input, find_witness
    from prodigy.analysis.session import parse_decisions
    unrolled = {"*": {"strategy": "unroll", "params": {"iterations": WITNESS_UNROLLING}}}
    attempts = [decisions, (unrolled, unrolled)]
    try:
        with _time_budget(seconds):
            for first, second in attempts:
                kept = session.provider.decisions
                session.provider.decisions = parse_decisions(first)
                try:
                    one = analyse_on_input(program, state, session)
                    with session.other_program(other, second):
                        two = analyse_on_input(other, state, session)
                    witness = find_witness(one, two)
                except WitnessTimeout:
                    raise
                except Exception:  # pylint: disable=broad-except
                    witness = None
                finally:
                    session.provider.decisions = kept
                if witness is not None:
                    return witness, False
    except WitnessTimeout:
        return None, True
    return None, False


def run_equivalence(request: Dict[str, Any], *, output: Optional[Callable[[str], None]] = None,
                    progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Checks whether two programs are equivalent for every input (in a job process): the second-order check
    (:func:`~prodigy.analysis.equivalence.verdict.run_check`) and, unless it proved the computed distributions
    equal, a witness search on the input of its counterexample (or every variable 0). No program runs on an
    input of the person's choosing, and there is no output distribution of either program.

    The result (``prodigy/equivalence-result@1``) has ``status``, ``error``, ``syntax``, ``loops`` and
    ``other_loops`` (what the loops of programs A and B were proven to be in the check), ``log``, ``timing``
    and ``equivalence``: the sound verdict (:mod:`prodigy.analysis.equivalence.verdict`), the witness if there
    is one (with its ``input``), the raw outcome of the check and, if the witness search ran out of time,
    ``witness_search: "timeout"``. Never raises for analysis errors."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    from prodigy.analysis.equivalence import verdict as equivalence
    from prodigy.analysis.result import prodigy_version
    from prodigy.analysis.session import AnalysisSession
    from prodigy.pgcl import parse_pgcl
    from prodigy.util.integers import unlimited_int_digits

    problems = validate_equivalence(request)
    if problems:
        return _error("request", " ".join(problems), problems=problems)
    start = time.perf_counter()
    first_request = prepare_request({key: value for key, value in request.items() if not key.startswith("other_")})
    second_request = prepare_request(other_program_request(request))
    programs: Dict[str, Any] = {}
    for which, part in (("A", first_request), ("B", second_request)):
        try:
            programs[which] = parse_pgcl(part["source"], request_syntax(part))
        except Exception as exc:  # pylint: disable=broad-except
            return _error("syntax", f"Program {which}: {str(exc).split(chr(10), maxsplit=1)[0]}")
    program, other = programs["A"], programs["B"]
    decisions = (first_request.get("decisions") or {}, second_request.get("decisions") or {})
    session = AnalysisSession(program, source=first_request["source"], syntax=request_syntax(first_request),
                              config=make_config(first_request), decisions=decisions[0], allow_files=False,
                              capture_output=True, output_callback=output, progress_callback=progress,
                              parameter_synthesis="allow")
    variables_differ = equivalence.different_variables(program, other)
    check = equivalence.CheckOutcome("failed", error={"type": "variables", "message": variables_differ or ""})
    witness: Optional[Dict[str, Any]] = None
    timed_out = False
    with session.activate(), session.config.activate(), session._output():  # pylint: disable=protected-access
        if variables_differ is None:
            print("Checking the equivalence for every input (second-order generating functions) …")
            # each program's loops are decided by their own ids and decisions
            check = equivalence.run_check(program, other, session, other_decisions=decisions[1])
        records = list(session.loop_records.items())
        if variables_differ is None and check.answer != "equal":
            state = _witness_input(check)
            shown = ", ".join(f"{k} = {v}" for k, v in sorted(state.items()) if v) or "every variable 0"
            print(f"Looking for a witness on the input {shown} …")
            limit = request.get("time_limit")
            seconds = WITNESS_SECONDS if not isinstance(limit, (int, float)) else min(WITNESS_SECONDS, limit / 3)
            with unlimited_int_digits():
                witness, timed_out = _search_witness(program, other, session, state, decisions, seconds)
            if witness is not None:
                witness["input"] = {name: state.get(name, 0) for name in sorted(program.variables)}
            elif timed_out:
                print(f"The witness search stopped after {seconds:g} s.")
    verdict = equivalence.decide(check, witness, variables_differ=variables_differ, input_tips=False)
    if timed_out and verdict["verdict"] == equivalence.UNKNOWN:
        verdict["reason"] += " The witness search ran out of time."

    def loops(other_program: bool) -> List[Dict[str, Any]]:
        found = [record.to_json() for key, record in records if isinstance(key, tuple) == other_program]
        return sorted(found, key=lambda r: (r["id"] is None, r["id"] if r["id"] is not None else 0))

    return {
        "schema": EQUIVALENCE_SCHEMA,
        "prodigy_version": prodigy_version(),
        "status": "ok",
        "error": None,
        "syntax": request_syntax(first_request),
        "loops": loops(False),
        "other_loops": loops(True),
        "log": session.log.getvalue(),
        "timing": {"total": round(time.perf_counter() - start, 6)},
        "equivalence": {
            **verdict,
            "witness": witness,
            "witness_search": "timeout" if timed_out else None,
            "check": check.to_json(),
        },
    }


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
        upper = bounds.get("upper")
        source = {"closed_form": distribution["closed_form"], "bound": bounds.get("kind", "unknown"),
                  "residual_mass": bounds.get("residual_mass"),
                  # the proven upper bound of the result (#451); a step snapshot has none
                  "upper_closed_form": upper.get("closed_form") if isinstance(upper, dict) else None}
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
    """Computes the coefficient grid of a :func:`joint_request` (in a job process); the cells go to
    ``progress`` while they are computed (#405)."""
    # pylint: disable=import-outside-toplevel,unused-argument
    from prodigy.analysis.joint_grid import GridError, joint_grid
    from prodigy.util.integers import unlimited_int_digits
    from prodigy.distribution import known_convergent_series
    config = make_config({"engine": request["engine"]})
    # the closed form is that of the server's result of a program analysis, a finite measure (#398)
    with config.activate(), unlimited_int_digits(), known_convergent_series(True):
        try:
            dist = config.factory.from_expr(request["closed_form"], *request["variables"])
            upper = request.get("upper_closed_form")
            # the server's own proven upper bound of the same result, a finite measure as well
            upper_dist = config.factory.from_expr(upper, *request["variables"]) if upper else None
            grid = joint_grid(dist, request["x"], request["y"], x_window=request.get("x_window"),
                              y_window=request.get("y_window"), kind=request["bound"],
                              residual=request.get("residual_mass") or 0, stream=progress, upper=upper_dist)
        except GridError as exc:
            return _error("request", str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            return _error(_error_type(exc), f"The grid could not be computed: {str(exc) or type(exc).__name__}")
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

    def __init__(self, task: Any) -> None:
        super().__init__("loop reached")
        self.task = task


def _stopping_session(target: int, **kwargs: Any) -> Any:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import AnalysisSession

    class StoppingSession(AnalysisSession):
        """Analyses the program up to the first time loop ``target`` is reached outside of
        another strategy's checks."""

        def run_loop(self, task: Any) -> Any:
            if task.loop is not None and task.loop.id == target and not self.silent:
                raise _LoopReached(task)
            return super().run_loop(task)

    return StoppingSession(**kwargs)


def _error_type(exc: BaseException) -> str:
    """The error type as in the analysis result (:func:`prodigy.analysis.result.error_type`)."""
    from prodigy.analysis.result import error_type  # pylint: disable=import-outside-toplevel
    return error_type(exc)


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
            return _error(_error_type(exc), f"The distribution entering loop {loop_id} could not be "
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
