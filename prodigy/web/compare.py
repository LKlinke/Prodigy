"""
Compare mode of the web GUI: two programs checked for equivalence on every input (``POST /api/equivalence``)
and a loop as a program of its own ("Compare this loop", ``POST /api/loop-program``).

An equivalence request is an analysis request (:mod:`prodigy.web.requests`) for program A plus
``other_source``, optional ``other_syntax`` and ``other_decisions`` for program B, without an initial
distribution. :func:`run_equivalence` runs in a job process.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from prodigy.web.requests import error_result, make_config, prepare_request, validate_request

if TYPE_CHECKING:
    from prodigy.analysis.session import AnalysisSession
    from prodigy.analysis.tools import CheckOutcome
    from prodigy.language import Program

EQUIVALENCE_SCHEMA = "prodigy/equivalence-result@1"
"""The schema of the result of a comparison (:func:`run_equivalence`)."""

WITNESS_SECONDS = 20.0
"""How long a comparison looks for a witness (at most a third of the job's time limit)."""

WITNESS_UNROLLING = 30
"""Iterations of each loop when the witness search unrolls the loops (lower bounds with residual mass)."""

Decisions = Tuple[Dict[str, Any], Dict[str, Any]]
"""The loop decisions of programs A and B, as in the request."""

WitnessInput = Dict[str, int]
"""The input state of a witness search: variable values, absent variables are 0."""


def loop_program(source: str, loop_id: int) -> Dict[str, Any]:
    """A loop of a program as a program of its own, for "Compare this loop" (#399): ``program`` (the
    program itself if the loop is all it does, else its declarations and the loop, see
    :func:`prodigy.analysis.loops.api.isolate_loop`), ``loop`` (its id there), ``declarations`` (to start
    program B from) and ``whole``. Raises ``ValueError`` with a message for the person otherwise."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import LoopRegistry
    from prodigy.analysis.loops.api import isolate_loop
    from prodigy.language import parse_program

    try:
        program = parse_program(source)
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
    isolated = isolate_loop(program, instruction, source)
    if isolated is None:
        raise ValueError(f"Loop {loop_id} cannot be written as a program of its own.")
    return {"program": isolated.program, "loop": isolated.loop, "declarations": isolated.declarations,
            "whole": isolated.whole}


def other_program_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """The request of the second program of an equivalence request: ``other_source`` (with an optional
    ``other_syntax``, checked like ``syntax``) with ``other_decisions``, and the options of the first."""
    other = {key: value for key, value in request.items()
             if key not in ("source", "syntax", "file_name", "decisions", "record_steps", "watches")}
    other["source"] = request.get("other_source")
    if request.get("other_syntax") is not None:
        other["syntax"] = request["other_syntax"]
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


def _witness_attempt(programs: Tuple["Program", "Program"], session: "AnalysisSession", state: WitnessInput,
                     decisions: Decisions) -> Optional[Dict[str, Any]]:
    """Both programs analysed on ``state`` with ``decisions``, and a witness from the results, or None."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.tools import analyse_on_input, find_witness
    from prodigy.analysis.session import parse_decisions
    program, other = programs
    kept = session.provider.decisions
    session.provider.decisions = parse_decisions(decisions[0])
    try:
        one = analyse_on_input(program, state, session)
        with session.other_program(other, decisions[1]):
            two = analyse_on_input(other, state, session)
        return find_witness(one, two)
    except WitnessTimeout:
        raise
    except Exception:  # pylint: disable=broad-except
        return None
    finally:
        session.provider.decisions = kept


def _search_witness(programs: Tuple["Program", "Program"], session: "AnalysisSession", state: WitnessInput,
                    decisions: Decisions, seconds: float) -> Tuple[Optional[Dict[str, Any]], bool]:
    """A witness on the input ``state`` (:func:`~prodigy.analysis.equivalence.verdict.find_witness`): both
    programs are analysed on it, first with the strategies of the request (``decisions`` of A and B), then
    with every loop unrolled (:data:`WITNESS_UNROLLING`; a lower bound with its residual mass is still
    sound for a witness). Returns the witness or None, and whether the search ran out of time."""
    unrolled = {"*": {"strategy": "unroll", "params": {"iterations": WITNESS_UNROLLING}}}
    try:
        with _time_budget(seconds):
            for attempt in (decisions, (unrolled, unrolled)):
                witness = _witness_attempt(programs, session, state, attempt)
                if witness is not None:
                    return witness, False
    except WitnessTimeout:
        return None, True
    return None, False


def _witness_on_check(programs: Tuple["Program", "Program"], session: "AnalysisSession", check: "CheckOutcome",
                      decisions: Decisions, time_limit: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    """The witness search after a check that did not prove the programs equal, on the input of its
    counterexample, within a third of the job's ``time_limit`` (at most :data:`WITNESS_SECONDS`)."""
    from prodigy.util.integers import unlimited_int_digits  # pylint: disable=import-outside-toplevel
    state = _witness_input(check)
    shown = ", ".join(f"{k} = {v}" for k, v in sorted(state.items()) if v) or "every variable 0"
    print(f"Looking for a witness on the input {shown} …")
    seconds = WITNESS_SECONDS if not isinstance(time_limit, (int, float)) else min(WITNESS_SECONDS, time_limit / 3)
    with unlimited_int_digits():
        witness, timed_out = _search_witness(programs, session, state, decisions, seconds)
    if witness is not None:
        witness["input"] = {name: state.get(name, 0) for name in sorted(programs[0].variables)}
    elif timed_out:
        print(f"The witness search stopped after {seconds:g} s.")
    return witness, timed_out


def _parse_both(first_request: Dict[str, Any], second_request: Dict[str, Any]
                ) -> Union[Tuple["Program", "Program"], Dict[str, Any]]:
    """Programs A and B, or the error result for the first that does not parse."""
    from prodigy.language import parse_program  # pylint: disable=import-outside-toplevel
    programs = []
    for which, part in (("A", first_request), ("B", second_request)):
        try:
            programs.append(parse_program(part["source"]))
        except Exception as exc:  # pylint: disable=broad-except
            return error_result("syntax", f"Program {which}: {str(exc).split(chr(10), maxsplit=1)[0]}")
    return programs[0], programs[1]


def _loops(records: List[Tuple[Any, Any]], other_program: bool) -> List[Dict[str, Any]]:
    """What the loops of program A (or B, ``other_program``) were proven to be, by loop id."""
    found = [record.to_json() for key, record in records if isinstance(key, tuple) == other_program]
    return sorted(found, key=lambda r: (r["id"] is None, r["id"] if r["id"] is not None else 0))


@dataclass
class _Outcome:
    """What a comparison found: the check, the witness search after it, and the loops' records."""
    check: "CheckOutcome"
    variables_differ: Optional[str]
    records: List[Tuple[Any, Any]] = field(default_factory=list)
    witness: Optional[Dict[str, Any]] = None
    timed_out: bool = False


def _compare(programs: Tuple["Program", "Program"], session: "AnalysisSession", decisions: Decisions,
             time_limit: Any) -> _Outcome:
    """The check for every input and, unless it proved the computed distributions equal, the witness search."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.tools import CheckOutcome, different_variables, run_check
    variables_differ = different_variables(*programs)
    outcome = _Outcome(CheckOutcome("failed", error={"type": "variables", "message": variables_differ or ""}),
                       variables_differ)
    with session.activate(), session.config.activate(), session._output():  # pylint: disable=protected-access
        if variables_differ is None:
            print("Checking the equivalence for every input (second-order generating functions) …")
            # each program's loops are decided by their own ids and decisions
            outcome.check = run_check(*programs, session, other_decisions=decisions[1])
        outcome.records = list(session.loop_records.items())
        if variables_differ is None and outcome.check.answer != "equal":
            outcome.witness, outcome.timed_out = _witness_on_check(programs, session, outcome.check, decisions,
                                                                   time_limit)
    return outcome


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
    from prodigy.analysis.session import AnalysisSession  # pylint: disable=import-outside-toplevel
    problems = validate_equivalence(request)
    if problems:
        return error_result("request", " ".join(problems), problems=problems)
    start = time.perf_counter()
    first_request = prepare_request({key: value for key, value in request.items() if not key.startswith("other_")})
    second_request = prepare_request(other_program_request(request))
    programs = _parse_both(first_request, second_request)
    if isinstance(programs, dict):
        return programs
    decisions = (first_request.get("decisions") or {}, second_request.get("decisions") or {})
    session = AnalysisSession(programs[0], source=first_request["source"],
                              config=make_config(first_request), decisions=decisions[0], allow_files=False,
                              capture_output=True, output_callback=output, progress_callback=progress,
                              parameter_synthesis="allow")
    return _equivalence_result(_compare(programs, session, decisions, request.get("time_limit")), session, start)


def _equivalence_result(outcome: _Outcome, session: "AnalysisSession", start: float) -> Dict[str, Any]:
    """The result ``prodigy/equivalence-result@1`` of :func:`run_equivalence` (``start``: its start time)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.tools import UNKNOWN, decide
    from prodigy.analysis.result import prodigy_version
    verdict = decide(outcome.check, outcome.witness, variables_differ=outcome.variables_differ, input_tips=False)
    if outcome.timed_out and verdict["verdict"] == UNKNOWN:
        verdict["reason"] += " The witness search ran out of time."
    return {
        "schema": EQUIVALENCE_SCHEMA,
        "prodigy_version": prodigy_version(),
        "status": "ok",
        "error": None,
        "syntax": "ppy",
        "loops": _loops(outcome.records, False),
        "other_loops": _loops(outcome.records, True),
        "log": session.log.getvalue(),
        "timing": {"total": round(time.perf_counter() - start, 6)},
        "equivalence": {
            **verdict,
            "witness": outcome.witness,
            "witness_search": "timeout" if outcome.timed_out else None,
            "check": outcome.check.to_json(),
        },
    }
