"""
The checks of the fuzzing tests: one program, analysed on every engine (and with simplification on
and off), compared with each other and with the simulator (:mod:`prodigy.simulator`).

:func:`check_program` returns the list of disagreements; an empty list means the program passed.
"""
from __future__ import annotations

import contextlib
import math
import multiprocessing
import random
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.equality import decide_equal_strings
from prodigy.pgcl.parser import parse_pgcl
from prodigy.simulator import SimulationError, UnsupportedConstruct, compare, simulate

Engine = ForwardAnalysisConfig.Engine


class ProgramTimeout(BaseException):
    """A program used up its time budget (:func:`time_budget`). A BaseException, so that the
    analysis's ``except Exception`` (errors end up in the result) does not swallow it."""


@contextlib.contextmanager
def time_budget(seconds: float) -> Iterator[None]:
    """
    Raises :class:`ProgramTimeout` in the block after ``seconds`` (wall clock, SIGALRM).

    pytest-timeout uses the same timer: its handler and remaining time are saved and restored, and
    the budget never outlasts it. Native code (the GiNaC bindings) is interrupted only when it
    returns to Python. Without SIGALRM (Windows) or outside the main thread there is no budget.
    """
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expire(_signum, _frame):
        raise ProgramTimeout(f"the program needs more than {seconds} s")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous_remaining) if previous_remaining else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining:
            signal.setitimer(signal.ITIMER_REAL, max(previous_remaining - (time.monotonic() - started), 0.001))


MEMORY_LIMIT = 3 * 2**30
"""Bytes of heap a program run in :func:`isolated` may allocate (four xdist workers fit on a CI runner)."""


class ProgramCrash(Exception):
    """A program run in :func:`isolated` ended its process: out of memory, or killed by a signal."""


class ProgramError(Exception):
    """A program run in :func:`isolated` raised an exception; the message holds its traceback."""


def _limit_memory(limit: int) -> None:
    try:
        import resource  # pylint: disable=import-outside-toplevel
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    except (ImportError, ValueError, OSError):  # Windows, or a platform that does not enforce it
        pass


def _run_child(connection, function: Callable[..., Any], args: tuple, memory: int, seed: Any) -> None:
    _limit_memory(memory)
    random.seed(seed)
    try:
        value = function(*args)
    except MemoryError:
        message = ("crash", f"out of memory (more than {memory // 2**20} MiB)")
    except BaseException:  # pylint: disable=broad-exception-caught
        message = ("error", traceback.format_exc())
    else:
        message = ("ok", value)
    try:
        connection.send(message)
    except MemoryError:
        connection.send(("crash", f"out of memory (more than {memory // 2**20} MiB)"))
    connection.close()


def isolated(function: Callable[..., Any], *args: Any, seconds: float, memory: int = MEMORY_LIMIT,
             seed: Any = 0) -> Any:
    """
    ``function(*args)`` in a child process with a hard limit of ``seconds`` (wall clock) and
    ``memory`` bytes of heap (``RLIMIT_DATA``, enforced on Linux).

    Unlike :func:`time_budget`, this stops native code too (a SymPy or GiNaC computation on huge
    numbers that never returns to the interpreter) and an analysis that eats all memory, so one
    program cannot hang or kill the whole test run. Raises :class:`ProgramTimeout` when over time,
    :class:`ProgramCrash` when the child runs out of memory or dies, and :class:`ProgramError` with
    the child's traceback when ``function`` raises. ``function`` and ``args`` must be picklable;
    the child seeds :mod:`random` with ``seed``. Without the start method ``forkserver``
    (Windows) the call runs in this process under :func:`time_budget`.
    """
    if "forkserver" not in multiprocessing.get_all_start_methods():
        with time_budget(seconds):
            return function(*args)
    context = multiprocessing.get_context("forkserver")
    # the server imports Prodigy once; every child is a cheap fork of it
    context.set_forkserver_preload([__name__])
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_child, args=(sender, function, args, memory, seed), daemon=True)
    process.start()
    sender.close()
    try:
        if not receiver.poll(seconds):
            raise ProgramTimeout(f"the program needs more than {seconds} s")
        try:
            kind, value = receiver.recv()
        except EOFError:
            process.join(5)
            raise ProgramCrash(f"the process ended with exit code {process.exitcode} "
                               f"(killed or out of memory)") from None
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(5)
    if kind == "crash":
        raise ProgramCrash(value)
    if kind == "error":
        raise ProgramError(value)
    return value


@dataclass(frozen=True)
class Setup:
    """One analysis configuration."""
    name: str
    engine: Any
    simplification: bool = False


SETUPS = (Setup("sympy", Engine.SYMPY), Setup("sympy+simplify", Engine.SYMPY, True),
          Setup("ginac", Engine.GINAC), Setup("symengine", Engine.SYMENGINE))
_NON_FINITE = ("nan", "zoo", "oo", "inf")


def analyse(source: str, syntax: str, setup: Setup, decisions: Mapping[str, Any]) -> Dict[str, Any]:
    """The JSON result of analysing ``source`` with ``setup``."""
    program = parse_pgcl(source, syntax)
    config = ForwardAnalysisConfig(engine=setup.engine, use_simplification=setup.simplification)
    session = AnalysisSession(program, source=source, syntax=syntax, config=config, decisions=dict(decisions),
                              capture_output=True, coefficient_limit=12)
    return session.run().to_json()


def _non_finite(text: Optional[str]) -> bool:
    if not text:
        return False
    tokens = set(text.replace("(", " ").replace(")", " ").replace("*", " ").replace("-", " ").split())
    return any(word in tokens for word in _NON_FINITE)


def _numbers(result: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """The exact numbers of a result, keyed by what they are."""
    numbers: Dict[str, Optional[str]] = {}
    distribution = result.get("distribution") or {}
    numbers["distribution"] = distribution.get("closed_form")
    numbers["mass"] = distribution.get("mass")
    for query in result.get("queries", []):
        if query.get("kind") in ("probability", "expectation"):
            numbers[f"query {query.get('index')} {query.get('expr')}"] = query.get("value")
    return numbers


def check_program(source: str, syntax: str = "ppy", decisions: Optional[Mapping[str, Any]] = None, *,
                  setups=SETUPS, runs: int = 2000, seed: Any = 0,
                  report: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Analyses ``source`` with every setup and returns the disagreements:

    * a result with status ``ok`` that contains ``nan``, ``zoo`` or ``oo`` (#165);
    * two ``ok`` results whose distribution, mass or query values are provably different
      (:func:`~prodigy.distribution.equality.decide_equal_strings`), or that differ in the kind of
      guarantee (exact, lower, upper);
    * an ``ok`` result outside the confidence intervals of the simulation.

    Results with another status (e.g. a "not implemented" error on one engine) are incomplete, not
    wrong, and are not compared. ``report`` (if given) receives the results by setup.
    """
    decisions = decisions or {}
    problems: List[str] = []
    results = {setup.name: analyse(source, syntax, setup, decisions) for setup in setups}
    if report is not None:
        report.update(results)
    ok = {name: result for name, result in results.items() if result["status"] == "ok"}
    for name, result in ok.items():
        for what, value in _numbers(result).items():
            if _non_finite(value) and not (what.startswith("query") and value in ("oo", "inf", "\\infty")):
                problems.append(f"{name}: status ok, but {what} is {value}")
    names = sorted({var for result in ok.values() for var in (result.get("distribution") or {}).get("variables", [])}
                   | {"x", "y", "z", "c"})
    reference_name = next(iter(ok), None)
    if reference_name is not None:
        reference = ok[reference_name]
        expected = _numbers(reference)
        for name, result in ok.items():
            if name == reference_name:
                continue
            if result["bounds"]["kind"] != reference["bounds"]["kind"]:
                problems.append(f"{name}: guarantee {result['bounds']['kind']}, "
                                f"{reference_name}: {reference['bounds']['kind']}")
            for what, value in _numbers(result).items():
                other = expected.get(what)
                if value is None or other is None:
                    continue
                if decide_equal_strings(value, other, names) is False:
                    problems.append(f"{what}: {name} gives {value}, {reference_name} gives {other}")
        try:
            simulation = simulate(parse_pgcl(source, syntax), runs, seed=seed)
        except UnsupportedConstruct:
            simulation = None
        except SimulationError as error:
            problems.append(f"the analysis accepts the program, but a run fails: {error}")
            simulation = None
        if simulation is not None:
            for name, result in ok.items():
                problems.extend(f"{name} vs. simulation: {m}"
                                for m in compare(parse_pgcl(source, syntax), result, simulation))
    return problems


def statuses(source: str, syntax: str = "ppy", decisions: Optional[Mapping[str, Any]] = None,
             setups=SETUPS) -> Dict[str, str]:
    """The status of every setup (for reports)."""
    return {setup.name: analyse(source, syntax, setup, decisions or {})["status"] for setup in setups}


def finite(value: Optional[float]) -> bool:
    """Whether ``value`` is a finite float."""
    return value is not None and math.isfinite(value)
