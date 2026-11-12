"""
EVT invariant synthesis: enumerates templates (rational functions or polynomials with unknown
coefficients), solves the fixed-point equation I = G + ⟦body⟧(⟨I⟩_guard) for the coefficients and
keeps the solutions that pass the exact fixed-point check.

The search is bounded: the template heuristic enumerates finitely many templates up to a maximal
degree, and an optional time limit stops it (also in the middle of a solver call, which can take
arbitrarily long for a single template). A second, shorter time limit per template skips a template
whose system the solver does not solve quickly, so that one expensive template does not use up the whole
time. If no invariant is found within the bounds, :class:`SynthesisBoundReached` is raised; the analysis
then has no result (``unknown``).
"""
import contextlib
import logging
import signal
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import sympy
from sympy.polys.polyerrors import BasePolynomialError
from prodigy.pgcl import WhileInstr, Instr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategy
from prodigy.analysis.evtinvariants.heuristics.templates.polynomial_function import PolynomialEnumeration
from prodigy.analysis.evtinvariants.heuristics.templates.rational_function import MAX_NUMERATOR_COEFFICIENTS
from prodigy.analysis.evtinvariants.invariant_verification import is_evt_fixed_point
from prodigy.analysis.evtinvariants.template_solver import solve_template
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[[Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                    Tuple[Distribution, Distribution]]
Candidates = List[Tuple[Distribution, Union[bool, None]]]

DEFAULT_SYNTHESIS_TIME_LIMIT = 60
"""Default time limit of the synthesis in seconds (strategy parameter ``time_limit`` of ``evt_synthesis``)."""

DEFAULT_TEMPLATE_TIME_LIMIT = 5
"""Default time limit per template in seconds (strategy parameter ``template_time_limit`` of
``evt_synthesis``): a template whose fixed-point system takes longer is skipped."""


class SynthesisBoundReached(VerificationError):
    """No EVT invariant was found within the bounds of the synthesis (all templates up to the maximal
    degree tried, or the time limit reached). ``details["reason"]`` is ``"degree"`` or ``"time_limit"``;
    the verdict is ``unknown``: an invariant of a higher degree, or a non-rational one, may exist.

    ``details["candidates"]`` lists the fixed points found whose coefficients could not be shown to be
    non-negative (at most :data:`MAX_REPORTED_CANDIDATES`); they are only candidates, not results, and
    ``messages`` names them for the loop's messages."""

    def __init__(self, message: str = "", details: Optional[Dict[str, Any]] = None,
                 messages: Sequence[str] = ()):
        super().__init__(message, details)
        self.messages: List[str] = list(messages)


MAX_REPORTED_CANDIDATES = 5
"""At most this many undecided fixed points are reported as candidates."""


class _TimeLimitReached(BaseException):
    """Raised by the alarm of :func:`_time_limit`. A ``BaseException`` so that no ``except Exception``
    inside SymPy or a heuristic swallows it."""


class _TemplateTimeLimitReached(_TimeLimitReached):
    """Raised by the alarm of the time limit of a single template; the synthesis goes on with the next one."""


@contextlib.contextmanager
def _time_limit(seconds: Optional[float],
                exception: type = _TimeLimitReached) -> Iterator[Callable[[], bool]]:
    """Interrupts the block with ``exception`` (a :class:`_TimeLimitReached`) after ``seconds`` (``None``:
    no limit).

    Yields a function that tells whether the time is up, for checks between steps. The interruption
    uses ``SIGALRM`` and therefore only works in the main thread on POSIX systems; elsewhere only these
    checks stop the block (after the step that is running). An alarm that is already set and fires
    earlier (e.g. the timeout of a test) is kept, a later one is restored afterwards."""
    deadline = None if seconds is None else time.monotonic() + seconds

    def expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    if seconds is None or not hasattr(signal, "setitimer") \
            or threading.current_thread() is not threading.main_thread():
        yield expired
        return
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    if old_delay and old_delay <= seconds:
        yield expired
        return
    armed = [True]

    def on_alarm(_signum, _frame):
        if armed[0]:
            raise exception()

    start = time.monotonic()
    old_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield expired
    finally:
        armed[0] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay:
            signal.setitimer(signal.ITIMER_REAL, max(old_delay - (time.monotonic() - start), 1e-3), old_interval)


def _template_kind(strategy: SynthesisStrategy) -> str:
    return "polynomial" if isinstance(strategy.template_heuristics, PolynomialEnumeration) else "rational"


def _max_degree(strategy: SynthesisStrategy) -> Optional[int]:
    """The degree bound of the strategy's templates (``None`` if they are enumerated without bound)."""
    degree = getattr(strategy.template_heuristics, "max_deg", None)
    return degree if isinstance(degree, int) and degree >= 0 else None


def _template_solutions(template: Distribution, phi: Distribution,
                        prog_info: ProgramInfo) -> Optional[List[Dict[sympy.Symbol, sympy.Expr]]]:
    """Solutions of the template's fixed-point equation ``template = phi`` (see :mod:`template_solver`);
    ``None`` if the conditions on the denominator could not be solved."""
    program_parameters = {str(param) for param in prog_info.program.parameters}
    unknowns = sorted(template.get_parameters() - program_parameters)
    # unknowns that occur only linearly in the template's numerator are solved by linear algebra, all others
    # (those of the denominator) by the conditions on them
    numerator, denominator = sympy.fraction(sympy.together(sympy.S(str(template).replace("^", "**"))))
    linear = [sympy.Symbol(name) for name in unknowns if sympy.Symbol(name) not in denominator.free_symbols]
    if linear and sympy.Poly(numerator, *linear).total_degree() > 1:
        linear = []
    numerator_unknowns = linear
    denominator_unknowns = [sympy.Symbol(name) for name in unknowns if sympy.Symbol(name) not in linear]
    variables = [sympy.Symbol(str(var)) for var in prog_info.program.variables]
    parameters = [sympy.Symbol(name) for name in sorted((template.get_parameters() | phi.get_parameters())
                                                        - set(unknowns))]
    # the difference in SymPy: together() in the solver combines it, cheaper than the engine's subtraction
    equation = sympy.S(str(template).replace("^", "**")) - sympy.S(str(phi).replace("^", "**"))
    return solve_template(equation, numerator_unknowns, denominator_unknowns, variables, parameters)


def _solutions(evt_candidate: Distribution, loop: WhileInstr, prog_info: ProgramInfo, distribution: Distribution,
               config: ForwardAnalysisConfig, strategy: SynthesisStrategy, analyzer: Analyzer,
               stats: Dict[str, int], undecided: List[str]) -> Tuple[Candidates, bool]:
    """The certified fixed points among the instances of the template ``evt_candidate`` and whether one of
    them was shown to have non-negative coefficients. Certified fixed points whose non-negativity is not
    decided are also added to ``undecided`` (once per function)."""
    zero_dist: Distribution = config.factory.from_expr("0", *prog_info.program.variables)
    # Compute one iteration step.
    evt_inv = evt_candidate
    one_step_dist, one_step_err = analyzer(loop.body, prog_info, evt_inv.filter(loop.cond), zero_dist, config)
    # is_zero_dist, not `!= zero_dist`: equality also compares the parameters, and the program's parameters
    # (e.g. a probability p) occur in the error term but not in the constant 0
    if not one_step_err.is_zero_dist():
        raise NotImplementedError("Currently invariant synthesis does not support conditioned distributions.")
    phi_inv = distribution + one_step_dist

    logger.debug("Check Invariant candidate %s", evt_inv)
    if config.solver_type == SolverType.SYMPY:
        # linear in the numerator coefficients (#281)
        solution_candidates = _template_solutions(evt_inv, phi_inv, prog_info)
        if solution_candidates is None:
            stats["failed"] += 1
            return [], False
    else:
        if config.solver_type == SolverType.Z3:
            solver = SolverType.make(config.solver_type, config.factory)
        else:
            solver = SolverType.make(config.solver_type)
        # Check equality between the iterated expression and the invariant.
        is_solution, solution_candidates = solver.solve(evt_inv, phi_inv)
        if is_solution is False or is_solution is None:
            return [], False
    logger.debug("Filter solutions in: %s", solution_candidates)

    # Exclude "all zero" solutions, as well as solutions which make the denominator 0.
    solutions = []
    # only computed when there are candidates (the engine's subtraction is not free)
    denominator = sympy.S(str(evt_inv - phi_inv)).as_numer_denom()[1] if solution_candidates else sympy.S.One
    for candidate in solution_candidates:
        if all(map(lambda x: x == 0, candidate.values())):
            continue

        # we have excluded all zero solutions and now also exclude the solutions
        # which make the denominator zero
        if denominator.subs(candidate).equals(0):
            continue

        if candidate not in solutions:
            solutions.append(candidate)

    # In case there are still some solutions we check them for actual solutions in the FPS domain with non-negative
    # coefficients. This is in general a hard problem (not known to be decidable), thus we use heuristics.
    result: Candidates = []
    inv_found = False
    for sol in solutions:
        sol_inv = sympy.S(str(evt_inv)).subs(sol).ratsimp()
        # The solver's solution is only a proposal (its values pass through string conversions).
        sol_dist = config.factory.from_expr(str(sol_inv).replace("**", "^"), *prog_info.program.variables)
        if not is_evt_fixed_point(loop, prog_info, distribution, sol_dist, config, analyzer):
            logger.info("Discarding the solution %s: the fixed-point equation is not proven.", sol)
            continue
        sol_is_positive = strategy.positivity_heuristics.is_positive(sol_inv)

        logger.debug("Possible solution: %s", sol)
        if sol_is_positive is None:
            print(f"{Style.CYAN}Possible invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")
            result.append((sol_dist, sol_is_positive))
            key = str(sympy.cancel(sol_inv))
            if key not in undecided:
                undecided.append(key)
        elif sol_is_positive is True:
            print(f"{Style.GREEN}Invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")
            result.append((sol_dist, sol_is_positive))
            inv_found = True
        else:
            if config.show_all_invs:
                print(f"{Style.RED}Spurious invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")
    return result, inv_found


def _effective_degree(strategy: SynthesisStrategy) -> Optional[int]:
    """The largest degree the templates actually reach (smaller than the bound for templates too large to
    build, see ``RationalFunctionMaxDeg.effective_max_deg``)."""
    degree = getattr(strategy.template_heuristics, "effective_max_deg", None)
    return degree if isinstance(degree, int) else _max_degree(strategy)


def _bound_message(reason: str, strategy: SynthesisStrategy, time_limit: Optional[float],
                   template_time_limit: Optional[float], stats: Dict[str, int], undecided: Sequence[str]) -> str:
    degree = _max_degree(strategy)
    found = f"No {_template_kind(strategy)} EVT invariant"
    effective = _effective_degree(strategy)
    left_out = ""
    if degree is not None and effective is not None and effective < degree:
        found += f" up to degree {effective}"
        left_out = f" Templates of a higher degree up to {degree} have more than {MAX_NUMERATOR_COEFFICIENTS} " \
                   "coefficients in the numerator and are left out."
    elif degree is not None:
        found += f" up to degree {degree}"
    tried = f"{stats['tried']} template{'s' if stats['tried'] != 1 else ''} tried"
    if stats["skipped"]:
        tried += f", {stats['skipped']} of them skipped after the time limit of {template_time_limit:g} s " \
                 "per template"
    if reason == "time_limit":
        message = f"Stopped after the time limit of {time_limit:g} s: {found[0].lower()}{found[1:]} found " \
                  f"({tried})."
    else:
        message = f"{found} found ({tried})."
    message += left_out
    if undecided:
        message += f" {len(undecided)} fixed point(s) were found whose coefficients could not be shown to be " \
                   "non-negative (reported as candidates, not as results)."
    if stats["failed"]:
        message += f" SymPy could not solve {stats['failed']} template(s)."
    return message


def evt_invariant_synthesis(loop: WhileInstr,
                            prog_info: ProgramInfo,
                            distribution: Distribution,
                            config: ForwardAnalysisConfig,
                            strategy: SynthesisStrategy,
                            analyzer: Analyzer,
                            time_limit: Optional[float] = None,
                            progress: Optional[Callable[[int], None]] = None,
                            template_time_limit: Optional[float] = None,
                            ) -> Candidates:
    """Searches the templates of ``strategy`` for EVT invariants of ``loop``.

    Returns the certified fixed points of the first template that has one with provably non-negative
    coefficients. Raises :class:`SynthesisBoundReached` if there is none among the templates (the
    template heuristic bounds their degree) or the search takes longer than ``time_limit`` seconds. A
    template that takes longer than ``template_time_limit`` seconds is skipped (only where the time limit
    can interrupt a solver call, see :func:`_time_limit`). ``progress`` is called with the number of
    templates tried so far."""
    logger.debug("Invariant Synthesis for loop %s with initial distribution %s.", loop, distribution)
    print(f"{Style.YELLOW}Invariant synthesis initiated...{Style.RESET}")
    if distribution.is_zero_dist():
        # Ψ(I) = ⟦body⟧(⟨I⟩_guard) is linear, so 0 is its least fixed point (e.g. an inner loop that is never
        # entered); the templates exclude the all-zero solution
        print(f"{Style.GREEN}Invariant: 0{Style.RESET}{Style.CLEARTOEND}")
        return [(distribution, True)]
    stats = {"tried": 0, "failed": 0, "skipped": 0}
    undecided: List[str] = []
    reason = "degree"
    try:
        with _time_limit(time_limit) as expired:
            # enumerate potential candidates given by a heuristic
            for evt_candidate in strategy.template_heuristics.generate():
                if expired():
                    raise _TimeLimitReached()
                print(f"{Style.YELLOW}Invariant candidate: {evt_candidate}{Style.RESET}{Style.CLEARTOEND}", end="\r")
                stats["tried"] += 1
                try:
                    with _time_limit(template_time_limit, _TemplateTimeLimitReached):
                        result, inv_found = _solutions(evt_candidate, loop, prog_info, distribution, config,
                                                       strategy, analyzer, stats, undecided)
                except _TemplateTimeLimitReached:
                    logger.info("Skipping the template %s after %s s.", evt_candidate, template_time_limit)
                    stats["skipped"] += 1
                    continue
                except BasePolynomialError as exc:
                    # e.g. HeuristicGCDFailed('no luck'): this template cannot be decided, try the next one
                    logger.info("SymPy failed on the template %s: %r", evt_candidate, exc)
                    stats["failed"] += 1
                    continue
                finally:
                    if progress is not None:
                        progress(stats["tried"])
                # Just return if we know that not all invariants are spurious.
                if inv_found:
                    return result
    except _TimeLimitReached:
        reason = "time_limit"

    print(Style.CLEARTOEND, end="")
    message = _bound_message(reason, strategy, time_limit, template_time_limit, stats, undecided)
    candidates = undecided[:MAX_REPORTED_CANDIDATES]
    details: Dict[str, Any] = {
        "type": "evt_synthesis",
        "verdict": "unknown",
        "reason": reason,
        "template": _template_kind(strategy),
        "max_degree": _max_degree(strategy),
        "effective_max_degree": _effective_degree(strategy),
        "time_limit": time_limit,
        "template_time_limit": template_time_limit,
        "templates_tried": stats["tried"],
        "templates_failed": stats["failed"],
        "templates_skipped": stats["skipped"],
        "unproven_fixed_points": len(undecided),
        "candidates": candidates,
    }
    messages = [f"Candidate EVT invariant (not certified, its coefficients could not be shown to be non-negative): "
                f"{candidate}" for candidate in candidates]
    if len(undecided) > len(candidates):
        messages.append(f"… {len(undecided) - len(candidates)} further candidate(s) omitted.")
    raise SynthesisBoundReached(message, details=details, messages=messages)
