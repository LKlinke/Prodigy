"""
While loops.

The handler asks the current :class:`~prodigy.analysis.session.AnalysisSession` which
:mod:`loop strategy <prodigy.analysis.loop_strategies>` to use for a loop and runs it.
The six historic strategies are implemented and registered here; in the interactive CLI
they show up in the same menu, with the same prompts, as before.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy
from lark.exceptions import LarkError

from prodigy.pgcl import Instr, parse_pgcl, Program, WhileInstr

from prodigy.analysis import evt_exploration
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants.heuristics.strategies import (DEFAULT_MAX_DEGREE, SynthesisStrategy,
                                                                  SynthesisStrategies)
from prodigy.analysis.evtinvariants.invariant_synthesis import (DEFAULT_SYNTHESIS_TIME_LIMIT,
                                                                DEFAULT_TEMPLATE_TIME_LIMIT, evt_invariant_synthesis)
from prodigy.analysis.evtinvariants.invariant_verification import FixedPointCheckFailed, evt_invariant_check
from prodigy.analysis.evtinvariants.series_identity import converges_on_unit_polydisc, formal_loop
from prodigy.analysis.exceptions import ForwardAnalysisError, VerificationError
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.distribution.equality import sympify_distribution
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.loop_strategies import (Bound, LoopOutcome, LoopStrategy, LoopTask, ParamSpec,
                                              get_loop_strategy, register_loop_strategy)
from prodigy.analysis.session import LoopDecision, current_session
from prodigy.distribution import Distribution, State
from prodigy.distribution.generating_function import SympyPGF
from prodigy.pgcl.pgcl_operations import cav_phi
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[
    [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
    Tuple[Distribution, Distribution]
]


class UnsupportedLoopError(ForwardAnalysisError):
    """The chosen strategy cannot handle this loop (e.g. EVT reasoning with conditioning)."""


class InvariantRefutedError(VerificationError):
    """
    A proposed invariant was not certified. ``details`` holds the evidence; its ``verdict`` is ``refuted``
    if the invariant is provably wrong and ``unknown`` if it could be neither proven nor refuted.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _mass(dist: Distribution) -> sympy.Expr:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    return sym(dist.get_probability_mass())


def _exit_part(dist: Distribution, cond) -> Distribution:
    """The part of ``dist`` that leaves a loop with guard ``cond``."""
    return dist - dist.filter(cond)


def _require_unconditioned(task: LoopTask) -> None:
    if not task.error_prob.is_zero_dist():
        raise UnsupportedLoopError(
            "Currently EVT reasoning does not support conditioning: the strategy "
            f"'{task.strategy.title}' cannot be used after an observe statement.")


def finite_mass(invariant: Any, variables: Sequence[str]) -> Tuple[Optional[bool], Optional[sympy.Expr]]:
    """Decides ``|I| < ∞`` for a generating function I with non-negative coefficients.

    Substitutes t for every variable; f(t) = I(t, …, t) has the same mass. For a rational f
    with f's denominator q(0) ≠ 0, Pringsheim's theorem says the radius of convergence is a
    singularity on the positive axis, so ``|I| < ∞`` iff q has no root in (0, 1]; then
    ``|I| = f(1)``. Returns (None, None) when this cannot be decided (parameters, non-rational).
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    try:
        expr = sym(invariant)
        t = sympy.Symbol("t__mass", positive=True)
        substitution = {s: t for s in expr.free_symbols if s.name in set(variables)}
        if any(s.name not in set(variables) for s in expr.free_symbols):
            return None, None
        f = sympy.cancel(sympy.together(expr.subs(substitution)))
        if not f.is_rational_function(t):
            return None, None
        _, denominator = sympy.fraction(f)
        poly = sympy.Poly(denominator, t)
        if poly.eval(0) == 0:
            return None, None
        if poly.count_roots(0, 1) > 0:
            return False, None
        return True, sympy.nsimplify(f.subs(t, 1))
    except Exception:  # pylint: disable=broad-except
        return None, None


def _positivity(config: ForwardAnalysisConfig, variables, invariant: Any) -> Optional[bool]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.lowerbounds.subinvariant import nonneg_by_construction
    # Sound criterion first: it covers polynomials and P / prod (c - R), for which the heuristic fails on
    # several variables (SymPy has no multivariate partial fraction decomposition).
    if nonneg_by_construction(str(invariant), [str(v) for v in variables]):
        return True
    try:
        strategy: SynthesisStrategy = SynthesisStrategies.from_config(config, variables)
        verdict = strategy.positivity_heuristics.is_positive(sympify_distribution(invariant))
        return verdict if verdict in (True, False) else None
    except Exception:  # pylint: disable=broad-except
        return None


def _evt_outcome_kind(scope_kind: str, positive: Optional[bool], finite: Optional[bool],
                      series: Optional[str] = None) -> Tuple[str, List[str]]:
    """Guarantee for a fixed point I of Ψ_G(H) = G + ⟦body⟧(⟨H⟩_B).

    ``scope_kind`` says how the body was analysed while checking I: exactly, or with inner
    loops over-approximated (then Ψ' ⪰ Ψ, so I = Ψ'(I) ⪰ Ψ(I) and Park still applies, but the
    finite-mass rule for the other direction does not).

    ``series`` names the reason why the equation, checked on closed forms, also holds for the power
    series of I (:mod:`prodigy.analysis.evtinvariants.series_identity`), or is None if there is none
    (#276). Finite mass is such a reason; without one, Park's lemma does not apply."""
    messages = []
    if scope_kind not in (Bound.EXACT, Bound.UPPER):
        messages.append("Loops inside the body were under-approximated while checking the invariant; "
                        "no guarantee.")
        return Bound.UNKNOWN, messages
    if positive is False:
        messages.append("The invariant has negative coefficients, so Park's lemma does not apply.")
        return Bound.UNKNOWN, messages
    if positive is None:
        messages.append("Non-negativity of the invariant's coefficients could not be established; "
                        "no guarantee.")
        return Bound.UNKNOWN, messages
    if series is None and finite is not True:
        messages.append("I = G + ⟦body⟧(⟨I⟩_B) holds for the closed form of I, but this does not prove it for the "
                        "power series: the series of I diverges inside the unit polydisc (or this could not be "
                        "decided), and the loop body evaluates I there, e.g. by marginalising a variable. I may be "
                        "no fixed point of the program's semantics; no guarantee.")
        return Bound.UNKNOWN, messages
    if scope_kind == Bound.UPPER:
        messages.append("Upper bound (Park's lemma): loops inside the body were over-approximated while "
                        "checking the invariant, so exactness cannot be certified.")
        return Bound.UPPER, messages
    if finite is True:
        messages.append("Exact: I is a non-negative fixed point of Ψ_G with finite mass. "
                        "Park's lemma gives ⪯, the finite-mass (lower induction) rule gives ⪰.")
        return Bound.EXACT, messages
    if finite is False:
        messages.append("Upper bound only: I has infinite mass and may be a spurious fixed point "
                        "(Park's lemma gives the result ⪯ ⟨I⟩_¬B).")
        return Bound.UPPER, messages
    messages.append("Upper bound (Park's lemma); finiteness of the mass of I could not be decided, "
                    "so exactness is not certified.")
    return Bound.UPPER, messages


# ---------------------------------------------------------------------------------------
# The six built-in strategies
# ---------------------------------------------------------------------------------------

def _strategy_invariant(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config = task.instruction, task.prog_info, task.config
    inv_src = task.param("invariant")
    # an invariant is written in the syntax of the program
    syntax = task.session.syntax
    try:
        inv_prog = parse_pgcl(inv_src, syntax)
    except (SyntaxError, LarkError) as exc:
        raise ValueError(f"The invariant is not a valid program in the {syntax} syntax "
                         f"(an invariant uses the syntax of the program): {exc}") from exc

    prog = Program(declarations=[],
                   variables=prog_info.variables,
                   constants=prog_info.constants,
                   parameters=prog_info.parameters,
                   instructions=[instruction],
                   functions=prog_info.functions)
    print(f"{Style.YELLOW}Verifying invariant...{Style.RESET}")

    # First we create the modified input program in order to fit the premise of Park's Lemma
    phi_inv = cav_phi(prog, inv_prog)

    with task.session.sub_analysis("invariant check") as scope:
        answer, result = check_equivalence(inv_prog, phi_inv, config, task.analyzer)

    certificate: Dict[str, Any] = {
        "type": "invariant",
        "rule": "Park",
        "claim": "Φ(I) = I for all inputs, hence ⟦while⟧ ⪯ ⟦I⟧",
        "invariant": inv_src,
    }
    if answer:
        assert isinstance(result, list)
        constraints = [{str(k): str(v) for k, v in solution.items()} for solution in result]
        certificate["verdict"] = "verified"
        certificate["constraints"] = constraints
        messages = []
        if len(result) == 0:
            print(Style.OKGREEN + "Invariant successfully validated!" + Style.RESET)
        else:
            print(
                f"{Style.OKGREEN}Invariant validated under the following constraints:{Style.RESET} {result}"
            )
            messages.append(f"The invariant holds under the constraints {constraints}; "
                            "the result still contains the parameters.")
        if config.show_intermediate_steps:
            print(Style.YELLOW +
                  "Compute the result using the invariant" +
                  Style.RESET)
        dist, error_prob = task.analyzer(inv_prog.instructions, prog_info, task.distribution, task.error_prob,
                                         config)
        if scope.kind == Bound.EXACT:
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; exact if the loop terminates almost surely.")
        elif scope.kind == Bound.UPPER:
            # an over-approximated body Φ' ⪰ Φ with Φ'(I) = I gives Φ(I) ⪯ I, so Park still applies
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; loops inside the body were over-approximated "
                            "during the check, which keeps the argument sound.")
        else:
            kind = Bound.UNKNOWN
            messages.append("Loops inside the body were under-approximated during the check; no guarantee.")
        return LoopOutcome(dist, error_prob, kind=kind, certificate=certificate, messages=messages)

    if answer is False:
        assert isinstance(result, State)
        print(
            f'{Style.OKRED}Invariant could not be verified.{Style.RESET} Counterexample: {result.valuations}'
        )
        certificate["verdict"] = "refuted"
        certificate["counterexample"] = {str(k): v for k, v in result.valuations.items()}
        raise InvariantRefutedError(
            f"Invariant could not be determined as such. Counterexample: {result.valuations}",
            details=certificate)

    assert answer is None
    assert isinstance(result, Distribution)
    print(
        f'{Style.OKRED}Could not determine whether invariant is valid.{Style.RESET} Phi(I) - I: {result}'
    )
    certificate["verdict"] = "unknown"
    certificate["difference"] = str(result)
    raise InvariantRefutedError("Could not determine whether invariant is valid.", details=certificate)


def _strategy_unroll(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations", max_iter)

    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    task.iterate(0, non_sat_part, inside=sat_part)
    fixpoint_after = None
    for i in range(max_iter):
        task.progress(i + 1, max_iter)
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        if iterated_non_sat == SympyPGF.zero(
        ) and iterated_sat == sat_part:
            print(
                f"\n{Style.OKGREEN}Terminated already after {i + 1} step(s)!{Style.RESET}"
            )
            fixpoint_after = i + 1
            break
        non_sat_part += iterated_non_sat
        sat_part = iterated_sat
        task.iterate(i + 1, non_sat_part, inside=sat_part)

    remaining = _mass(sat_part)
    certificate: Dict[str, Any] = {"type": "unrolling", "iterations": max_iter,
                                   "iterations_done": fixpoint_after or max_iter,
                                   "remaining_mass": str(remaining)}
    if fixpoint_after is not None:
        # One more iteration changes nothing: the mass still inside the loop never leaves it.
        certificate["fixpoint"] = True
        messages = [f"Exact: the iteration reached a fixed point after {fixpoint_after} step(s)."]
        if remaining != 0:
            messages.append(f"Mass {remaining} stays inside the loop forever (divergence).")
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=messages)
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {max_iter} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining,
                       certificate=certificate,
                       messages=[f"Lower bound after {max_iter} iterations; mass {remaining} is still "
                                 "inside the loop."])


def _strategy_mass_threshold(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    captured_probability_threshold = Fraction(task.param("threshold"))
    max_iterations = task.param("max_iterations")
    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    captured_part = non_sat_part + error_prob
    iterations = 0
    task.iterate(0, non_sat_part, inside=sat_part)
    while Fraction(captured_part.get_probability_mass()
                   ) < captured_probability_threshold:
        if max_iterations is not None and iterations >= max_iterations:
            task.note(f"Stopped after {iterations} iterations before reaching the threshold.")
            break
        logger.info("Collected %f of the desired mass", (float(
            (Fraction(captured_part.get_probability_mass()) /
             captured_probability_threshold)) * 100))
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        non_sat_part += iterated_non_sat
        captured_part = non_sat_part + error_prob
        sat_part = iterated_sat
        iterations += 1
        task.iterate(iterations, non_sat_part, inside=sat_part)
        task.progress(min(100, int(
            (Fraction(captured_part.get_probability_mass()) /
             captured_probability_threshold) * 100)), 100,
            f"{iterations} iterations")
    remaining = _mass(sat_part)
    certificate = {"type": "mass_threshold", "threshold": str(captured_probability_threshold),
                   "iterations": iterations, "captured_mass": str(captured_part.get_probability_mass()),
                   "remaining_mass": str(remaining)}
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {iterations} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound after {iterations} iterations; mass {remaining} is still "
                                 "inside the loop."])


def _strategy_evt_iterate(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    distribution, error_prob = task.distribution, task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations of EVT operator", max_iter)

    evt = distribution * 0
    mass_in = _mass(distribution)
    # iterates kept for the exploration (#324): H_N is no output distribution, but shows the expected visiting
    # times, from which a user (or a rational fit) guesses an EVT invariant
    samples = set(evt_exploration.sample_points(max_iter)) if task.session.collect_details else set()
    kept: List[Tuple[int, Distribution]] = []
    recent: List[Distribution] = []
    with task.session.sub_analysis("EVT iteration") as scope:
        for i in range(max_iter):
            logger.debug("Iteration %i", i)
            task.progress(i + 1, max_iter)
            evt = distribution + \
                  analyzer(instruction.body, prog_info, evt.filter(instruction.cond), error_prob, config)[0]
            # the terminating part of the i-th iterate, a lower bound unless inner loops were
            # over-approximated; the mass not accounted for is what has not left the loop yet
            task.iterate(i + 1, functools.partial(_exit_part, evt, instruction.cond),
                         residual=lambda exit_dist: mass_in - _mass(exit_dist),
                         kind=Bound.LOWER if scope.kind in (Bound.EXACT, Bound.LOWER) else Bound.UNKNOWN)
            if samples:
                recent = (recent + [evt])[-3:]
                if i + 1 in samples:
                    kept.append((i + 1, evt))
    result = evt - evt.filter(instruction.cond)
    remaining = sympy.simplify(mass_in - _mass(result))
    certificate: Dict[str, Any] = {"type": "evt_iteration", "iterations": max_iter, "remaining_mass": str(remaining),
                                   "evt": str(evt)}
    if kept:
        try:
            certificate["exploration"] = evt_exploration.exploration(
                kept, [str(v) for v in prog_info.program.variables.keys()], task.session.syntax, recent)
        except Exception as exc:  # pylint: disable=broad-except  # the exploration is optional
            logger.debug("EVT exploration failed: %s", exc)
    if scope.kind not in (Bound.EXACT, Bound.LOWER):
        return LoopOutcome(result, error_prob, kind=Bound.UNKNOWN, certificate=certificate,
                           messages=["Loops inside the body were over-approximated; no guarantee."])
    if remaining == 0:
        return LoopOutcome(result, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: all mass has left the loop within {max_iter} iterations."])
    return LoopOutcome(result, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound from {max_iter} iterations of the EVT operator; "
                                 f"mass {remaining} is not accounted for. The iterates H_N are expected "
                                 "visiting times, not the output distribution."])


def _series_identity(invariant: Distribution, variables: Sequence[str], loop: WhileInstr,
                     finite: Optional[bool]) -> Optional[str]:
    """Why the fixed-point equation of the non-negative closed form I also holds for its power series, or
    None (#276; the criteria and their proofs are in :mod:`prodigy.analysis.evtinvariants.series_identity`)."""
    if finite is True:
        return "finite mass"
    if converges_on_unit_polydisc(str(invariant), variables) is True:
        return "converges on the open unit polydisc"
    if formal_loop(loop, variables):
        return "formal loop body"
    return None


def _evt_certificate(kind_name: str, invariant: Distribution, variables, config, scope_kind: str,
                     loop: WhileInstr, loop_cond: str) -> Tuple[str, Dict[str, Any], List[str]]:
    positive = _positivity(config, variables, invariant)
    finite, mass = finite_mass(str(invariant), list(variables)) if positive else (None, None)
    series = _series_identity(invariant, [str(v) for v in variables], loop, finite) if positive else None
    kind, messages = _evt_outcome_kind(scope_kind, positive, finite, series)
    certificate = {
        "type": kind_name,
        "rule": "Park + finite-mass lower induction" if kind == Bound.EXACT else "Park",
        "equation": f"I = G + ⟦body⟧(⟨I⟩_({loop_cond}))",
        "invariant": str(invariant),
        "nonnegative": positive,
        "finite_mass": finite,
        "mass": str(mass) if mass is not None else None,
        "series_identity": series,
    }
    return kind, certificate, messages


def _strategy_evt_invariant(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    variables = list(prog_info.program.variables.keys())
    evt_inv = config.factory.from_expr(task.param("invariant"), *variables)
    try:
        with task.session.sub_analysis("EVT invariant check") as scope:
            solution, full_invariant, solutions = evt_invariant_check(instruction, prog_info, task.distribution,
                                                                      evt_inv, config, analyzer)
    except FixedPointCheckFailed as exc:
        message = f"The EVT invariant {evt_inv} is refuted" if exc.verdict == "refuted" \
            else f"The EVT invariant {evt_inv} is not proven (it may still be correct)"
        raise InvariantRefutedError(message, details={"type": "evt_invariant", "invariant": str(evt_inv),
                                                      "verdict": exc.verdict, "difference": exc.difference,
                                                      "reason": str(exc)}) from exc
    except VerificationError as exc:
        raise InvariantRefutedError(f"Could not validate the EVT invariant {evt_inv}",
                                    details={"type": "evt_invariant", "invariant": str(evt_inv),
                                             "verdict": "unknown", "reason": str(exc)}) from exc
    print(f"{Style.OKGREEN}Continuing with {solution}{Style.RESET}")
    kind, certificate, messages = _evt_certificate("evt_invariant", full_invariant, variables, config,
                                                   scope.kind, instruction, task.session.spell(instruction.cond))
    certificate["candidate"] = str(evt_inv)
    certificate["solutions"] = solutions
    return LoopOutcome(solution, task.error_prob, kind=kind, certificate=certificate, messages=messages)


def _strategy_evt_synthesis(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    logger.debug("Using invariant synthesis.")

    # Build the strategy.
    max_degree = task.param("max_degree")
    strategy: SynthesisStrategy = SynthesisStrategies.from_config(config, prog_info.variables.keys(), max_degree)

    # generate the invariants using the strategy; ends with SynthesisBoundReached (verdict unknown) if
    # there is none up to the degree bound or within the time limit
    with task.session.sub_analysis("EVT invariant synthesis") as scope:
        invariants = evt_invariant_synthesis(
            instruction, prog_info, task.distribution, config, strategy, analyzer,
            time_limit=task.param("time_limit"), template_time_limit=task.param("template_time_limit"),
            progress=lambda tried: task.progress(tried, None, f"{tried} template(s) tried"))

    def _sorting_key(elem):
        return 1 if elem[1] else 0

    # certified candidates first, so that the default candidate 0 is a certified one (#337)
    invariants = sorted(invariants, key=_sorting_key, reverse=True)

    # give associated distributions:
    distributions = [inv - inv.filter(instruction.cond) for inv, inv_type in invariants]
    print("Continuing with the following distributions is possible:")
    for i, d in enumerate(distributions):
        print(f"{i + 1}.\t{d}")

    # Choose one solution
    index = task.session.provider.choose_candidate(task, [str(d) for d in distributions])
    print(f"Continue with distribution: {Style.CYAN}{distributions[index]}{Style.RESET}")

    variables = list(prog_info.program.variables.keys())
    candidates = []
    for i, (inv, positive) in enumerate(invariants):
        finite, mass = finite_mass(str(inv), variables) if positive else (None, None)
        candidates.append({"index": i, "invariant": str(inv), "result": str(distributions[i]),
                           "nonnegative": positive, "finite_mass": finite,
                           "mass": str(mass) if mass is not None else None})
    chosen, positive = invariants[index]
    kind, certificate, messages = _evt_certificate("evt_synthesis", chosen, variables, config, scope.kind,
                                                   instruction, task.session.spell(instruction.cond))
    certificate["chosen"] = index
    certificate["candidates"] = candidates
    certificate["synthesis_strategy"] = config.strategy.name
    certificate["max_degree"] = max_degree
    return LoopOutcome(distributions[index], task.error_prob, kind=kind, certificate=certificate,
                       messages=messages)


_INVARIANT_PARAMS = (
    ParamSpec("invariant", "pgcl", "Invariant (loop-free pGCL program)",
              prompt="Invariant file:\t",
              help="A loop-free program I with the loop's variables; checked against one unrolling "
                   "of the loop (Park's lemma)."),
    ParamSpec("invariant_file", "file", "Invariant file", prompt="Invariant file:\t",
              alias_of="invariant", help="Path of a file containing the invariant (CLI only)."),
)

register_loop_strategy(LoopStrategy(
    name="invariant",
    title="Check invariant",
    group="exact",
    outcome="upper bound",
    summary="Exact if the loop terminates almost surely, otherwise an upper bound; you provide a loop-free "
            "program that behaves like the loop; one check, usually fast.",
    legacy_key="1",
    soundness=Bound.UPPER,
    description="Checks that a loop-free program I satisfies I ≡ if (guard) { body; I } on all "
                "inputs and continues with I. By Park's lemma the result over-approximates the "
                "loop; it is exact if the loop terminates almost surely.",
    params=_INVARIANT_PARAMS,
    run=_strategy_invariant,
))

register_loop_strategy(LoopStrategy(
    name="unroll",
    title="Unroll N iterations",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound whose missing mass makes every query an interval; you provide N; the cost grows "
            "with N.",
    legacy_key="2",
    soundness=Bound.LOWER,
    description="Executes at most N iterations. Mass that is still inside the loop afterwards is "
                "reported as residual mass, which turns every query into a sound interval.",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Specify a maximum iteration limit: ",
                      default=50, minimum=0, help="Maximal number of loop iterations."),),
    run=_strategy_unroll,
))

register_loop_strategy(LoopStrategy(
    name="mass_threshold",
    title="Iterate until mass captured",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound that captures at least the given mass; you provide the mass, e.g. 0.99; may not "
            "stop if the loop rarely terminates.",
    legacy_key="3",
    soundness=Bound.LOWER,
    description="Iterates until the terminated (or conditioned-away) mass reaches the threshold. "
                "Does not terminate if the loop does not terminate with at least this probability.",
    params=(ParamSpec("threshold", "probability", "Threshold", prompt="Enter the probability threshold: ",
                      default="0.99", help="Probability mass to capture, e.g. 0.999 or 999/1000."),
            ParamSpec("max_iterations", "int", "Iteration cap", default=None, minimum=1, optional=True,
                      help="Optional safety cap on the number of iterations.")),
    run=_strategy_mass_threshold,
))

register_loop_strategy(LoopStrategy(
    name="evt_iterate",
    title="Expected visiting times H_N",
    group="explore",
    outcome="exploration",
    summary="Shows the iterates H_N of the expected visiting times (not the output distribution) to help you "
            "guess an EVT invariant; you provide N; the result is only a lower bound.",
    legacy_key="4",
    soundness=Bound.LOWER,
    description="Iterates the expected-visiting-time operator Ψ_G(H) = G + ⟦body⟧(⟨H⟩_guard) N times "
                "from 0. The coefficient of a state in H_N is the expected number of times the loop head "
                "is reached in that state within N iterations; the GUI shows how they grow with N and a "
                "guessed EVT invariant to check with Check EVT invariant. The loop's result is the "
                "terminating part of H_N, a lower bound (the mass that left the loop within N iterations).",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Enter the number of iterations: ",
                      default=50, minimum=0, help="Number of applications of the EVT operator."),),
    run=_strategy_evt_iterate,
    requires_unconditioned=True,
))

register_loop_strategy(LoopStrategy(
    name="evt_invariant",
    title="Check EVT invariant",
    group="exact",
    outcome="exact or upper bound",
    summary="Exact if your invariant has finite mass, otherwise an upper bound; you provide a generating "
            "function of the expected visiting times; one check, usually fast.",
    legacy_key="5",
    soundness=Bound.UPPER,
    description="Checks a generating function I against I = G + ⟦body⟧(⟨I⟩_guard), solving for its "
                "parameters. With non-negative coefficients the result is an upper bound (Park), and "
                "exact if I has finite mass.",
    params=(ParamSpec("invariant", "expression", "EVT invariant I", prompt="Enter EVT invariant: ",
                      help="Generating function over the program variables, e.g. 1/(1-c/2)."),),
    run=_strategy_evt_invariant,
    requires_unconditioned=True,
))

register_loop_strategy(LoopStrategy(
    name="evt_synthesis",
    title="Synthesise EVT invariant",
    group="exact",
    outcome="exact or upper bound",
    summary="Like Check EVT invariant, but Prodigy searches the invariant; you provide nothing; seconds up to "
            "the time limit (60 s), and no result if none is found.",
    legacy_key="6",
    soundness=Bound.UPPER,
    description="Searches rational-function templates up to a maximal degree for a fixed point of the "
                "EVT operator and checks non-negativity heuristically (synthesis strategy from the "
                "settings). Without an invariant within the degree bound or the time limit the result "
                "is unknown.",
    params=(ParamSpec("candidate", "int", "Candidate", default=None, minimum=0, optional=True,
                      help="Index of the synthesised candidate to continue with (default: the first "
                           "certified one)."),
            ParamSpec("max_degree", "int", "Maximal template degree", default=DEFAULT_MAX_DEGREE, minimum=0,
                      optional=True,
                      help="Templates N/D up to this degree of N in each variable (D has at most N's "
                           "degrees in total)."),
            ParamSpec("time_limit", "int", "Time limit (seconds)", default=DEFAULT_SYNTHESIS_TIME_LIMIT,
                      minimum=1, optional=True,
                      help="The synthesis stops after this time, also inside a solver call."),
            ParamSpec("template_time_limit", "int", "Time limit per template (seconds)",
                      default=DEFAULT_TEMPLATE_TIME_LIMIT, minimum=1, optional=True,
                      help="A template whose equations are not solved within this time is skipped.")),
    run=_strategy_evt_synthesis,
    requires_unconditioned=True,
))



# ---------------------------------------------------------------------------------------
# Lower bounds without a termination assumption (prodigy.analysis.lowerbounds)
# ---------------------------------------------------------------------------------------

def _exact(value: Any) -> Any:
    """Fraction (or ∞) from the lower-bound package -> sympy, ``None`` for ∞."""
    if value is None or value == float("inf"):
        return None
    if isinstance(value, Fraction):
        return sympy.Rational(value.numerator, value.denominator)
    return sympy.nsimplify(value)


def _strategy_evt_subinvariant(task: LoopTask) -> LoopOutcome:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.lowerbounds.subinvariant import evt_subinvariant_verification
    config = task.config
    variables = [str(v) for v in task.prog_info.program.variables.keys()]
    subinvariant = config.factory.from_expr(task.param("subinvariant"), *variables)
    details: Dict[str, Any] = {}
    lower = evt_subinvariant_verification(task.instruction, task.prog_info, task.distribution, subinvariant, config,
                                          task.analyzer, details=details)
    # The output lies coefficient-wise above `lower` (lower induction). The mass it misses is at most
    # |Phi(nu)| - |nu|, and at most the mass that entered the loop minus |lower|.
    residual = _mass(task.distribution) - _mass(lower)
    gap = _exact(details.get("gap"))
    if gap is not None:
        residual = sympy.Min(residual, gap)
    certificate = {
        "type": "evt_subinvariant",
        "subinvariant": str(subinvariant),
        "mass": str(details.get("mass")),
        "nonnegative": details.get("nonneg_nu"),
        "subinvariance": details.get("nonneg_gap"),
        "missing_mass": str(residual),
    }
    # |Phi(nu)| = |nu|: nu is the occupation measure itself, so the result is exact
    kind = Bound.EXACT if residual == 0 else Bound.LOWER
    return LoopOutcome(lower, task.error_prob, kind=kind, residual=residual, certificate=certificate)


def _strategy_evt_sandwich(task: LoopTask) -> LoopOutcome:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.lowerbounds.egcalc import directed_decimal
    from prodigy.analysis.lowerbounds.problem import problem_from_loop, queries_after
    from prodigy.analysis.lowerbounds.sandwich import evt_sandwich
    _require_unconditioned(task)
    variables = [str(v) for v in task.prog_info.program.variables.keys()]
    problem = problem_from_loop(task.instruction, variables, str(task.distribution))
    problem.queries = queries_after(task.prog_info.program, task.instruction)
    result = evt_sandwich(problem)
    if result.lower is None:
        raise VerificationError("No finite-mass EVT sub-invariant found for the lower bound.")
    lower = task.config.factory.from_expr(result.lower.calc.to_expr(result.lower.exit_numerator), *variables)
    mass_low, mass_high = (_exact(m) for m in result.mass_bounds)
    if mass_low is not None and mass_high is not None:
        residual = mass_high - mass_low
    else:
        residual = _mass(task.distribution) - _mass(lower)
    low, high = result.mass_bounds
    certificate: Dict[str, Any] = {
        "type": "evt_sandwich",
        # rounded outwards, so the interval stays sound; the exact values follow
        "termination_probability": [directed_decimal(low, False), directed_decimal(high, True)],
        "termination_probability_exact": [str(mass_low), str(mass_high) if mass_high is not None else None],
    }
    for name, cert in (("lower", result.lower), ("upper", result.upper)):
        summary = None
        if cert is not None:
            summary = cert.to_json()
            summary.pop("numerator", None)  # the generating function says the same
            summary["exit_mass_rounded"] = directed_decimal(cert.exit_mass, name == "upper")
            summary["check"] = cert.extra.get("check")
        certificate[name] = summary
    query_bounds = {(bound.query.kind, bound.query.text): (_exact(bound.lower), _exact(bound.upper))
                    for bound in result.bounds}
    # the values on the lower bound itself, computed from the certificate (no epilogue: `lower` is its exit part)
    query_values = {}
    if problem.epilogue == ("seq", []):
        query_values = {(bound.query.kind, bound.query.text): _exact(bound.value)
                        for bound in result.bounds if bound.value is not None}
    return LoopOutcome(lower, task.error_prob, kind=Bound.LOWER, residual=residual, certificate=certificate,
                       query_bounds=query_bounds, query_values=query_values)


register_loop_strategy(LoopStrategy(
    name="evt_subinvariant",
    title="Check EVT sub-invariant",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound without termination assumption; you provide a generating function with finite "
            "mass below one more iteration of itself; one check, usually fast.",
    legacy_key="7",
    soundness=Bound.LOWER,
    description="Checks that a generating function nu with finite mass satisfies nu ≤ Φ(nu) "
                "coefficient-wise. By lower induction [¬guard]·nu is then a lower bound on the output, "
                "without any termination assumption; the missing mass is at most |Φ(nu)| − |nu|. "
                "Not for loops with observe in the body.",
    params=(ParamSpec("subinvariant", "expression", "EVT sub-invariant nu", prompt="Enter EVT sub-invariant: ",
                      help="Generating function over the program variables with finite mass, "
                           "e.g. (1 + d/3)/(1 - 3/5*c)."),),
    run=_strategy_evt_subinvariant,
))

register_loop_strategy(LoopStrategy(
    name="evt_sandwich",
    title="EVT sandwich",
    group="bounds",
    outcome="lower and upper bounds",
    summary="Lower and upper bounds for the queries right after the loop, both searched and checked by "
            "Prodigy; you provide nothing; needs a rational input and no observe.",
    legacy_key="8",
    soundness=Bound.LOWER,
    description="Synthesises a finite-mass EVT sub-invariant (lower bound) and an EVT super-invariant "
                "(upper bound, Park) with eventually geometric templates by linear programming, checks "
                "both exactly, and bounds the ?Pr and ?Ex queries right after the loop from both sides. "
                "Needs a rational input distribution and no observe.",
    run=_strategy_evt_sandwich,
    requires_unconditioned=True,
))


# ---------------------------------------------------------------------------------------
# Candidates from a language model (prodigy.analysis.ai_candidates)
# ---------------------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _larger(mass: sympy.Expr, best: sympy.Expr) -> bool:
    """``mass > best``, if this is proven (masses with parameters may be incomparable)."""
    return sympy.simplify(mass - best).is_positive is True


@dataclasses.dataclass
class _BestCandidate:
    """The proven candidate with the largest lower bound so far (strategy ai_invariant)."""
    outcome: LoopOutcome
    entry: Dict[str, Any]
    """Its attempt in the log."""
    mass: sympy.Expr
    """Mass of its lower bound."""


def _strategy_ai_invariant(task: LoopTask) -> LoopOutcome:
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements
    from prodigy.analysis import ai_candidates
    form = task.param("form")
    attempts = task.param("attempts")
    provider = ai_candidates.get_provider()
    target = get_loop_strategy(form)
    program = task.prog_info.program
    variables = [str(v) for v in program.variables.keys()]
    parameters = [str(p) for p in program.parameters.keys()]
    earlier: List[Tuple[str, str]] = []  # candidates that did not help, with Prodigy's verdict, for the model
    log: List[Dict[str, Any]] = []  # every attempt, for the certificate and the GUI
    best: Optional[_BestCandidate] = None
    for attempt in range(1, attempts + 1):
        proven = None
        if best is not None:
            proven = ai_candidates.Proven(best.entry["candidate"], str(best.mass), str(best.outcome.residual))
        task.progress(attempt - 1, attempts, f"Asking {provider.model}"
                      + (" for a larger sub-invariant" if proven else ""))
        request = ai_candidates.prompt(form, str(program), str(task.instruction), variables, parameters,
                                       str(task.distribution), earlier, proven)
        try:
            answer = provider.ask(ai_candidates.SYSTEM_PROMPT, request)
        except ai_candidates.AIRequestError as exc:
            if best is None:
                raise
            # a proven sub-invariant is in hand; a failed request for a larger one does not discard it
            log.append({"attempt": attempt, "answer": "", "candidate": None, "verdict": "error", "reason": str(exc)})
            print(f"Request {attempt} failed, keeping the sub-invariant of attempt {best.entry['attempt']}: {exc}")
            break
        entry: Dict[str, Any] = {"attempt": attempt, "answer": answer.strip()[-2000:], "candidate": None,
                                 **provider.diagnostics}
        log.append(entry)
        print(f"Answer {attempt} of {provider.model}:\n{answer.strip()[-2000:] or '(no text)'}")
        try:
            candidate = ai_candidates.candidate_from_answer(answer, variables + parameters)
        except ai_candidates.CandidateError as exc:
            entry["verdict"], entry["reason"] = "no_candidate", str(exc)
            if not answer.strip() and provider.diagnostics:
                entry["reason"] += (f" The model stopped with {provider.diagnostics.get('stop_reason')} "
                                    f"(content: {provider.diagnostics.get('content', '?')}).")
            lines = answer.strip().splitlines()
            earlier.append((lines[-1][:200] if lines else "(empty answer)", entry["reason"]))
            print(f"Rejected: {entry['reason']}")
            continue
        entry["candidate"] = candidate
        # The candidate goes through the same check as one typed in by hand.
        attempt_task = dataclasses.replace(task, strategy=target,
                                           decision=LoopDecision(form, {ai_candidates.TARGET_PARAMS[form]: candidate}))
        try:
            outcome = target.run(attempt_task)
        except Exception as exc:  # pylint: disable=broad-except  # any failed check rejects the candidate
            message = str(exc) or type(exc).__name__
            details = _json_safe(getattr(exc, "details", None) or {})
            entry["verdict"] = details.get("verdict") if details.get("verdict") in ("refuted", "unknown") \
                else "rejected"
            entry["reason"] = ai_candidates.describe_rejection(message, details)
            if details:
                entry["details"] = details
            # the model gets the full details of the check
            earlier.append((candidate, f"{entry['reason']} {message}"
                            + (f" Details: {json.dumps(details)[:1000]}" if details else "")))
            print(f"Rejected {candidate}: {entry['reason']}")
            continue
        entry["verdict"] = "proven"
        if form != "evt_subinvariant" or outcome.kind == Bound.EXACT:
            best = _BestCandidate(outcome, entry, sympy.S(0))
            break  # nothing to improve: exact, or an invariant
        mass = _mass(outcome.distribution)
        entry["mass"], entry["missing_mass"] = str(mass), str(outcome.residual)
        if best is not None and not _larger(mass, best.mass):
            entry["verdict"] = "not_larger"
            entry["reason"] = (f"Proven, but its lower bound has mass {mass}, not more than {best.mass} "
                               f"(attempt {best.entry['attempt']}).")
            earlier.append((candidate, entry["reason"]))
            print(f"Not larger: {candidate}")
            continue
        best = _BestCandidate(outcome, entry, mass)
        print(f"Proven sub-invariant {candidate}: lower bound with mass {mass}, missing mass {outcome.residual}")
    if best is None:
        raise InvariantRefutedError(
            f"None of {attempts} answers of {provider.model} gave a candidate that passed Prodigy's check. "
            f"Last reason: {log[-1].get('reason', 'none')[:300] if log else 'none'}",
            details={"type": "ai_attempts", "provider": provider.name, "model": provider.model, "attempts": log})
    outcome, chosen = best.outcome, best.entry
    proofs = sum(1 for entry in log if entry.get("verdict") in ("proven", "not_larger"))
    outcome.certificate["ai"] = {"provider": provider.name, "model": provider.model, "attempts": chosen["attempt"],
                                 "asked": len(log), "candidate": chosen["candidate"], "answer": chosen["answer"],
                                 "history": [entry for entry in log if entry is not chosen]}
    outcome.messages.append(f"Candidate proposed by {provider.model} (attempt {chosen['attempt']} of {len(log)}), "
                            "checked by Prodigy"
                            + (f"; the largest of {proofs} proven sub-invariants." if proofs > 1 else "."))
    return outcome


register_loop_strategy(LoopStrategy(
    name="ai_invariant",
    title="EVT invariant from a language model",
    group="exact",
    outcome="exact or a bound",
    summary="Like Check EVT invariant or Check EVT sub-invariant, with candidates from a language model that "
            "Prodigy checks; you provide an API key; one model request per attempt.",
    legacy_key="9",
    soundness=Bound.UPPER,
    description="A language model proposes an EVT invariant or a finite-mass sub-invariant. Prodigy checks "
                "every candidate exactly, like one typed in by hand, and tells the model why a candidate "
                "was rejected. After a proven sub-invariant, it asks for a larger one while attempts are left "
                "and keeps the one with the largest lower bound. Needs an API key in the environment of "
                "prodigy (see docs/gui.md).",
    params=(ParamSpec("form", "choice", "Kind of candidate", prompt="Kind (evt_invariant or evt_subinvariant): ",
                      default="evt_invariant", choices=("evt_invariant", "evt_subinvariant"),
                      help="evt_invariant: expected visiting times (exact with finite mass, otherwise an upper "
                           "bound); evt_subinvariant: a lower bound without termination assumption."),
            ParamSpec("attempts", "int", "Attempts", prompt="Number of attempts: ", default=3, minimum=1,
                      help="How often the model is asked in total: again after a rejected candidate, and "
                           "after a proven sub-invariant for a larger one.")),
    run=_strategy_ai_invariant,
))

# ---------------------------------------------------------------------------------------
# Automatic: a short synthesis, otherwise unrolling (the GUI's default, #324)
# ---------------------------------------------------------------------------------------

AUTO_MAX_DEGREE = 2
AUTO_TIME_LIMIT = 5
AUTO_TEMPLATE_TIME_LIMIT = 2


def _strategy_auto(task: LoopTask) -> LoopOutcome:
    """Synthesis first, accepted only if certified exact; otherwise unrolling. Each part is the
    unchanged strategy with its own checks, so the outcome is exactly as sound as theirs."""
    reason = ""
    if not task.error_prob.is_zero_dist():
        reason = "an observe statement comes before the loop, which EVT reasoning does not support"
    else:
        synthesis = get_loop_strategy("evt_synthesis")
        params = {"max_degree": task.param("max_degree"), "time_limit": task.param("time_limit"),
                  "template_time_limit": AUTO_TEMPLATE_TIME_LIMIT}
        try:
            outcome = synthesis.run(dataclasses.replace(task, strategy=synthesis,
                                                        decision=LoopDecision("evt_synthesis", params)))
        except Exception as exc:  # pylint: disable=broad-except  # any failure falls back to unrolling
            reason = (str(exc) or type(exc).__name__).splitlines()[0][:300]
        else:
            if outcome.kind == Bound.EXACT:
                outcome.certificate["automatic"] = {"chosen": "evt_synthesis"}
                outcome.messages.insert(0, "Automatic: an EVT invariant was synthesised and certified exact.")
                return outcome
            reason = (f"the synthesised invariant {outcome.certificate.get('invariant')} gives only "
                      f"{'an upper bound' if outcome.kind == Bound.UPPER else 'no guarantee'}")
    unroll = get_loop_strategy("unroll")
    outcome = unroll.run(dataclasses.replace(task, strategy=unroll, decision=LoopDecision(
        "unroll", {"iterations": task.param("iterations")})))
    outcome.certificate["automatic"] = {"chosen": "unroll", "synthesis": reason}
    outcome.messages.insert(0, f"Automatic: unrolled {task.param('iterations')} iterations, since no exact "
                               f"EVT invariant was found ({reason}).")
    return outcome


register_loop_strategy(LoopStrategy(
    name="auto",
    title="Automatic",
    group="automatic",
    outcome="exact or lower bound",
    summary="Exact if Prodigy finds an EVT invariant within a few seconds, otherwise a lower bound from "
            "unrolling N iterations; you provide nothing.",
    soundness=Bound.LOWER,
    description="Runs Synthesise EVT invariant with small templates (degree ≤ 2) and a short time limit and "
                "keeps its result only if it is certified exact. Otherwise (no invariant found, only an "
                "upper bound, an observe statement before the loop, or any failure) it unrolls the loop N "
                "times, which gives a lower bound whose missing mass turns every query into an interval. "
                "Both parts are the unchanged strategies with their own checks.",
    params=(ParamSpec("iterations", "int", "Iterations N", default=50, minimum=0,
                      prompt="Specify a maximum iteration limit: ",
                      help="Loop iterations to unroll when the synthesis finds no exact invariant."),
            ParamSpec("max_degree", "int", "Maximal template degree", default=AUTO_MAX_DEGREE, minimum=0,
                      optional=True, help="Templates of the synthesis up to this degree in each variable."),
            ParamSpec("time_limit", "int", "Synthesis time limit (seconds)", default=AUTO_TIME_LIMIT, minimum=1,
                      optional=True, help="How long to search for an invariant before unrolling.")),
    run=_strategy_auto,
))

# ---------------------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------------------

class WhileHandler(InstructionHandler):

    @staticmethod
    def _run(strategy_name: Optional[str], instruction: WhileInstr, prog_info: ProgramInfo,
             distribution: Distribution, error_prob: Distribution, config: ForwardAnalysisConfig,
             analyzer: Analyzer) -> Tuple[Distribution, Distribution]:
        session = current_session()
        with session.activate():
            loop = session.loops.lookup(instruction)
            decision = session.decide_loop(loop, instruction) if strategy_name is None \
                else LoopDecision(strategy_name)
            strategy = get_loop_strategy(decision.strategy)
            task = LoopTask(instruction=instruction, prog_info=prog_info, distribution=distribution,
                            error_prob=error_prob, config=config, analyzer=analyzer, session=session,
                            loop=loop, decision=decision, strategy=strategy)
            outcome = session.run_loop(task)
        return outcome.distribution, outcome.error_prob

    # The historic entry points, kept for callers that use them directly.

    @staticmethod
    def _analyze_with_invariant(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("invariant", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _compute_iterations(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("unroll", instruction, prog_info, distribution, error_prob, config, analyzer)

    @staticmethod
    def _compute_until_threshold(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("mass_threshold", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _approx_expected_visiting_times(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_iterate", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _evt_invariant(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_invariant", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _evt_invariant_synthesis(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_synthesis", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]],
            prog_info: ProgramInfo,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Analyzer
    ) -> tuple[Distribution, Distribution]:

        instruction = _assume(instruction, WhileInstr, 'WhileHandler')
        return WhileHandler._run(None, instruction, prog_info, distribution, error_prob, config, analyzer)
