"""
The strategies with exact EVT invariants: ``evt_invariant`` (check a given invariant) and ``evt_synthesis``
(synthesise one), with the side conditions that make the fixed point exact (non-negative coefficients, finite mass)
or an upper bound (the series identity, Park's lemma).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy

from prodigy.language import WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.strategies import (DEFAULT_MAX_DEGREE, SynthesisStrategy,
                                                                  SynthesisStrategies)
from prodigy.analysis.evtinvariants.invariant_synthesis import (DEFAULT_SYNTHESIS_TIME_LIMIT,
                                                                DEFAULT_TEMPLATE_TIME_LIMIT, evt_invariant_synthesis)
from prodigy.analysis.evtinvariants.invariant_verification import FixedPointCheckFailed, evt_invariant_check
from prodigy.analysis.evtinvariants.exp_polynomial import (entire_exp_polynomial, exp_polynomial_nonneg,
                                                           exp_polynomial_value_at_one, independent_factors)
from prodigy.analysis.evtinvariants.log_polynomial import log_form_mass, log_form_nonneg
from prodigy.analysis.evtinvariants.series_identity import converges_on_unit_polydisc, formal_loop
from prodigy.analysis.exceptions import VerificationError
from prodigy.distribution.equality import sympify_distribution, sympify_engine_string
from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup
from prodigy.analysis.loops.common import InvariantLike, InvariantRefutedError, _require_unconditioned


logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _names_as_symbols(invariant: InvariantLike, variables: Sequence[str]) -> sympy.Expr:
    """``invariant`` as a SymPy expression in which the variables are symbols, also when they are named like a
    function (``sum``, #321)."""
    if isinstance(invariant, sympy.Basic):
        return invariant
    if isinstance(invariant, Distribution):
        return sympify_distribution(invariant)
    names = {str(v) for v in variables}
    if hasattr(invariant, "get_parameters"):
        names |= set(invariant.get_parameters())
    return sympify_engine_string(str(invariant), names)


def finite_mass(invariant: InvariantLike, variables: Sequence[str]) -> Tuple[Optional[bool], Optional[sympy.Expr]]:
    """Decides ``|I| < ∞`` for a generating function I with non-negative coefficients.

    Substitutes t for every variable; f(t) = I(t, …, t) has the same mass. For a rational f
    with f's denominator q(0) ≠ 0, Pringsheim's theorem says the radius of convergence is a
    singularity on the positive axis, so ``|I| < ∞`` iff q has no root in (0, 1]; then
    ``|I| = f(1)``. An exponential polynomial ``A·exp(c·k) + B`` (#323) is entire, so its mass is
    ``I(1, …, 1)``; for ``P + Q·log(1 - c·k)`` see :func:`log_form_mass`. A product of factors over disjoint
    variables has the product of their masses. Returns (None, None) when this cannot be decided (parameters,
    other forms).
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import sym
    try:
        expr = sym(invariant)
        factors = independent_factors(_names_as_symbols(invariant, variables), list(variables))
        if factors is not None:
            masses = [finite_mass(factor, variables) for factor in factors]
            if all(finite is True for finite, _ in masses):
                return True, sympy.Mul(*(mass for _, mass in masses))
            # a non-zero factor with non-negative coefficients has positive mass
            if any(finite is False for finite, _ in masses) and all(finite is not None for finite, _ in masses):
                return False, None
        if entire_exp_polynomial(expr, list(variables)):
            return True, exp_polynomial_value_at_one(expr, list(variables))
        logarithmic = log_form_mass(expr, list(variables))
        if logarithmic is not None:
            return logarithmic
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


def _exp_polynomial_positivity(variables: Sequence[str], invariant: InvariantLike) -> Optional[bool]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import sym
    try:
        expr, names = sym(invariant), [str(v) for v in variables]
        verdict = exp_polynomial_nonneg(expr, names)
        return verdict if verdict is not None else log_form_nonneg(expr, names)
    except Exception:  # pylint: disable=broad-except
        return None


def _positivity(config: ForwardAnalysisConfig, variables: Sequence[str], invariant: InvariantLike) -> Optional[bool]:
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.coefficient_signs import nonneg_by_construction
    # Sound criterion first: it covers polynomials and P / prod (c - R), for which the heuristic fails on
    # several variables (SymPy has no multivariate partial fraction decomposition).
    if nonneg_by_construction(str(invariant), [str(v) for v in variables]):
        return True
    # A·exp(c·k) + B with polynomials A, B, and P + Q·log(1 - c·k): decided coefficient-wise (#323)
    verdict = _exp_polynomial_positivity(variables, invariant)
    if verdict is not None:
        return verdict
    # a product of factors over disjoint variables: its coefficients are products of theirs
    try:
        factors = independent_factors(_names_as_symbols(invariant, variables), [str(v) for v in variables])
    except Exception:  # pylint: disable=broad-except
        factors = None
    if factors is not None and all(_positivity(config, variables, factor) is True for factor in factors):
        return True
    if not isinstance(invariant, Distribution):
        return None  # the heuristics read a distribution (a text or a SymPy factor is not one)
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


def _exit_part_converges(result: Distribution, variables: Sequence[str]) -> bool:
    """Whether the closed form of the terminating part ⟨I⟩_¬B (non-negative coefficients) gives the values of
    its series where queries evaluate it, at 1: with finite mass, or converging on the open unit polydisc (by
    Abel's theorem, a finite value of the closed form at 1 is then the sum of the series)."""
    return finite_mass(str(result), list(variables))[0] is True \
        or converges_on_unit_polydisc(str(result), variables) is True


def _evt_certificate(kind_name: str, invariant: Distribution, variables: Sequence[str],
                     config: ForwardAnalysisConfig, scope_kind: str,
                     loop: WhileInstr, loop_cond: str,
                     result: Distribution) -> Tuple[str, Dict[str, Any], List[str]]:
    positive = _positivity(config, variables, invariant)
    finite, mass = finite_mass(str(invariant), list(variables)) if positive else (None, None)
    series = _series_identity(invariant, [str(v) for v in variables], loop, finite) if positive else None
    kind, messages = _evt_outcome_kind(scope_kind, positive, finite, series)
    if kind == Bound.UPPER and series == "formal loop body" \
            and not _exit_part_converges(result, [str(v) for v in variables]):
        # (B) holds for a series of any radius: Park bounds the coefficients of ⟨I⟩_¬B, but queries and
        # the rest of the program evaluate it at 1, where its series may diverge (#393)
        kind = Bound.UNKNOWN
        messages = ["I = G + ⟦body⟧(⟨I⟩_B) holds for the power series of I (formal loop body), but the series "
                    f"of the terminating part {result} is not proven to converge at 1, where queries and the rest "
                    "of the program evaluate it; its closed form need not be the sum of the series there. "
                    "No guarantee."]
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
                                                   scope.kind, instruction, task.session.spell(instruction.cond),
                                                   solution)
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

    def _sorting_key(elem: Tuple[Distribution, Optional[bool]]) -> int:
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
                                                   instruction, task.session.spell(instruction.cond),
                                                   distributions[index])
    certificate["chosen"] = index
    certificate["candidates"] = candidates
    certificate["synthesis_strategy"] = config.strategy.name
    certificate["max_degree"] = max_degree
    return LoopOutcome(distributions[index], task.error_prob, kind=kind, certificate=certificate,
                       messages=messages)

_EVT_INVARIANT_DELIVERS = (
    Delivers(Bound.EXACT, "I has non-negative coefficients and finite mass"),
    Delivers(Bound.UPPER, "I has non-negative coefficients and the equation holds for its power series"),
)

register_loop_strategy(LoopStrategy(
    name="evt_invariant",
    title="Check EVT invariant",
    group="exact",
    outcome="exact or upper bound",
    summary="Exact if your invariant has finite mass, otherwise an upper bound; you provide a generating "
            "function of the expected visiting times; one check, usually fast.",
    legacy_key="5",
    soundness=Bound.UPPER,
    delivers=_EVT_INVARIANT_DELIVERS,
    description="Checks a generating function I against I = G + ⟦body⟧(⟨I⟩_guard), solving for its "
                "parameters. With non-negative coefficients the result is an upper bound (Park), and "
                "exact if I has finite mass. Choice probabilities α + β/(k + a) in a counter k act as "
                "integral operators; then I may be an exponential polynomial A·exp(c·k) + B, e.g. for a "
                "Poisson output (#323).",
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
    delivers=_EVT_INVARIANT_DELIVERS,
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
