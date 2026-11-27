"""
The strategies with bounds that need no termination proof (:mod:`prodigy.analysis.lowerbounds`):
``evt_subinvariant`` (a lower bound from an EVT sub-invariant) and ``evt_sandwich`` (lower and upper bounds).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict

import sympy

from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution import known_convergent_series
from prodigy.analysis.loops.common import _mass, _require_unconditioned


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
    # the body runs on the candidate, which is not proven to be a finite measure yet (#393)
    with known_convergent_series(False):
        lower = evt_subinvariant_verification(task.instruction, task.prog_info, task.distribution, subinvariant,
                                              config, task.analyzer, details=details)
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
            summary["exit_mass_rounded"] = directed_decimal(cert.verified_exit_mass(), name == "upper")
            summary["check"] = cert.extra.get("check")
        certificate[name] = summary
    query_bounds = {bound.query.key: (_exact(bound.lower), _exact(bound.upper))
                    for bound in result.bounds}
    # the values on the lower bound itself, computed from the certificate (no epilogue: `lower` is its exit part)
    query_values = {}
    if problem.epilogue == ("seq", []):
        query_values = {bound.query.key: _exact(bound.value)
                        for bound in result.bounds if bound.value is not None}
    # Park: the output lies coefficient-wise below [not guard]·I. The certificate checked I >= 0, so this is a
    # power series with non-negative coefficients; it is kept (LoopOutcome.upper) if its mass is finite.
    upper = None
    if result.upper is not None and result.upper.verified_exit_mass() != float("inf"):
        upper = task.config.factory.from_expr(result.upper.calc.to_expr(result.upper.exit_numerator), *variables)
        certificate["upper"]["per_state"] = True
    elif result.upper is not None:
        certificate["upper"]["per_state"] = False
    return LoopOutcome(lower, task.error_prob, kind=Bound.LOWER, residual=residual, certificate=certificate,
                       query_bounds=query_bounds, query_values=query_values, upper=upper)


register_loop_strategy(LoopStrategy(
    name="evt_subinvariant",
    title="Check EVT sub-invariant",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound without termination assumption; you provide a generating function with finite "
            "mass below one more iteration of itself; one check, usually fast.",
    legacy_key="7",
    soundness=Bound.LOWER,
    delivers=(Delivers(Bound.LOWER), Delivers(Bound.EXACT, "the sub-invariant misses no mass")),
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
    delivers=(Delivers(Bound.LOWER),
              Delivers(Bound.UPPER, "a super-invariant is found: for the queries right after the loop, and with "
                                    "finite mass for every state")),
    description="Synthesises a finite-mass EVT sub-invariant (lower bound) and an EVT super-invariant "
                "(upper bound, Park) with eventually geometric templates by linear programming, checks "
                "both exactly, and bounds the ?Pr and ?Ex queries right after the loop from both sides. "
                "A super-invariant of finite mass also bounds every state from above, through the "
                "loop-free code after the loop. Needs a rational input distribution and no observe.",
    run=_strategy_evt_sandwich,
    requires_unconditioned=True,
))
