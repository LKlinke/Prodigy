# pylint: disable=invalid-name,import-outside-toplevel
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
r"""
Verification of user-provided EVT sub-invariants (lower induction).

For a loop ``while (G) { B }`` with input distribution ``g`` and ``Phi_g(nu) = g + [[B]](<nu>_G)``::

    if   nu >= 0,   nu <= Phi_g(nu)  (coefficient-wise)   and   |nu| = nu(1, ..., 1) < inf,
    then <nu>_{not G} <= [[while (G) { B }]](g)   and   |[[loop]](g)| - |<nu>_{not G}| <= |Phi_g(nu)| - |nu|.

No termination assumption is needed.  ``Phi_g(nu)`` is computed with Prodigy's own forward semantics.  The
coefficient-wise sign conditions are decided exactly when the generating functions have product denominators
``prod_i (1 - r_i X_i^{p_i})`` (finite positivity check); otherwise Prodigy's positivity heuristics are used and
the result may be inconclusive.
"""
from __future__ import annotations

import logging
import math
from fractions import Fraction
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from prodigy.distribution.equality import sympify_symbolic_string
from prodigy.pgcl import Instr, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.lowerbounds.egcalc import GFCalculus, directed_decimal
from prodigy.distribution.coefficient_signs import nonneg_by_construction
from prodigy.distribution.distribution import Distribution
from prodigy.distribution.product_form import rational_mass, to_product_form
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

Details = Dict[str, Any]
"""A JSON-like record for the result (heterogeneous values: ``Any`` is correct here)."""

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def decide_nonneg(expr_str: str, variables: Sequence[str], config: ForwardAnalysisConfig,
                  negative: Optional[Details] = None) -> Tuple[Optional[bool], str]:
    """Coefficient-wise nonnegativity: exact for product denominators, then a sufficient criterion for rational
    functions, otherwise Prodigy's heuristics.

    ``negative``, if given, receives the first negative coefficient found (``coefficient``, and the state ``at``
    as a mapping from variables to values)."""
    form = to_product_form(expr_str, variables)
    if form is not None:
        numer, den = form
        calc = GFCalculus(list(variables), den)
        neg = calc.first_negative(numer)
        if neg is None:
            return True, "exact (finite positivity check)"
        state = dict(zip(variables, neg[0]))
        if negative is not None:
            negative.update(coefficient=str(neg[1]), at=state)
        return False, f"exact: coefficient {neg[1]} at {state}"
    if nonneg_by_construction(expr_str, variables):
        return True, "P / prod (c - R) with non-negative P and R"
    from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
    try:
        heuristic = PositivityHeuristics.create(PositivityHeuristics.MIXED_RAT_FUNC, config.factory,
                                                variables=variables)
        res = heuristic.is_positive(sympify_symbolic_string(expr_str))
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"positivity heuristic failed: {exc}"
    return res, "positivity heuristic of Prodigy"


def exact_mass(expr_str: str, variables: Sequence[str]) -> Optional[Union[Fraction, float]]:
    form = to_product_form(expr_str, variables)
    if form is None:
        return None
    numer, den = form
    return GFCalculus(list(variables), den).mass(numer)


def evt_subinvariant_verification(loop: WhileInstr, prog_info: ProgramInfo, distribution: Distribution,
                                  invariant: Distribution, config: ForwardAnalysisConfig,
                                  analyzer: Callable[[Union[Instr, Sequence[Instr]], ProgramInfo, Distribution,
                                                      Distribution, ForwardAnalysisConfig],
                                                     Tuple[Distribution, Distribution]],
                                  details: Optional[Details] = None) -> Distribution:
    """Verify an EVT sub-invariant ``nu`` of finite mass and return the lower bound ``<nu>_{not G}`` on the output
    distribution of the loop (raises :class:`VerificationError` if the conditions cannot be established).

    ``details``, if given, receives ``mass`` (``|nu|``), ``gap`` (``|Phi(nu)| - |nu|``, which bounds the mass that the
    lower bound misses), ``exit_mass`` and how non-negativity was established (``nonneg_nu``, ``nonneg_gap``)."""
    print(f"{Style.YELLOW}Sub-invariant verification (lower induction) initiated...{Style.RESET}")
    from prodigy.analysis.lowerbounds.ir import instr_to_ir
    from prodigy.analysis.lowerbounds.problem import _contains
    if _contains(instr_to_ir(loop.body), "observe"):
        raise VerificationError("observe in the loop body is not supported by lower induction: conditioning has to be "
                                "distinguished from non-termination, which this mode does not do yet.")
    variables = [str(v) for v in prog_info.program.variables.keys()]
    zero = config.factory.from_expr("0", *variables)
    body_out, _ = analyzer(loop.body, prog_info, invariant.filter(loop.cond), zero, config)
    phi = distribution + body_out
    residual = phi - invariant
    logger.debug("Phi(nu) - nu = %s", residual)
    negative: Details = {}
    nonneg_nu, how1 = decide_nonneg(str(invariant), variables, config, negative)
    if nonneg_nu is not True:
        raise VerificationError(f"Could not establish nu >= 0 ({how1}).",
                                details={"type": "evt_subinvariant", "condition": "nonnegative",
                                         "verdict": "refuted" if nonneg_nu is False else "unknown",
                                         "subinvariant": str(invariant), "method": how1, **negative})
    nonneg_res, how2 = decide_nonneg(str(residual), variables, config, negative)
    if nonneg_res is not True:
        raise VerificationError(f"Could not establish nu <= Phi(nu) ({how2}).",
                                details={"type": "evt_subinvariant", "condition": "subinvariance",
                                         "verdict": "refuted" if nonneg_res is False else "unknown",
                                         "subinvariant": str(invariant), "difference": str(residual),
                                         "method": how2, **negative})
    mass = exact_mass(str(invariant), variables)
    if mass is None:
        try:
            mass = rational_mass(str(invariant), variables)
        except Exception as exc:  # pylint: disable=broad-except
            raise VerificationError(f"Could not determine the mass of nu: {exc}") from exc
        if mass is None:
            raise VerificationError("Could not determine the mass of nu: the closed form is not rational, "
                                    "and evaluating "
                                    "it at 1 is only sound inside the radius of convergence.")
    if mass == math.inf:
        raise VerificationError("nu has infinite mass; lower induction needs |nu| < inf.",
                                details={"type": "evt_subinvariant", "condition": "finite_mass", "verdict": "refuted",
                                         "subinvariant": str(invariant)})
    exit_part = invariant - invariant.filter(loop.cond)
    gap = exact_mass(str(residual), variables)
    exit_mass = exact_mass(str(exit_part), variables)
    print(f"{Style.OKGREEN}Sub-invariant verified:{Style.RESET} nu >= 0 [{how1}], nu <= Phi(nu) [{how2}], "
          f"|nu| = {mass} < inf.")
    if details is not None:
        details.update(mass=mass, gap=gap, exit_mass=exit_mass, nonneg_nu=how1, nonneg_gap=how2)
    if gap is not None and exit_mass is not None:
        print(f"{Style.GREEN}Lower bound on the output distribution:{Style.RESET} {exit_part}")
        print(f"{Style.GREEN}Termination probability in{Style.RESET} [{directed_decimal(exit_mass, False)}, "
              f"{directed_decimal(min(exit_mass + gap, Fraction(1)), True)}]   "
              f"(gap <= |Phi(nu)| - |nu| = {directed_decimal(gap, True)})")
    return exit_part
