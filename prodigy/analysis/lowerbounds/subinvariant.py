# pylint: disable=invalid-name,import-outside-toplevel
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
r"""
Verification of user-provided EVT sub-invariants (lower induction).

For a loop ``while (G) { B }`` with input distribution ``g`` and ``Phi_g(nu) = g + [[B]](<nu>_G)``:

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

import sympy
from prodigy.pgcl import Instr, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus, Num, directed_decimal
from prodigy.distribution.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def to_product_form(expr: Union[str, sympy.Expr], variables: Sequence[str]) -> Optional[Tuple[Num, Denominator]]:
    """Write a rational generating function as ``P / prod_i (1 - r_i X_i^{p_i})`` (at most one factor per variable,
    ``r_i >= 0``), or return None if this is not possible."""
    syms = [sympy.Symbol(v) for v in variables]
    e = sympy.sympify(expr, locals=dict(zip(variables, syms))) if isinstance(expr, str) else expr
    e = sympy.together(sympy.expand(e))
    num, den = sympy.fraction(e)
    if any(s not in syms for s in e.free_symbols):
        return None
    const, factors = sympy.factor_list(sympy.expand(den), *syms)
    const = sympy.Rational(const)
    rates = [Fraction(0)] * len(syms)
    periods = [1] * len(syms)
    scale = const
    for fac, mult in factors:
        if mult != 1:
            return None
        free = fac.free_symbols
        if len(free) != 1:
            return None
        x = free.pop()
        i = syms.index(x)
        poly = sympy.Poly(fac, x)
        terms = poly.terms()
        if len(terms) != 2:
            return None
        (deg_a, a), (deg_b, b) = sorted(((t[0][0], t[1]) for t in terms))
        if deg_a != 0 or a == 0:
            return None
        r = -sympy.Rational(b) / sympy.Rational(a)
        if r < 0 or rates[i] != 0:
            return None
        rates[i], periods[i] = Fraction(int(r.p), int(r.q)), int(deg_b)
        scale *= a
    poly_num = sympy.Poly(sympy.expand(num / scale), *syms)
    numer: Num = {}
    for monom, c in poly_num.terms():
        c = sympy.Rational(c)
        numer[tuple(int(m) for m in monom)] = Fraction(int(c.p), int(c.q))
    return numer, Denominator(rates, periods)


def nonneg_by_construction(expr_str: str, variables: Sequence[str]) -> bool:
    """Sufficient criterion for non-negative coefficients: the expression is ``P / prod_i Q_i^k_i`` where ``P`` has
    non-negative coefficients and every ``Q_i = c_i - R_i`` has a constant ``c_i > 0`` and ``R_i`` non-negative
    coefficients, as ``1/(c - R) = sum_n R^n / c^(n+1)``. Covers denominators that are not products of univariate
    factors, e.g. ``x*y/(1 - 9/50*(x + y))``."""
    syms = [sympy.Symbol(v) for v in variables]
    try:
        expr = sympy.cancel(sympy.together(sympy.sympify(expr_str, locals=dict(zip(variables, syms)))))
        num, den = sympy.fraction(expr)
        coeff, factors = sympy.factor_list(den, *syms)
        for factor, power in factors:
            poly = sympy.Poly(factor, *syms)
            constant = poly.coeff_monomial(1)
            if constant == 0:
                return False
            if constant < 0:
                poly, coeff = -poly, coeff * (-1) ** power
            if any(c > 0 for monomial, c in poly.terms() if any(monomial)):
                return False
        return all(c >= 0 for c in sympy.Poly(num / coeff, *syms).coeffs())
    except (TypeError, ValueError, sympy.SympifyError, sympy.PolynomialError):
        return False


def decide_nonneg(expr_str: str, variables: Sequence[str], config: ForwardAnalysisConfig) -> Tuple[Optional[bool], str]:
    """Coefficient-wise nonnegativity: exact for product denominators, then a sufficient criterion for rational
    functions, otherwise Prodigy's heuristics."""
    form = to_product_form(expr_str, variables)
    if form is not None:
        numer, den = form
        calc = GFCalculus(list(variables), den)
        neg = calc.first_negative(numer)
        if neg is None:
            return True, "exact (finite positivity check)"
        return False, f"exact: coefficient {neg[1]} at {dict(zip(variables, neg[0]))}"
    if nonneg_by_construction(expr_str, variables):
        return True, "P / prod (c - R) with non-negative P and R"
    from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
    try:
        heuristic = PositivityHeuristics.create(PositivityHeuristics.MIXED_RAT_FUNC, config.factory)
        res = heuristic.is_positive(sympy.S(expr_str))
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"positivity heuristic failed: {exc}"
    return res, "positivity heuristic of Prodigy"


def exact_mass(expr_str: str, variables: Sequence[str]):
    form = to_product_form(expr_str, variables)
    if form is None:
        return None
    numer, den = form
    return GFCalculus(list(variables), den).mass(numer)


def rational_mass(expr_str: str, variables: Sequence[str]):
    """Total mass of a rational generating function with NONNEGATIVE coefficients, or ``None`` if the closed form is
    not rational.  Let ``f(t) = F(t, ..., t)`` in lowest terms.  By Pringsheim's theorem, the radius of convergence of
    ``f`` is its smallest pole on the positive real axis; hence the mass ``sum of coefficients = lim_{t -> 1-} f(t)`` is
    infinite iff ``f`` has a pole in ``(0, 1]``, and equals ``f(1)`` otherwise.  Evaluating the closed form at ``1``
    directly is unsound: ``(1 + X)/(1 - 2X) = 1 + 3X + 6X^2 + ...`` evaluates to ``-2``."""
    t = sympy.Symbol("_t", positive=True)
    expr = sympy.S(expr_str).subs({sympy.Symbol(v): t for v in variables}, simultaneous=True)
    f = sympy.cancel(sympy.together(expr))
    numer, den = sympy.fraction(f)
    if not (numer.is_polynomial(t) and den.is_polynomial(t)):
        return None
    den_poly = sympy.Poly(den, t)
    if den_poly.eval(0) == 0:
        return None  # not a power series at the origin
    if den_poly.degree() > 0 and den_poly.count_roots(0, 1) > 0:  # exact (Sturm sequences); no root at 0
        return math.inf
    value = f.subs(t, 1)
    return Fraction(str(sympy.nsimplify(value))) if value.is_rational else float(value)


def evt_subinvariant_verification(loop: WhileInstr, prog_info: ProgramInfo, distribution: Distribution,
                                  invariant: Distribution, config: ForwardAnalysisConfig,
                                  analyzer: Callable[[Union[Instr, Sequence[Instr]], ProgramInfo, Distribution,
                                                      Distribution, ForwardAnalysisConfig],
                                                     Tuple[Distribution, Distribution]],
                                  details: Optional[Dict[str, Any]] = None) -> Distribution:
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
    nonneg_nu, how1 = decide_nonneg(str(invariant), variables, config)
    if nonneg_nu is not True:
        raise VerificationError(f"Could not establish nu >= 0 ({how1}).")
    nonneg_res, how2 = decide_nonneg(str(residual), variables, config)
    if nonneg_res is not True:
        raise VerificationError(f"Could not establish nu <= Phi(nu) ({how2}).")
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
        raise VerificationError("nu has infinite mass; lower induction needs |nu| < inf.")
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
