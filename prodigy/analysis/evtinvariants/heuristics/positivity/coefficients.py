"""
Signs of power-series coefficients, shared by the positivity heuristics.

A positivity verdict is three-valued: ``True`` means that every coefficient of the power series in the
*variables* is proven non-negative, ``False`` that some coefficient is proven negative, ``None`` that neither is
proven. Every other symbol of an expression is a program *parameter*. Parameters are coefficients, never series
variables: ``1/(1 - p)`` is the constant ``1/(1 - p)``, which is negative for ``p = 3`` (#295). Parameters are
declared ``real`` or ``nat`` and have no bounds, so the only fact used about them is that they are real; a sign
that depends on their value is not decided.
"""
from __future__ import annotations

from typing import Collection, Iterable, List, Optional, Sequence, Tuple, Union

import sympy

MAX_DIVISION_ORDERS = 6
"""How many variable orders :func:`nonneg_by_construction` tries for the polynomial division."""


def series_form(f: Union[str, sympy.Expr], variables: Collection[str]) -> Tuple[sympy.Expr, List[sympy.Symbol]]:
    """``f`` as a SymPy expression whose parameters (free symbols that are not ``variables``) are real symbols,
    and the symbols of the variables that occur in it (sorted by name)."""
    names = {str(v) for v in variables}
    expr = sympy.S(f.replace("^", "**")) if isinstance(f, str) else sympy.S(f)
    parameters = {s: sympy.Symbol(s.name, real=True) for s in expr.free_symbols if s.name not in names}
    if parameters:
        expr = expr.xreplace(parameters)
    gens = sorted((s for s in expr.free_symbols if s.name in names), key=lambda s: s.name)
    return expr, gens


def sign(coefficient: sympy.Expr) -> Optional[bool]:
    """``True`` if the coefficient is proven non-negative, ``False`` if it is proven negative, else ``None``."""
    coefficient = sympy.S(coefficient)
    if coefficient.is_nonnegative is True:
        return True
    if coefficient.is_negative is True:
        return False
    return None


def all_nonneg(coefficients: Iterable[sympy.Expr]) -> Optional[bool]:
    """Three-valued: ``False`` if one coefficient is proven negative, ``True`` if all are proven non-negative."""
    verdict: Optional[bool] = True
    for coefficient in coefficients:
        result = sign(coefficient)
        if result is False:
            return False
        if result is None:
            verdict = None
    return verdict


def polynomial_verdict(expr: sympy.Expr, gens: Sequence[sympy.Symbol]) -> Optional[bool]:
    """The verdict for a polynomial in ``gens`` (its coefficients are the coefficients of its power series)."""
    return all_nonneg(sympy.Poly(expr, *(gens or [sympy.Dummy("c")])).coeffs())


def nonneg_quotient(numerator: sympy.Expr, denominator: sympy.Expr, gens: Sequence[sympy.Symbol]) -> bool:
    """Whether ``numerator / denominator`` is shown to have non-negative coefficients: the denominator is
    ``c - R`` with ``c > 0`` and ``R`` a polynomial without constant term whose coefficients are non-negative, so
    ``1 / (c - R) = sum_k R^k / c^(k+1)``, and the numerator has non-negative coefficients. Coefficients may
    contain parameters; their signs must be proven for every real value."""
    gens = list(gens) or [sympy.Dummy("c")]
    num = sympy.Poly(numerator, *gens)
    den = sympy.Poly(denominator, *gens)
    constant = den.as_dict().get((0,) * len(gens), sympy.S.Zero)
    if sympy.S(constant).is_positive is not True:
        return False
    rest = [-coeff for monom, coeff in den.terms() if any(monom)]
    return all_nonneg(rest) is True and all_nonneg(num.coeffs()) is True


def _normalised_denominator(denominator: sympy.Expr, gens: Sequence[sympy.Symbol]) \
        -> Optional[Tuple[sympy.Expr, List[sympy.Expr]]]:
    """``(k, [D_1, ..., D_m])`` with ``denominator = k * D_1 * ... * D_m`` where every ``D_i`` is a factor
    ``c - R`` with ``c > 0`` and ``R`` with non-negative coefficients and no constant term (so ``1/D_i`` has
    non-negative coefficients; factors are repeated by their multiplicity), or None. Covers denominators that
    are not products of univariate factors, e.g. ``1 - 9/50*(x + y)``."""
    coeff, factors = sympy.factor_list(denominator, *gens)
    normalised: List[sympy.Expr] = []
    for factor, power in factors:
        poly = sympy.Poly(factor, *gens)
        constant = poly.coeff_monomial(1)
        if sign(constant) is False:
            poly, coeff = -poly, coeff * (-1) ** power
        if not nonneg_quotient(sympy.S.One, poly.as_expr(), gens):
            return None
        normalised.extend([poly.as_expr()] * power)
    return coeff, normalised


def _quotient_by_construction(expr: sympy.Expr, gens: Sequence[sympy.Symbol]) -> bool:
    """``expr = N / (k * D)`` with ``D = D_1 * ... * D_m`` as in :func:`_normalised_denominator`, and either
    ``N / k`` has non-negative coefficients, or the division ``N / k = Q * E + r`` by ``E = D`` or by one factor
    ``E = D_i`` (in some lex order of the variables) gives a quotient ``Q`` and a remainder ``r`` with
    non-negative coefficients: then ``expr = Q / (D / E) + r / D`` (#297)."""
    num, den = sympy.fraction(sympy.cancel(sympy.together(expr)))
    if not (num.is_polynomial(*gens) and den.is_polynomial(*gens)):
        return False
    normalised = _normalised_denominator(den, gens) if gens else (den, [])
    if normalised is None:
        return False
    coeff, factors = normalised
    if sympy.S(coeff).is_positive is not True:
        coeff, num = -coeff, -num
        if sympy.S(coeff).is_positive is not True:
            return False
    numerator = sympy.expand(num / coeff)
    if polynomial_verdict(numerator, gens) is True:
        return True
    if not factors:
        return False
    divisors = [sympy.expand(sympy.Mul(*factors))] + list(dict.fromkeys(factors)) if len(factors) > 1 else factors
    orders = [list(gens[i:]) + list(gens[:i]) for i in range(min(len(gens), MAX_DIVISION_ORDERS))]
    for divisor in divisors:
        for order in orders:
            (quotient,), remainder = sympy.reduced(numerator, [divisor], *order, order="lex")
            if polynomial_verdict(quotient, gens) is True and polynomial_verdict(remainder, gens) is True:
                return True
    return False


def _by_construction(expr: sympy.Expr, gens: Sequence[sympy.Symbol], depth: int) -> bool:
    if depth > 0 and isinstance(expr, (sympy.Add, sympy.Mul)):
        # a sum or a product of non-negative series is non-negative; checking the summands before combining
        # them over a common denominator keeps e.g. 5/(2 - x) + t*(1/2 + x^5/2) (#297)
        if all(_by_construction(arg, gens, depth - 1) for arg in expr.args):
            return True
    if isinstance(expr, sympy.Pow) and expr.exp.is_Integer and expr.exp > 0 and depth > 0:
        if _by_construction(expr.base, gens, depth - 1):
            return True
    return _quotient_by_construction(expr, gens)


def nonneg_by_construction(f: Union[str, sympy.Expr], variables: Collection[str]) -> bool:
    """Sufficient criterion for non-negative coefficients in ``variables``: ``f`` is a sum or product of terms
    ``P / prod_i Q_i^k_i`` where every ``Q_i = c_i - R_i`` has a constant ``c_i > 0`` and ``R_i`` non-negative
    coefficients, as ``1/(c - R) = sum_n R^n / c^(n+1)``, and ``P`` has non-negative coefficients or is
    ``Q * prod_i Q_i^k_i + r`` with ``Q`` and ``r`` non-negative. Parameters are real coefficients
    (:func:`series_form`)."""
    try:
        expr, gens = series_form(f, variables)
        return _by_construction(expr, gens, depth=3)
    except (TypeError, ValueError, AttributeError, sympy.SympifyError, sympy.PolynomialError):
        return False
