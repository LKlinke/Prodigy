"""
Taylor coefficients of rational generating functions without limits.

For f = P/Q rational in x with Q(0) != 0, the power series sum_n s_n x^n of f satisfies Q * S = P, so

    s_n = (p_n - sum_{k=1}^{n} q_k s_{n-k}) / q_0

with p_k and q_k the coefficients of P and Q. The coefficients of P and Q are rational functions in the
other symbols, so the recursion is exact. If Q is constant after cancelling (e.g. the closed form
`(x^4 - 1) / (4 (x - 1))` of `unif(0, 3)`), f is a polynomial and all coefficients beyond its degree are 0.

Computing s_n as the limit of the n-th derivative at 0 is equivalent but SymPy's `limit` can take
exponentially long on the growing derivatives (Prodigy#163). Other symbols and non-rational subexpressions
without x are handled as in `prodigy.distribution.multisection`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import sympy

from prodigy.distribution.multisection import _to_rational


def rational_taylor_coefficients(function: sympy.Expr, variable: sympy.Symbol,
                                 count: int) -> Optional[List[sympy.Expr]]:
    """
    Returns the coefficients of x^0, ..., x^(count-1) of the power series of `function` in
    x = `variable` around 0, or None if `function` is not a rational function in `variable` or has a
    pole at 0. Other symbols are treated as constants. If `function` is a polynomial in `variable`, the
    list may be shorter than `count`; the missing coefficients are 0.
    """
    if count <= 0:
        return []
    replacements: Dict[sympy.Expr, sympy.Dummy] = {}
    rational = _to_rational(sympy.sympify(function), variable, replacements)
    if rational is None:
        return None

    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(rational)))
    others = sorted(rational.free_symbols - {variable}, key=sympy.default_sort_key)
    domain = sympy.QQ.frac_field(*others) if others else sympy.QQ
    try:
        p = [domain.from_sympy(c) for c in reversed(sympy.Poly(numerator, variable, domain=domain).all_coeffs())]
        q = [domain.from_sympy(c) for c in reversed(sympy.Poly(denominator, variable, domain=domain).all_coeffs())]
    except (sympy.polys.polyerrors.PolynomialError, sympy.CoercionFailed):  # pragma: no cover - defensive
        return None
    if not q[0]:
        return None  # pole at 0: not a power series

    if len(q) == 1:
        count = min(count, len(p))
    coefficients: List = []
    for n in range(count):
        value = p[n] if n < len(p) else domain.zero
        for k in range(1, min(n, len(q) - 1) + 1):
            value -= q[k] * coefficients[n - k]
        coefficients.append(domain.quo(value, q[0]))

    back = {dummy: original for original, dummy in replacements.items()}
    result = [_normalized(domain.to_sympy(c), others) for c in coefficients]
    return [c.xreplace(back) for c in result] if back else result


def _normalized(coefficient: sympy.Expr, symbols: List[sympy.Symbol]) -> sympy.Expr:
    """
    Writes a rational coefficient P/Q in the other symbols with the constant term of Q scaled to 1, e.g.
    `1/(1 - 3*c/5)` instead of `-5/(3*c - 5)`, the usual form of generating functions (and the one other
    parts of Prodigy, e.g. the product form of the lower bounds, recognise).
    """
    numerator, denominator = sympy.fraction(coefficient)
    if not symbols or not denominator.free_symbols:
        return coefficient
    constant = sympy.Poly(denominator, *symbols).coeff_monomial(1)
    if constant in (0, 1):
        return coefficient
    return sympy.expand(numerator / constant) / sympy.expand(denominator / constant)
