"""
Multisection of rational generating functions without roots of unity.

For f in x and a modulus q, the r-th section f_r keeps the terms x^k of the power series of f with
k = r (mod q); with a primitive q-th root of unity w it is (1/q) * sum_m w^(-m r) f(w^m x). For a rational
function f = P/Q this sum can be computed exactly over the coefficient field of f:

    N(x) := prod_m Q(w^m x) = (-1)^(q deg Q) * Res_y(Q(y), y^q - x^q)

is a polynomial in x^q (so N(w x) = N(x)), and Q divides it. Writing f = P (N / Q) / N, the substitution
x -> w^m x only changes the numerator M = P N / Q, so f_r is the r-th section of the polynomial M divided
by N. The coefficients of P, Q, N and M are polynomials in the other symbols, so no root of unity and no
floating point number occurs.

Other variables and parameters are coefficients. Subexpressions without x that are not rational (e.g.
`exp(-l)` or `sqrt(p)`) are replaced by fresh symbols for the computation and substituted back afterwards;
the identities above are polynomial identities in these symbols, so they remain valid after the substitution
wherever f is defined. A function that is not rational in x (e.g. `exp(x)`) is left to the caller, which
falls back to the roots of unity.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import sympy


def _to_rational(expression: sympy.Expr, variable: sympy.Symbol,
                 replacements: Dict[sympy.Expr, sympy.Dummy]) -> Optional[sympy.Expr]:
    """
    Rewrites `expression` as a rational function in `variable` and other symbols by replacing the maximal
    subexpressions without `variable` that are not built from symbols and rational numbers by `+`, `*`
    and integer powers with dummies (recorded in `replacements`). Returns None if `expression` is not a
    rational function in `variable`.
    """
    if isinstance(expression, (sympy.Symbol, sympy.Rational)):
        return expression
    if isinstance(expression, (sympy.Add, sympy.Mul)):
        args = [_to_rational(argument, variable, replacements) for argument in expression.args]
        if any(argument is None for argument in args):
            return None
        return expression.func(*args)
    if isinstance(expression, sympy.Pow) and isinstance(expression.exp, sympy.Integer):
        base = _to_rational(expression.base, variable, replacements)
        return None if base is None else base ** expression.exp
    if expression.has(variable):
        return None
    if expression not in replacements:
        replacements[expression] = sympy.Dummy("c")
    return replacements[expression]


def _norm(denominator: sympy.Poly, modulus: int) -> sympy.Poly:
    """N(x) = prod_m Q(w^m x) for the q-th roots of unity w, a polynomial in x^q."""
    variable = denominator.gens[0]
    if denominator.degree() <= 0:
        return denominator
    y = sympy.Dummy("y")
    gens = (y,) + denominator.gens
    q_y = sympy.Poly(denominator.as_expr().subs(variable, y), *gens)
    power = sympy.Poly(y ** modulus - variable ** modulus, *gens)
    resultant = q_y.resultant(power)
    sign = -1 if modulus * denominator.degree() % 2 else 1
    return sympy.Poly(resultant.as_expr() * sign, *denominator.gens)


def rational_sections(function: sympy.Expr, variable: sympy.Symbol, modulus: int) -> Optional[List[sympy.Expr]]:
    """
    Returns the sections f_0, ..., f_{q-1} of `function` in `variable` for q = `modulus`, or None if
    `function` is not a rational function in `variable` (see the module documentation). Other symbols are
    treated as constants.

    :raises ValueError: if `modulus` is not positive.
    """
    if modulus < 1:
        raise ValueError(f"The modulus has to be positive, got {modulus}.")
    replacements: Dict[sympy.Expr, sympy.Dummy] = {}
    rational = _to_rational(sympy.sympify(function), variable, replacements)
    if rational is None:
        return None

    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(rational)))
    others = sorted((rational.free_symbols - {variable}), key=sympy.default_sort_key)
    gens = (variable, *others)
    numerator_poly = sympy.Poly(numerator, *gens)
    denominator_poly = sympy.Poly(denominator, *gens)
    if not numerator_poly.domain.is_Exact or not denominator_poly.domain.is_Exact:
        return None  # pragma: no cover - only symbols and rational numbers are left

    norm = _norm(denominator_poly, modulus)
    cofactor, remainder = norm.div(denominator_poly)
    if not remainder.is_zero:
        raise AssertionError(f"The norm of {denominator} is not divisible by it.")  # pragma: no cover
    product = numerator_poly * cofactor

    parts: List[Dict[tuple, sympy.Expr]] = [{} for _ in range(modulus)]
    for monomial, coefficient in product.as_dict().items():
        parts[monomial[0] % modulus][monomial] = coefficient
    back = {dummy: original for original, dummy in replacements.items()}
    sections: List[sympy.Expr] = []
    for part in parts:
        if not part:
            sections.append(sympy.Integer(0))
            continue
        section = sympy.Poly.from_dict(part, *gens, domain=product.domain)
        common, section_numerator, section_denominator = section.cancel(norm)
        expression = common * section_numerator.as_expr() / section_denominator.as_expr()
        sections.append(expression.xreplace(back) if back else expression)
    return sections


def rational_multisection(function: sympy.Expr, variable: sympy.Symbol, modulus: int) -> List[sympy.Expr]:
    """
    Returns the sections f_0, ..., f_{q-1} of the rational function `function` in `variable` for q = `modulus`.
    Other symbols are treated as constants.

    :raises ValueError: if `function` is not a rational function in `variable` or `modulus` is not positive.
    """
    sections = rational_sections(function, variable, modulus)
    if sections is None:
        raise ValueError(f"{function} is not a rational function in {variable}.")
    return sections
