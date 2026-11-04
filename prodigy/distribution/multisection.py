"""
Multisection of univariate rational functions without roots of unity.

For f in x and a modulus q, the r-th section f_r keeps the terms x^k of the power series of f with
k = r (mod q); with a primitive q-th root of unity w it is (1/q) * sum_m w^(-m r) f(w^m x). For a rational
function f = P/Q this sum can be computed exactly over the coefficient field of f:

    N(x) := prod_m Q(w^m x) = (-1)^(q deg Q) * Res_y(Q(y), y^q - x^q)

is a polynomial in x^q (so N(w x) = N(x)), and Q divides it. Writing f = P (N / Q) / N, the substitution
x -> w^m x only changes the numerator M = P N / Q, so f_r is the r-th section of the polynomial M divided
by N.
"""
from __future__ import annotations

from typing import List

import sympy


def rational_multisection(function: sympy.Expr, variable: sympy.Symbol, modulus: int) -> List[sympy.Expr]:
    """
    Returns the sections f_0, ..., f_{q-1} of the rational function `function` in `variable` for q = `modulus`.
    Other symbols are treated as constants.

    :raises ValueError: if `function` is not a rational function in `variable` or `modulus` is not positive.
    """
    if modulus < 1:
        raise ValueError(f"The modulus has to be positive, got {modulus}.")
    if not function.is_rational_function(variable):
        raise ValueError(f"{function} is not a rational function in {variable}.")

    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(function)))
    numerator_poly = sympy.Poly(numerator, variable)
    denominator_poly = sympy.Poly(denominator, variable)

    y = sympy.Dummy("y")
    norm = sympy.resultant(denominator_poly.as_expr().subs(variable, y), y ** modulus - variable ** modulus, y)
    norm_poly = sympy.Poly(sympy.expand(norm * (-1) ** (modulus * denominator_poly.degree())), variable)
    cofactor, remainder = sympy.div(norm_poly, denominator_poly)
    if not remainder.is_zero:
        raise AssertionError(f"The norm of {denominator} is not divisible by it.")  # pragma: no cover

    product = numerator_poly * cofactor
    norm_expr = norm_poly.as_expr()
    sections: List[sympy.Expr] = []
    for residue in range(modulus):
        section = sum((coefficient * variable ** exponent
                       for (exponent,), coefficient in product.terms() if exponent % modulus == residue),
                      sympy.Integer(0))
        sections.append(sympy.cancel(section / norm_expr))
    return sections
