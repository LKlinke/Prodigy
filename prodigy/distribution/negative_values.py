"""
Exact check whether a generating function puts mass on negative values of a variable.

A subtraction such as `x := y - 1` is executed on a generating function by substituting and multiplying with
`x^(-1)`. The result encodes a valid distribution only if no term with a negative power of `x` remains. The engines
check this cheaply via the marginal of `x`; that check needs all other variables at 1 and therefore fails if the
function has infinite mass in another variable, as in the second-order generating functions of the equivalence check.
`has_negative_powers` decides the question for rational functions without any marginal.
"""
from __future__ import annotations

import sympy


def has_negative_powers(function: sympy.Expr, variable: sympy.Symbol) -> bool | None:
    """
    Decides whether the Laurent expansion of `function` in `variable` has a term with a negative power of `variable`.

    `function` must be a formal Laurent series in `variable` whose coefficients are power series in the other
    symbols, i.e., a power series divided by a power of `variable`. This holds for every result of a subtraction on a
    generating function.

    For a rational function in lowest terms P/Q this is the case if and only if `variable` divides Q: a power series
    that is rational has a reduced denominator with a non-zero constant term, so a factor `variable` in Q can only
    come from negative powers, and a pole at `variable` = 0 means that the expansion is not a power series in
    `variable`.

    A function that is not rational is only decided if `variable` is its only symbol: then it is analytic at 0 up to
    the power of `variable` it is divided by, so its limit at 0 is infinite if and only if negative powers remain.

    :returns: True or False if `function` is a rational function or univariate, None if the question cannot be
        decided.
    """
    if not function.is_rational_function():
        if function.free_symbols != {variable}:
            return None
        limit = sympy.limit(function, variable, 0)
        if limit.is_finite:
            return False
        if limit in (sympy.oo, -sympy.oo, sympy.zoo):
            return True
        return None
    _, denominator = sympy.fraction(sympy.cancel(sympy.together(function)))
    return sympy.expand(denominator.subs(variable, 0)) == 0
