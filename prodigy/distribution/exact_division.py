"""
Exact division of a variable with infinite marginal by a natural number.

The update `target := numerator / q` is only defined if `numerator` is a multiple of `q` in every
state. For an infinite marginal this cannot be checked state by state. If the generating function
is rational in `numerator`, it can be decided exactly:

Write f = P / Q with polynomials P and Q in x = `numerator` and let D(y) = Res_t(t^q - y, Q(t)), so
that D(x^q) is the product of Q(w x) over all q-th roots of unity w. Q divides D(x^q), hence
f = N(x) / D(x^q) for the polynomial N = P D(x^q) / Q. As 1 / D(x^q) only has exponents of x that
are multiples of q, the exponents of x in f are all multiples of q if and only if those of N are.
In that case f with x^k replaced by x^k target^(k/q) is N(x) with x^(qk) replaced by
x^(qk) target^k, divided by D(x^q target).

The computation is engine independent and works on SymPy expressions.
"""
from __future__ import annotations

import sympy


def _symbol(function: sympy.Expr, name: str, like: sympy.Symbol | None = None) -> sympy.Symbol | None:
    """The symbol called `name` in `function`, or a new one with the assumptions of `like`."""
    for symbol in function.free_symbols:
        if isinstance(symbol, sympy.Symbol) and symbol.name == name:
            return symbol
    if like is None:
        return None
    return sympy.Symbol(name, **like.assumptions0)


def divide_exponents(function: sympy.Expr, numerator: str, target: str,
                     divisor: int) -> sympy.Expr | None:
    """
    The generating function after the update `target := numerator / divisor`, or None if it is not
    proven that `numerator` is a multiple of `divisor` in every state. The result is None as well if
    `function` is not rational in `numerator`; the caller has to reject the update then.
    """
    x = _symbol(function, numerator)
    if x is None or divisor < 1:
        return None
    target_symbol = _symbol(function, target, like=x)
    assert target_symbol is not None
    f = function.subs(target_symbol, 1)

    num, den = sympy.fraction(sympy.together(f))
    if not (num.is_polynomial(x) and den.is_polynomial(x)) or den == 0:
        return None
    t, y = sympy.Dummy('t'), sympy.Dummy('y')
    try:
        d = sympy.resultant(sympy.Poly(den.subs(x, t), t), sympy.Poly(t**divisor - y, t))
        quotient, remainder = sympy.div(sympy.Poly(num * d.as_expr().subs(y, x**divisor), x),
                                        sympy.Poly(den, x))
    except (sympy.PolynomialError, ZeroDivisionError):
        return None
    if d.is_zero or not remainder.is_zero:
        return None

    n = 0
    for (exponent,), coefficient in quotient.terms():
        if exponent % divisor != 0:
            return None
        n += coefficient * x**exponent * target_symbol**(exponent // divisor)
    return sympy.cancel(n / d.as_expr().subs(y, x**divisor * target_symbol))
