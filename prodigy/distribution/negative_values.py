"""
Exact check whether a subtraction `minuend - subtrahend` can be negative on the support of a generating function.

The check works on the generating function before the subtraction and does not use marginals. Marginals are not
enough for two reasons:

- A marginal sums coefficients over the other variables. If coefficients may be negative (e.g. templates with
  parameters of unknown sign in invariant synthesis), terms on negative values can cancel in the marginal.
- If the subtrahend has infinite support, as in `x := x - y` for a geometric `y`, the result has infinitely many
  negative powers of `x`. Its marginal is then the expansion of a rational function at infinity, which looks finite at
  `x = 0` although the result has mass on negative values.

The question is a property of the support of `function`: it has a term `minuend^a * subtrahend^b` with `a < b`.
With a fresh symbol `t`, the substitution `minuend -> minuend * t`, `subtrahend -> subtrahend / t` maps such a term to
`t^(a - b)`, so the subtraction is never negative if and only if the substituted function is a formal power series.
A rational function in lowest terms is a power series if and only if its denominator does not vanish when all
variables are 0: a denominator that does not vanish there is invertible in the power series ring, and a power series
that is rational has, by the faithful flatness of the completion, a reduced denominator with a non-zero constant term.
This needs no assumption on the signs of coefficients or parameters.
"""
from __future__ import annotations

from collections.abc import Iterable

import sympy


def can_be_negative(function: sympy.Expr, variables: Iterable[sympy.Symbol], minuend: sympy.Expr,
                    subtrahend: sympy.Expr) -> bool | None:
    """
    Decides whether `minuend - subtrahend` is negative for some term of the formal power series `function`.

    `minuend` and `subtrahend` are variables of `function` (from `variables`) or constants; parameters are all symbols
    that are not in `variables`. Parameters are treated as indeterminates, so the answer holds for all values of the
    parameters except for the roots of a non-zero expression.

    :returns: True or False if the question is decided, None if it cannot be decided (e.g. the subtrahend is a
        parameter, or `function` is not rational in its variables, the subtrahend is a variable and the check of
        :func:`_power_series_with_exponentials` fails).
    """
    variables = set(variables)
    is_variable = {minuend in variables, subtrahend in variables}
    if minuend == subtrahend:
        return False
    if is_variable == {False}:
        difference = sympy.S(minuend - subtrahend)
        if difference.is_nonnegative:
            return False
        if difference.is_negative:
            return True
        return None
    if function.is_polynomial(*variables):
        return _polynomial_can_be_negative(function, minuend, subtrahend)
    if function.has(sympy.Float):
        # floats are rounded, so neither zero nor non-zero is certain
        return None
    if minuend in variables and subtrahend.is_Integer:
        return _below_constant(function, minuend, int(subtrahend))
    shift = sympy.Dummy("t")
    if subtrahend in variables:
        substitution = {subtrahend: subtrahend / shift}
        factor: sympy.Expr = sympy.S.One
    elif subtrahend.is_Integer:
        substitution, factor = {}, shift ** (-subtrahend)
    else:
        return None
    if minuend in variables:
        substitution[minuend] = minuend * shift
    elif minuend.is_Integer:
        factor *= shift ** minuend
    else:
        return None
    shifted = function.subs(substitution, simultaneous=True) * factor
    origin = {var: sympy.S.Zero for var in variables | {shift}}
    if shifted.is_rational_function(*variables, shift):
        _, denominator = sympy.fraction(sympy.cancel(sympy.together(shifted)))
        # negative values remain if and only if the constant term of the reduced denominator is zero
        return _is_zero(denominator.subs(origin, simultaneous=True))
    return _power_series_with_exponentials(shifted, origin)


def _power_series_with_exponentials(function: sympy.Expr, origin: dict[sympy.Symbol, sympy.Expr]) -> bool | None:
    """
    False if `function` is proven to be a power series in the symbols of `origin`, None otherwise.

    Every `exp(e)` whose argument `e` is a power series (a rational function whose denominator does not vanish at the
    origin) is a power series, too. With these replaced by fresh symbols `s`, `function` becomes a rational function
    `P/Q`, and it is a power series if `Q` does not vanish at the origin with `s = exp(e(0))`. Unlike for rational
    functions, a vanishing `Q` does not prove negative values, so the result is then None.
    """
    variables = list(origin)
    replacements: dict[sympy.Expr, sympy.Symbol] = {}
    at_origin: dict[sympy.Symbol, sympy.Expr] = dict(origin)
    for exponential in function.atoms(sympy.exp):
        argument = exponential.args[0]
        if not argument.is_rational_function(*variables):
            return None
        _, denominator = sympy.fraction(sympy.cancel(sympy.together(argument)))
        if _is_zero(denominator.subs(origin, simultaneous=True)) is not False:
            return None
        symbol = sympy.Dummy("s")
        replacements[exponential] = symbol
        at_origin[symbol] = sympy.exp(argument.subs(origin, simultaneous=True))
    replaced = function.xreplace(replacements)
    if not replaced.is_rational_function(*variables, *replacements.values()):
        return None
    _, denominator = sympy.fraction(sympy.cancel(sympy.together(replaced)))
    return False if _is_zero(denominator.subs(at_origin, simultaneous=True)) is False else None


def _polynomial_can_be_negative(function: sympy.Expr, minuend: sympy.Expr, subtrahend: sympy.Expr) -> bool | None:
    """
    The check for a finite support: the difference of the exponents in every term of the expanded polynomial. This
    needs no zero test of coefficients, so floats are allowed (as in the per-term check of the GiNaC bindings).
    """
    if not all(operand_.is_Symbol or operand_.is_Integer for operand_ in (minuend, subtrahend)):
        return None
    for term in sympy.Add.make_args(sympy.expand(function)):
        if term == 0:
            continue
        powers = term.as_powers_dict()
        value = [powers.get(operand_, 0) if operand_.is_Symbol else operand_ for operand_ in (minuend, subtrahend)]
        if value[0] < value[1]:
            return True
    return False


def _below_constant(function: sympy.Expr, variable: sympy.Symbol, constant: int) -> bool | None:
    """
    Whether `function` has a term in which `variable` is less than `constant`, i.e. whether one of the coefficients of
    `variable^0` to `variable^(constant-1)` is not zero.

    A formal power series in several variables converges near 0 if it is rational or built from the elementary
    functions of the engines, so its coefficients in `variable` are the coefficients of its Taylor series in
    `variable`, which `sympy.series` computes exactly.
    """
    if constant <= 0:
        return False
    try:
        polynomial = sympy.expand(sympy.series(function, variable, 0, constant).removeO())
    except (NotImplementedError, ValueError, TypeError, sympy.PoleError):
        return None
    if not polynomial.is_polynomial(variable) or polynomial.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo):
        return None
    undecided = False
    for power in range(constant):
        zero = _is_zero(polynomial.coeff(variable, power))
        if zero is False:
            return True
        if zero is None:
            undecided = True
    return None if undecided else False


def _is_zero(expression: sympy.Expr) -> bool | None:
    """
    Decides whether `expression` is zero; None if this cannot be decided.

    Exact for rational functions of symbols with rational coefficients. Other expressions (e.g. with `exp(-l)` or
    `sqrt(2)`) are zero if SymPy simplifies them to zero and non-zero if SymPy proves it.
    """
    numerator, _ = sympy.fraction(sympy.together(expression))
    numerator = sympy.expand(numerator)
    if numerator == 0:
        return True
    if numerator.is_Rational:
        return False
    if numerator.free_symbols:
        try:
            domain = sympy.Poly(numerator, *sorted(numerator.free_symbols, key=str)).domain
        except sympy.PolynomialError:
            domain = None
        if domain is not None and (domain.is_ZZ or domain.is_QQ):
            return False
    simplified = sympy.simplify(expression)
    if simplified == 0:
        return True
    if simplified.is_zero is False:
        return False
    return None


def operand(text: str | int, variables: Iterable[str]) -> sympy.Expr:
    """An operand of a subtraction as passed to the engines: a variable name from `variables`, or a constant."""
    if str(text) in set(variables):
        return sympy.Symbol(str(text))
    try:
        return sympy.Integer(str(text))
    except (TypeError, ValueError):
        return sympy.Symbol(str(text))
