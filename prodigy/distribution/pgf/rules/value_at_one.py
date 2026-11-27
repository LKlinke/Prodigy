"""
Values of SymPy closed forms at 1 (#511, P4c): the sum of the coefficients of a power series as the limit from
below at 1 (Abel's theorem), computed cheaply where it is provably the value at 1, and the fast cancellation it
uses. Shared by the engines (marginals, expected values); the SymPy engine uses them for all its closed forms.
"""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, Tuple

import sympy

__all__ = ["_UNDEFINED_VALUES", "_cancel", "_cheap_value_at_one", "_constants_as_symbols", "_continuous_at_one",
           "_exponentials_as_powers", "_substitute_one"]


def _cancel(function: sympy.Expr) -> sympy.Expr:
    """``function`` as one fraction in lowest terms, like ``sympy.cancel``, but much faster on long sums of
    fractions with many irrational constants (normalised residue classes of Poisson draws, #472, #486).

    Every subexpression that is not built from symbols and rationals by ``+``, ``*`` and integer powers is
    replaced by a fresh symbol (:func:`_constants_as_symbols`), after the exponentials of rationals became
    powers of one symbol (:func:`_exponentials_as_powers`). The summands are added one by one in the field of
    rational functions over the rationals (sparse arithmetic, already in lowest terms), and the replacements
    are substituted back. Cancelling is an identity of rational functions, so it holds for every value of the
    symbols where the function is defined; relations between the replaced parts are not used, as by
    ``cancel``, which treats them as independent generators too. ``sympy.cancel`` if the field cannot hold
    a summand."""
    rewritten, back = _exponentials_as_powers(function)
    replacements: Dict[sympy.Expr, sympy.Dummy] = {}
    frozen = _constants_as_symbols(rewritten, frozenset(), replacements)
    generators = sorted(frozen.free_symbols, key=sympy.default_sort_key)
    if not generators:
        return sympy.cancel(function)
    try:
        field = sympy.polys.fields.FracField(generators, sympy.QQ)
        lowest = sum((field.from_expr(term) for term in sympy.Add.make_args(frozen)), field.zero)
    except (sympy.polys.polyerrors.CoercionFailed, sympy.polys.polyerrors.PolynomialError):
        return sympy.cancel(function)
    back.update({symbol: expression for expression, symbol in replacements.items()})
    return lowest.as_expr().xreplace(back)


def _exponentials_as_powers(expression: sympy.Expr) -> Tuple[sympy.Expr, Dict[sympy.Expr, sympy.Expr]]:
    """``expression``, and the substitution back, with ``exp(r)`` for rational ``r`` (``E`` included)
    written as the power ``t**(r*L)`` of one fresh symbol ``t`` standing for ``exp(1/L)``, ``L`` the least
    common denominator of the ``r``.

    Equal as functions, since ``exp(r) = exp(1/L)**(r*L)`` with an integer exponent; a cancellation then
    sees one generator instead of one per constant (sums of many exponential constants come from normalising
    Poisson draws, #486)."""
    exponents = {power.args[0] for power in expression.atoms(sympy.exp) if power.args[0].is_Rational}
    if expression.has(sympy.E):
        exponents.add(sympy.Integer(1))
    if not exponents:
        return expression, {}
    common = math.lcm(*(int(exponent.q) for exponent in exponents))
    base = sympy.Dummy("t")
    replacements: Dict[sympy.Expr, sympy.Expr] = {sympy.exp(exponent): base ** int(exponent * common)
                                                  for exponent in exponents}
    replacements[sympy.E] = base ** common
    return expression.xreplace(replacements), {base: sympy.exp(sympy.Rational(1, common))}


def _constants_as_symbols(expression: sympy.Expr, variables: FrozenSet[sympy.Symbol],
                          replacements: Dict[sympy.Expr, sympy.Dummy]) -> sympy.Expr:
    """``expression`` with each maximal subexpression without a variable that is not built from symbols and
    rational numbers by ``+``, ``*`` and integer powers replaced by a fresh symbol (the same one for equal
    subexpressions, recorded in ``replacements``)."""
    if isinstance(expression, (sympy.Symbol, sympy.Rational)):
        return expression
    if isinstance(expression, (sympy.Add, sympy.Mul)) or (
            isinstance(expression, sympy.Pow) and isinstance(expression.exp, sympy.Integer)):
        return expression.func(*(_constants_as_symbols(arg, variables, replacements) for arg in expression.args))
    if expression.free_symbols & variables:
        return expression.func(*(_constants_as_symbols(arg, variables, replacements) for arg in expression.args))
    if expression not in replacements:
        replacements[expression] = sympy.Dummy("c")
    return replacements[expression]


def _substitute_one(function: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr:
    """
    The value of `function` at `variable` = 1, i.e. the variable marginalised out.

    For a generating function with non-negative coefficients whose sum is finite, Abel's theorem makes the
    sum of the coefficients the limit of `function` from below at 1. This limit is computed cheaply where
    that is sound: if `function` (or its cancelled form, which removes a removable singularity of a
    rational expression such as `(x**3 - 1)/(x - 1)`) is provably continuous at 1 (see
    :func:`_continuous_at_one`) and finite there, the limit is its value at 1. Otherwise SymPy's `limit` is
    used, which can be very slow on unsimplified expressions (#259).

    Like `limit`, this assumes that `function` is such a generating function: for `x/(x - 1/2)`, whose power
    series diverges at 1, both give 2.
    """
    function = sympy.sympify(function)
    value = _cheap_value_at_one(function, variable)
    return function.limit(variable, 1, '-') if value is None else value


def _cheap_value_at_one(function: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr | None:
    """
    The value of `function` at `variable` = 1 where it is provably the limit from below: `function` or its
    cancelled form is continuous at 1 (:func:`_continuous_at_one`) and finite there. None otherwise.
    """
    if variable not in function.free_symbols:
        return function
    # the cancelled form only if the function itself fails: cancelling is the expensive part (Prodigy#472)
    for form in (lambda: function, lambda: _cancel(function)):
        candidate = form()
        if _continuous_at_one(candidate, variable):
            value = candidate.subs(variable, 1)
            if not value.has(*_UNDEFINED_VALUES):
                return value
    return None


_UNDEFINED_VALUES = (sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)


def _continuous_at_one(function: sympy.Expr, variable: sympy.Symbol) -> bool:
    # pylint: disable=too-many-return-statements  # one verdict per case of the rule in the docstring
    """
    True only if `function` is continuous in `variable` at 1 wherever its value there is finite.

    An allowlist: sums, products, `exp` and integer powers are continuous wherever they are finite;
    `log` and powers with other exponents only where their argument (base) is positive at 1 (there they
    are analytic, away from the branch cut), or for a base 0 under a positive constant exponent.
    A part that does not depend on `variable` is constant. Anything else (e.g. `floor`, `Piecewise`,
    `Abs`) is not shown continuous, and the caller takes the limit.
    """
    if variable not in function.free_symbols:
        return True
    if isinstance(function, sympy.Symbol):
        return True
    if isinstance(function, (sympy.Add, sympy.Mul, sympy.exp)):
        return all(_continuous_at_one(arg, variable) for arg in function.args)
    if isinstance(function, sympy.log) and len(function.args) == 1:
        (argument,) = function.args
        return _continuous_at_one(argument, variable) \
            and argument.subs(variable, 1).is_extended_positive is True
    if isinstance(function, sympy.Pow):
        base, exponent = function.args
        if not (_continuous_at_one(base, variable) and _continuous_at_one(exponent, variable)):
            return False
        if exponent.is_Integer:
            return True
        base_at_one = base.subs(variable, 1)
        if base_at_one.is_extended_positive is True:
            return True
        return base_at_one.is_zero is True and variable not in exponent.free_symbols \
            and exponent.is_extended_positive is True
    return False
