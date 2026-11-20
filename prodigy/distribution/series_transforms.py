"""
Two changes of a generating function that keep the signs of its coefficients (#431), used by the proofs of
:mod:`~prodigy.distribution.finite_measure` (and through them by the order of
:mod:`~prodigy.distribution.ordering`). Both are statements about classes of power series, independent of the
expression they are applied to:

* **Substitution of a monomial.** If every occurrence of a variable ``x`` in ``f`` is a power ``x^k`` with
  ``m | k`` for some ``m >= 2``, then ``f(x) = G(x^m)`` as functions, where ``G`` is ``f`` with every ``x^k``
  replaced by ``x^(k/m)`` (an identity, since ``x^k = (x^m)^(k/m)`` for integer ``k/m``). If ``G`` is a
  power series ``sum_n g_n x^n``, then ``f = sum_n g_n x^(m*n)``: the coefficients of ``f`` are those of
  ``G`` and zeros, and the mass is the same (``f(1) = G(1)``). A positive factor ``r`` of the monomial needs
  no rule of its own: ``F(r*x^m) = G(x^m)`` with ``G(u) = F(r*u)``, whose coefficients ``f_n r^n`` have the
  signs of those of ``F`` for ``r > 0``, and the proofs of ``G`` see ``r*u`` directly (e.g. the fractional
  powers ``(1 - r*u)^a``). This covers e.g. ``1 - sqrt(1 - c^2)`` from ``1 - sqrt(1 - u)``.
* **Shift.** If ``f = G / x^k`` with ``k >= 1`` and the coefficients of ``G`` of degree ``< k`` in ``x``
  vanish (checked exactly), then ``f = sum_{n >= k} g_n x^(n - k)`` is a power series with the coefficients
  of ``G`` from degree ``k`` on, and the same mass (``f(1) = G(1)``). This covers e.g.
  ``(1 - sqrt(1 - c^2)) / c``.

Neither rule decides a sign by itself; they reduce the question to the same question for ``G``.
"""
from __future__ import annotations

import math
from typing import Optional, Set, Tuple

import sympy

MonomialSubstitution = Tuple[sympy.Expr, int]
"""``(G, m)`` with ``f(x) = G(x^m)``; ``G`` is written in the same symbol ``x``."""
Shift = Tuple[sympy.Expr, int]
"""``(G, k)`` with ``f = G / x^k``."""


def _exponents(expr: sympy.Expr, symbol: sympy.Symbol) -> Optional[Set[int]]:
    """The exponents ``k`` of all occurrences ``x^k`` of ``x = symbol`` in ``expr`` (a bare ``x`` is ``k = 1``),
    or None if ``x`` occurs in another way (e.g. ``sqrt(x)`` or ``x^p``)."""
    if expr == symbol:
        return {1}
    if isinstance(expr, sympy.Pow) and expr.base == symbol:
        return {int(expr.exp)} if expr.exp.is_Integer else None
    exponents: Set[int] = set()
    for arg in expr.args:
        inner = _exponents(arg, symbol)
        if inner is None:
            return None
        exponents |= inner
    return exponents


def _substituted(expr: sympy.Expr, symbol: sympy.Symbol, m: int) -> sympy.Expr:
    """``expr`` with every ``x^k`` replaced by ``x^(k/m)`` (all ``k`` are multiples of ``m``)."""
    if isinstance(expr, sympy.Pow) and expr.base == symbol:
        return symbol ** (expr.exp / m)
    if not expr.args:
        return expr
    return expr.func(*(_substituted(arg, symbol, m) for arg in expr.args))


def monomial_substitution(expr: sympy.Expr, symbol: sympy.Symbol) -> Optional[MonomialSubstitution]:
    """``(G, m)`` with ``expr = G(x^m)`` for ``x = symbol`` and the largest such ``m >= 2``, or None if there is
    none (``x`` does not occur, occurs in a non-integer power, or the exponents have greatest common divisor 1).
    See the module documentation for why the coefficients of ``expr`` are those of ``G``."""
    exponents = _exponents(expr, symbol)
    if not exponents:
        return None
    m = math.gcd(*exponents)
    if m < 2:
        return None
    return _substituted(expr, symbol, m), m


def shift(expr: sympy.Expr, symbol: sympy.Symbol) -> Optional[Shift]:
    """``(G, k)`` with ``expr = G / x^k`` for ``x = symbol``, ``k >= 1`` the total exponent of the factors
    ``x^(-j)`` of the product ``expr``, and ``G`` the product of its other factors, or None if ``expr`` has no
    such factor. Whether ``G / x^k`` is a power series is decided by :func:`low_coefficients_vanish`."""
    if not isinstance(expr, sympy.Mul):
        return None
    k = 0
    rest = []
    for factor in expr.args:
        if isinstance(factor, sympy.Pow) and factor.base == symbol and factor.exp.is_Integer and factor.exp < 0:
            k -= int(factor.exp)
        else:
            rest.append(factor)
    if k == 0:
        return None
    return sympy.Mul(*rest), k


def low_coefficients_vanish(expr: sympy.Expr, symbol: sympy.Symbol, k: int) -> bool:
    """Whether the Taylor coefficients of ``expr`` at ``x = 0`` of degree ``< k`` are proven zero (as
    expressions in the other symbols): the truncated series of degree ``< k`` simplifies to 0. False if the
    series has negative or fractional powers, cannot be computed, or a coefficient is not proven zero."""
    try:
        truncated = sympy.series(expr, symbol, 0, k).removeO()
        return truncated == 0 or sympy.simplify(truncated).is_zero is True
    except Exception:  # pylint: disable=broad-except  # SymPy's series raises many kinds of errors
        return False
