"""
Streaming the Taylor coefficients of a generating function (Prodigy#405).

The GUI draws the coefficients of the output distribution while they are computed. The streams here yield
them one at a time, in order, and each value is exactly the value a single expansion gives:

* A rational function in x (the common case) has a power series whose coefficients satisfy the linear
  recursion of its denominator (:func:`prodigy.distribution.rational_series.rational_taylor_stream`): each
  coefficient comes from the previous ones, cheaper than a series expansion.
* An exponential of a polynomial in x times a rational function (e.g. compound Poisson distributions), or a
  finite sum of such terms, has coefficients that satisfy the linear recursion of ``F' = g'·F``
  (:func:`prodigy.distribution.exponential_series.exp_polynomial_taylor_stream`, Prodigy#455).
* Other closed forms (``exp``, ``log``, algebraic functions) are expanded with doubling orders 4, 8, 16, ...
  (:func:`doubling`): each step is a complete prefix of the series, and the coefficients it adds are yielded
  at once. Without a consumer that wants them early (``incremental=False``), the stream expands to the full
  order in one step, as before.
* An engine may compute the coefficients itself (:meth:`Distribution.native_coefficients`, e.g. GiNaC); then
  the stream is the engine's.

This is shared logic for all engines; the series expansion and the polynomial arithmetic of the recursion
are the CAS operations (SymPy for the engines that have no native stream).
"""
from __future__ import annotations

import itertools
import math
import time
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import sympy

from prodigy.distribution.exponential_series import Window, exp_polynomial_columns, exp_polynomial_taylor_stream
from prodigy.distribution.rational_series import rational_taylor_stream

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.distribution import Distribution

Coefficient = Union[str, sympy.Expr]
"""A coefficient: a SymPy expression, or the text of a coefficient an engine computed natively."""

Clock = Callable[[], float]
Waiting = Optional[Callable[[], None]]
"""Called right before a step that may take long (a series expansion), e.g. to send what is computed so far."""

FIRST_ORDER = 4
"""The first order of a doubling expansion."""
STEP_BUDGET = 0.25
"""Seconds: if the expansion to the full order is estimated to take at most this long, it is taken at once
(streaming it would only add work)."""


def doubling(prefix: Callable[[int], Sequence[sympy.Expr]], limit: int, *, incremental: bool = True,
             clock: Clock = time.perf_counter, on_wait: Waiting = None) -> Iterator[sympy.Expr]:
    """
    The coefficients 0, ..., ``limit - 1`` of a series, from ``prefix(order)`` (the coefficients
    0, ..., ``order - 1``; missing ones are 0). With ``incremental``, the orders are 4, 8, 16, ... (at most
    about twice the work of one expansion, as the cost of an expansion grows at least linearly with the
    order), and each step yields the coefficients it adds; the step to ``limit`` is taken at once when
    the steps so far suggest that it is cheap (:func:`_estimate`). Without ``incremental``, one expansion to ``limit``.
    ``on_wait`` is called before each expansion.
    """
    done = 0
    order = min(limit, FIRST_ORDER) if incremental else limit
    last: Optional[Tuple[int, float]] = None
    while done < limit:
        if on_wait is not None:
            on_wait()
        start = clock()
        coefficients = list(prefix(order))
        elapsed = clock() - start
        coefficients += [sympy.Integer(0)] * (order - len(coefficients))
        yield from coefficients[done:order]
        cheap = _estimate(last, (order, elapsed), limit) <= STEP_BUDGET
        done, last = order, (order, elapsed)
        order = limit if cheap else min(limit, 2 * order)


def _estimate(previous: Optional[Tuple[int, float]], current: Tuple[int, float], limit: int) -> float:
    """The estimated time of an expansion to ``limit`` from the times of the last two steps (order, seconds):
    the cost grows like order^k with k measured between them (between 1 and 3; 2 after the first step)."""
    order, elapsed = current
    exponent = 2.0
    if previous is not None and previous[1] > 0 and elapsed > 0 and order > previous[0]:
        exponent = min(3.0, max(1.0, math.log(elapsed / previous[1]) / math.log(order / previous[0])))
    return elapsed * (limit / order) ** exponent


def series_prefix(expr: sympy.Expr, symbol: sympy.Symbol, order: int) -> List[sympy.Expr]:
    """The coefficients of symbol^0, ..., symbol^(order-1) of the power series of ``expr`` around 0."""
    series = sympy.expand(sympy.series(expr, symbol, 0, order).removeO())
    return [series.coeff(symbol, n) for n in range(order)]


def taylor_stream(expr: sympy.Expr, symbol: sympy.Symbol, limit: int, *,
                  incremental: bool = True, on_wait: Waiting = None) -> Iterator[sympy.Expr]:
    """
    Exactly ``limit`` coefficients of symbol^0, symbol^1, ... of the power series of ``expr`` (a SymPy
    expression; other symbols are constants): by the recursion of the denominator if ``expr`` is rational
    in ``symbol``, by the recursion of an exponential of a polynomial if it is one (times a rational function,
    or a sum of such terms), else by :func:`doubling` of :func:`series_prefix` (``on_wait`` as there).
    """
    if limit <= 0:
        return
    if symbol not in expr.free_symbols:
        yield expr
        yield from itertools.repeat(sympy.Integer(0), limit - 1)
        return
    rational = rational_taylor_stream(expr, symbol)
    if rational is None:
        exponential = exp_polynomial_taylor_stream(expr, symbol)
        if exponential is not None:
            yield from itertools.islice(exponential, limit)
            return
        yield from doubling(lambda order: series_prefix(expr, symbol, order), limit, incremental=incremental,
                            on_wait=on_wait)
        return
    count = 0
    for coefficient in itertools.islice(rational, limit):
        yield coefficient
        count += 1
    # a polynomial: the coefficients beyond its degree are 0
    yield from itertools.repeat(sympy.Integer(0), limit - count)


def taylor_window(expr: sympy.Expr, first: sympy.Symbol, second: sympy.Symbol, first_window: Window,
                  second_window: Window, *, incremental: bool = True,
                  on_wait: Waiting = None) -> Iterator[Iterator[sympy.Expr]]:
    """
    The coefficients of ``first^i second^j`` of ``expr`` for ``i`` and ``j`` in the windows (inclusive ranges
    ``(low, high)``) as columns, lazily: the column of ``i`` yields the coefficients of ``j = low, ..., high``.
    A function of the class of :mod:`prodigy.distribution.exponential_series` (rational functions included) is
    expanded in both variables without leaving the exact kernel, so no column is re-read by SymPy (Prodigy#455);
    other functions by :func:`taylor_stream` in ``first`` and then in ``second`` (``incremental`` and
    ``on_wait`` as there).
    """
    columns = exp_polynomial_columns(expr, first, second, first_window, second_window)
    if columns is not None:
        yield from columns
        return
    (first_low, first_high), (second_low, second_high) = first_window, second_window
    outer = taylor_stream(expr, first, first_high + 1, incremental=incremental, on_wait=on_wait)
    for coefficient in itertools.islice(outer, first_low, None):
        yield itertools.islice(taylor_stream(coefficient, second, second_high + 1, incremental=incremental,
                                             on_wait=on_wait), second_low, None)


def sympy_form(dist: "Distribution") -> Optional[sympy.Expr]:
    """The generating function of ``dist`` as a SymPy expression, or None if it cannot be read."""
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.generating_function import _parse_to_sympy
    function = getattr(dist, "_function", None)
    if isinstance(function, sympy.Basic):
        return function
    try:
        return _parse_to_sympy(str(dist))
    except Exception:  # pylint: disable=broad-except
        return None


def marginal_stream(marginal: "Distribution", variable: str, limit: int, *,
                    incremental: bool = True, on_wait: Waiting = None) -> Iterator[Coefficient]:
    """
    The coefficients Pr[variable = 0], ..., Pr[variable = limit - 1] of the univariate distribution
    ``marginal`` (with infinite support), one at a time: the engine's own stream
    (:meth:`Distribution.native_coefficients`) if it has one, else :func:`taylor_stream` of its SymPy form.
    """
    native = marginal.native_coefficients(variable)
    if native is not None:
        yield from itertools.islice(native, limit)
        return
    expr = sympy_form(marginal)
    if expr is None:
        raise ValueError(f"The generating function of the marginal of {variable} cannot be read.")
    symbol = next((s for s in expr.free_symbols if s.name == variable), sympy.Symbol(variable))
    yield from taylor_stream(expr, symbol, limit, incremental=incremental, on_wait=on_wait)
