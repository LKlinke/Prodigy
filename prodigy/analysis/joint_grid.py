"""
-----------------------------------------
Coefficient grids of joint distributions
-----------------------------------------

:func:`joint_grid` lists the exact probabilities Pr[x = i, y = j] of two variables for a window
``i in [x_from, x_to]``, ``j in [y_from, y_to]``; the other variables are marginalised by the
engine. The GUI draws them as a heatmap and a 3D view (``POST /api/jobs/<id>/joint``).

The coefficients are Taylor coefficients of the marginal generating function
(:func:`prodigy.distribution.coefficient_stream.taylor_stream`): for a rational function they come
from the exact recursion of :mod:`prodigy.distribution.rational_series`, first in x (the
coefficients are rational functions in y) and then in y; other functions (e.g. ``exp`` of a Poisson
distribution) are expanded with SymPy's ``series``. No value is approximated. They are computed by
growing total degree, so that a streamed grid grows from its corner (#405).

The result carries the guarantee of the analysis it comes from (``bound``): for a lower bound,
every cell is a lower bound and at most ``residual_mass`` is not accounted for (the "?" mass,
which may still end up in any cell, outside the window, or nowhere); ``outside_mass`` is the
mass of the computed distribution outside the window.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple

import sympy

from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import exact_and_float, sym, tidy, to_sympy, upper_end
from prodigy.analysis.streaming import Batcher, Sink
from prodigy.distribution.coefficient_stream import Waiting, taylor_stream

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution import Distribution

MAX_AXIS = 50
"""At most this many values per axis."""
MAX_VALUE = 1000
"""The largest value a window may reach (the expansion always starts at 0)."""
DEFAULT_AXIS = 16
"""Values per axis of the default window of an infinite distribution."""


class GridError(ValueError):
    """The grid cannot be computed for this request (unknown variable, window too large, ...)."""


def check_window(window: Any, name: str) -> Optional[Tuple[int, int]]:
    """``window`` as ``(from, to)``, or None for the default window; raises :class:`GridError`."""
    if window is None:
        return None
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in window)):
        raise GridError(f"The window of {name} must be two whole numbers [from, to].")
    low, high = window
    if not 0 <= low <= high <= MAX_VALUE:
        raise GridError(f"The window of {name} must satisfy 0 ≤ from ≤ to ≤ {MAX_VALUE}.")
    if high - low + 1 > MAX_AXIS:
        raise GridError(f"At most {MAX_AXIS} values of {name} at once.")
    return low, high


def _support(expr: sympy.Expr, x: sympy.Symbol, y: sympy.Symbol) -> Optional[Dict[str, List[int]]]:
    """The bounding box of the exponents of a polynomial in x and y, or None."""
    try:
        poly = sympy.Poly(sympy.expand(sympy.cancel(expr)), x, y)
    except sympy.PolynomialError:
        return None
    monomials = [m for m, c in poly.terms() if c != 0]
    if not monomials:
        return None
    return {str(x): [min(m[0] for m in monomials), max(m[0] for m in monomials)],
            str(y): [min(m[1] for m in monomials), max(m[1] for m in monomials)]}


def _default_window(support: Optional[Sequence[int]]) -> Tuple[int, int]:
    if support is None:
        return 0, DEFAULT_AXIS - 1
    low, high = support
    return low, min(high, low + MAX_AXIS - 1)


def _cells_by_total_degree(expr: sympy.Expr, x: sympy.Symbol, y: sympy.Symbol, x_window: Tuple[int, int],
                           y_window: Tuple[int, int], *, incremental: bool,
                           on_wait: Waiting = None) -> Iterator[Tuple[int, int, sympy.Expr]]:
    """
    The coefficients of x^i y^j of ``expr`` in the windows as ``(r, c, coefficient)`` with ``i = x_low + c``
    and ``j = y_low + r``, by growing total degree ``r + c`` (the heatmap grows from its corner, #405): the
    coefficients of x^i are expanded once, lazily (:func:`taylor_stream`, rational functions in y), and each of
    them is expanded in y as far as the diagonal needs it.
    """
    (x_low, x_high), (y_low, y_high) = x_window, y_window
    cols, rows = x_high - x_low + 1, y_high - y_low + 1
    in_x = itertools.islice(taylor_stream(expr, x, x_high + 1, incremental=incremental, on_wait=on_wait), x_low, None)
    in_y: List[Iterator[sympy.Expr]] = []
    for degree in range(rows + cols - 1):
        if degree < cols:
            in_y.append(itertools.islice(taylor_stream(_next(in_x), y, y_high + 1, incremental=incremental,
                                                       on_wait=on_wait), y_low, None))
        for c in range(max(0, degree - rows + 1), min(cols - 1, degree) + 1):
            yield degree - c, c, _next(in_y[c])


def _next(stream: Iterator[sympy.Expr]) -> sympy.Expr:
    """The next coefficient of a stream of :func:`taylor_stream`, which yields as many as asked for."""
    for value in stream:
        return value
    raise AssertionError("the Taylor stream ended early")  # pragma: no cover - taylor_stream yields `limit` values


def _cell(prob: sympy.Expr, kind: str, residual: sympy.Expr) -> Dict[str, Any]:
    exact, number = exact_and_float(tidy(prob))
    cell: Dict[str, Any] = {"prob": exact, "p": number}
    if kind == Bound.LOWER and residual != 0:
        upper, upper_float = exact_and_float(tidy(upper_end(prob, residual)))
        cell.update({"upper": upper, "upper_p": upper_float})
    return cell


def joint_grid(dist: "Distribution", x: str, y: str, *, x_window: Optional[Tuple[int, int]] = None,
               y_window: Optional[Tuple[int, int]] = None, kind: str = Bound.EXACT,
               residual: Any = 0, stream: Optional[Sink] = None) -> Dict[str, Any]:
    """
    The exact coefficients Pr[x = i, y = j] of ``dist`` for the windows (default: the support of a
    finite distribution, else the first :data:`DEFAULT_AXIS` values), as JSON. ``cells[r][c]`` is
    the cell of ``y = y_values[r]`` and ``x = x_values[c]``. With ``stream``, the cells also go there in
    batches by growing total degree while they are computed (:mod:`prodigy.analysis.streaming`).
    """
    # pylint: disable=too-many-locals
    variables = sorted(dist.get_variables())
    for name in (x, y):
        if name not in variables:
            raise GridError(f"{name!r} is not a variable of the distribution ({', '.join(variables)}).")
    if x == y:
        raise GridError("Choose two different variables.")
    marginal = dist.marginal(x, y)
    expr = to_sympy(marginal)
    if expr is None:
        raise GridError("The generating function of the marginal cannot be read.")
    sx = next((s for s in expr.free_symbols if s.name == x), sympy.Symbol(x))
    sy = next((s for s in expr.free_symbols if s.name == y), sympy.Symbol(y))
    finite = bool(marginal.is_finite())
    support = _support(expr, sx, sy) if finite else None
    x_low, x_high = x_window or _default_window(support[x] if support else None)
    y_low, y_high = y_window or _default_window(support[y] if support else None)
    check_window([x_low, x_high], x)
    check_window([y_low, y_high], y)

    residual_expr = sym(residual) if residual is not None else sympy.Integer(0)
    x_values, y_values = list(range(x_low, x_high + 1)), list(range(y_low, y_high + 1))
    batcher = None
    if stream is not None:
        residual_exact, residual_float = exact_and_float(tidy(residual_expr) if kind == Bound.LOWER else None)
        batcher = Batcher(stream, "cells", {"kind": "grid"})
        stream({"kind": "grid", "cells": [], "header": {
            "x": x, "y": y, "variables": variables, "x_values": x_values, "y_values": y_values,
            "finite_support": finite, "support": support, "bound": kind, "residual_mass": residual_exact,
            "residual_mass_float": residual_float}})
    rows: List[List[Optional[Dict[str, Any]]]] = [[None] * len(x_values) for _ in y_values]
    shown = sympy.Integer(0)
    for r, c, prob in _cells_by_total_degree(expr, sx, sy, (x_low, x_high), (y_low, y_high),
                                             incremental=stream is not None,
                                             on_wait=None if batcher is None else batcher.flush):
        shown += prob
        rows[r][c] = _cell(prob, kind, residual_expr)
        if batcher is not None:
            batcher.add([r, c, rows[r][c]])
    if batcher is not None:
        batcher.flush()

    mass = sym(marginal.get_probability_mass())
    mass_exact, mass_float = exact_and_float(tidy(mass))
    shown_exact, shown_float = exact_and_float(tidy(shown))
    outside_exact, outside_float = exact_and_float(tidy(mass - shown))
    residual_exact, residual_float = exact_and_float(tidy(residual_expr))
    return {
        "x": x,
        "y": y,
        "variables": variables,
        "x_values": x_values,
        "y_values": y_values,
        "cells": rows,
        "finite_support": finite,
        "support": support,
        "bound": kind,
        "mass": mass_exact,
        "mass_float": mass_float,
        "shown_mass": shown_exact,
        "shown_mass_float": shown_float,
        "outside_mass": outside_exact,
        "outside_mass_float": outside_float,
        "residual_mass": residual_exact if kind == Bound.LOWER else None,
        "residual_mass_float": residual_float if kind == Bound.LOWER else None,
        "limits": {"max_axis": MAX_AXIS, "max_value": MAX_VALUE},
    }
