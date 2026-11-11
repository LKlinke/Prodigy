"""
-----------------------------------------
Coefficient grids of joint distributions
-----------------------------------------

:func:`joint_grid` lists the exact probabilities Pr[x = i, y = j] of two variables for a window
``i in [x_from, x_to]``, ``j in [y_from, y_to]``; the other variables are marginalised by the
engine. The GUI draws them as a heatmap and a 3D view (``POST /api/jobs/<id>/joint``).

The coefficients are Taylor coefficients of the marginal generating function: for a rational
function they come from the exact recursion of :mod:`prodigy.distribution.rational_series`, first
in x (the coefficients are rational functions in y) and then in y; other functions (e.g. ``exp``
of a Poisson distribution) are expanded with SymPy's ``series``. No value is approximated.

The result carries the guarantee of the analysis it comes from (``bound``): for a lower bound,
every cell is a lower bound and at most ``residual_mass`` is not accounted for (the "?" mass,
which may still end up in any cell, outside the window, or nowhere); ``outside_mass`` is the
mass of the computed distribution outside the window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import sympy

from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import exact_and_float, sym, tidy, to_sympy
from prodigy.distribution.rational_series import rational_taylor_coefficients

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


def _taylor(expr: sympy.Expr, symbol: sympy.Symbol, count: int) -> List[sympy.Expr]:
    """The coefficients of symbol^0, ..., symbol^(count-1) of the power series of ``expr``."""
    coefficients = rational_taylor_coefficients(expr, symbol, count)
    if coefficients is None:
        series = sympy.expand(sympy.series(expr, symbol, 0, count).removeO())
        coefficients = [series.coeff(symbol, n) for n in range(count)]
    return list(coefficients) + [sympy.Integer(0)] * (count - len(coefficients))


def _cell(prob: sympy.Expr, kind: str, residual: sympy.Expr) -> Dict[str, Any]:
    exact, number = exact_and_float(tidy(prob))
    cell: Dict[str, Any] = {"prob": exact, "p": number}
    if kind == Bound.LOWER and residual != 0:
        upper, upper_float = exact_and_float(tidy(prob + residual))
        cell.update({"upper": upper, "upper_p": upper_float})
    return cell


def joint_grid(dist: "Distribution", x: str, y: str, *, x_window: Optional[Tuple[int, int]] = None,
               y_window: Optional[Tuple[int, int]] = None, kind: str = Bound.EXACT,
               residual: Any = 0) -> Dict[str, Any]:
    """
    The exact coefficients Pr[x = i, y = j] of ``dist`` for the windows (default: the support of a
    finite distribution, else the first :data:`DEFAULT_AXIS` values), as JSON. ``cells[r][c]`` is
    the cell of ``y = y_values[r]`` and ``x = x_values[c]``.
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
    columns = _taylor(expr, sx, x_high + 1)[x_low:]
    rows: List[List[Optional[Dict[str, Any]]]] = [[None] * len(columns) for _ in range(y_high - y_low + 1)]
    shown = sympy.Integer(0)
    for c, column in enumerate(columns):
        for r, prob in enumerate(_taylor(column, sy, y_high + 1)[y_low:]):
            shown += prob
            rows[r][c] = _cell(prob, kind, residual_expr)

    mass = sym(marginal.get_probability_mass())
    mass_exact, mass_float = exact_and_float(tidy(mass))
    shown_exact, shown_float = exact_and_float(tidy(shown))
    outside_exact, outside_float = exact_and_float(tidy(mass - shown))
    residual_exact, residual_float = exact_and_float(tidy(residual_expr))
    return {
        "x": x,
        "y": y,
        "variables": variables,
        "x_values": list(range(x_low, x_high + 1)),
        "y_values": list(range(y_low, y_high + 1)),
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
