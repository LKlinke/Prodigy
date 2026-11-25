"""
-------------------------------
Values of the analysis results
-------------------------------

The spelling of values (closed forms, numbers, tables of coefficients) and the values of queries
(:class:`QueryResult`, intervals with their guarantee). Used while the analysis runs (query handler,
step trace, nested inference) and when :mod:`prodigy.analysis.result` assembles the result; imports
only the vocabulary of the analysis, the distributions and the language.
"""
from __future__ import annotations

import functools
import logging
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, TypeVar

import sympy
from prodigy.language import (Binop, BinopExpr, BoolType, Expr, NatLitExpr, NatType, Program, RealLitExpr, UnopExpr,
                              VarExpr)

from prodigy.analysis.loops.api import Bound
from prodigy.analysis.streaming import Batcher, Sink
from prodigy.distribution.coefficient_stream import marginal_stream
from prodigy.distribution.display import display_form
from prodigy.distribution.equality import sympy_text
from prodigy.util.logger import log_setup

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution

T = TypeVar("T")

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

UpperSource = Literal["certificate", "residual"]
"""Where the upper end of an interval comes from: a loop's upper certificate, or the residual mass."""

CERTIFICATE: UpperSource = "certificate"
RESIDUAL: UpperSource = "residual"

_INFINITY_SPELLINGS = {"\\infty", "∞", "oo", "inf", "infinity", "Infinity"}  # Infinity: str(RealLitExpr.infinity())


def error_type(exc: BaseException) -> str:
    """
    The `type` of an error in the result: the class name, except that every subclass of
    NotImplementedError (e.g. `ComparisonNotImplemented`) is reported as `NotImplementedError`,
    the type consumers of the result check for; the reason is in the message.
    """
    return "NotImplementedError" if isinstance(exc, NotImplementedError) else type(exc).__name__


def sym(value: Any) -> sympy.Expr:
    """Parses Prodigy's string notation (``1/3``, ``p^2``, ``\\infty``) into sympy."""
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.generating_function import _parse_to_sympy
    if isinstance(value, sympy.Basic):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return sympy.nsimplify(value) if isinstance(value, float) else sympy.Integer(value)
    text = str(value).strip()
    if text in _INFINITY_SPELLINGS:
        return sympy.oo
    return _parse_to_sympy(text)

_TIDY_MAX_OPS = 40

_TIDY_MAX_BITS = 128


def tidy(expr: sympy.Expr) -> sympy.Expr:
    """
    A readable form of an exact value for display, with bounded effort; the value never changes.

    ``sympy.simplify`` is only used on small expressions (at most ``_TIDY_MAX_OPS`` operations and
    integers below ``2**_TIDY_MAX_BITS``) without ``log``: its ``logcombine`` turns ``q*log(3/2)``
    into ``log((3/2)**q)`` and computes that power, which for the tail mass of ``logdist(1/3)`` (a
    rational ``q`` with a 90-bit denominator) does not finish. A rational function over the
    rationals is put over a common denominator instead (exact polynomial arithmetic); any other
    expression is returned as it is.

    A small expression is simplified as a rational multiple ``q * P`` of its primitive part ``P``
    (``as_content_primitive``: ``q`` a positive rational, ``P`` without rational content), and the
    simplified form of ``P`` is kept (:func:`_simplified`). The cells of a joint grid and the rows of
    a marginal are multiples of few primitive parts (#469): every coefficient of a Poisson factor is
    ``q * exp(-λ)``, and for a fixed value of one variable the coefficients of a product form share
    one polynomial in the parameters, so that one ``simplify`` serves them all. ``q * P == expr``, so
    the value never changes. If the simplified ``P`` is a product or a power, ``q`` stays a factor in
    front of it and the form is that of ``simplify(expr)``; if it is a sum, SymPy would distribute ``q``
    over its terms, so ``expr`` itself is simplified (and kept). ``tests/analysis/test_result_display.py``
    and ``tests/web/test_grid_cache.py`` compare the forms with ``simplify(expr)`` on generic values.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.equality import is_rational_over_q
    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        return expr
    small = sympy.count_ops(expr) <= _TIDY_MAX_OPS and all(
        abs(atom.p).bit_length() <= _TIDY_MAX_BITS and abs(atom.q).bit_length() <= _TIDY_MAX_BITS
        for atom in expr.atoms(sympy.Rational))
    if small and not expr.has(sympy.log):
        content, primitive = expr.as_content_primitive()
        simplified = _simplified(primitive)
        # q times a sum is distributed over its terms by SymPy, and simplify may write the multiple as a
        # product instead: then simplify decides the form of the multiple itself
        return _simplified(expr) if simplified.is_Add and content != 1 else content * simplified
    if is_rational_over_q(expr):
        return sympy.cancel(sympy.together(expr))
    return expr


@functools.lru_cache(maxsize=1024)
def _simplified(expr: sympy.Expr) -> sympy.Expr:
    """``sympy.simplify(expr)``, kept for the next value with the same primitive part (see :func:`tidy`)."""
    return sympy.simplify(expr)


def to_float(expr: Any) -> Optional[float]:
    try:
        expr = sym(expr)
        if expr.free_symbols:
            return None
        if expr in (sympy.oo, -sympy.oo, sympy.zoo) or expr is sympy.nan:
            return None
        number = float(expr)
        return number if math.isfinite(number) else None
    except Exception:  # pylint: disable=broad-except
        return None


def show(expr: Any) -> Optional[str]:
    """Prodigy's notation for an exact value (``^`` for powers, ``\\infty``)."""
    if expr is None:
        return None
    if isinstance(expr, str):
        return expr
    if expr == sympy.oo:
        return "\\infty"
    # Euler's number as exp(1): sym reads "E" back as a symbol
    return sympy_text(expr).replace("**", "^")


def exact_and_float(value: Any) -> Tuple[Optional[str], Optional[float]]:
    if value is None:
        return None, None
    try:
        expr = sym(value)
    except Exception:  # pylint: disable=broad-except
        return str(value), None
    return show(expr), to_float(expr)

_NON_FINITE_NAMES = frozenset({"nan", "NaN", "zoo", "oo", "inf", "Inf", "infinity", "Infinity",
                                "ComplexInfinity", "UnsignedInfinity"})


def is_non_finite(value: Any, *, names: frozenset = frozenset(), infinity: bool = False) -> bool:
    """Whether ``value`` (a SymPy expression or a string in any engine's notation) contains ``nan``,
    ``zoo`` or ``oo``; ``infinity=True`` accepts a value that is exactly ∞.

    ``names`` are the program's variables and parameters, which are never taken for one of
    the spellings of ``nan`` or ∞ (a variable may be called ``inf``).
    """
    if value is None:
        return False
    if isinstance(value, sympy.Basic):
        expr: Optional[sympy.Basic] = value
    else:
        try:
            expr = sym(value)
        except Exception:  # pylint: disable=broad-except
            expr = None
    if expr is not None:
        if infinity and expr == sympy.oo:
            return False
        if expr.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            return True
        return any(str(symbol) in _NON_FINITE_NAMES and str(symbol) not in names
                   for symbol in expr.free_symbols)
    text = str(value).strip()
    if infinity and text in _INFINITY_SPELLINGS:
        return False
    tokens = set(re.findall(r"[A-Za-z_]\w*", text))
    return "∞" in text or bool((tokens & _NON_FINITE_NAMES) - names)


def to_sympy(dist: "Distribution") -> Optional[sympy.Expr]:
    function = getattr(dist, "_function", None)
    if isinstance(function, sympy.Basic):
        return function
    try:
        return sym(str(dist))
    except Exception:  # pylint: disable=broad-except
        return None


def to_latex(dist: "Distribution") -> Optional[str]:
    expr = to_sympy(dist)
    if expr is None:
        return None
    try:
        return sympy.latex(expr)
    except Exception:  # pylint: disable=broad-except
        return None


def upper_end(prob: Any, residual: Any) -> sympy.Expr:
    """The upper end of the interval of a coefficient of a lower bound: ``min(1, prob + residual)``; the cap
    applies where both are numbers (with parameters the sum is kept)."""
    upper = sym(prob) + sym(residual)
    try:
        if upper.is_number and bool(upper > 1):
            return sympy.Integer(1)
    except TypeError:  # not comparable (e.g. a complex number): keep the sum
        pass
    return upper


def upper_with_source(prob: Any, residual: Any, upper: Any = None) -> Tuple[sympy.Expr, UpperSource]:
    """The upper end of the interval of a coefficient (or event) of a lower bound ``prob`` with residual mass
    ``residual``, and where it comes from: ``min(U, prob + residual, 1)``, where ``upper`` is the value U of a
    proven upper bound (:mod:`prodigy.analysis.upper_track`) or ``None``. All three are proven upper bounds of
    the exact value. U is taken only where it is a finite number that is provably below ``prob + residual``
    (capped at 1) and not below ``prob`` (which would contradict the bounds: then neither is trusted to be
    tighter, and the residual end is kept)."""
    fallback = upper_end(prob, residual)
    if upper is None:
        return fallback, RESIDUAL
    try:
        value = tidy(sym(upper))
        if (value.is_number and value.is_real and value.is_finite and bool(value < fallback)
                and not bool(value < sym(prob))):
            return value, CERTIFICATE
        if value.is_number and value.is_real and bool(value < sym(prob)):
            logger.warning("An upper bound %s lies below the lower bound %s; it is not used.", value, prob)
    except (TypeError, ValueError, ArithmeticError, sympy.SympifyError):
        pass  # not comparable (e.g. parameters): the residual end stays
    return fallback, RESIDUAL


def _row(value: int, prob: Any, residual: Any = None, kind: str = Bound.EXACT, upper: Any = None
         ) -> Dict[str, Any]:
    exact, number = exact_and_float(prob)
    row: Dict[str, Any] = {"value": value, "prob": exact, "p": number}
    if kind == Bound.LOWER and residual is not None:
        end, source = upper_with_source(prob, residual, upper)
        upper_exact, upper_float = exact_and_float(end)
        row.update({"upper": upper_exact, "upper_p": upper_float, "upper_source": source})
    return row

MAX_UPPER_VALUE = 2000
"""The largest value of a variable whose coefficient in a marginal of the upper track is computed; beyond it the
upper ends come from the residual mass."""


class UpperCoefficients:
    """The coefficients of the marginal of one variable of a proven upper bound (:mod:`prodigy.analysis.upper_track`),
    computed on demand by series expansion. A coefficient that cannot be computed is ``None``: its cell falls back
    to the residual mass."""

    def __init__(self, upper: Optional["Distribution"], variable: str, size: int):
        self._values: List[sympy.Expr] = []
        self._stream: Optional[Iterator[Any]] = None
        if upper is None or size <= 0:
            return
        try:
            marginal = upper.marginal(variable)
            self._stream = marginal_stream(marginal, variable, min(size, MAX_UPPER_VALUE + 1), incremental=False)
        except Exception:  # pylint: disable=broad-except  # the upper ends are only an addition
            logger.debug("No coefficients of the upper bound for %s", variable, exc_info=True)

    def get(self, value: int) -> Optional[sympy.Expr]:
        """The coefficient of ``variable = value``, or ``None``."""
        while self._stream is not None and len(self._values) <= value:
            try:
                self._values.append(tidy(sym(next(self._stream))))
            except StopIteration:
                self._stream = None
            except Exception:  # pylint: disable=broad-except  # the upper ends are only an addition
                logger.debug("A coefficient of the upper bound failed", exc_info=True)
                self._stream = None
        return self._values[value] if 0 <= value < len(self._values) else None


def univariate_table(dist: "Distribution", variable: str, limit: int, *,
                     kind: str = Bound.EXACT, residual: Any = 0,
                     max_finite_terms: int = 2000, stream: Optional[Batcher] = None,
                     upper: Optional["Distribution"] = None) -> Dict[str, Any]:
    """The first ``limit`` coefficients of the marginal of ``variable``.

    Finite marginals are listed completely (up to ``max_finite_terms`` states);
    infinite ones by series expansion around 0 (:func:`~prodigy.distribution.coefficient_stream.marginal_stream`).
    With ``stream``, each row also goes to the batcher as soon as it is computed (#405); the rows are the
    same. For a lower bound, each row has an upper end (:func:`upper_with_source`), also from ``upper``, a
    proven upper bound on the exact distribution, if given."""
    marginal = dist.marginal(variable)
    upper_known = upper is not None and kind == Bound.LOWER
    total, total_float = exact_and_float(_safe(marginal.get_probability_mass))
    rows: List[Dict[str, Any]] = []
    finite = bool(_safe(marginal.is_finite))
    truncated = False
    if finite:
        terms = list(itertools.islice(iter(marginal), max_finite_terms + 1))
        truncated = len(terms) > max_finite_terms
        pairs = sorted(((state[variable], prob) for prob, state in terms[:max_finite_terms]),
                       key=lambda pair: pair[0])
        bounds = UpperCoefficients(upper if upper_known else None, variable,
                                   max((int(value) for value, _ in pairs), default=-1) + 1)
        for value, prob in pairs:
            if sym(prob) != 0:
                rows.append(_row(value, prob, residual, kind, bounds.get(int(value))))
        if stream is not None:
            for row in rows:
                stream.add(row)
    else:
        coefficients = marginal_stream(marginal, variable, limit, incremental=stream is not None,
                                       on_wait=None if stream is None else stream.flush)
        bounds = UpperCoefficients(upper if upper_known else None, variable, limit)
        for value, prob in enumerate(coefficients):
            row = _row(value, tidy(prob), residual, kind, bounds.get(value))
            rows.append(row)
            if stream is not None:
                stream.add(row)
        truncated = True
    shown = sum((sym(r["prob"]) for r in rows), sympy.Integer(0)) if rows else sympy.Integer(0)
    tail = None
    if total is not None:
        try:
            tail = tidy(sym(total) - shown)
        except Exception:  # pylint: disable=broad-except
            tail = None
    tail_exact, tail_float = exact_and_float(tail)
    shown_exact, shown_float = exact_and_float(tidy(shown))
    return {
        "variable": variable,
        "finite_support": finite,
        "rows": rows,
        "truncated": truncated,
        "mass": total,
        "mass_float": total_float,
        "shown_mass": shown_exact,
        "shown_mass_float": shown_float,
        "tail_mass": tail_exact,
        "tail_mass_float": tail_float,
    }


def joint_table(dist: "Distribution", limit: int) -> Optional[Dict[str, Any]]:
    """All states of a finite distribution, sorted, if there are at most ``limit`` of them."""
    if not _safe(dist.is_finite):
        return None
    terms = list(itertools.islice(iter(dist), limit + 1))
    if len(terms) > limit:
        return {"variables": sorted(dist.get_variables()), "rows": [], "truncated": True}
    variables = sorted(dist.get_variables())
    rows: List[Dict[str, Any]] = []
    for prob, state in terms:
        if sym(prob) == 0:
            continue
        exact, number = exact_and_float(prob)
        rows.append({"state": {v: state[v] for v in variables if v in state.valuations},
                     "prob": exact, "p": number})
    rows.sort(key=lambda r: tuple(r["state"].get(v, 0) for v in variables))
    return {"variables": variables, "rows": rows, "truncated": False}


@dataclass
class QueryResult:
    """The answer to one query of the program (``?Pr``, ``?Ex``, ``!Print``, ...)."""

    kind: str
    """``probability``, ``marginal``, ``expectation``, ``variance``, ``covariance``, ``print``, ``plot`` or
    ``optimization``."""
    expr: Optional[str]
    value: Optional[str] = None
    value_float: Optional[float] = None
    lower: Optional[str] = None
    lower_float: Optional[float] = None
    upper: Optional[str] = None
    upper_float: Optional[float] = None
    upper_source: Optional[UpperSource] = None
    """For a lower bound with an upper end: whether it comes from a certificate or the residual mass."""
    bound: str = Bound.EXACT
    note: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    line: Optional[int] = None
    column: Optional[int] = None
    error: Optional[Dict[str, str]] = None
    """``{"type", "message"}`` of a query that could not be computed, else None."""

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "expr": self.expr,
            "value": self.value,
            "value_float": self.value_float,
            "lower": self.lower,
            "lower_float": self.lower_float,
            "upper": self.upper,
            "upper_float": self.upper_float,
            "upper_source": self.upper_source,
            "bound": self.bound,
            "note": self.note,
            "line": self.line,
            "column": self.column,
            "data": self.data,
            "error": self.error,
        }


def is_monotone_nonnegative(expr: Expr, program: Optional[Program] = None) -> bool:
    """True if E[expr] can only grow when probability mass is added, i.e. ``expr`` is
    built from natural/boolean variables and non-negative literals with ``+``, ``*``, ``/`` (by a
    positive literal) and ``^`` (by a natural literal)."""
    variables = getattr(program, "variables", {}) or {}
    if isinstance(expr, VarExpr):
        typ = variables.get(expr.var)
        return typ is None or isinstance(typ, (NatType, BoolType))
    if isinstance(expr, NatLitExpr):
        return True
    if isinstance(expr, RealLitExpr):
        try:
            return expr.to_fraction() >= 0
        except Exception:  # pylint: disable=broad-except
            return False
    if isinstance(expr, BinopExpr):
        if expr.operator in (Binop.PLUS, Binop.TIMES):
            return is_monotone_nonnegative(expr.lhs, program) and \
                is_monotone_nonnegative(expr.rhs, program)
        if expr.operator == Binop.DIVIDE:
            return is_monotone_nonnegative(expr.lhs, program) and \
                isinstance(expr.rhs, (NatLitExpr, RealLitExpr)) and _positive_literal(expr.rhs)
        if expr.operator == Binop.POWER:
            return is_monotone_nonnegative(expr.lhs, program) and isinstance(expr.rhs, NatLitExpr)
    if isinstance(expr, UnopExpr):
        return False
    return False


def _positive_literal(expr: Expr) -> bool:
    if isinstance(expr, NatLitExpr):
        return expr.value > 0
    if not isinstance(expr, RealLitExpr):
        return False
    try:
        return expr.to_fraction() > 0
    except Exception:  # pylint: disable=broad-except
        return False


def make_interval(value: Any, kind: str, residual: Any, *, monotone: bool,
                  probability: bool) -> Tuple[Any, Any, str, str]:
    """(lower, upper, bound, note) for a query value, see the module documentation."""
    if kind == Bound.EXACT:
        return value, value, Bound.EXACT, ""
    if not monotone:
        return None, None, Bound.UNKNOWN, \
            "No bound: the expression is not monotone in the distribution."
    if kind == Bound.LOWER:
        try:
            residual_expr = sym(residual)
        except Exception:  # pylint: disable=broad-except
            residual_expr = None
        if residual_expr is not None and residual_expr == 0:
            return value, value, Bound.EXACT, ""
        if probability and residual_expr is not None:
            try:
                upper = tidy(upper_end(value, residual_expr))
            except Exception:  # pylint: disable=broad-except
                upper = None
            return value, upper, Bound.LOWER, ""
        return value, None, Bound.LOWER, \
            "Lower bound only: mass not yet accounted for may carry arbitrarily large values."
    if kind == Bound.UPPER:
        return None, value, Bound.UPPER, ""
    return None, None, Bound.UNKNOWN, "No guaranteed bound for this approximation."


def tighten_interval(lower: Any, upper: Any, bound: str, known: Tuple[Any, Any]) -> Tuple[Any, Any, str]:
    """Intersects an interval with bounds known from elsewhere (``None``: unknown side)."""
    if bound not in (Bound.LOWER, Bound.EXACT):
        return lower, upper, bound
    known_lower, known_upper = known
    try:
        if known_lower is not None and (lower is None or sym(known_lower) > sym(lower)):
            lower = sym(known_lower)
        if known_upper is not None and (upper is None or sym(known_upper) < sym(upper)):
            upper = sym(known_upper)
    except TypeError:  # symbolic values that cannot be compared
        return lower, upper, bound
    if lower is not None and upper is not None and sym(lower) == sym(upper):
        bound = Bound.EXACT
    return lower, upper, bound


def with_upper(known: Optional[Tuple[Any, Any]], upper: Any) -> Optional[Tuple[Any, Any]]:
    """``known`` bounds of a query (``None``: none) with the upper bound ``upper`` added (``None``: none): both
    are proven, so the smaller one holds. Undecided comparisons keep the bound known before."""
    if upper is None:
        return known
    if known is None:
        return None, upper
    lower, before = known
    if before is None:
        return lower, upper
    try:
        return lower, upper if bool(sym(upper) < sym(before)) else before
    except TypeError:
        return known


def finish_query(session: "AnalysisSession", query: QueryResult, *, monotone: bool,
                 probability: bool, known: Optional[Tuple[Any, Any]] = None) -> QueryResult:
    """Fills value_float and the interval of ``query`` from the session's bound state, intersected
    with ``known`` bounds (e.g. from the strategy of the loop right before the query, or from the upper track,
    :mod:`prodigy.analysis.upper_track`). ``upper_source`` says whether the upper end of a lower bound is one of
    them (``certificate``) or the residual end (``residual``)."""
    query.value, query.value_float = exact_and_float(query.value)
    lower, upper, bound, note = make_interval(query.value, session.bound_kind, session.residual,
                                              monotone=monotone, probability=probability)
    residual_upper = upper
    if known is not None and monotone:
        lower, upper, bound = tighten_interval(lower, upper, bound, known)
        if bound == Bound.LOWER and lower is not None:
            # the value of a lower-bound query is its lower end
            query.value, query.value_float = exact_and_float(lower)
        if upper is not None:
            note = ""
    query.lower, query.lower_float = exact_and_float(lower)
    query.upper, query.upper_float = exact_and_float(upper)
    query.bound = bound
    if bound == Bound.LOWER and upper is not None:
        query.upper_source = RESIDUAL if residual_upper is not None and upper is residual_upper else CERTIFICATE
    if note and not query.note:
        query.note = note
    return query


def describe_distribution(dist: "Distribution", *, limit: int, kind: str = Bound.EXACT,
                          residual: Any = 0, expectations: bool = True,
                          stream: Optional[Sink] = None, simplify: bool = False,
                          upper: Optional["Distribution"] = None) -> Dict[str, Any]:
    """The JSON summary of an output distribution; with ``stream``, the coefficients of the marginals also
    go to ``stream`` in batches while they are computed (:mod:`prodigy.analysis.streaming`).

    With ``simplify``, ``closed_form`` and ``latex`` show the form that the bounded search of
    :func:`~prodigy.distribution.display.display_form` finds, ``display`` says which, and ``computed_form``
    and ``computed_latex`` keep the computed form if it differs. Everything else is computed from ``dist``."""
    variables = sorted(dist.get_variables())
    parameters = sorted(dist.get_parameters())
    mass, mass_float = exact_and_float(_safe(dist.get_probability_mass))
    finite = bool(_safe(dist.is_finite))
    shown = display_form(dist) if simplify else None
    summary: Dict[str, Any] = {
        "closed_form": str(dist if shown is None else shown.distribution),
        "latex": to_latex(dist if shown is None else shown.distribution),
        "variables": variables,
        "parameters": parameters,
        "finite_support": finite,
        "mass": mass,
        "mass_float": mass_float,
        "marginals": {},
        "expectations": {},
        "joint": None,
    }
    if shown is not None:
        summary["display"] = shown.to_json()
        if shown.simplified:
            summary["computed_form"] = str(dist)
            summary["computed_latex"] = to_latex(dist)
    if limit > 0:
        if stream is not None:
            residual_exact, residual_float = exact_and_float(residual if kind == Bound.LOWER else None)
            stream({"kind": "coefficients", "event": "start", "variables": list(variables), "limit": limit,
                    "bound": kind, "residual_mass": residual_exact, "residual_mass_float": residual_float})
        for variable in variables:
            batcher = None if stream is None else Batcher(stream, "rows", {"kind": "coefficients",
                                                                             "variable": variable})
            try:
                table = univariate_table(dist, variable, limit, kind=kind, residual=residual, stream=batcher,
                                         upper=upper)
                summary["marginals"][variable] = table
                if batcher is not None:
                    batcher.flush(done=True, finite_support=table["finite_support"])
            except Exception as exc:  # pylint: disable=broad-except
                summary["marginals"][variable] = {"variable": variable, "error": str(exc), "rows": []}
                if batcher is not None:
                    batcher.flush(done=True, error=str(exc))
        summary["joint"] = _safe(lambda: joint_table(dist, limit * 4))
    if expectations:
        for variable in variables:
            try:
                value, number = exact_and_float(dist.get_expected_value_of(variable))
                summary["expectations"][variable] = {"value": value, "value_float": number}
            except Exception as exc:  # pylint: disable=broad-except
                summary["expectations"][variable] = {"value": None, "value_float": None,
                                                     "error": str(exc)}
    return summary


def bounds_explanation(kind: str, residual: Optional[str], upper: bool = False) -> str:
    if kind == Bound.EXACT:
        return "Exact: every loop was handled by a certified method or terminated during unrolling."
    if kind == Bound.LOWER:
        return ("Lower bound: the result lies coefficient-wise below the exact distribution; "
                f"at most {residual} probability mass is unaccounted for. "
                "No termination assumption is involved."
                + (" A proven upper bound per state (from a loop's certificate) tightens the upper ends."
                   if upper else ""))
    if kind == Bound.UPPER:
        return ("Upper bound: the result lies coefficient-wise above the exact distribution "
                "(Park's lemma). It is exact if the loops terminate almost surely.")
    return ("Approximation without guaranteed direction: approximations of different kinds were "
            "combined or a side condition could not be checked.")


def _safe(function: Callable[[], T]) -> Optional[T]:
    try:
        return function()
    except Exception:  # pylint: disable=broad-except
        return None
