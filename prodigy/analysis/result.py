"""
------------------------------
Analysis results (JSON format)
------------------------------

:func:`build_result` runs an :class:`~prodigy.analysis.session.AnalysisSession` and turns
everything it recorded into an :class:`AnalysisResult`. :meth:`AnalysisResult.to_json`
is the exchange format between Prodigy and its front ends (``prodigy analyze --json``,
the web GUI, a language server). Its top-level keys::

    schema, prodigy_version, status, error,
    program      variables, parameters, constants, functions, loops, queries
    input        distribution, engine, normalize, decisions
    distribution closed_form, latex, variables, parameters, finite_support, mass,
                 error_probability, expectations, marginals
    bounds       kind, residual_mass, explanation
    queries      kind, expr, value, lower, upper, bound, line, data
    loops        id, line, cond, strategy, params, kind, residual_mass, certificates, ...
    steps, warnings, log, timing

Exact values are strings in Prodigy's notation; every exact value that is a number also
comes as ``..._float``.

Sound intervals
---------------
Each query carries ``lower`` and ``upper``. What they mean depends on the strategies that
were used for the loops (``bounds.kind``):

* ``exact``: lower = upper = value.
* ``lower``: the computed distribution μ_N lies coefficient-wise below the exact output
  distribution μ, and at most the residual mass r is unaccounted for (mass still inside a
  loop after the last iteration, summed over all loops). For an event φ,
  μ_N(φ) ≤ μ(φ) ≤ μ_N(φ) + r. With conditioning, Prodigy divides by D = 1 − (known
  violation mass); the residual is rescaled by 1/D at the same moment, and
  μ_N(φ)/D ≤ μ(φ)/(1 − err) ≤ (μ_N(φ) + r)/D still holds, whatever the unaccounted mass
  does later (terminate in φ, outside φ, violate an observation, or diverge). No
  termination assumption is involved. Expectations of monotone, non-negative expressions
  get the lower bound only (the upper bound is ∞ while r > 0).
* ``upper``: the computed distribution lies coefficient-wise above μ (Park's lemma); with
  conditioning the violation mass is over-approximated as well, so values are upper bounds.
* ``unknown``: no guaranteed direction (e.g. lower and upper approximations were mixed,
  or a side condition could not be checked); values are reported without interval.
"""
from __future__ import annotations

import itertools
import math
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sympy
from prodigy.pgcl import (Binop, BinopExpr, BoolType, Expr, NatLitExpr, NatType, RealLitExpr,
                           UnopExpr, VarExpr)

from prodigy.analysis.loop_strategies import Bound

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution

SCHEMA = "prodigy/analysis-result@1"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_INFINITY_SPELLINGS = {"\\infty", "∞", "oo", "inf", "infinity"}


def prodigy_version() -> str:
    try:
        from importlib.metadata import version  # pylint: disable=import-outside-toplevel
        return version("prodigy")
    except Exception:  # pylint: disable=broad-except
        return "3.0.0"


# ---------------------------------------------------------------------------------------
# Exact values
# ---------------------------------------------------------------------------------------

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
    return str(expr).replace("**", "^")


def exact_and_float(value: Any) -> Tuple[Optional[str], Optional[float]]:
    if value is None:
        return None, None
    try:
        expr = sym(value)
    except Exception:  # pylint: disable=broad-except
        return str(value), None
    return show(expr), to_float(expr)


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


# ---------------------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------------------

def _row(value: int, prob: Any, residual: Any = None, kind: str = Bound.EXACT) -> Dict[str, Any]:
    exact, number = exact_and_float(prob)
    row: Dict[str, Any] = {"value": value, "prob": exact, "p": number}
    if kind == Bound.LOWER and residual is not None:
        upper, upper_float = exact_and_float(sym(prob) + sym(residual))
        row.update({"upper": upper, "upper_p": upper_float})
    return row


def univariate_table(dist: "Distribution", variable: str, limit: int, *,
                     kind: str = Bound.EXACT, residual: Any = 0,
                     max_finite_terms: int = 2000) -> Dict[str, Any]:
    """The first ``limit`` coefficients of the marginal of ``variable``.

    Finite marginals are listed completely (up to ``max_finite_terms`` states);
    infinite ones by series expansion around 0."""
    marginal = dist.marginal(variable)
    total, total_float = exact_and_float(_safe(marginal.get_probability_mass))
    rows: List[Dict[str, Any]] = []
    finite = bool(_safe(marginal.is_finite))
    truncated = False
    if finite:
        terms = list(itertools.islice(iter(marginal), max_finite_terms + 1))
        truncated = len(terms) > max_finite_terms
        pairs = sorted(((state[variable], prob) for prob, state in terms[:max_finite_terms]),
                       key=lambda pair: pair[0])
        for value, prob in pairs:
            if sym(prob) != 0:
                rows.append(_row(value, prob, residual, kind))
    else:
        coefficients = _series_coefficients(marginal, variable, limit)
        for value, prob in enumerate(coefficients):
            rows.append(_row(value, prob, residual, kind))
        truncated = True
    shown = sum((sym(r["prob"]) for r in rows), sympy.Integer(0)) if rows else sympy.Integer(0)
    tail = None
    if total is not None:
        try:
            tail = sympy.simplify(sym(total) - shown)
        except Exception:  # pylint: disable=broad-except
            tail = None
    tail_exact, tail_float = exact_and_float(tail)
    return {
        "variable": variable,
        "finite_support": finite,
        "rows": rows,
        "truncated": truncated,
        "mass": total,
        "mass_float": total_float,
        "tail_mass": tail_exact,
        "tail_mass_float": tail_float,
    }


def _series_coefficients(marginal: "Distribution", variable: str, limit: int) -> List[Any]:
    # GiNaC-backed distributions (FPS) expand quickly on their own; for sympy-based ones a
    # single series expansion is much faster than iterating term by term.
    expr = None if type(marginal).__name__ == "FPS" else to_sympy(marginal)
    if expr is not None:
        symbol = next((s for s in expr.free_symbols if s.name == variable), None)
        if symbol is None:
            return [expr] + [0] * (limit - 1)
        series = sympy.series(expr, symbol, 0, limit).removeO()
        series = sympy.expand(series)
        return [sympy.simplify(series.coeff(symbol, n)) for n in range(limit)]
    terms = list(itertools.islice(iter(marginal), limit))
    return [prob for prob, _ in terms]


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


# ---------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------

@dataclass
class QueryResult:
    """The answer to one query of the program (``?Pr``, ``?Ex``, ``!Print``, ...)."""

    kind: str
    """``probability``, ``marginal``, ``expectation``, ``print``, ``plot`` or ``optimization``."""
    expr: Optional[str]
    value: Optional[str] = None
    value_float: Optional[float] = None
    lower: Optional[str] = None
    lower_float: Optional[float] = None
    upper: Optional[str] = None
    upper_float: Optional[float] = None
    bound: str = Bound.EXACT
    note: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    line: Optional[int] = None
    column: Optional[int] = None

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
            "bound": self.bound,
            "note": self.note,
            "line": self.line,
            "column": self.column,
            "data": self.data,
        }


def is_monotone_nonnegative(expr: Expr, program=None) -> bool:
    """True if E[expr] can only grow when probability mass is added, i.e. ``expr`` is
    built from natural/boolean variables and non-negative literals with +, *, / (by a
    positive literal) and ^ (by a natural literal)."""
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
    try:
        return expr.to_fraction() > 0  # type: ignore[attr-defined]
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
                upper = sympy.simplify(sym(value) + residual_expr)
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


def finish_query(session: "AnalysisSession", query: QueryResult, *, monotone: bool,
                 probability: bool, known: Optional[Tuple[Any, Any]] = None) -> QueryResult:
    """Fills value_float and the interval of ``query`` from the session's bound state, intersected
    with ``known`` bounds (e.g. from the strategy of the loop right before the query)."""
    query.value, query.value_float = exact_and_float(query.value)
    lower, upper, bound, note = make_interval(query.value, session.bound_kind, session.residual,
                                              monotone=monotone, probability=probability)
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
    if note and not query.note:
        query.note = note
    return query


# ---------------------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------------------

def describe_distribution(dist: "Distribution", *, limit: int, kind: str = Bound.EXACT,
                          residual: Any = 0, expectations: bool = True) -> Dict[str, Any]:
    variables = sorted(dist.get_variables())
    parameters = sorted(dist.get_parameters())
    mass, mass_float = exact_and_float(_safe(dist.get_probability_mass))
    finite = bool(_safe(dist.is_finite))
    summary: Dict[str, Any] = {
        "closed_form": str(dist),
        "latex": to_latex(dist),
        "variables": variables,
        "parameters": parameters,
        "finite_support": finite,
        "mass": mass,
        "mass_float": mass_float,
        "marginals": {},
        "expectations": {},
        "joint": None,
    }
    if limit > 0:
        for variable in variables:
            try:
                summary["marginals"][variable] = univariate_table(dist, variable, limit, kind=kind,
                                                                  residual=residual)
            except Exception as exc:  # pylint: disable=broad-except
                summary["marginals"][variable] = {"variable": variable, "error": str(exc), "rows": []}
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


def bounds_explanation(kind: str, residual: Optional[str]) -> str:
    if kind == Bound.EXACT:
        return "Exact: every loop was handled by a certified method or terminated during unrolling."
    if kind == Bound.LOWER:
        return ("Lower bound: the result lies coefficient-wise below the exact distribution; "
                f"at most {residual} probability mass is unaccounted for. "
                "No termination assumption is involved.")
    if kind == Bound.UPPER:
        return ("Upper bound: the result lies coefficient-wise above the exact distribution "
                "(Park's lemma). It is exact if the loops terminate almost surely.")
    return ("Approximation without guaranteed direction: approximations of different kinds were "
            "combined or a side condition could not be checked.")


# ---------------------------------------------------------------------------------------
# Program outline
# ---------------------------------------------------------------------------------------

def outline_program(program, source: Optional[str] = None) -> Dict[str, Any]:
    """Variables, parameters, loops and queries of ``program`` (for editors and forms)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import _QUERY_RE, LoopRegistry, query_instructions, token_positions
    loops = LoopRegistry(program, source)
    positions = token_positions(source, _QUERY_RE) if source is not None else []
    queries = []
    for index, instr in enumerate(query_instructions(program)):
        line, column = positions[index] if index < len(positions) else (None, None)
        queries.append({"index": index, "kind": _query_kind(instr), "text": str(instr),
                        "line": line, "column": column})
    return {
        "variables": {name: type_name(typ) for name, typ in program.variables.items()},
        "parameters": {name: type_name(typ) for name, typ in program.parameters.items()},
        "constants": {name: str(value) for name, value in program.constants.items()},
        "functions": list(program.functions.keys()),
        "loops": [loop.to_json() for loop in loops],
        "queries": queries,
    }


def type_name(typ) -> str:
    """pGCL spelling of a declared type (``nat``, ``nat [0, 5]``, ``bool``, ``real``, ...)."""
    name = type(typ).__name__
    if name == "NatType":
        bounds = getattr(typ, "bounds", None)
        return f"nat [{bounds.lower}, {bounds.upper}]" if bounds is not None else "nat"
    return {"BoolType": "bool", "RealType": "real", "DistributionType": "dist"}.get(name, str(typ))


def _query_kind(instr) -> str:
    name = type(instr).__name__
    if name == "ExpectationInstr":
        return "expectation"
    if name == "ProbabilityQueryInstr":
        return "marginal" if isinstance(instr.expr, VarExpr) else "probability"
    if name == "PrintInstr":
        return "print"
    if name == "PlotInstr":
        return "plot"
    return "optimization"


# ---------------------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    status: str = "ok"
    """``ok``, ``needs_decision``, ``error``; front ends add ``timeout`` and ``cancelled``."""
    error: Optional[Dict[str, Any]] = None
    program: Optional[Dict[str, Any]] = None
    input: Dict[str, Any] = field(default_factory=dict)
    distribution: Optional[Dict[str, Any]] = None
    error_probability: Optional[str] = None
    error_probability_float: Optional[float] = None
    bounds: Dict[str, Any] = field(default_factory=dict)
    queries: List[QueryResult] = field(default_factory=list)
    loops: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    log: str = ""
    timing: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        distribution = dict(self.distribution) if self.distribution is not None else None
        if distribution is not None:
            distribution["error_probability"] = self.error_probability
            distribution["error_probability_float"] = self.error_probability_float
        return {
            "schema": SCHEMA,
            "prodigy_version": prodigy_version(),
            "status": self.status,
            "error": self.error,
            "program": self.program,
            "input": self.input,
            "distribution": distribution,
            "bounds": self.bounds,
            "queries": [q.to_json() for q in self.queries],
            "loops": self.loops,
            "steps": self.steps,
            "warnings": self.warnings,
            "log": self.log,
            "timing": self.timing,
        }


def _engine_name(config) -> str:
    return config.engine.name.lower()


def build_result(session: "AnalysisSession") -> AnalysisResult:
    """Runs ``session`` and collects the result; analysis errors end up in the result."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import compute_discrete_distribution
    from prodigy.analysis.exceptions import ForwardAnalysisError, VerificationError
    from prodigy.analysis.session import DecisionRequired

    result = AnalysisResult()
    config = session.config
    started = time.perf_counter()
    decisions = {str(k): v.to_json() for k, v in session.provider.decisions.items()}
    result.input = {
        "distribution": str(session.input_distribution) if session.input_distribution is not None else None,
        "engine": _engine_name(config),
        "normalize": config.normalize,
        "decisions": decisions,
    }
    if session.program is not None:
        result.program = outline_program(session.program, session.source)

    analysis_time = 0.0
    with session.activate(), session._output():  # pylint: disable=protected-access
        try:
            program = session.program
            assert program is not None
            if session.input_distribution is None:
                dist = config.factory.one(*program.variables.keys())
            elif isinstance(session.input_distribution, str):
                dist = config.factory.from_expr(session.input_distribution, *program.variables.keys())
            else:
                dist = session.input_distribution
            analysis_start = time.perf_counter()
            final, error_prob = compute_discrete_distribution(
                program, dist, config,
                step_callback=session.record_step if session.record_steps else None)
            analysis_time = time.perf_counter() - analysis_start
            residual, residual_float = exact_and_float(session.residual)
            result.bounds = {
                "kind": session.bound_kind,
                "residual_mass": residual,
                "residual_mass_float": residual_float,
                "explanation": bounds_explanation(session.bound_kind, residual),
            }
            summary_start = time.perf_counter()
            result.distribution = describe_distribution(final, limit=session.coefficient_limit,
                                                        kind=session.bound_kind,
                                                        residual=session.residual)
            result.timing["summary"] = round(time.perf_counter() - summary_start, 6)
            result.error_probability, result.error_probability_float = \
                exact_and_float(_safe(error_prob.get_probability_mass))
        except DecisionRequired as exc:
            analysis_time = time.perf_counter() - started
            result.status = "needs_decision"
            result.error = {"type": "decision", **exc.to_json()}
        except (VerificationError, ForwardAnalysisError) as exc:
            analysis_time = time.perf_counter() - started
            result.status = "error"
            result.error = {"type": type(exc).__name__, "message": str(exc),
                            "details": getattr(exc, "details", None)}
        except Exception as exc:  # pylint: disable=broad-except
            analysis_time = time.perf_counter() - started
            result.status = "error"
            result.error = {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__,
                            "traceback": traceback.format_exc()}

    if result.status != "ok":
        residual, residual_float = exact_and_float(session.residual)
        result.bounds = {"kind": session.bound_kind, "residual_mass": residual,
                         "residual_mass_float": residual_float,
                         "explanation": bounds_explanation(session.bound_kind, residual)}

    result.queries = list(session.queries)
    result.loops = [record.to_json() for record in session.loop_records.values()]
    result.loops.sort(key=lambda r: (r["id"] is None, r["id"] if r["id"] is not None else 0))
    result.steps = session.steps
    result.warnings = list(session.warnings)
    result.log = _ANSI.sub("", session.log.getvalue())
    result.timing["analysis"] = round(analysis_time, 6)
    result.timing["total"] = round(time.perf_counter() - started, 6)
    return result


def _safe(function):
    try:
        return function()
    except Exception:  # pylint: disable=broad-except
        return None
