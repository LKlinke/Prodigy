"""
------------------------------
Analysis results (JSON format)
------------------------------

:meth:`AnalysisSession.run <prodigy.analysis.session.AnalysisSession.run>` runs an analysis and turns
everything it recorded into an :class:`AnalysisResult`. :meth:`AnalysisResult.to_json`
is the exchange format between Prodigy and its front ends (``prodigy analyze --json``,
the web GUI, a language server). Its top-level keys::

    schema, prodigy_version, status, error, syntax,
    program      variables, parameters, constants, functions, loops, queries
    input        distribution, engine, normalize, decisions
    distribution closed_form, latex, variables, parameters, finite_support, mass,
                 error_probability, expectations, marginals, display, computed_form, computed_latex
    bounds       kind, residual_mass, explanation, upper (closed_form, origin) or null
    distributions variable, depends_on_state, values (probability, infer, function, arguments,
                 closed, closed_form, latex, normalizing_constant, mean, finite_support, table)
    queries      kind, expr, value, lower, upper, bound, line, data, error
    loops        id, line, cond, strategy, params, kind, residual_mass, certificates, ...
    steps        index, line, column, kind, instruction, before, after, bound, time, loop, watches, ...
    trace        limits, watches, hints
    warnings, log, timing
    equivalence  only in the result of a comparison of two programs (the GUI's equivalence view):
                 verdict, reason, relation, constraints, witness, check, input, other

Comparing two programs
----------------------
The result of comparing two programs (``POST /api/equivalence``,
:func:`prodigy.web.requests.run_equivalence`) is the result of the first program with the extra key
``equivalence``; ``equivalence.other`` is the result of the second program, in this format as well.
``verdict`` is ``equivalent`` only if equality was proven with both programs analysed exactly,
``not_equivalent`` only with a ``witness`` (an output state whose probabilities provably differ), and
``unknown`` otherwise, with the ``reason``; see :mod:`prodigy.analysis.equivalence.verdict`. Consumers
that do not know the key read the result of the first program as before.

Exact values are strings in Prodigy's notation; every exact value that is a number also
comes as ``..._float``. Expressions taken from the program (query expressions, loop guards,
the instructions of ``steps``) are written in ``.ppy``, e.g. ``x == 1 and y ** 2 < 3``; ``syntax`` is always
``ppy`` (kept for compatibility; the ``.pgcl`` syntax was removed in v0.6.0).

The shown form
--------------
The option ``use_simplification`` does not change the analysis. With it (the GUI's
"Simplify the shown result", on unless ``prodigy --no-simplification``), ``distribution.closed_form`` and
``distribution.latex`` are the shortest form that a bounded search finds
(:mod:`prodigy.distribution.display`): the same generating function, checked to be equal, perhaps
cancelled or factored. ``distribution.display`` says which form is shown (``form``: ``computed``,
``together``, ``cancel`` or ``factor``; ``simplified``; ``note``, e.g. why the computed form was kept;
``computed_ops`` and ``shown_ops``, the sizes by ``count_ops``; ``seconds``). If the shown form differs,
``computed_form`` and ``computed_latex`` are the form as computed. Without the option there is no
``display``, and ``closed_form`` is the computed form. Coefficients, mass and expectations never depend on it.

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

  Where a loop strategy proved an upper bound U on the exact output (the EVT sandwich), U is carried
  through the rest of the program (:mod:`prodigy.analysis.upper_track`), and each upper end is
  min(U, μ_N + r, 1). ``upper_source`` (rows of marginal tables and joint grids, queries) says which end it
  is: ``certificate`` (from U, strictly below μ_N + r) or ``residual`` (μ_N + r, capped at 1).
  ``bounds.upper`` holds U at the end of the program (``closed_form``, ``origin``), or is ``null``.
* ``upper``: the computed distribution lies coefficient-wise above μ (Park's lemma); with
  conditioning the violation mass is over-approximated as well, so values are upper bounds.
* ``unknown``: no guaranteed direction (e.g. lower and upper approximations were mixed,
  or a side condition could not be checked); values are reported without interval.

A result whose ``status`` is not ``ok`` (``error``, ``needs_decision``, and ``timeout`` or
``cancelled`` from front ends) never carries a guarantee: its ``bounds.kind`` is ``unknown``
(results made up by a front end may omit ``bounds``) and its residual mass is ``null``.
Queries answered before the failure keep their own intervals, which were computed from the
distribution at that point.

Queries that cannot be computed
-------------------------------
A query does not change the distribution, so a query that cannot be computed (``NotImplementedError``,
e.g. a condition on variables with infinite support that the engine cannot filter on, or a marginal
that is not a finite measure) fails only itself: it keeps its place in ``queries`` with ``error``
``{"type", "message"}``, ``value``, ``lower`` and ``upper`` ``null``, ``bound`` ``unknown`` and the
message in ``note``, and ``warnings`` names it. ``status``, the distribution and the other queries
are not affected. ``error`` is ``null`` for every query that was computed; results from before this
field have no ``error``.

Values of type dist
-------------------
``distributions`` lists the ``dist`` variables at the end of the program (nested inference, see
:mod:`prodigy.analysis.nested_inference`), apart from the joint generating function of the program
state in ``distribution``, which does not contain them. A variable has one entry in ``values`` per
value it can have; ``probability`` is the probability of that value (1 if it does not depend on
the state) and carries the guarantee of ``bounds``. ``closed_form`` is the normalised generating
function of the value, in the name of the variable; it is always exact (``infer`` is only computed
from an exact analysis), as are ``normalizing_constant`` (Z), ``mean`` (``Ex[d]``, maybe ``oo``) and
the coefficients in ``table`` (the format of a marginal). A value whose arguments have infinite
support (``closed`` false) has no closed form. Older results have no ``distributions``; it is
empty for programs without ``dist``.

Undefined values
----------------
A distribution, mass or error probability that contains ``nan``, ``zoo`` (complex infinity) or ``oo``
is not a result: such a run ends with ``status`` ``error`` and ``error.type`` ``NonFiniteResult``. A
query value that contains them gets ``bound`` ``unknown`` without interval and a warning, but does not
fail the analysis (#316). The value ∞ of an expectation or variance query is no such value
(``Ex[x]`` may diverge): it is reported with the guarantee of the result, e.g. ``Ex[x] ≤ ∞`` under an
upper bound.

Failed invariant checks
-----------------------
If a proposed invariant is not certified, ``status`` is ``error`` and ``error.verdict`` (also in
``error.details`` and in the loop's certificates) says why:

* ``refuted``: the invariant is provably wrong (a counterexample, or Φ(I) − I is a nonzero rational
  function after exact cancellation).
* ``unknown``: the check could neither prove nor refute it (e.g. non-rational expressions, an
  engine whose equality is incomplete, or no parameter values found); the invariant may be correct.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from prodigy.analysis.loops.api import Bound
from prodigy.analysis.values import QueryResult, is_non_finite, to_sympy
from prodigy.util.logger import log_setup

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


SCHEMA = "prodigy/analysis-result@1"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def prodigy_version() -> str:
    """The version of prodigy-ppl, taken from the Git tag ``v<version>`` when it was built or installed.

    It comes from the package metadata, or, when the sources run without being installed, from
    ``prodigy/_version.py``, which every build writes (see ``[tool.hatch.build.hooks.vcs]``).
    """
    # pylint: disable=import-outside-toplevel
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("prodigy-ppl")
    except PackageNotFoundError:
        pass
    try:
        from prodigy._version import __version__  # type: ignore[import-not-found,unused-ignore]
        return str(__version__)
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------------------
# Exact values
# ---------------------------------------------------------------------------------------


def non_finite_parts(result: "AnalysisResult", dist: "Distribution") -> List[str]:
    """The parts of an otherwise finished ``result`` that are undefined (see :func:`is_non_finite`);
    the affected queries lose their interval."""
    names = frozenset(map(str, dist.get_variables())) | frozenset(map(str, dist.get_parameters()))
    parts = []
    if is_non_finite(to_sympy(dist), names=names) or is_non_finite(str(dist), names=names):
        parts.append("the distribution")
    if result.distribution is not None and is_non_finite(result.distribution.get("mass"), names=names):
        parts.append("its mass")
    if is_non_finite(result.error_probability, names=names):
        parts.append("the error probability")
    for query in result.queries:
        if (is_non_finite(query.value, names=names, infinity=query.kind in ("expectation", "variance"))
                or is_non_finite(query.lower, names=names)
                or is_non_finite(query.upper, names=names, infinity=True)):
            query.bound = Bound.UNKNOWN
            query.lower = query.upper = None
            query.lower_float = query.upper_float = None
            query.note = "Undefined value (nan or ∞): no guarantee."
            where = f" (line {query.line})" if query.line is not None else ""
            parts.append(f"the query {query.expr}{where}")
    return parts


# ---------------------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------------------


def failure_bounds(status: str) -> Dict[str, Any]:
    """``bounds`` of a result that did not finish (``status`` other than ``ok``): no guarantee.

    The bound state of the session at the moment of a failure describes a distribution that was
    never computed (e.g. the loop whose invariant was refuted has not been booked), so it must
    not be reported."""
    return {
        "kind": Bound.UNKNOWN,
        "residual_mass": None,
        "residual_mass_float": None,
        "explanation": f"No guarantee: the analysis did not finish (status '{status}').",
        "upper": None,
    }


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
    distributions: List[Dict[str, Any]] = field(default_factory=list)
    """The values of the ``dist`` variables (see "Values of type dist")."""
    queries: List[QueryResult] = field(default_factory=list)
    loops: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None
    """Limits, watches and diagnosis hints of the step trace (see :mod:`prodigy.analysis.trace`)."""
    warnings: List[str] = field(default_factory=list)
    log: str = ""
    timing: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        distribution = dict(self.distribution) if self.distribution is not None else None
        if distribution is not None:
            distribution["error_probability"] = self.error_probability
            distribution["error_probability_float"] = self.error_probability_float
        # A result that did not finish never carries a guarantee, whoever set ``bounds``.
        bounds = self.bounds if self.status == "ok" else failure_bounds(self.status)
        return {
            "schema": SCHEMA,
            "prodigy_version": prodigy_version(),
            "status": self.status,
            "error": self.error,
            # kept in the schema for compatibility: programs and the expressions from them are .ppy
            "syntax": "ppy",
            "program": self.program,
            "input": self.input,
            "distribution": distribution,
            "bounds": bounds,
            "distributions": self.distributions,
            "queries": [q.to_json() for q in self.queries],
            "loops": self.loops,
            "steps": self.steps,
            "trace": self.trace,
            "warnings": self.warnings,
            "log": self.log,
            "timing": self.timing,
        }


def engine_name(config: ForwardAnalysisConfig) -> str:
    return config.engine.name.lower()


def language_warnings(session: Any) -> List[str]:
    """The language warnings of the program (monus, implicit cast; see :mod:`prodigy.language.language_warnings`) as
    ``"line L, column C: message"``; the entries of ``warnings`` are strings, so the position is part of the text."""
    if session.program is None or session.source is None or session.silent:
        return []
    # pylint: disable=import-outside-toplevel
    from prodigy.language.diagnostics import language_warning_diagnostics
    found = language_warning_diagnostics(session.program, session.source)
    return [f"line {d['line']}, column {d['column']}: {d['message']}" for d in found]
