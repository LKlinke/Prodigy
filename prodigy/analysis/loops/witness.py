"""
---------
Witnesses
---------

A *witness* that two programs differ: an output state whose probabilities provably differ, i.e. the sound
intervals of the two analyses on the same input (exact value, lower bound plus residual mass, or Park upper
bound) are disjoint. The comparison of two programs (:mod:`prodigy.analysis.equivalence.verdict`) and the check
of an invariant ``Φ(I) = I`` by the while handler need it to call a difference *refuted*; without a witness the
verdict stays open. The analyses run in a sub-analysis of the current context, with the ``analyzer`` the caller
passes (:func:`~prodigy.analysis.analyzer.compute_semantics`), so this module imports no analysis above the
vocabulary layer.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional, Tuple

import sympy

from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.loops.api import Bound
from prodigy.analysis.values import show, sym, to_float
from prodigy.distribution.distribution import known_convergent_series

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.context import AnalysisContext, Analyzer
    from prodigy.distribution import Distribution
    from prodigy.language import Program


WITNESS_TERMS = 64
"""How many terms of each output distribution :func:`find_witness` examines."""


def show_valuation(valuation: Mapping[str, Any], empty: str = "every variable 0") -> str:
    """A state or input as text in the style of the programs, ``x = 1, n = 3`` (``empty`` if there is
    no variable)."""
    return ", ".join(f"{name} = {value}" for name, value in valuation.items()) or empty


@dataclass
class Side:
    """One program's output on the shown input, with its guarantee (from its analysis result)."""

    distribution: "Distribution"
    kind: str
    residual: Any = 0

    def interval(self, state: Any) -> Optional[Tuple[sympy.Expr, sympy.Expr]]:
        """The sound interval of the probability of ``state`` (None without a guarantee)."""
        if self.kind == Bound.UNKNOWN:
            return None
        value = sym(self.distribution.filter_state(state).get_probability_mass())
        if self.kind == Bound.EXACT:
            return value, value
        if self.kind == Bound.LOWER:
            return value, value + sym(self.residual or 0)
        return sympy.Integer(0), value  # upper bound (Park's lemma)

    def support(self, limit: int) -> Iterator[Any]:
        """States with a positive lower bound, in the order the engine lists them."""
        if self.kind not in (Bound.EXACT, Bound.LOWER):
            return iter(())
        terms = itertools.islice(iter(self.distribution), limit)
        return (state for probability, state in terms if probability != "0")


def _provably_positive(value: sympy.Expr) -> bool:
    return bool(sympy.nsimplify(value).is_positive is True)


def find_witness(first: Side, second: Side, *, limit: int = WITNESS_TERMS) -> Optional[Dict[str, Any]]:
    """The first output state whose probabilities provably differ between the two programs: the
    sound intervals of the two (exact value, lower bound plus residual mass, or Park upper bound)
    are disjoint. Only states that one of the programs reaches with a positive lower bound are
    candidates (examined in the engine's order, at most ``limit`` terms each). None if there is none
    among them, or if the difference depends on parameters."""

    def side(interval: Tuple[sympy.Expr, sympy.Expr], kind: str) -> Dict[str, Any]:
        return {"lower": show(interval[0]), "lower_float": to_float(interval[0]),
                "upper": show(interval[1]), "upper_float": to_float(interval[1]), "kind": kind}

    seen = set()
    for state in itertools.chain(first.support(limit), second.support(limit)):
        key = tuple(sorted(state.valuations.items()))
        if key in seen:
            continue
        seen.add(key)
        one, two = first.interval(state), second.interval(state)
        if one is None or two is None:
            return None
        if _provably_positive(one[0] - two[1]) or _provably_positive(two[0] - one[1]):
            return {
                "state": {str(k): int(v) for k, v in state.valuations.items()},
                "first": side(one, first.kind),
                "second": side(two, second.kind),
            }
    return None


def point_input(program: "Program", state: Dict[str, int], config: "ForwardAnalysisConfig") -> "Distribution":
    """The distribution that is ``state`` with probability 1 (absent variables are 0)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.language import Binop, BinopExpr, BoolLitExpr, BoolType, NatLitExpr, VarExpr
    dist = config.factory.one(*program.variables)
    for name, value in sorted(state.items()):
        if value:
            literal = BoolLitExpr(True) if isinstance(program.variables.get(name), BoolType) \
                else NatLitExpr(int(value))
            dist = dist.update(BinopExpr(Binop.EQ, VarExpr(name), literal))
    return dist


def analyse_on_input(program: "Program", state: Dict[str, int], session: "AnalysisContext",
                     analyzer: "Analyzer") -> Side:
    """The output of ``program`` on the input ``state``, before conditioning on observations (as in the
    check), with the guarantee of its analysis by ``analyzer``. Its loops are decided in ``session`` as
    usual; the analysis does not count for the session's own result (a sub-analysis)."""
    config = session.config
    start = point_input(program, state, config)
    zero = config.factory.one(*program.variables) * 0
    # the input is a point distribution, a finite measure: the marginals of the analysis are sums of series
    with session.sub_analysis("witness search") as scope, known_convergent_series(True):
        dist, error = analyzer(program.instructions, ProgramInfo(program), start, zero, config)
    residual: sympy.Expr = sympy.Integer(0)
    if scope.kind == Bound.LOWER:
        # every output state misses at most the mass the analysis did not account for (the input has mass 1)
        residual = 1 - sym(dist.get_probability_mass()) - sym(error.get_probability_mass())
    return Side(dist, scope.kind, residual)


def search_witness(program: "Program", other: "Program", session: "AnalysisContext", state: Dict[str, int],
                   analyzer: "Analyzer") -> Optional[Dict[str, Any]]:
    """A witness (:func:`find_witness`) on the input ``state``, from fresh analyses of both programs by
    ``analyzer``, or None (also if an analysis fails: a missing witness only leaves the verdict open)."""
    try:
        return find_witness(analyse_on_input(program, state, session, analyzer),
                            analyse_on_input(other, state, session, analyzer))
    except Exception:  # pylint: disable=broad-except
        return None
