"""
Sound verdicts for the comparison of two programs (the GUI's equivalence view).

:func:`~prodigy.analysis.equivalence.equivalence_check.check_equivalence` compares the
second-order generating functions of two programs: it runs both on a test distribution that
stands for every input at once. Its answer is only as good as the two analyses behind it. If a
loop was handled by an invariant (Park's lemma), the computed output is an upper bound that is
exact only if the loop terminates almost surely; if it was unrolled, the output is a lower bound.
This module turns the raw answer into a verdict that never claims more than was proven:

* ``equivalent``: the check proved equality, and both analyses were exact.
* ``not_equivalent``: only with a *witness*, a concrete output state on a concrete input where the
  probabilities of the two programs provably differ (their sound intervals are disjoint, see
  :func:`find_witness`). A raw "not equal" of the check is not enough: it may come from an
  approximation, or from programs that declare different variables.
* ``unknown``: everything else, with the reason. A "not implemented" of the analysis stays a
  "not implemented"; it never becomes "not equivalent".

If equality was proven while one side is only a bound, the verdict is ``unknown`` and ``relation``
says what was proven, e.g. ``first_below_second``: for every input, the output distribution of the
first program lies coefficient-wise below that of the second (equal if its loops terminate almost
surely).

The case analysis lives here and uses only the :class:`~prodigy.distribution.Distribution`
interface and exact SymPy arithmetic on the coefficients the engines print, so it is the same for
every engine.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import sympy

from prodigy.analysis.loop_strategies import Bound

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution
    from prodigy.pgcl import Program

EQUIVALENT = "equivalent"
NOT_EQUIVALENT = "not_equivalent"
UNKNOWN = "unknown"
VERDICTS = (EQUIVALENT, NOT_EQUIVALENT, UNKNOWN)

FIRST_BELOW_SECOND = "first_below_second"
SECOND_BELOW_FIRST = "second_below_first"

WITNESS_TERMS = 64
"""How many terms of each output distribution :func:`find_witness` examines."""


@dataclass
class CheckOutcome:
    """What the second-order check (:func:`check_equivalence`) found, before any verdict.

    ``answer`` is ``equal`` (equality proven for the computed distributions), ``different`` (the
    solver found no way to make them equal), ``undecided`` (neither), or ``failed`` (the check
    raised; ``error`` says why). ``first`` and ``second`` are the bound kinds (:class:`Bound`) of
    the two analyses inside the check."""

    answer: str
    first: str = Bound.EXACT
    second: str = Bound.EXACT
    constraints: List[Dict[str, str]] = field(default_factory=list)
    counterexample: Optional[Dict[str, int]] = None
    difference: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    time: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "first": self.first,
            "second": self.second,
            "constraints": self.constraints,
            "counterexample": self.counterexample,
            "difference": self.difference,
            "error": self.error,
            "time": round(self.time, 6),
        }


def different_variables(program: "Program", other: "Program") -> Optional[str]:
    """Why two programs cannot be compared state by state (their variables differ), or None."""
    first, second = set(program.variables), set(other.variables)
    if first == second:
        return None
    parts = []
    if first - second:
        parts.append(f"only the first program declares {', '.join(sorted(first - second))}")
    if second - first:
        parts.append(f"only the second program declares {', '.join(sorted(second - first))}")
    return ("The programs declare different variables (" + "; ".join(parts) + "). The comparison needs the "
            "same variables in both programs; declare the missing ones (they keep the value 0).")


def _failure(exc: BaseException) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import error_type
    from prodigy.analysis.session import DecisionRequired
    from prodigy.distribution.distribution import NonFiniteMarginalError
    error: Dict[str, Any] = {"type": error_type(exc), "message": str(exc) or type(exc).__name__}
    if isinstance(exc, NonFiniteMarginalError):
        # The test distribution of the check stands for every input at once and has infinite mass by
        # design; a step that needs a finite measure is a limitation of the check, not of the programs.
        error.update(type="NotImplementedError", message=(
            "a step of the check for every input needs a finite measure, but the check works on a generating "
            f"function with infinite mass ({error['message']})"))
    if isinstance(exc, DecisionRequired):
        error.update(type="decision", **exc.to_json())
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        error["details"] = {key: value for key, value in details.items() if key != "invariant"}
        if details.get("verdict") in ("refuted", "unknown"):
            error["verdict"] = details["verdict"]
    return error


def _loop_constraints(session: "AnalysisSession") -> List[Dict[str, str]]:
    """Parameter values under which the invariants of the loops were verified (#84: the check of a
    parametrised invariant finds them, the comparison afterwards does not see them)."""
    found: List[Dict[str, str]] = []
    for record in session.loop_records.values():
        for certificate in record.certificates:
            for constraint in certificate.get("constraints") or []:
                if constraint and constraint not in found:
                    found.append(dict(constraint))
    return found


def run_check(program: "Program", other: "Program", session: "AnalysisSession") -> CheckOutcome:
    """Runs the second-order check of ``program`` against ``other`` in ``session`` (which must be
    active, with its configuration) and records the bound kind of each analysis. Never raises for
    analysis errors: they end up in :attr:`CheckOutcome.error`."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import compute_semantics
    from prodigy.analysis.equivalence.equivalence_check import check_equivalence
    from prodigy.distribution import Distribution, State

    kinds: List[str] = []

    def analyzer(*args: Any) -> Any:
        with session.sub_analysis("equivalence check") as scope:
            outcome = compute_semantics(*args)
        kinds.append(scope.kind)
        return outcome

    start = time.perf_counter()
    try:
        answer, result = check_equivalence(program, other, session.config, analyzer)
    except Exception as exc:  # pylint: disable=broad-except
        return CheckOutcome("failed", error=_failure(exc), time=time.perf_counter() - start,
                            constraints=_loop_constraints(session))
    # check_equivalence analyses the second program first
    second, first = (kinds + [Bound.EXACT, Bound.EXACT])[:2]
    outcome = CheckOutcome("undecided", first=first, second=second, time=time.perf_counter() - start)
    if answer is True:
        outcome.answer = "equal"
        assert isinstance(result, list)
        outcome.constraints = [{str(k): str(v) for k, v in solution.items()} for solution in result] \
            or _loop_constraints(session)
    elif answer is False:
        outcome.answer = "different"
        if isinstance(result, State) and result.valuations:
            outcome.counterexample = {str(k): int(v) for k, v in result.valuations.items()}
    else:
        outcome.difference = str(result) if isinstance(result, Distribution) else None
    return outcome


# ---------------------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------------------

@dataclass
class Side:
    """One program's output on the shown input, with its guarantee (from its analysis result)."""

    distribution: "Distribution"
    kind: str
    residual: Any = 0

    def interval(self, state: Any) -> Optional[Tuple[sympy.Expr, sympy.Expr]]:
        """The sound interval of the probability of ``state`` (None without a guarantee)."""
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.result import sym
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
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import show, to_float

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


# ---------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------

_KIND_WORDS = {Bound.LOWER: "a lower bound (unrolling)", Bound.UPPER: "an upper bound (Park's lemma)",
               Bound.UNKNOWN: "an approximation without guarantee"}


def _relation(check: CheckOutcome) -> Optional[str]:
    """What proven equality of the computed distributions means if one of them is a bound."""
    kinds = (check.first, check.second)
    if kinds in ((Bound.UPPER, Bound.EXACT), (Bound.EXACT, Bound.LOWER)):
        return FIRST_BELOW_SECOND
    if kinds in ((Bound.EXACT, Bound.UPPER), (Bound.LOWER, Bound.EXACT)):
        return SECOND_BELOW_FIRST
    return None


def _failure_reason(error: Dict[str, Any]) -> str:
    message = error.get("message", "")
    kind = error.get("type")
    if kind == "NotImplementedError" or "not implemented" in message.lower():
        return f"Not implemented: {message}"
    if kind == "decision":
        return f"The check needs a strategy for a loop: {message}"
    if error.get("verdict") == "refuted":
        return f"The invariant chosen for a loop was refuted while checking it: {message}"
    if error.get("verdict") == "unknown":
        return f"The invariant chosen for a loop could not be proven: {message}"
    return f"The check failed: {message}"


def _proven_relation_reason(check: CheckOutcome, relation: str) -> str:
    lower, upper = ("first", "second") if relation == FIRST_BELOW_SECOND else ("second", "first")
    bounded = "first" if check.first != Bound.EXACT else "second"
    kind = check.first if bounded == "first" else check.second
    tail = ("They are equal if the loops of the first program terminate almost surely, which Prodigy does not "
            "check." if kind == Bound.UPPER and bounded == "first" else
            "They are equal if the loops of the second program terminate almost surely, which Prodigy does not "
            "check." if kind == Bound.UPPER else
            "Equality is not proven: the unrolled loops may still contribute more mass.")
    return (f"Proven for every input: the output distribution of the {lower} program is coefficient-wise at most "
            f"that of the {upper} program, because the analysis of the {bounded} program is "
            f"{_KIND_WORDS[kind]}. {tail}")


def decide(check: CheckOutcome, witness: Optional[Dict[str, Any]], *,
           variables_differ: Optional[str] = None) -> Dict[str, Any]:
    """The verdict (``verdict``, ``reason``, ``relation``, ``constraints``) from the check and an
    optional witness; see the module documentation for the rules."""
    if witness is not None:
        state = ", ".join(f"{k} = {v}" for k, v in witness["state"].items()) or "the empty state"
        reason = f"The programs give provably different probabilities to the output state {state} (witness below)."
        if check.answer == "failed" and check.error is not None:
            reason += f" The check for all inputs did not finish: {_failure_reason(check.error)}"
        return {"verdict": NOT_EQUIVALENT, "reason": reason, "relation": None, "constraints": []}
    unknown: Dict[str, Any] = {"verdict": UNKNOWN, "relation": None, "constraints": check.constraints}
    if variables_differ:
        return {**unknown, "reason": variables_differ}
    if check.answer == "failed":
        return {**unknown, "reason": _failure_reason(check.error or {})}
    if check.answer == "equal":
        if check.first == Bound.EXACT and check.second == Bound.EXACT:
            reason = ("Proven: both programs have the same output distribution for every input"
                      + (" for the parameter values below." if check.constraints else "."))
            return {"verdict": EQUIVALENT, "reason": reason, "relation": None, "constraints": check.constraints}
        relation = _relation(check)
        if relation is not None:
            return {**unknown, "relation": relation, "reason": _proven_relation_reason(check, relation)}
        return {**unknown, "reason": "The computed distributions are equal, but both analyses are approximations "
                                     f"({check.first} and {check.second}), so this proves nothing."}
    if check.answer == "different":
        if check.first == Bound.EXACT and check.second == Bound.EXACT:
            reason = ("The check found the generating functions to differ, but no witness was confirmed: no output "
                      "state on the input shown has provably different probabilities. Try the input of the "
                      "counterexample, or turn off conditioning on observations (the check compares the "
                      "outputs before conditioning).")
        else:
            reason = ("The computed distributions differ, but one of them is "
                      f"{_KIND_WORDS.get(check.first if check.first != Bound.EXACT else check.second, 'a bound')}, "
                      "and no output state on the input shown has provably different probabilities.")
        return {**unknown, "reason": reason}
    return {**unknown, "reason": "The check could neither prove nor refute that the generating functions are equal "
                                 "(their difference is shown below)."}
