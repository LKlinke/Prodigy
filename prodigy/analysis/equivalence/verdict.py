"""
Sound verdicts for the comparison of two programs (``prodigy check_equality`` and the GUI's Compare mode).

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

:func:`compare` runs the whole comparison (check, witness search, verdict); the command line uses
it, the GUI combines :func:`run_check`, :func:`analyse_on_input`, :func:`find_witness` and :func:`decide`
with a time limit for the witness search (:func:`prodigy.web.requests.run_equivalence`).

The case analysis lives here and uses only the :class:`~prodigy.distribution.Distribution`
interface and exact SymPy arithmetic on the coefficients the engines print, so it is the same for
every engine.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional


# re-exported: the web reads both from the verdict (`as` makes the re-export explicit, #379)
from prodigy.analysis.equivalence.equivalence_check import (  # pylint: disable=unused-import,useless-import-alias
    DifferentVariablesError as DifferentVariablesError, different_variables as different_variables)
from prodigy.analysis.loops import witness as witnesses
from prodigy.analysis.loops.api import Bound
# re-exported: the front ends read them from the verdict (`as` makes the re-export explicit, #379)
from prodigy.analysis.loops.witness import (  # pylint: disable=unused-import,useless-import-alias
    WITNESS_TERMS as WITNESS_TERMS, Side as Side, find_witness as find_witness, show_valuation as show_valuation)

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution
    from prodigy.language import Program

EQUIVALENT = "equivalent"
NOT_EQUIVALENT = "not_equivalent"
UNKNOWN = "unknown"
VERDICTS = (EQUIVALENT, NOT_EQUIVALENT, UNKNOWN)

FIRST_BELOW_SECOND = "first_below_second"
SECOND_BELOW_FIRST = "second_below_first"


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


def _failure(exc: BaseException) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import error_type
    from prodigy.analysis.loops.api import DecisionRequired
    # a NonFiniteMarginalError of the check arrives as CheckNotImplemented, a NotImplementedError
    error: Dict[str, Any] = {"type": error_type(exc), "message": str(exc) or type(exc).__name__}
    if isinstance(exc, DifferentVariablesError):
        error["type"] = "variables"
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


def run_check(program: "Program", other: "Program", session: "AnalysisSession",
              other_decisions: Optional[Mapping[str, Any]] = None) -> CheckOutcome:
    """Runs the second-order check of ``program`` against ``other`` in ``session`` (which must be
    active, with its configuration) and records the bound kind of each analysis. Never raises for
    analysis errors: they end up in :attr:`CheckOutcome.error`. With ``other_decisions`` the loops of
    ``other`` are decided by their own ids (:meth:`AnalysisSession.other_program`); without, they are
    looked up among the loops of ``program``."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import analyzer_in
    from prodigy.analysis.equivalence.equivalence_check import check_equivalence
    from prodigy.distribution import Distribution, State

    kinds: List[str] = []
    semantics = analyzer_in(session)

    def analyzer(*args: Any) -> Any:
        own = other_decisions is not None and getattr(args[1], "program", None) is other
        decided = session.other_program(other, other_decisions) if own else contextlib.nullcontext()
        with decided, session.sub_analysis("equivalence check") as scope:
            outcome = semantics(*args)
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


def analyse_on_input(program: "Program", state: Dict[str, int], session: "AnalysisSession") -> Side:
    """:func:`~prodigy.analysis.loops.witness.analyse_on_input` with the forward analysis."""
    from prodigy.analysis.analyzer import analyzer_in  # pylint: disable=import-outside-toplevel
    return witnesses.analyse_on_input(program, state, session, analyzer_in(session))


def search_witness(program: "Program", other: "Program", session: "AnalysisSession",
                   state: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """:func:`~prodigy.analysis.loops.witness.search_witness` with the forward analysis."""
    from prodigy.analysis.analyzer import analyzer_in  # pylint: disable=import-outside-toplevel
    return witnesses.search_witness(program, other, session, state, analyzer_in(session))


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
    lower, upper = ("A", "B") if relation == FIRST_BELOW_SECOND else ("B", "A")
    bounded = "A" if check.first != Bound.EXACT else "B"
    kind = check.first if bounded == "A" else check.second
    tail = (f"They are equal if the loops of program {bounded} terminate almost surely, which Prodigy does not "
            "check." if kind == Bound.UPPER else
            "Equality is not proven: the unrolled loops may still contribute more mass.")
    return (f"Proven for every input: the output distribution of program {lower} is coefficient-wise at most "
            f"that of program {upper}, because the analysis of program {bounded} is "
            f"{_KIND_WORDS[kind]}. {tail}")


def decide(check: CheckOutcome, witness: Optional[Dict[str, Any]], *,
           variables_differ: Optional[str] = None, input_tips: bool = True) -> Dict[str, Any]:
    """The verdict (``verdict``, ``reason``, ``relation``, ``constraints``) from the check and an
    optional witness; see the module documentation for the rules. ``input_tips`` adds what to try if no
    witness was found (for a caller whose input can be chosen and is conditioned on observations)."""
    if witness is not None:
        # the witness itself (input, output state, probabilities) is shown next to the verdict, not repeated here
        reason = "The programs give provably different probabilities to an output state (witness below)."
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
                      "state on the input shown has provably different probabilities.")
            if input_tips:
                reason += (" Try the input of the counterexample, or turn off conditioning on observations (the "
                           "check compares the outputs before conditioning).")
        else:
            reason = ("The computed distributions differ, but one of them is "
                      f"{_KIND_WORDS.get(check.first if check.first != Bound.EXACT else check.second, 'a bound')}, "
                      "and no output state on the input shown has provably different probabilities.")
        return {**unknown, "reason": reason}
    return {**unknown, "reason": "The check could neither prove nor refute that the generating functions are equal "
                                 "(their difference is shown below)."}


# ---------------------------------------------------------------------------------------
# The whole comparison (the command line)
# ---------------------------------------------------------------------------------------

@dataclass
class Comparison:
    """The outcome of :func:`compare`: the check, the witness (if any), the verdict (:func:`decide`)
    and the input on which the witness was searched (variable values; absent variables are 0)."""

    check: CheckOutcome
    witness: Optional[Dict[str, Any]]
    verdict: Dict[str, Any]
    input: Dict[str, int] = field(default_factory=dict)


def compare(program: "Program", other: "Program", session: "AnalysisSession") -> Comparison:
    """Compares two programs soundly: the second-order check for every input (:func:`run_check`), then,
    unless it proved the computed distributions equal (no witness can exist then), a witness search on
    the input of its counterexample (or on the input where every variable is 0), and the verdict
    (:func:`decide`). ``session`` must be active, with its configuration; its decision provider decides
    the loops. Never raises for analysis errors."""
    differ = different_variables(program, other)
    if differ is not None:
        check = CheckOutcome("failed", error={"type": "variables", "message": differ})
        return Comparison(check, None, decide(check, None, variables_differ=differ))
    unproven = len(session.unproven_conditions)
    check = run_check(program, other, session)
    state = dict(check.counterexample or {})
    witness = None if check.answer == "equal" else search_witness(program, other, session, state)
    # the witness search takes the counterexample's input and compares before conditioning already
    verdict = decide(check, witness, input_tips=False)
    if verdict["verdict"] == UNKNOWN and session.unproven_conditions[unproven:]:
        # e.g. a generating function of sample_pgf that is not proven to be one (#424)
        verdict["reason"] += " " + " ".join(session.unproven_conditions[unproven:])
    return Comparison(check, witness, verdict, state)
