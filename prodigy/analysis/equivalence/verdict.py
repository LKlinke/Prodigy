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
import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Tuple

import sympy

# re-exported: the web reads both from the verdict (`as` makes the re-export explicit, #379)
from prodigy.analysis.equivalence.equivalence_check import (  # pylint: disable=unused-import,useless-import-alias
    DifferentVariablesError as DifferentVariablesError, different_variables as different_variables)
from prodigy.analysis.loops.api import Bound

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution
    from prodigy.language import Program

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


def show_valuation(valuation: Mapping[str, Any], empty: str = "every variable 0") -> str:
    """A state or input as text in the style of the programs, ``x = 1, n = 3`` (``empty`` if there is
    no variable)."""
    return ", ".join(f"{name} = {value}" for name, value in valuation.items()) or empty


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
    from prodigy.analysis.analyzer import compute_semantics
    from prodigy.analysis.equivalence.equivalence_check import check_equivalence
    from prodigy.distribution import Distribution, State

    kinds: List[str] = []

    def analyzer(*args: Any) -> Any:
        own = other_decisions is not None and getattr(args[1], "program", None) is other
        decided = session.other_program(other, other_decisions) if own else contextlib.nullcontext()
        with decided, session.sub_analysis("equivalence check") as scope:
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
        from prodigy.analysis.values import sym
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
    from prodigy.analysis.values import show, to_float

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


def analyse_on_input(program: "Program", state: Dict[str, int], session: "AnalysisSession") -> Side:
    """The output of ``program`` on the input ``state``, before conditioning on observations (as in the
    check), with the guarantee of its analysis. Its loops are decided in ``session`` as usual; the
    analysis does not count for the session's own result (a sub-analysis)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import compute_semantics
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.values import sym
    from prodigy.distribution.distribution import known_convergent_series
    config = session.config
    start = point_input(program, state, config)
    zero = config.factory.one(*program.variables) * 0
    # the input is a point distribution, a finite measure: the marginals of the analysis are sums of series
    with session.sub_analysis("witness search") as scope, known_convergent_series(True):
        dist, error = compute_semantics(program.instructions, ProgramInfo(program), start, zero, config)
    residual: sympy.Expr = sympy.Integer(0)
    if scope.kind == Bound.LOWER:
        # every output state misses at most the mass the analysis did not account for (the input has mass 1)
        residual = 1 - sym(dist.get_probability_mass()) - sym(error.get_probability_mass())
    return Side(dist, scope.kind, residual)


def search_witness(program: "Program", other: "Program", session: "AnalysisSession",
                   state: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """A witness (:func:`find_witness`) on the input ``state``, from fresh analyses of both programs, or
    None (also if an analysis fails: a missing witness only leaves the verdict open)."""
    try:
        return find_witness(analyse_on_input(program, state, session), analyse_on_input(other, state, session))
    except Exception:  # pylint: disable=broad-except
        return None


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
