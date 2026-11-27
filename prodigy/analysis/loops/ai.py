"""
The strategy ``ai_invariant``: candidates from a language model (:mod:`prodigy.analysis.ai_candidates`), each checked
like one typed in by hand.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple

import sympy

from prodigy.analysis.loops.api import Bound, Delivers, LoopDecision, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import get_loop_strategy, register_loop_strategy
from prodigy.analysis.loops.common import InvariantRefutedError, _mass


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _larger(mass: sympy.Expr, best: sympy.Expr) -> bool:
    """``mass > best``, if this is proven (masses with parameters may be incomparable)."""
    return sympy.simplify(mass - best).is_positive is True


@dataclasses.dataclass
class _BestCandidate:
    """The proven candidate with the largest lower bound so far (strategy ai_invariant)."""
    outcome: LoopOutcome
    entry: Dict[str, Any]
    """Its attempt in the log."""
    mass: sympy.Expr
    """Mass of its lower bound."""


def _strategy_ai_invariant(task: LoopTask) -> LoopOutcome:
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements
    from prodigy.analysis import ai_candidates
    form = task.param("form")
    attempts = task.param("attempts")
    provider = ai_candidates.get_provider()
    target = get_loop_strategy(form)
    program = task.prog_info.program
    variables = [str(v) for v in program.variables.keys()]
    parameters = [str(p) for p in program.parameters.keys()]
    earlier: List[Tuple[str, str]] = []  # candidates that did not help, with Prodigy's verdict, for the model
    log: List[Dict[str, Any]] = []  # every attempt, for the certificate and the GUI
    best: Optional[_BestCandidate] = None
    for attempt in range(1, attempts + 1):
        proven = None
        if best is not None:
            proven = ai_candidates.Proven(best.entry["candidate"], str(best.mass), str(best.outcome.residual))
        task.progress(attempt - 1, attempts, f"Asking {provider.model}"
                      + (" for a larger sub-invariant" if proven else ""))
        request = ai_candidates.prompt(form, str(program), task.session.spell(task.instruction), variables, parameters,
                                       str(task.distribution), earlier, proven)
        try:
            answer = provider.ask(ai_candidates.SYSTEM_PROMPT, request)
        except ai_candidates.AIRequestError as exc:
            if best is None:
                raise
            # a proven sub-invariant is in hand; a failed request for a larger one does not discard it
            log.append({"attempt": attempt, "answer": "", "candidate": None, "verdict": "error", "reason": str(exc)})
            print(f"Request {attempt} failed, keeping the sub-invariant of attempt {best.entry['attempt']}: {exc}")
            break
        entry: Dict[str, Any] = {"attempt": attempt, "answer": answer.strip()[-2000:], "candidate": None,
                                 **provider.diagnostics}
        log.append(entry)
        print(f"Answer {attempt} of {provider.model}:\n{answer.strip()[-2000:] or '(no text)'}")
        try:
            candidate = ai_candidates.candidate_from_answer(answer, variables + parameters)
        except ai_candidates.CandidateError as exc:
            entry["verdict"], entry["reason"] = "no_candidate", str(exc)
            if not answer.strip() and provider.diagnostics:
                entry["reason"] += (f" The model stopped with {provider.diagnostics.get('stop_reason')} "
                                    f"(content: {provider.diagnostics.get('content', '?')}).")
            lines = answer.strip().splitlines()
            earlier.append((lines[-1][:200] if lines else "(empty answer)", entry["reason"]))
            print(f"Rejected: {entry['reason']}")
            continue
        entry["candidate"] = candidate
        # The candidate goes through the same check as one typed in by hand.
        attempt_task = dataclasses.replace(task, strategy=target,
                                           decision=LoopDecision(form, {ai_candidates.TARGET_PARAMS[form]: candidate}))
        try:
            outcome = target.run(attempt_task)
        except Exception as exc:  # pylint: disable=broad-except  # any failed check rejects the candidate
            message = str(exc) or type(exc).__name__
            details = _json_safe(getattr(exc, "details", None) or {})
            entry["verdict"] = details.get("verdict") if details.get("verdict") in ("refuted", "unknown") \
                else "rejected"
            entry["reason"] = ai_candidates.describe_rejection(message, details)
            if details:
                entry["details"] = details
            # the model gets the full details of the check
            earlier.append((candidate, f"{entry['reason']} {message}"
                            + (f" Details: {json.dumps(details)[:1000]}" if details else "")))
            print(f"Rejected {candidate}: {entry['reason']}")
            continue
        entry["verdict"] = "proven"
        if form != "evt_subinvariant" or outcome.kind == Bound.EXACT:
            best = _BestCandidate(outcome, entry, sympy.S(0))
            break  # nothing to improve: exact, or an invariant
        mass = _mass(outcome.distribution)
        entry["mass"], entry["missing_mass"] = str(mass), str(outcome.residual)
        if best is not None and not _larger(mass, best.mass):
            entry["verdict"] = "not_larger"
            entry["reason"] = (f"Proven, but its lower bound has mass {mass}, not more than {best.mass} "
                               f"(attempt {best.entry['attempt']}).")
            earlier.append((candidate, entry["reason"]))
            print(f"Not larger: {candidate}")
            continue
        best = _BestCandidate(outcome, entry, mass)
        print(f"Proven sub-invariant {candidate}: lower bound with mass {mass}, missing mass {outcome.residual}")
    if best is None:
        raise InvariantRefutedError(
            f"None of {attempts} answers of {provider.model} gave a candidate that passed Prodigy's check. "
            f"Last reason: {log[-1].get('reason', 'none')[:300] if log else 'none'}",
            details={"type": "ai_attempts", "provider": provider.name, "model": provider.model, "attempts": log})
    outcome, chosen = best.outcome, best.entry
    proofs = sum(1 for entry in log if entry.get("verdict") in ("proven", "not_larger"))
    outcome.certificate["ai"] = {"provider": provider.name, "model": provider.model, "attempts": chosen["attempt"],
                                 "asked": len(log), "candidate": chosen["candidate"], "answer": chosen["answer"],
                                 "history": [entry for entry in log if entry is not chosen]}
    outcome.messages.append(f"Candidate proposed by {provider.model} (attempt {chosen['attempt']} of {len(log)}), "
                            "checked by Prodigy"
                            + (f"; the largest of {proofs} proven sub-invariants." if proofs > 1 else "."))
    return outcome


register_loop_strategy(LoopStrategy(
    name="ai_invariant",
    title="EVT invariant from a language model",
    group="exact",
    outcome="exact or a bound",
    summary="Like Check EVT invariant or Check EVT sub-invariant, with candidates from a language model that "
            "Prodigy checks; you provide an API key; one model request per attempt.",
    legacy_key="9",
    soundness=Bound.UPPER,
    delivers=(Delivers(Bound.EXACT, "a candidate EVT invariant has non-negative coefficients and finite mass, or a "
                                    "candidate sub-invariant misses no mass"),
              Delivers(Bound.UPPER, "a candidate EVT invariant has non-negative coefficients"),
              Delivers(Bound.LOWER, "a candidate sub-invariant is proven")),
    description="A language model proposes an EVT invariant or a finite-mass sub-invariant. Prodigy checks "
                "every candidate exactly, like one typed in by hand, and tells the model why a candidate "
                "was rejected. After a proven sub-invariant, it asks for a larger one while attempts are left "
                "and keeps the one with the largest lower bound. Needs an API key in the environment of "
                "prodigy (see docs/gui.md).",
    params=(ParamSpec("form", "choice", "Kind of candidate", prompt="Kind (evt_invariant or evt_subinvariant): ",
                      default="evt_invariant", choices=("evt_invariant", "evt_subinvariant"),
                      help="evt_invariant: expected visiting times (exact with finite mass, otherwise an upper "
                           "bound); evt_subinvariant: a lower bound without termination assumption."),
            ParamSpec("attempts", "int", "Attempts", prompt="Number of attempts: ", default=3, minimum=1,
                      help="How often the model is asked in total: again after a rejected candidate, and "
                           "after a proven sub-invariant for a larger one.")),
    run=_strategy_ai_invariant,
))
