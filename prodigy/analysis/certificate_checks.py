"""
The checks of a loop certificate in words.

A loop strategy books a certificate (:attr:`~prodigy.analysis.loops.api.LoopOutcome.certificate`) or, when it fails,
the ``details`` of its error. :func:`certificate_checks` turns either into checks for the step trace and the GUI:
a label, a status and a sentence. A check is ``passed`` only if the certificate says that Prodigy proved it; what is
not established is ``unknown``, what is refuted is ``failed``, and plain facts (iterations, masses) are ``info``.

Every certificate type has its own function in :data:`_BY_TYPE`; the checks common to all types (the rejection of a
failed certificate, the automatic choice, the language model's candidate) are added around it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = ["certificate_checks"]

Check = Dict[str, str]
Certificate = Dict[str, Any]


def certificate_checks(certificate: Optional[Mapping[str, Any]], failed: Optional[str] = None) -> List[Check]:
    """The checks of a loop certificate (or of the ``details`` of a failed check) in words; ``failed`` is the error
    message of a failed strategy. A check is only ``passed`` if the certificate says that Prodigy proved it."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import describe_rejection  # the language-model layer is above this one
    cert = dict(certificate or {})
    verdict = cert.get("verdict")
    checks: List[Check] = []
    if failed is not None and (verdict in ("refuted", "unknown") or cert.get("condition")):
        status = "unknown" if verdict == "unknown" else "failed"
        checks.append(_check("Certificate", status, describe_rejection(failed, cert)))
    automatic = cert.get("automatic")
    if automatic:
        checks.append(_check("Automatic", "info",
                             "An EVT invariant was synthesised and certified exact."
                             if automatic.get("chosen") == "evt_synthesis" else
                             f"No exact EVT invariant was found ({automatic.get('synthesis')}); unrolled instead."))
    describe = _BY_TYPE.get(cert.get("type"))
    if describe is not None:
        checks.extend(describe(cert, failed))
    if cert.get("ai"):
        ai = cert["ai"]
        checks.append(_check("Language model", "info",
                             f"Candidate {ai.get('candidate')} proposed by {ai.get('model')} "
                             f"(attempt {ai.get('attempts')} of {ai.get('asked')}), checked by Prodigy."))
    if failed is not None and not checks:
        checks.append(_check("Strategy", "failed", describe_rejection(failed, cert)))
    return checks


# ---------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------

def _check(label: str, status: str, text: str) -> Check:
    """``status``: ``passed`` (proven by Prodigy), ``failed``, ``unknown`` (not established) or ``info``."""
    return {"label": label, "status": status, "text": text}


def _valuation(state: Dict[str, Any]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in state.items()) or "where every variable is 0"


def _remaining(certificate: Mapping[str, Any], key: str = "remaining_mass") -> Optional[Check]:
    remaining = certificate.get(key)
    if remaining is None:
        return None
    if str(remaining) == "0":
        return _check("Residual mass", "passed", "No mass is left inside the loop: the result is exact.")
    return _check("Residual mass", "info",
                  f"Mass {remaining} is still inside the loop, so the result is a lower bound with this residual "
                  "mass (it may still terminate with any value, or never).")


def _tristate(value: Any, label: str, yes: str, no: str, undecided: str, *, no_status: str = "failed") -> Check:
    """A three-valued fact of the certificate: proven (``True`` or a reason), refuted (``False``) or undecided."""
    if value is True or (isinstance(value, str) and value not in ("", "None", "False")):
        return _check(label, "passed", yes)
    if value is False:
        return _check(label, no_status, no)
    return _check(label, "unknown", undecided)


def _with_remaining(checks: List[Check], cert: Certificate) -> List[Check]:
    found = _remaining(cert)
    return checks + [found] if found else checks


# ---------------------------------------------------------------------------------------
# One function per certificate type
# ---------------------------------------------------------------------------------------

def _unrolling(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    done = cert.get("iterations_done", cert.get("iterations"))
    fixpoint = cert.get("fixpoint")
    checks = [_check("Fixed point", "passed", f"After {done} iteration(s) one more iteration changes nothing.")
              if fixpoint else _check("Iterations", "info", f"{done} iteration(s) computed.")]
    found = _remaining(cert)
    if found and not (fixpoint and found["status"] == "info"):
        checks.append(found)
    elif fixpoint and found:
        checks.append(_check("Divergence", "info", f"Mass {cert.get('remaining_mass')} stays inside the loop forever."))
    return checks


def _mass_threshold(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    return _with_remaining([_check("Threshold", "info", f"Captured mass {cert.get('captured_mass')} after "
                                                        f"{cert.get('iterations')} iteration(s); threshold "
                                                        f"{cert.get('threshold')}.")], cert)


def _evt_iteration(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    checks = [_check("Iterations", "info", f"{cert.get('iterations')} application(s) of the EVT operator."),
              _check("Exploration", "info", "The iterates H_N are expected visiting times, not the output "
                                            "distribution; the result is their terminating part.")]
    guess = (cert.get("exploration") or {}).get("guess")
    if guess:
        checks.append(_check("Candidate", "info", f"Guessed EVT invariant {guess.get('candidate')} (not "
                                                  "checked; use Check EVT invariant)."))
    return _with_remaining(checks, cert)


def _invariant(cert: Certificate, failed: Optional[str]) -> List[Check]:
    verdict = cert.get("verdict")
    if verdict == "verified":
        checks = [_check("Fixed point", "passed",
                         "Φ(I) = I holds for all inputs: I equals one unrolling of the loop followed by I.")]
        if cert.get("constraints"):
            checks.append(_check("Constraints", "info", f"Only under the constraints {cert['constraints']}."))
        checks.append(_check("Park's lemma", "info", "So the result is an upper bound; it is exact if the loop "
                                                     "terminates almost surely, which is not checked."))
        return checks
    if verdict == "refuted" and failed is None:
        return [_check("Fixed point", "failed", f"Φ(I) ≠ I: witness {cert.get('witness')}.")]
    if verdict == "unknown" and failed is None:
        return [_check("Fixed point", "unknown", f"Fixed point not proven: Φ(I) − I = {cert.get('difference')} "
                                                 "could not be shown to vanish.")]
    if verdict == "refuted" and cert.get("witness") is not None:
        witness = cert["witness"]
        return [_check("Witness", "failed",
                       "I and one unrolling of the loop give provably different probabilities to the output state "
                       + _valuation(witness.get("state") or {}) + " on the input "
                       + _valuation(cert.get("counterexample") or {}) + ".")]
    if verdict == "unknown" and cert.get("counterexample") is not None:
        # a raw "different" of the check without a witness is no refutation (#403)
        return [_check("Counterexample", "unknown",
                       "Candidate only: the check found I and one unrolling of the loop to differ on the input "
                       + _valuation(cert["counterexample"]) + ", but no witness was confirmed.")]
    return []


def _evt_invariant(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    """``evt_invariant`` and ``evt_synthesis``."""
    if cert.get("verdict") in ("refuted", "unknown"):
        return []
    checks = [
        _check("Fixed point", "passed", f"I = G + ⟦body⟧(⟨I⟩_guard) holds for I = {cert.get('invariant')}."),
        _tristate(cert.get("nonnegative"), "Non-negative", "I has non-negative coefficients.",
                  "I has negative coefficients, so Park's lemma does not apply.",
                  "Non-negativity of the coefficients of I could not be established."),
        _tristate(cert.get("finite_mass"), "Finite mass",
                  f"I has finite mass |I| = {cert.get('mass')}, so the fixed point is the least one: exact.",
                  "I has infinite mass and may be a spurious fixed point: at most an upper bound.",
                  "Finite mass of I was not shown, so exactness is not certified.", no_status="unknown"),
    ]
    if cert.get("nonnegative") is True and cert.get("finite_mass") is not True:
        reason = cert.get("series_identity")
        checks.append(_check("Series", "passed" if reason else "unknown",
                             f"The equation also holds for the power series of I ({reason}), so Park's "
                             "lemma gives an upper bound." if reason else
                             "The equation was shown for the closed form of I only: the series of I may "
                             "diverge where the loop body evaluates it (e.g. a marginal), so Park's "
                             "lemma does not apply."))
    return checks


def _evt_subinvariant(cert: Certificate, failed: Optional[str]) -> List[Check]:
    if failed is not None:
        return []
    checks = [
        _tristate(cert.get("nonnegative"), "ν ≥ 0", f"ν has non-negative coefficients ({cert.get('nonnegative')}).",
                  "ν has negative coefficients.", "Non-negativity of ν was not established."),
        _tristate(cert.get("subinvariance"), "ν ≤ Φ(ν)", f"ν ≤ Φ(ν) coefficient-wise ({cert.get('subinvariance')}).",
                  "ν ≤ Φ(ν) fails.", "ν ≤ Φ(ν) was not established."),
        _check("Finite mass", "passed", f"|ν| = {cert.get('mass')} is finite."),
    ]
    missing = cert.get("missing_mass")
    if missing is not None:
        exact = str(missing) == "0"
        checks.append(_check("Residual mass", "passed" if exact else "info",
                             "Nothing is missing: exact." if exact
                             else f"Lower bound; at most {missing} of the mass is not accounted for."))
    return checks


def _evt_sandwich(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    low, high = (cert.get("termination_probability") or [None, None])[:2]
    checks = [_check("Termination probability", "info", f"Between {low} and {high}.")]
    for name, label in (("lower", "Sub-invariant"), ("upper", "Super-invariant")):
        part = cert.get(name)
        if part:
            checks.append(_check(label, "passed", f"Checked exactly: {part.get('check') or 'verified'}."))
        elif name == "upper":
            checks.append(_check(label, "unknown", "No super-invariant found: no upper bounds."))
    return checks


_ATTEMPT_STATUS = {"proven": "passed", "not_larger": "info", "unknown": "unknown"}


def _ai_attempts(cert: Certificate, _failed: Optional[str]) -> List[Check]:
    return [_check(f"Attempt {attempt.get('attempt')}", _ATTEMPT_STATUS.get(attempt.get("verdict"), "failed"),
                   attempt.get("reason") or f"Candidate {attempt.get('candidate')}.")
            for attempt in cert.get("attempts") or []]


_BY_TYPE: Dict[Any, Callable[[Certificate, Optional[str]], List[Check]]] = {
    "unrolling": _unrolling,
    "mass_threshold": _mass_threshold,
    "evt_iteration": _evt_iteration,
    "invariant": _invariant,
    "evt_invariant": _evt_invariant,
    "evt_synthesis": _evt_invariant,
    "evt_subinvariant": _evt_subinvariant,
    "evt_sandwich": _evt_sandwich,
    "ai_attempts": _ai_attempts,
}
"""The checks of each certificate type (``certificate["type"]``)."""
