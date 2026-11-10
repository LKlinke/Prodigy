"""
----------------------------------
Step trace of the forward analysis
----------------------------------

With ``AnalysisSession(record_steps=True)`` (``prodigy analyze --steps``, the GUI's stepper) the
analysis records a *trace*: one step per instruction, with its source position, the distribution
before and after it, the guarantee that holds for it, the time it took and the size of the closed
form. Front ends navigate the trace afterwards (forward, back, jump to a line); the whole trace is
recorded within the time limit of the analysis.

Which instructions are steps
----------------------------
* every top-level instruction of the program, and
* the instructions inside the branches of ``if`` (and ``switch``), of probabilistic choices and of
  query blocks; their distributions are the *sub-distributions* of that branch (restricted to the
  guard, or weighted by the probability of the branch), not normalised.

Instructions inside loop bodies, function bodies and checks of invariants (sub-analyses) are not
steps. A ``while`` loop is one step whose ``loop`` entry tells what its strategy did:

* unrolling strategies (``unroll``, ``mass_threshold``, ``evt_iterate``) record their iterates:
  the part that has left the loop after *i* iterations, which grows towards the output of the
  loop, and the mass still inside the loop, which shrinks; at most ``TraceLimits.iterates`` of
  them, spread evenly over the run, plus the last one;
* every strategy records the checks of its certificate in words (fixed point, non-negativity,
  finite mass, ...) and, if a check failed, why.

Soundness
---------
Each distribution of the trace carries the guarantee that holds for it at that point, with the
same kinds as the result (``exact``, ``lower`` with residual mass, ``upper``, ``unknown``; see
:mod:`prodigy.analysis.result`): the kind of the session at that moment, combined with the kind of
the loop iterate for iterates. A step whose instruction failed has no distribution after it. An
iterate of an unrolling is a lower bound for the output of the loop; its residual is the mass still
inside the loop plus the residual of earlier loops. Watches (queries evaluated after every
top-level step, see :func:`parse_watch`) get intervals exactly like queries at that place.

The ``dist`` variables of nested inference are not part of the distributions of the trace; a
snapshot lists their values under ``dists`` (variable, probability, ``infer``, closed form).

Per distribution the trace stores the closed form (cut after ``TraceLimits.closed_form_chars``
characters), its mass, the first ``TraceLimits.coefficients`` coefficients of the marginals of up
to ``TraceLimits.max_variables`` variables and the size of the closed form (its length in
characters, the same measure for all engines). Marginals are skipped for larger closed forms and
once computing them took ``TraceLimits.snapshot_budget`` seconds in total.

A marginal that was not recorded is *unknown*, not zero. Every snapshot says so explicitly:
``marginals_not_recorded`` maps each variable of the distribution without a marginal table to the
reason (closed form too large, time budget used up, more than ``TraceLimits.max_variables``
variables, recording turned off), and ``marginals_omitted`` gives the reason when the snapshot has
no marginals at all. A table that could not be computed has an ``error`` and no rows. Front ends
must not draw any of them as probability 0.

The JSON form is the ``steps`` list and the ``trace`` object of the result
(:meth:`Tracer.steps_json`, :meth:`Tracer.to_json`).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from prodigy.pgcl import (AbortInstr, AsgnInstr, ChoiceInstr, Expr, IfInstr, Instr, LoopInstr, ObserveInstr,
                          QueryInstr, SkipInstr, VarExpr, WhileInstr)
from prodigy.pgcl.positions import span_of

from prodigy.analysis.loop_strategies import Bound

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.loop_strategies import LoopOutcome, LoopTask
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution


MARGINALS_TOO_LARGE = "closed form too large"
MARGINALS_OUT_OF_TIME = "time budget used up"
MARGINALS_TURNED_OFF = "recording of marginals is turned off"


@dataclass
class TraceLimits:
    """How much of each distribution the trace keeps (see the module documentation)."""

    coefficients: int = 10
    """Coefficients per marginal (0: no marginals)."""
    max_variables: int = 6
    """Marginals of at most this many variables (in alphabetical order)."""
    closed_form_chars: int = 3000
    """Longer closed forms are cut, and their marginals skipped."""
    iterates: int = 40
    """Iterates kept per loop (spread evenly over the iterations; the last one is always kept)."""
    snapshot_budget: float = 20.0
    """Seconds that computing marginals may take in total; afterwards only closed forms are kept."""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------
# Watches: queries evaluated after every top-level step
# ---------------------------------------------------------------------------------------

@dataclass
class Watch:
    """A query such as ``Pr[x == 3]`` or ``Ex[x]`` that the trace evaluates after every top-level
    step, as if it stood right after that instruction."""

    text: str
    kind: str
    """``probability`` or ``expectation``."""
    expr: Expr

    def to_json(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind}


_WATCH_PREFIXES = (("query ", ""), ("?", ""))
_WATCH_KINDS = {"Pr": "probability", "Ex": "expectation", "E": "expectation"}


def parse_watch(text: str, syntax: str, names: Optional[Sequence[str]] = None) -> Watch:
    """Parses a watch ``Pr[cond]``, ``Ex[expr]`` or ``E[expr]`` (``query Pr[…]`` and ``?Pr[…]`` are
    accepted too) with the expression parser of ``syntax``; untrusted text never reaches SymPy.
    With ``names``, the expression may only use these names. Raises ``ValueError``."""
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.ast.walk import Walk, walk_expr
    from prodigy.pgcl.parser import parse_expr
    from prodigy.pgcl.ppy_parser import parse_ppy_expr
    from prodigy.pgcl.util.ref import Mut
    original = text
    text = text.strip().rstrip(";").strip()
    for prefix, replacement in _WATCH_PREFIXES:
        if text.startswith(prefix):
            text = replacement + text[len(prefix):].lstrip()
    head, bracket, rest = text.partition("[")
    if not bracket or not rest.endswith("]") or head.strip() not in _WATCH_KINDS:
        raise ValueError(f"{original.strip()!r} is not a query: write Pr[condition] or Ex[expression].")
    inner = rest[:-1].strip()
    if not inner:
        raise ValueError(f"{original.strip()!r} has no expression.")
    try:
        expr = parse_ppy_expr(inner) if syntax == "ppy" else parse_expr(inner)
    except Exception as exc:  # pylint: disable=broad-except  # lark and syntax errors
        message = str(exc).split("\n", maxsplit=1)[0]
        raise ValueError(f"{original.strip()!r}: not an expression in the {syntax} syntax ({message})") from None
    if names is not None:
        allowed = set(names)
        for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
            if isinstance(ref.val, VarExpr) and ref.val.var not in allowed:
                raise ValueError(f"{original.strip()!r}: '{ref.val.var}' is not a variable of the program.")
    return Watch(original.strip(), _WATCH_KINDS[head.strip()], expr)


# ---------------------------------------------------------------------------------------
# Certificate checks in words
# ---------------------------------------------------------------------------------------

def _check(label: str, status: str, text: str) -> Dict[str, str]:
    """``status``: ``passed`` (proven by Prodigy), ``failed``, ``unknown`` (not established) or ``info``."""
    return {"label": label, "status": status, "text": text}


def _remaining(certificate: Mapping[str, Any], key: str = "remaining_mass") -> Optional[Dict[str, str]]:
    remaining = certificate.get(key)
    if remaining is None:
        return None
    if str(remaining) == "0":
        return _check("Residual mass", "passed", "No mass is left inside the loop: the result is exact.")
    return _check("Residual mass", "info",
                  f"Mass {remaining} is still inside the loop, so the result is a lower bound with this residual "
                  "mass (it may still terminate with any value, or never).")


def _tristate(value: Any, label: str, yes: str, no: str, undecided: str, *, no_status: str = "failed"
              ) -> Dict[str, str]:
    if value is True or (isinstance(value, str) and value not in ("", "None", "False")):
        return _check(label, "passed", yes)
    if value is False:
        return _check(label, no_status, no)
    return _check(label, "unknown", undecided)


def certificate_checks(certificate: Optional[Mapping[str, Any]], failed: Optional[str] = None
                       ) -> List[Dict[str, str]]:
    """The checks of a loop certificate (or of the ``details`` of a failed check) in words. A check is
    only ``passed`` if the certificate says that Prodigy proved it."""
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements
    from prodigy.analysis.ai_candidates import describe_rejection
    cert = dict(certificate or {})
    kind = cert.get("type")
    verdict = cert.get("verdict")
    checks: List[Dict[str, str]] = []
    if failed is not None and (verdict in ("refuted", "unknown") or cert.get("condition")):
        status = "failed" if verdict == "refuted" else "unknown" if verdict == "unknown" else "failed"
        checks.append(_check("Certificate", status, describe_rejection(failed, cert)))
    if kind == "unrolling":
        done = cert.get("iterations_done", cert.get("iterations"))
        if cert.get("fixpoint"):
            checks.append(_check("Fixed point", "passed",
                                 f"After {done} iteration(s) one more iteration changes nothing."))
        else:
            checks.append(_check("Iterations", "info", f"{done} iteration(s) computed."))
        found = _remaining(cert)
        if found and not (cert.get("fixpoint") and found["status"] == "info"):
            checks.append(found)
        elif cert.get("fixpoint") and found:
            checks.append(_check("Divergence", "info",
                                 f"Mass {cert.get('remaining_mass')} stays inside the loop forever."))
    elif kind == "mass_threshold":
        checks.append(_check("Threshold", "info", f"Captured mass {cert.get('captured_mass')} after "
                                                   f"{cert.get('iterations')} iteration(s); threshold "
                                                   f"{cert.get('threshold')}."))
        found = _remaining(cert)
        if found:
            checks.append(found)
    elif kind == "evt_iteration":
        checks.append(_check("Iterations", "info", f"{cert.get('iterations')} application(s) of the EVT operator."))
        found = _remaining(cert)
        if found:
            checks.append(found)
    elif kind == "invariant":
        if verdict == "verified":
            checks.append(_check("Fixed point", "passed",
                                 "Φ(I) = I holds for all inputs: I equals one unrolling of the loop followed by I."))
            if cert.get("constraints"):
                checks.append(_check("Constraints", "info", f"Only under the constraints {cert['constraints']}."))
            checks.append(_check("Park's lemma", "info",
                                 "So the result is an upper bound; it is exact if the loop terminates almost "
                                 "surely, which is not checked."))
        elif verdict == "refuted" and failed is None:
            checks.append(_check("Fixed point", "failed",
                                 f"Φ(I) ≠ I: counterexample {cert.get('counterexample')}."))
        elif verdict == "unknown" and failed is None:
            checks.append(_check("Fixed point", "unknown",
                                 f"Fixed point not proven: Φ(I) − I = {cert.get('difference')} could not be shown "
                                 "to vanish."))
        elif verdict == "refuted" and cert.get("counterexample") is not None:
            checks.append(_check("Counterexample", "failed",
                                 "I and one unrolling of the loop differ on the input "
                                 + ", ".join(f"{k} = {v}" for k, v in cert["counterexample"].items()) + "."))
    elif kind in ("evt_invariant", "evt_synthesis"):
        if verdict not in ("refuted", "unknown"):
            checks.append(_check("Fixed point", "passed",
                                 f"I = G + ⟦body⟧(⟨I⟩_guard) holds for I = {cert.get('invariant')}."))
            checks.append(_tristate(cert.get("nonnegative"), "Non-negative", "I has non-negative coefficients.",
                                    "I has negative coefficients, so Park's lemma does not apply.",
                                    "Non-negativity of the coefficients of I could not be established."))
            mass = cert.get("mass")
            checks.append(_tristate(cert.get("finite_mass"), "Finite mass",
                                    f"I has finite mass |I| = {mass}, so the fixed point is the least one: exact.",
                                    "I has infinite mass and may be a spurious fixed point: upper bound only.",
                                    "Finite mass of I was not shown, so exactness is not certified.",
                                    no_status="unknown"))
    elif kind == "evt_subinvariant":
        if failed is None:
            checks.append(_tristate(cert.get("nonnegative"), "ν ≥ 0",
                                    f"ν has non-negative coefficients ({cert.get('nonnegative')}).",
                                    "ν has negative coefficients.", "Non-negativity of ν was not established."))
            checks.append(_tristate(cert.get("subinvariance"), "ν ≤ Φ(ν)",
                                    f"ν ≤ Φ(ν) coefficient-wise ({cert.get('subinvariance')}).",
                                    "ν ≤ Φ(ν) fails.", "ν ≤ Φ(ν) was not established."))
            checks.append(_check("Finite mass", "passed", f"|ν| = {cert.get('mass')} is finite."))
            missing = cert.get("missing_mass")
            if missing is not None:
                checks.append(_check("Residual mass", "passed" if str(missing) == "0" else "info",
                                     "Nothing is missing: exact." if str(missing) == "0"
                                     else f"Lower bound; at most {missing} of the mass is not accounted for."))
    elif kind == "evt_sandwich":
        low, high = (cert.get("termination_probability") or [None, None])[:2]
        checks.append(_check("Termination probability", "info", f"Between {low} and {high}."))
        for name, label in (("lower", "Sub-invariant"), ("upper", "Super-invariant")):
            part = cert.get(name)
            if part:
                checks.append(_check(label, "passed", f"Checked exactly: {part.get('check') or 'verified'}."))
            elif name == "upper":
                checks.append(_check(label, "unknown", "No super-invariant found: no upper bounds."))
    elif kind == "ai_attempts":
        for attempt in cert.get("attempts") or []:
            status = {"proven": "passed", "not_larger": "info", "unknown": "unknown"}.get(attempt.get("verdict"),
                                                                                         "failed")
            checks.append(_check(f"Attempt {attempt.get('attempt')}", status,
                                 attempt.get("reason") or f"Candidate {attempt.get('candidate')}."))
    if cert.get("ai"):
        ai = cert["ai"]
        checks.append(_check("Language model", "info",
                             f"Candidate {ai.get('candidate')} proposed by {ai.get('model')} "
                             f"(attempt {ai.get('attempts')} of {ai.get('asked')}), checked by Prodigy."))
    if failed is not None and not checks:
        checks.append(_check("Strategy", "failed", describe_rejection(failed, cert)))
    return checks


# ---------------------------------------------------------------------------------------
# The tracer
# ---------------------------------------------------------------------------------------

_KINDS = (
    (WhileInstr, "while"), (IfInstr, "if"), (ChoiceInstr, "choice"), (AsgnInstr, "assign"),
    (ObserveInstr, "observe"), (LoopInstr, "loop"), (QueryInstr, "query_block"), (SkipInstr, "skip"),
    (AbortInstr, "abort"),
)
_CONTAINERS = (IfInstr, ChoiceInstr, QueryInstr)


def _kind(instr: Instr) -> str:
    for cls, name in _KINDS:
        if isinstance(instr, cls):
            return name
    return "query"


def _branch(parent: Instr, instr: Instr) -> Optional[str]:
    def within(block: Optional[List[Instr]]) -> bool:
        return any(item is instr for item in block or [])
    if isinstance(parent, IfInstr):
        return "then" if within(parent.true) else "else" if within(parent.false) else None
    if isinstance(parent, ChoiceInstr):
        return "left" if within(parent.lhs) else "right" if within(parent.rhs) else None
    if isinstance(parent, QueryInstr):
        return "block"
    return None


@dataclass
class _Open:
    """A step whose instruction is running."""
    step: Dict[str, Any]
    instr: Instr
    started: float
    overhead_before: float


@dataclass
class _LoopFrame:
    """The trace of one run of a loop strategy (the ``loop`` entry of its step)."""
    task: Any
    entry: Dict[str, Any]
    kind_before: str
    residual_before: Any
    stride: int = 1
    iterates: List[Dict[str, Any]] = field(default_factory=list)
    last_offer: Optional[Any] = None
    """(iteration, terminated, inside, residual, kind) of the last iterate offered."""
    last_recorded: int = -1


def _first_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return ""
    if lines[0].endswith("{"):  # a compound statement: its head, e.g. "while (x == 1)"
        return lines[0][:-1].rstrip()
    return lines[0] + (" …" if len(lines) > 1 else "")


class Tracer:
    """Records the trace of one analysis (see the module documentation)."""

    def __init__(self, session: "AnalysisSession", limits: Optional[TraceLimits] = None,
                 watches: Sequence[Watch] = (), on_step: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.session = session
        self.limits = limits or TraceLimits()
        self.watches = list(watches)
        self.on_step = on_step
        self.steps: List[Dict[str, Any]] = []
        self._open: List[_Open] = []
        self._loop_depth = 0
        self._frames: List[_LoopFrame] = []
        self.snapshot_time = 0.0
        """Seconds spent on snapshots (not counted as time of the steps)."""
        self.marginals_skipped = False
        self._last: Optional[tuple] = None
        """(distribution, error_prob, kind, residual, snapshot) of the last snapshot after a step."""

    # -- snapshots -------------------------------------------------------------------

    def snapshot(self, dist: "Distribution", error_prob: Optional["Distribution"], kind: str,
                 residual: Any) -> Dict[str, Any]:
        """The bounded summary of a distribution that the trace stores (with its guarantee)."""
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.nested_inference import for_trace
        from prodigy.analysis.result import exact_and_float, sym, to_latex, univariate_table
        start = time.perf_counter()
        limits = self.limits
        dist, dist_values = for_trace(dist, self.session.program, self.session.syntax)
        try:
            closed = str(dist)
        except Exception:  # pylint: disable=broad-except
            closed = "?"
        mass, mass_float = exact_and_float(_safe(dist.get_probability_mass))
        error_mass, error_float = (exact_and_float(_safe(error_prob.get_probability_mass))
                                   if error_prob is not None else (None, None))
        if kind == Bound.LOWER:
            residual_exact, residual_float = exact_and_float(residual)
        else:
            residual_exact, residual_float = None, None
        snap: Dict[str, Any] = {
            "closed_form": closed[:limits.closed_form_chars],
            "truncated": len(closed) > limits.closed_form_chars,
            "size": len(closed),
            "latex": None,
            "mass": mass,
            "mass_float": mass_float,
            "error_probability": error_mass,
            "error_probability_float": error_float,
            "bound": kind,
            "residual_mass": residual_exact,
            "residual_mass_float": residual_float,
            "marginals": {},
        }
        if dist_values:
            snap["dists"] = dist_values
        small = len(closed) <= limits.closed_form_chars
        if small and len(closed) <= 600:
            snap["latex"] = _safe(lambda: to_latex(dist))
        variables = sorted(map(str, _safe(dist.get_variables) or ()))
        omitted = self._marginals_omitted(small)
        if omitted is None:
            for variable in variables[:limits.max_variables]:
                try:
                    snap["marginals"][variable] = univariate_table(
                        dist, variable, limits.coefficients, kind=kind,
                        residual=sym(residual) if kind == Bound.LOWER else 0, max_finite_terms=200)
                except Exception as exc:  # pylint: disable=broad-except
                    snap["marginals"][variable] = {"variable": variable, "error": str(exc), "rows": []}
            plural = "" if limits.max_variables == 1 else "s"
            beyond = (f"beyond the limit of {limits.max_variables} variable{plural} per distribution "
                      "(alphabetical order)")
            snap["marginals_not_recorded"] = {variable: beyond for variable in variables[limits.max_variables:]}
        else:
            snap["marginals_omitted"] = omitted
            snap["marginals_not_recorded"] = {variable: omitted for variable in variables}
            self.marginals_skipped = self.marginals_skipped or omitted != MARGINALS_TURNED_OFF
        self.snapshot_time += time.perf_counter() - start
        return snap

    def _marginals_omitted(self, small: bool) -> Optional[str]:
        """Why a snapshot gets no marginals at all, or ``None`` if it gets them."""
        if self.limits.coefficients <= 0:
            return MARGINALS_TURNED_OFF
        if not small:
            return MARGINALS_TOO_LARGE
        if self.snapshot_time >= self.limits.snapshot_budget:
            return MARGINALS_OUT_OF_TIME
        return None

    def _reuse(self, dist: "Distribution", error_prob: "Distribution", kind: str, residual: Any) -> Dict[str, Any]:
        """The snapshot after the previous step if it shows the same distribution, else a new one."""
        last = self._last
        if last is not None and last[0] is dist and last[1] is error_prob and last[2] == kind \
                and str(last[3]) == str(residual):
            return last[4]
        return self.snapshot(dist, error_prob, kind, residual)

    def _guarantee(self) -> tuple:
        return self.session.bound_kind, self.session.residual

    # -- steps -----------------------------------------------------------------------

    def wants(self, instr: Any) -> bool:  # pylint: disable=unused-argument
        """Whether ``instr``, about to run, is a step (see the module documentation)."""
        if self.session.silent or self._loop_depth > 0:
            return False
        if not self._open:
            return True
        return isinstance(self._open[-1].instr, _CONTAINERS)

    def begin(self, instr: Any, dist: "Distribution", error_prob: "Distribution") -> Optional[_Open]:
        """Opens a step for ``instr`` if it is one (see the module documentation)."""
        if not self.wants(instr):
            return None
        parent = self._open[-1] if self._open else None
        span = span_of(instr)
        text = self.session.spell(instr)
        kind, residual = self._guarantee()
        step: Dict[str, Any] = {
            "index": len(self.steps),
            "parent": parent.step["index"] if parent else None,
            "depth": len(self._open),
            "branch": _branch(parent.instr, instr) if parent else None,
            "kind": _kind(instr),
            "instruction": text,
            "summary": _first_line(text),
            "line": span.line if span else None,
            "column": span.column if span else None,
            "end_line": span.end_line if span else None,
            "end_column": span.end_column if span else None,
            "generated": span is None,
            "before": self._reuse(dist, error_prob, kind, residual),
            "after": None,
            "time": 0.0,
            "self_time": 0.0,
            "watches": [],
            "loop": None,
            "error": None,
        }
        self.steps.append(step)
        token = _Open(step, instr, time.perf_counter(), self.snapshot_time)
        self._open.append(token)
        return token

    def _close(self, token: _Open) -> None:
        if self._open and self._open[-1] is token:
            self._open.pop()
        elif token in self._open:
            self._open.remove(token)
        step = token.step
        elapsed = time.perf_counter() - token.started - (self.snapshot_time - token.overhead_before)
        step["time"] = round(max(0.0, elapsed), 6)
        children = sum(s["time"] for s in self.steps if s["parent"] == step["index"])
        step["self_time"] = round(max(0.0, step["time"] - children), 6)

    def finish(self, token: _Open, dist: "Distribution", error_prob: "Distribution") -> None:
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.result import exact_and_float
        self._close(token)
        step = token.step
        kind, residual = self._guarantee()
        after = self.snapshot(dist, error_prob, kind, residual)
        self._last = (dist, error_prob, kind, residual, after)
        step["after"] = after
        step["distribution"] = after["closed_form"]
        step["mass"], step["mass_float"] = after["mass"], after["mass_float"]
        step["error_probability"] = after["error_probability"] or "0"
        step["bound"] = kind
        step["residual_mass"], step["residual_mass_float"] = (exact_and_float(residual) if kind == Bound.LOWER
                                                              else (None, None))
        step["size_before"], step["size_after"] = step["before"]["size"], after["size"]
        if step["depth"] == 0 and self.watches:
            step["watches"] = [self._evaluate(watch, dist, error_prob, kind, residual) for watch in self.watches]
        if self.on_step is not None:
            self.on_step(step)

    def fail(self, token: _Open, exc: BaseException) -> None:
        self._close(token)
        step = token.step
        step["error"] = str(exc) or type(exc).__name__
        step["bound"] = Bound.UNKNOWN
        step["distribution"] = None
        step["size_before"], step["size_after"] = step["before"]["size"], None
        if self.on_step is not None:
            self.on_step(step)

    def pseudo_step(self, kind: str, text: str, before: "Distribution", error_before: "Distribution",
                    after: "Distribution", error_after: "Distribution", started: float) -> None:
        """A step without an instruction of its own (e.g. conditioning on the observations at the end)."""
        if self.session.silent:
            return
        guarantee, residual = self._guarantee()
        token = _Open({
            "index": len(self.steps), "parent": None, "depth": 0, "branch": None, "kind": kind,
            "instruction": text, "summary": text, "line": None, "column": None, "end_line": None,
            "end_column": None, "generated": True,
            "before": self.snapshot(before, error_before, guarantee, residual),
            "after": None, "time": 0.0, "self_time": 0.0, "watches": [], "loop": None, "error": None,
        }, SkipInstr(), started, self.snapshot_time)
        self.steps.append(token.step)
        self._open.append(token)
        self.finish(token, after, error_after)

    # -- watches ---------------------------------------------------------------------

    def _evaluate(self, watch: Watch, dist: "Distribution", error_prob: "Distribution", kind: str,
                  residual: Any) -> Dict[str, Any]:
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.result import (QueryResult, exact_and_float, is_monotone_nonnegative, make_interval,
                                             sym)
        entry: Dict[str, Any] = {"text": watch.text, "kind": watch.kind}
        start = time.perf_counter()
        try:
            if watch.kind == "probability":
                value: Any = sym(dist.filter(watch.expr).get_probability_mass())
            else:
                value = sym(dist.get_expected_value_of(watch.expr))
            error_mass = sym(error_prob.get_probability_mass()) if error_prob is not None else 0
            if self.session.config.normalize and error_mass != 0:
                # as a query at this place: conditioned on the observations so far (see result.py)
                value = value / (1 - error_mass)
                residual = sym(residual) / (1 - error_mass) if kind == Bound.LOWER else residual
            program = self.session.program
            lower, upper, bound, note = make_interval(
                value, kind, residual, monotone=watch.kind == "probability"
                or is_monotone_nonnegative(watch.expr, program), probability=watch.kind == "probability")
            query = QueryResult(watch.kind, watch.text, value=value, bound=bound, note=note)
            query.value, query.value_float = exact_and_float(value)
            query.lower, query.lower_float = exact_and_float(lower)
            query.upper, query.upper_float = exact_and_float(upper)
            data = query.to_json()
            entry.update({key: data[key] for key in ("value", "value_float", "lower", "lower_float", "upper",
                                                     "upper_float", "bound", "note")})
        except Exception as exc:  # pylint: disable=broad-except
            entry.update({"value": None, "value_float": None, "lower": None, "lower_float": None,
                          "upper": None, "upper_float": None, "bound": Bound.UNKNOWN,
                          "note": f"Not available: {exc}"})
        self.snapshot_time += time.perf_counter() - start
        return entry

    # -- loops -----------------------------------------------------------------------

    def loop_started(self, task: "LoopTask") -> Optional[_LoopFrame]:
        """Called by :meth:`AnalysisSession.run_loop`; opens the loop entry of the running ``while`` step."""
        nested = self._loop_depth > 0
        self._loop_depth += 1
        if nested or self.session.silent or not self._open or self._open[-1].instr is not task.instruction:
            return None
        kind, residual = self._guarantee()
        entry: Dict[str, Any] = {
            "id": task.loop.id if task.loop is not None else None,
            "strategy": task.strategy.name,
            "title": task.strategy.title,
            "params": {k: v for k, v in task.decision.params.items() if not str(k).endswith("_file")},
            "kind": None,
            "iterates": [],
            "iterations": 0,
            "stride": 1,
            "checks": [],
            "messages": [],
            "failed": None,
        }
        self._open[-1].step["loop"] = entry
        frame = _LoopFrame(task, entry, kind, residual)
        self._frames.append(frame)
        return frame

    def loop_iterate(self, task: "LoopTask", iteration: int,
                     terminated: Union["Distribution", Callable[[], "Distribution"]],
                     inside: Optional["Distribution"] = None, residual: Any = None,
                     kind: str = Bound.LOWER) -> None:
        """An iterate of an unrolling strategy: ``terminated`` has left the loop after ``iteration``
        iterations (a lower bound on the loop's output), ``inside`` is still in the loop (its mass is
        the residual), or ``residual`` gives that mass directly."""
        frame = next((f for f in self._frames if f.task is task), None)
        if frame is None:
            return
        frame.entry["iterations"] = iteration
        frame.last_offer = (iteration, terminated, inside, residual, kind)
        if iteration % frame.stride:
            return
        self._record_iterate(frame, iteration, terminated, inside, residual, kind)
        if len(frame.iterates) > self.limits.iterates:
            frame.iterates = frame.iterates[::2]
            frame.stride *= 2
            frame.entry["stride"] = frame.stride

    def _record_iterate(self, frame: _LoopFrame, iteration: int, terminated: Any, inside: Any, residual: Any,
                        kind: str) -> None:
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.result import exact_and_float, sym
        dist = terminated() if callable(terminated) else terminated
        start = time.perf_counter()
        try:
            if inside is not None:
                remaining = sym(inside.get_probability_mass())
            else:
                remaining = sym(residual(dist) if callable(residual) else residual)
        except Exception:  # pylint: disable=broad-except
            remaining = None
        self.snapshot_time += time.perf_counter() - start
        own_kind = kind
        if remaining is not None and remaining == 0 and kind == Bound.LOWER:
            own_kind = Bound.EXACT
        if remaining is None:
            own_kind = Bound.UNKNOWN
        combined = Bound.combine(frame.kind_before, own_kind)
        total = None
        if combined == Bound.LOWER:
            total = (sym(frame.residual_before) if frame.kind_before == Bound.LOWER else 0) + remaining
        inside_exact, inside_float = exact_and_float(remaining)
        frame.iterates.append({
            "iteration": iteration,
            "inside_mass": inside_exact,
            "inside_mass_float": inside_float,
            "distribution": self.snapshot(dist, None, combined, total),
        })
        frame.last_recorded = iteration

    def loop_finished(self, task: "LoopTask", frame: Optional[_LoopFrame],  # pylint: disable=unused-argument
                      outcome: Optional["LoopOutcome"], exc: Optional[BaseException] = None) -> None:
        self._loop_depth = max(0, self._loop_depth - 1)
        if frame is None:
            return
        if frame in self._frames:
            self._frames.remove(frame)
        if frame.last_offer is not None and frame.last_offer[0] != frame.last_recorded:
            self._record_iterate(frame, *frame.last_offer)
        entry = frame.entry
        entry["iterates"] = frame.iterates
        if outcome is not None:
            entry["kind"] = outcome.kind
            entry["messages"] = list(outcome.messages)
            entry["checks"] = certificate_checks(outcome.certificate)
            # ai_invariant: the attempts that were rejected before
            history = (outcome.certificate.get("ai") or {}).get("history") or []
            if history:
                entry["checks"] = certificate_checks({"type": "ai_attempts", "attempts": history}) + entry["checks"]
        else:
            message = str(exc) if exc is not None else "failed"
            entry["kind"] = Bound.UNKNOWN
            entry["failed"] = message or type(exc).__name__
            entry["checks"] = certificate_checks(getattr(exc, "details", None) or {}, failed=entry["failed"])

    # -- the result ------------------------------------------------------------------

    def steps_json(self) -> List[Dict[str, Any]]:
        return self.steps

    def hints(self) -> List[Dict[str, Any]]:
        """Where the expression grows most and where most of the time goes."""
        hints: List[Dict[str, Any]] = []
        done = [s for s in self.steps if s.get("size_after") is not None]

        def where(step: Dict[str, Any]) -> str:
            return f"line {step['line']} ({step['summary'].rstrip(';')})" if step["line"] else step["summary"]

        if done:
            growth = max(done, key=lambda s: s["size_after"] - s["size_before"])
            delta = growth["size_after"] - growth["size_before"]
            if delta > 0:
                factor = growth["size_after"] / max(1, growth["size_before"])
                hints.append({"kind": "growth", "step": growth["index"], "line": growth["line"],
                              "text": f"The closed form grows most at {where(growth)}: from "
                                      f"{growth['size_before']} to {growth['size_after']} characters "
                                      f"(×{factor:.1f})."})
        timed = [s for s in self.steps if s["self_time"] > 0]
        total = sum(s["time"] for s in self.steps if s["depth"] == 0)
        if timed and total > 0:
            slow = max(timed, key=lambda s: s["self_time"])
            share = 100 * slow["self_time"] / total
            hints.append({"kind": "time", "step": slow["index"], "line": slow["line"],
                          "text": f"Most of the time goes to {where(slow)}: {slow['self_time']:.3g} s of "
                                  f"{total:.3g} s ({share:.0f} %)."})
        if self.marginals_skipped:
            hints.append({"kind": "limits", "step": None, "line": None,
                          "text": "Some steps show no marginals: their closed form is too large or the time "
                                  "budget for the trace was used up."})
        return hints

    def to_json(self) -> Dict[str, Any]:
        return {
            "limits": self.limits.to_json(),
            "watches": [watch.to_json() for watch in self.watches],
            "hints": self.hints(),
            "snapshot_time": round(self.snapshot_time, 6),
        }


_GUARANTEE_TEXT = {Bound.EXACT: "exact", Bound.LOWER: "lower bound", Bound.UPPER: "upper bound",
                   Bound.UNKNOWN: "no guarantee"}


def describe_step(step: Mapping[str, Any]) -> str:
    """One step of the trace as text for the terminal (``--intermediate-results``, ``analyze --steps``)."""
    where = f"line {step['line']}" if step.get("line") else "(generated)" if step.get("kind") != "normalize" else "end"
    indent = "  " * int(step.get("depth") or 0)
    branch = f"[{step['branch']}] " if step.get("branch") and step["branch"] != "block" else ""
    head = f"{indent}{where}: {branch}{step.get('summary') or step.get('instruction')}"
    if step.get("error"):
        return f"{head}\n{indent}    failed: {step['error']}"
    after = step.get("after") or {}
    guarantee = _GUARANTEE_TEXT.get(step.get("bound", Bound.UNKNOWN), "no guarantee")
    if step.get("bound") == Bound.LOWER and step.get("residual_mass") is not None:
        guarantee += f", residual mass {step['residual_mass']}"
    closed = after.get("closed_form", step.get("distribution", ""))
    if after.get("truncated"):
        closed += " …"
    lines = [head, f"{indent}    → {closed}", f"{indent}    mass {after.get('mass')} · {guarantee} · "
                                               f"{step.get('time', 0):.3g} s"]
    loop = step.get("loop")
    if loop:
        lines.append(f"{indent}    loop {loop.get('id')} ({loop.get('strategy')}): {len(loop.get('iterates') or [])} "
                     f"iterate(s) recorded, {loop.get('iterations')} iteration(s)")
        for check in loop.get("checks") or []:
            lines.append(f"{indent}    {check['status']}: {check['label']}: {check['text']}")
    for watch in step.get("watches") or []:
        value = watch.get("value")
        interval = "" if watch.get("bound") == Bound.EXACT else \
            f" ∈ [{watch.get('lower') or '?'}, {watch.get('upper') or '?'}] ({watch.get('bound')})"
        lines.append(f"{indent}    {watch['text']} = {value}{interval}")
    return "\n".join(lines)


def print_step(step: Mapping[str, Any]) -> None:
    """Prints a finished step (the historic ``--intermediate-results`` output, now with lines)."""
    print(describe_step(step))


def _safe(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except Exception:  # pylint: disable=broad-except
        return None
