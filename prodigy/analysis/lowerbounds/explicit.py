r"""
Explicit-state semantics of loop-free statements of the intermediate representation, on measures given as
dicts ``state tuple -> weight`` (exact :class:`fractions.Fraction` or ``float``), and a sparse Kleene iteration for
loops.  Used to validate the generating-function calculus, to choose template parameters, and as the unrolling
baseline.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from prodigy.analysis.lowerbounds.ir import IR, UnsupportedProgram, eval_expr, eval_number

State = Tuple[int, ...]
Measure = Dict[State, object]


def _weight(p, like):
    return p if isinstance(like, Fraction) else float(p)


def run_state(stmt: IR, state: dict, w, out: Measure, variables: Sequence[str], geo_cut: int = 400):
    """Execute ``stmt`` from a single state with weight ``w`` and add the output sub-measure to ``out``."""
    t = stmt[0]
    if t == "seq":
        cur = {tuple(state[v] for v in variables): w}
        for sub in stmt[1]:
            nxt: Measure = {}
            for key, ww in cur.items():
                run_state(sub, dict(zip(variables, key)), ww, nxt, variables, geo_cut)
            cur = nxt
        for key, ww in cur.items():
            out[key] = out.get(key, 0) + ww
        return
    if t == "skip":
        key = tuple(state[v] for v in variables)
        out[key] = out.get(key, 0) + w
        return
    if t == "abort":
        return
    if t == "observe":
        if eval_expr(stmt[1], state):
            key = tuple(state[v] for v in variables)
            out[key] = out.get(key, 0) + w
        return
    if t == "choice":
        p = _weight(eval_number(stmt[1]), w)
        run_state(stmt[2], state, w * p, out, variables, geo_cut)
        run_state(stmt[3], state, w * (1 - p), out, variables, geo_cut)
        return
    if t == "if":
        branch = stmt[2] if eval_expr(stmt[1], state) else stmt[3]
        run_state(branch, state, w, out, variables, geo_cut)
        return
    if t == "assign":
        var, e = stmt[1], stmt[2]
        if e[0] == "call":
            name, args = e[1], e[2]
            if name in ("unif", "unif_d"):
                lo, hi = int(eval_number(args[0])), int(eval_number(args[1]))
                dist = [(v, Fraction(1, hi - lo + 1)) for v in range(lo, hi + 1)]
            elif name == "bernoulli":
                q = eval_number(args[0])
                dist = [(1, q), (0, 1 - q)]
            elif name == "binomial":
                n, q = int(eval_number(args[0])), eval_number(args[1])
                dist = [(v, Fraction(math.comb(n, v)) * q ** v * (1 - q) ** (n - v)) for v in range(n + 1)]
            elif name == "geometric":
                q = eval_number(args[0])
                dist = [(v, q * (1 - q) ** v) for v in range(geo_cut)]
            else:
                raise UnsupportedProgram(f"unsupported distribution {name}")
            for v, pr in dist:
                st2 = dict(state)
                st2[var] = v
                key = tuple(st2[x] for x in variables)
                out[key] = out.get(key, 0) + w * _weight(pr, w)
            return
        val = eval_expr(e, state)
        if isinstance(val, Fraction):
            if val.denominator != 1:
                raise UnsupportedProgram(f"non-integral value in {var} := {e}")
            val = int(val)
        st2 = dict(state)
        st2[var] = max(int(val), 0)
        key = tuple(st2[x] for x in variables)
        out[key] = out.get(key, 0) + w
        return
    raise UnsupportedProgram(f"unsupported statement {t}")


def apply(stmt: IR, measure: Measure, variables: Sequence[str], geo_cut: int = 400) -> Measure:
    """Forward semantics of a loop-free statement on an explicit measure."""
    out: Measure = {}
    for key, w in measure.items():
        run_state(stmt, dict(zip(variables, key)), w, out, variables, geo_cut)
    return {k: w for k, w in out.items() if w != 0}


class Kernel:
    """The transition kernel of a loop ``("loop", guard, body)``: ``step(s)`` is None if the guard is false in
    ``s`` and the list of successor states with probabilities otherwise (cached)."""

    def __init__(self, guard: IR, body: IR, variables: Sequence[str], exact: bool = False):
        self.guard, self.body, self.vars = guard, body, list(variables)
        self.exact = exact
        self.cache: Dict[State, Optional[List[Tuple[State, object]]]] = {}

    def step(self, s: State):
        if s in self.cache:
            return self.cache[s]
        st = dict(zip(self.vars, s))
        if not eval_expr(self.guard, st):
            res = None
        else:
            out: Measure = {}
            run_state(self.body, st, Fraction(1) if self.exact else 1.0, out, self.vars)
            res = [(k, w) for k, w in out.items() if w != 0]
        self.cache[s] = res
        return res


class Exploration:
    """Sparse floating-point Kleene iteration of the occupation measure of a loop, restricted to a box.

    After ``run(n)``: ``occupation`` approximates the occupation measure on the box (sum of the first ``n``
    iterates), ``exit`` the output measure of the runs that terminated within ``n`` iterations, ``remaining`` the
    mass still inside the loop, and ``leaked`` the mass that left the box (it is dropped).
    """

    def __init__(self, init: Measure, guard: IR, body: IR, variables: Sequence[str],
                 box: Optional[Sequence[int]] = None, max_states: int = 2_000_000):
        self.kernel = Kernel(guard, body, variables)
        self.vars = list(variables)
        self.box = list(box) if box is not None else None
        self.max_states = max_states
        self.current: Dict[State, float] = {k: float(w) for k, w in init.items()}
        self.occupation: Dict[State, float] = {}
        self.exit: Dict[State, float] = {}
        self.exit_by_iteration: List[float] = []
        self.leaked = 0.0
        self.iterations = 0

    def _inside(self, s: State) -> bool:
        return self.box is None or all(0 <= s[i] <= self.box[i] for i in range(len(s)))

    @property
    def remaining(self) -> float:
        return sum(self.current.values())

    def run(self, n: int, tol: float = 0.0, max_work: Optional[int] = None, time_limit: Optional[float] = None):
        """Perform up to ``n`` iterations; stop early when the mass inside the loop drops to ``tol``, after
        ``max_work`` state expansions, or after ``time_limit`` seconds."""
        import time as _time
        t0 = _time.perf_counter()
        work = 0
        for _ in range(n):
            if not self.current or self.remaining <= tol:
                break
            work += len(self.current)
            if max_work is not None and work > max_work:
                break
            if time_limit is not None and _time.perf_counter() - t0 > time_limit:
                break
            nxt: Dict[State, float] = {}
            exited = 0.0
            for s, w in self.current.items():
                self.occupation[s] = self.occupation.get(s, 0.0) + w
                tr = self.kernel.step(s)
                if tr is None:
                    self.exit[s] = self.exit.get(s, 0.0) + w
                    exited += w
                    continue
                for t, p in tr:
                    if self._inside(t):
                        nxt[t] = nxt.get(t, 0.0) + w * float(p)
                    else:
                        self.leaked += w * float(p)
            self.exit_by_iteration.append(exited)
            self.current = nxt
            self.iterations += 1
            if len(self.occupation) > self.max_states:
                break
        return self


def reachable(init_states, guard: IR, body: IR, variables: Sequence[str], box: Sequence[int],
              limit: int = 1_000_000) -> set:
    """States reachable from ``init_states`` without leaving ``box`` (support only, no weights)."""
    kernel = Kernel(guard, body, variables)
    seen = set(s for s in init_states if all(0 <= s[i] <= box[i] for i in range(len(s))))
    frontier = list(seen)
    while frontier:
        s = frontier.pop()
        tr = kernel.step(s)
        if tr is None:
            continue
        for t, _ in tr:
            if t not in seen and all(0 <= t[i] <= box[i] for i in range(len(t))):
                seen.add(t)
                frontier.append(t)
                if len(seen) > limit:
                    return seen
    return seen
