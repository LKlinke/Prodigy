r"""
Automatic choice of template parameters (rates, periods, tail starts) for the LP synthesis of EVT sub- and
superinvariants.

The heuristic explores the loop numerically (a sparse floating-point Kleene iteration of the occupation measure,
:class:`prodigy.analysis.lowerbounds.explicit.Exploration`) and derives, for every program variable,

* its *period* ``p_i``: the gcd of the differences of values of ``x_i`` among explored states that agree on all
  other variables (e.g. 2 for the step counter of a +-1 random walk, whose parity is tied to the walk);
* its *kind*: ``bounded`` (only finitely many small values), ``decaying`` (the marginal of the occupation measure
  decays geometrically, with an estimated rate per period), or ``persistent`` (no decay, e.g. the height of a
  transient random walk);
* a *scale*: the value below which (almost) all of the explored occupation mass lies.

Lower certificates truncate bounded and persistent variables and decaying variables with negligible tails
(rate 0: a truncated sub-invariant is again a sub-invariant) and give the remaining variables geometric tails with
rates slightly below the decay rate of the *truncated* loop.  Upper certificates need geometric tails in every
unbounded variable; their rates must make the template a super-invariant far out in the tail, where the body acts
as a random walk whose increments ``delta`` depend on a finite *phase* (the bounded variables and the residues of the
unbounded ones): for per-unit rates ``s_i`` this requires that the matrix
``M(s)[e', e] = sum_delta P(e -> e', delta) prod_i s_i^(-delta_i)`` has spectral radius at most one.  We minimise
the rates of the variables that are unbounded at loop exit subject to this condition.  The heuristic only proposes candidates: every certificate is
verified exactly, so the parameters affect precision and running time, never soundness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prodigy.analysis.lowerbounds.explicit import Exploration, Measure, reachable, run_state
from prodigy.analysis.lowerbounds.ir import IR, eval_expr


@dataclass
class TemplateShape:
    rates: List[Fraction]
    periods: List[int]
    tail_start: List[int]
    reach_box: Optional[List[int]] = None     # restrict the template support to the states reachable in this box
    group: int = 0                            # shapes of a group differ only in their rates (ascending)

    def size(self) -> int:
        return int(np.prod([(b + p) if r != 0 else (b + 1)
                            for r, p, b in zip(self.rates, self.periods, self.tail_start)]))

    def describe(self) -> str:
        return f"rates={[str(r) for r in self.rates]}, periods={self.periods}, tail starts={self.tail_start}"


@dataclass
class VariableProfile:
    name: str
    kind: str                 # bounded | decaying | persistent
    period: int
    rate: float               # estimated decay per period (decaying), 1.0 otherwise
    scale_trunc: int          # range that holds all but 1e-9 of the occupation mass (decaying variables)
    scale_bulk: int           # range that holds all but 1e-3 of the occupation mass
    max_value: int
    exit_unbounded: bool      # the variable takes many values at loop exit


@dataclass
class ParameterPlan:
    lower: List[TemplateShape]
    upper: List[TemplateShape]
    profiles: List[VariableProfile] = field(default_factory=list)
    exploration: Dict[str, object] = field(default_factory=dict)


def nice_rate(x: float, digits: int = 3, up: bool = False) -> Fraction:
    """Round a positive rate to a fraction with few significant digits (keeps exact arithmetic cheap)."""
    if x <= 0:
        return Fraction(0)
    e = math.floor(math.log10(x))
    scale = 10 ** (digits - 1 - e)
    v = x * scale
    return Fraction(math.ceil(v) if up else math.floor(v), scale) if scale >= 1 else Fraction(round(x))


def _periods(support: List[tuple], d: int, cap: int = 6) -> List[int]:
    periods = []
    for i in range(d):
        groups: Dict[tuple, List[int]] = {}
        for s in support:
            groups.setdefault(s[:i] + s[i + 1:], []).append(s[i])
        g = 0
        for vals in groups.values():
            for v in vals[1:]:
                g = math.gcd(g, abs(v - vals[0]))
        periods.append(g if 0 < g <= cap else 1)
    return periods


def decay_rate(marg: Dict[int, float], period: int) -> float:
    """Estimated geometric decay per ``period`` of a marginal ``v -> mass`` (least-squares fit of
    ``log m(v) = a + b log v + c v`` on the tail, which removes polynomial factors such as ``v^(-3/2)``)."""
    if not marg:
        return 0.0
    total = sum(marg.values())
    classes: Dict[int, float] = {}
    for v, m in marg.items():
        classes[v % period] = classes.get(v % period, 0.0) + m
    cls = max(classes, key=classes.get)
    pts = [(v, marg[v]) for v in sorted(marg) if v % period == cls and marg[v] > 1e-14 * total]
    if len(pts) < 3:
        return 0.0
    peak = max(range(len(pts)), key=lambda k: pts[k][1])
    cum, cut = 0.0, len(pts)
    for k, (_, m) in enumerate(pts):
        cum += m
        if cum >= (1 - 1e-7) * classes[cls]:
            cut = k + 1
            break
    tail = pts[peak:cut]
    if len(tail) < 4:
        tail = pts[peak:]
    if len(tail) < 2:
        return 0.0
    tail = tail[len(tail) // 4:] if len(tail) >= 12 else tail
    v = np.array([t[0] for t in tail], dtype=float)
    y = np.log(np.array([t[1] for t in tail]))
    if len(tail) >= 6:
        a = np.stack([np.ones_like(v), np.log(v + 1.0), v], axis=1)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        slope = coef[2]
    else:
        slope = (y[-1] - y[0]) / (v[-1] - v[0])
    return float(math.exp(slope * period))


def _quantile(marg: Dict[int, float], q: float) -> int:
    total = sum(marg.values())
    if total <= 0:
        return 0
    cum = 0.0
    for v in sorted(marg):
        cum += marg[v]
        if cum >= q * total:
            return v
    return max(marg)


def _marginal(occ: Dict[tuple, float], i: int) -> Dict[int, float]:
    marg: Dict[int, float] = {}
    for s, w in occ.items():
        marg[s[i]] = marg.get(s[i], 0.0) + w
    return marg


def profile_loop(init: Measure, guard: IR, body: IR, variables: Sequence[str], iterations: int = 1500,
                 max_states: int = 300_000, cap: int = 5000, max_work: int = 600_000,
                 time_limit: float = 8.0) -> Tuple[List[VariableProfile], Exploration]:
    """Explore the loop numerically (about ``max_work`` state expansions) and classify its variables."""
    d = len(variables)
    explo = Exploration(init, guard, body, variables, box=[cap] * d, max_states=max_states)
    explo.run(iterations // 2, tol=1e-13, max_work=max_work // 3, time_limit=time_limit / 3)
    half = dict(explo.occupation)
    half_support_max = [max((s[i] for s in half), default=0) for i in range(d)]
    explo.run(explo.iterations, tol=1e-13, max_work=2 * max_work // 3, time_limit=2 * time_limit / 3)
    occ = explo.occupation
    top = max(occ.values()) if occ else 1.0
    support = [s for s, w in occ.items() if w > 1e-15 * top]
    periods = _periods(support, d)
    profiles = []
    for i, name in enumerate(variables):
        marg = _marginal(occ, i)
        vmax = max((s[i] for s in support), default=0)
        exit_vals = {s[i] for s, w in explo.exit.items() if w > 1e-12}
        exit_unb = len(exit_vals) > 8
        if vmax <= 8 and vmax == half_support_max[i]:
            profiles.append(VariableProfile(name, "bounded", 1, 1.0, vmax, vmax, vmax, exit_unb))
            continue
        # only the part of the marginal that has converged (it did not change in the second half of the
        # exploration) carries information about the decay; for transient loops the frontier is an artefact
        marg_half = _marginal(half, i)
        conv: Dict[int, float] = {}
        for v in sorted(marg):
            if abs(marg[v] - marg_half.get(v, 0.0)) > 1e-3 * marg[v]:
                break
            conv[v] = marg[v]
        rho = decay_rate(conv, periods[i]) if len(conv) >= 4 else 0.0
        converged = len(conv) == len(marg)
        if 0 < rho < 0.985 and (converged or explo.remaining < 1e-6 or len(conv) >= 24):
            kind = "decaying"
        elif rho == 0.0 and converged:
            kind = "decaying"
            rho = decay_rate(marg, periods[i]) or 0.5
        else:
            kind = "persistent"
        if kind == "persistent" and converged and vmax == half_support_max[i] and vmax <= 64:
            # the variable keeps to a fixed small range (e.g. it is resampled in every iteration)
            profiles.append(VariableProfile(name, "bounded", 1, 1.0, vmax, vmax, vmax, exit_unb))
            continue
        q_trunc = _quantile(marg, 1 - 1e-9) if kind == "decaying" else vmax
        q_bulk = _quantile(marg, 1 - 1e-3) if kind == "decaying" else min(vmax, 24)
        profiles.append(VariableProfile(name, kind, periods[i], rho if kind == "decaying" else 1.0,
                                        max(q_trunc, 1), max(q_bulk, 1), vmax, exit_unb))
    return profiles, explo


# ------------------------------------------------------------------------------------------- tail kernel
@dataclass
class TailKernel:
    """The loop far out in the tail as a Markov-modulated random walk: the *phase* of a state consists of the values
    of the bounded variables and the residues of the unbounded ones modulo their periods; ``transitions`` lists
    ``(target phase, source phase, increment of the unbounded variables, probability)`` for guard-satisfying
    source phases."""
    phases: List[tuple]
    transitions: List[Tuple[int, int, tuple, float]]

    def _entries(self, log_s: np.ndarray):
        rows = np.fromiter((j for j, _, _, _ in self.transitions), dtype=np.int64, count=len(self.transitions))
        cols = np.fromiter((i for _, i, _, _ in self.transitions), dtype=np.int64, count=len(self.transitions))
        deltas = np.array([d for _, _, d, _ in self.transitions], dtype=float).reshape(len(self.transitions), -1)
        weights = np.fromiter((w for _, _, _, w in self.transitions), dtype=float, count=len(self.transitions))
        # cap the exponent: an overflowing entry only means that these rates are far from feasible
        vals = weights * np.exp(np.minimum(-(deltas @ np.asarray(log_s, dtype=float)), 700.0))
        return rows, cols, vals

    def matrix(self, log_s: np.ndarray) -> np.ndarray:
        """``M(s)[e', e] = sum P(e -> e', delta) * prod_i s_i^(-delta_i)`` for per-unit rates ``s = exp(log_s)``."""
        n = len(self.phases)
        m = np.zeros((n, n))
        if self.transitions:
            rows, cols, vals = self._entries(log_s)
            np.add.at(m, (rows, cols), vals)
        return m

    def log_radius(self, log_s: np.ndarray) -> float:
        """Logarithm of the spectral radius of ``M(s)`` (``-inf`` if it is nilpotent)."""
        n = len(self.phases)
        if not self.transitions:
            return -math.inf
        rho = None
        if n > 400:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigs
            rows, cols, vals = self._entries(log_s)
            if not np.all(np.isfinite(vals)):
                return math.inf
            try:
                ev = eigs(csr_matrix((vals, (rows, cols)), shape=(n, n)), k=1, which="LM", tol=1e-10,
                          maxiter=20 * n, return_eigenvectors=False)
                rho = float(np.max(np.abs(ev)))
            except (ArpackNoConvergence, ArpackError, ValueError):
                rho = None
        if rho is None:
            m = self.matrix(log_s)
            if not np.all(np.isfinite(m)):
                return math.inf
            rho = float(np.max(np.abs(np.linalg.eigvals(m))))
        if not math.isfinite(rho):
            return math.inf
        return math.log(rho) if rho > 1e-300 else -math.inf


def tail_kernel(guard: IR, body: IR, variables: Sequence[str], profiles: List[VariableProfile],
                seeds: Optional[Sequence[tuple]] = None, base: int = 10_080, max_phases: int = 2048) -> TailKernel:
    """Explore the phases reachable (far out in the tail) from the phases of the states ``seeds`` -- the states visited
    by the exploration of the loop -- or, without seeds, from all combinations of bounded values and residues.
    ``base`` is a multiple of every possible period, so that the residue of a seed is that of its tail copy."""
    d = len(variables)
    unb = [i for i, p in enumerate(profiles) if p.kind != "bounded"]

    def phase_of(t):
        return tuple(t[i] if profiles[i].kind == "bounded" else (t[i] - base) % profiles[i].period for i in range(d))

    def state_of(ph):
        return tuple(ph[i] if profiles[i].kind == "bounded" else base + ph[i] for i in range(d))

    if seeds is not None:
        queue = list(dict.fromkeys(phase_of(t) for t in seeds))
    else:
        choices = [list(range(p.max_value + 1)) if p.kind == "bounded" else list(range(p.period)) for p in profiles]
        queue = list(product(*choices))
    index: Dict[tuple, int] = {}
    pending: List[Tuple[tuple, tuple, tuple, float]] = []
    head = 0
    while head < len(queue) and len(index) < max_phases:
        ph = queue[head]
        head += 1
        if ph in index:
            continue
        index[ph] = len(index)
        st = state_of(ph)
        env = dict(zip(variables, st))
        if not eval_expr(guard, env):
            continue                       # exit phase: no outgoing transitions
        out: Dict[tuple, float] = {}
        run_state(body, env, 1.0, out, variables)
        for t, w in out.items():
            tph = phase_of(t)
            pending.append((tph, ph, tuple(t[i] - st[i] for i in unb), w))
            if tph not in index:
                queue.append(tph)
    transitions = [(index[tph], index[ph], delta, w) for tph, ph, delta, w in pending if tph in index]
    return TailKernel(list(index), transitions)


def upper_tail_rates(kernel: TailKernel, profiles: List[VariableProfile], slack: float = 0.0,
                     fixed: Optional[Dict[int, float]] = None):
    """Per-unit rates ``s_i`` of the unbounded variables minimising the rates of the exit-relevant variables
    subject to ``rho(M(s)) <= 1``, the condition under which a product tail ``a_e * prod_i s_i^(n_i)`` with
    phase-dependent coefficients ``a_e > 0`` can be super-invariant far out in the tail (Perron--Frobenius:
    ``M(s) a <= a`` has a solution ``a > 0`` iff the spectral radius is at most one).  ``log rho(M(e^t))`` is convex
    in ``t`` (Kingman), so the problem is convex.  The per-unit rates ``fixed`` are not optimised: a variable that is
    sampled from a geometric distribution before the loop keeps the rate of that distribution, since the denominator of
    the template has to contain the denominator of the initial measure.  Returns None if the optimisation fails."""
    from scipy.optimize import minimize
    unb = [i for i, p in enumerate(profiles) if p.kind != "bounded"]
    if not kernel.transitions or not unb:
        return None
    fixed = {i: s for i, s in (fixed or {}).items() if i in unb}
    free = [k for k, i in enumerate(unb) if i not in fixed]
    base_t = np.array([math.log(fixed[i]) if i in fixed else 0.0 for i in unb])
    weights = np.array([1.0 if profiles[unb[k]].exit_unbounded else 1e-3 for k in free])

    def full(t):
        out = base_t.copy()
        out[free] = t
        return out

    def margin(t):
        lr = kernel.log_radius(full(t))
        return 50.0 if lr == -math.inf else -lr - math.log1p(slack)

    if not free:
        return {i: fixed[i] for i in unb} if margin(np.zeros(0)) >= -1e-9 else None
    cons = [{"type": "ineq", "fun": margin}]
    best = None
    for start in (np.zeros(len(free)), np.full(len(free), -0.1), np.full(len(free), 0.3)):
        res = minimize(lambda t: float(weights @ t), start, constraints=cons, method="SLSQP",
                       bounds=[(-12.0, 3.0)] * len(free), options={"maxiter": 500, "ftol": 1e-12})
        if res.success and margin(res.x) >= -1e-9:
            if best is None or weights @ res.x < weights @ best:
                best = res.x
    if best is None:
        return None
    rates = {i: float(math.exp(t)) for i, t in zip(unb, full(best))}
    return {i: (fixed[i] if i in fixed else 1.0 if abs(s - 1.0) < 5e-3 else s) for i, s in rates.items()}


# ------------------------------------------------------------------------------------------- sizes
def _reach_count(init_states, guard, body, variables, shape: TemplateShape, cap: int = 100_000) -> int:
    """Number of template values that survive the restriction to reachable states (the box size itself if the box
    has more than ``cap`` states, which avoids exploring huge boxes)."""
    full = int(np.prod([b + 1 for b in shape.reach_box]))
    if full > cap:
        return full
    reach = reachable(init_states, guard, body, variables, shape.reach_box)
    hi = [b + p - 1 if r != 0 else b for r, p, b in zip(shape.rates, shape.periods, shape.tail_start)]
    return sum(1 for s in reach if all(s[i] <= hi[i] for i in range(len(hi))))


def plan_parameters(init: Measure, guard: IR, body: IR, variables: Sequence[str], size: int = 1200,
                    upper_size: Optional[int] = None, lower_multipliers=(0.95, 0.985, 0.998),
                    upper_steps=(0.01, 0.05, 0.15, 0.35), aspect_factors=(1.0, 0.6),
                    forced: Optional[Dict[str, Tuple[Fraction, int]]] = None) -> ParameterPlan:
    """Candidate template shapes for lower and upper certificates.  ``forced`` maps the variables that are sampled
    from a geometric distribution before the loop to the rate and period of that distribution."""
    profiles, explo = profile_loop(init, guard, body, variables)
    d = len(variables)
    upper_size = upper_size or max(size // 2, 64)
    init_states = list(init.keys())

    # ---------------------------------------------------------------- lower certificates
    # Truncate what can be truncated cheaply; variables that take many values at loop exit and do not have a
    # negligible tail get geometric tails, provided they decay in the truncated loop.
    trunc: Dict[int, int] = {}
    geometric: List[int] = []
    bulk_low: Dict[int, int] = {}
    for i, pr in enumerate(profiles):
        if pr.kind == "bounded":
            trunc[i] = pr.max_value
        elif pr.kind == "persistent" and not pr.exit_unbounded:
            trunc[i] = min(pr.max_value, 40)
        elif pr.kind == "decaying" and pr.scale_trunc <= 48:
            trunc[i] = pr.scale_trunc
        else:
            geometric.append(i)
    if geometric:
        box = [trunc.get(i, 5000) for i in range(d)]
        tr = Exploration(init, guard, body, variables, box=box, max_states=300_000).run(
            1000, tol=1e-13, max_work=400_000, time_limit=6.0)
        for i in list(geometric):
            rho = decay_rate(_marginal(tr.occupation, i), profiles[i].period)
            if not 0 < rho < 0.995:
                geometric.remove(i)
                trunc[i] = min(profiles[i].max_value, 40)
            elif profiles[i].kind == "persistent":
                marg = _marginal(tr.occupation, i)
                bulk_low[i] = max(_quantile(marg, 1 - 1e-3), 1)
    periods_low = [profiles[i].period if i in geometric else 1 for i in range(d)]
    bulk = {i: max(bulk_low.get(i, profiles[i].scale_bulk), 4 * profiles[i].period) for i in geometric}

    def lower_shape(lam_t, lam_g):
        tail = [0] * d
        for i in range(d):
            if i in geometric:
                b = max(int(math.ceil(lam_g * bulk[i])), 2 * profiles[i].period)
                tail[i] = b - b % profiles[i].period
            elif profiles[i].kind == "bounded":
                tail[i] = trunc[i]
            else:
                tail[i] = max(int(math.ceil(lam_t * trunc[i])), 2)
        rates = [Fraction(1, 2) if i in geometric else Fraction(0) for i in range(d)]
        box = [tail[i] + periods_low[i] + 1 if i in geometric else tail[i] for i in range(d)]
        return TemplateShape(rates, list(periods_low), tail, reach_box=box)

    lam_t, lam_g = 1.0, 1.0
    shape = lower_shape(lam_t, lam_g)
    for _ in range(12):
        count = _reach_count(init_states, guard, body, variables, shape)
        if count <= size * 1.1 and (count >= size * 0.75 or lam_g >= 4):
            break
        factor = (size / max(count, 1)) ** (1.0 / max(len(geometric), 1))
        if geometric:
            lam_g = min(max(lam_g * factor, 0.1), 4.0)
            if count > size and lam_g <= 0.1:
                lam_t *= 0.8
        else:
            lam_t = min(lam_t * (size / max(count, 1)) ** (1.0 / max(d, 1)), 1.0)
            if count <= size:
                break
        shape = lower_shape(lam_t, lam_g)
    # decay of the truncated loop (the template must lie below its occupation measure)
    rates_low: Dict[int, float] = {}
    if geometric:
        box = [shape.tail_start[i] if i not in geometric else 5000 for i in range(d)]
        tr = Exploration(init, guard, body, variables, box=box, max_states=400_000).run(
            1500, tol=1e-13, max_work=400_000, time_limit=6.0)
        for i in geometric:
            rho = decay_rate(_marginal(tr.occupation, i), profiles[i].period)
            rates_low[i] = rho if 0 < rho < 1 else profiles[i].rate
    lower = []
    for m in lower_multipliers:
        rates = [nice_rate(min(rates_low[i] * m, 0.999)) if i in geometric else Fraction(0) for i in range(d)]
        lower.append(TemplateShape(rates, list(periods_low), list(shape.tail_start), reach_box=shape.reach_box))

    # ---------------------------------------------------------------- upper certificates
    unb = [i for i, p in enumerate(profiles) if p.kind != "bounded"]
    kernel = tail_kernel(guard, body, variables, profiles, seeds=list(explo.occupation))
    fixed = {variables.index(v): float(r) ** (1.0 / p) for v, (r, p) in (forced or {}).items() if v in variables}
    per_unit = upper_tail_rates(kernel, profiles, fixed=fixed)
    periods_up = [profiles[i].period if i in unb else 1 for i in range(d)]

    def extent(i):
        pr = profiles[i]
        if pr.kind == "persistent":
            return max(pr.max_value, 4)
        return max(pr.scale_bulk, 2 * pr.period)

    def upper_box(tail):
        return [tail[i] + periods_up[i] - 1 if i in unb else tail[i] for i in range(d)]

    upper = []
    seen = set()
    for group, factor in enumerate(aspect_factors):
        base_up = {i: extent(i) * (factor if profiles[i].exit_unbounded else 1.0) for i in unb}
        top = max(base_up.values(), default=1.0)

        def upper_tail(lam, base_up=base_up, top=top):
            tail = [0] * d
            for i in range(d):
                if i in unb:
                    b = max(int(math.ceil(lam * base_up[i] / top)), 2 * profiles[i].period)
                    tail[i] = b - b % profiles[i].period
                else:
                    tail[i] = profiles[i].max_value
            return tail

        def upper_count(lam):
            tail = upper_tail(lam)
            full = int(np.prod([b + 1 for b in upper_box(tail)]))
            if full <= upper_size:
                return full
            shape = TemplateShape([Fraction(1, 2) if i in unb else Fraction(0) for i in range(d)], periods_up, tail,
                                  reach_box=upper_box(tail))
            return _reach_count(init_states, guard, body, variables, shape)

        lam = 400.0
        while lam > 1 and upper_count(lam) > upper_size:
            lam *= 0.9
        tail_up = upper_tail(lam)
        if tuple(tail_up) in seen:
            continue
        seen.add(tuple(tail_up))
        full_box = int(np.prod([b + 1 for b in upper_box(tail_up)]))
        up_reach = upper_box(tail_up) if full_box > upper_size else None
        for mu in upper_steps:
            rates = []
            for i, pr in enumerate(profiles):
                if i not in unb:
                    rates.append(Fraction(0))
                elif per_unit is not None:
                    s = per_unit[i]
                    if pr.exit_unbounded and s < 1:
                        s = s + (1 - s) * mu          # towards 1: heavier tails are easier to dominate
                    rates.append(nice_rate(s ** pr.period, up=True) if abs(s - 1) > 1e-9 else Fraction(1))
                elif pr.kind == "persistent":
                    rates.append(Fraction(1) if mu < 0.1 else Fraction(2))
                else:
                    rho = pr.rate ** (1.0 / pr.period)
                    rates.append(nice_rate((rho + (1 - rho) * mu) ** pr.period, up=True))
            upper.append(TemplateShape(rates, list(periods_up), list(tail_up), reach_box=up_reach, group=group))
        # The tail condition only sees the states far from all boundaries.  If the estimated decay rate of a decaying
        # variable that is unbounded at exit exceeds all rates tried so far (e.g. because the decay is governed by
        # states near a boundary, as for a walk in the quadrant), also try rates between that decay rate and 1.
        if per_unit is not None:
            emp = {i: profiles[i].rate ** (1.0 / profiles[i].period) for i in unb
                   if profiles[i].kind == "decaying" and profiles[i].exit_unbounded}
            if any(emp[i] > per_unit[i] + (1 - per_unit[i]) * max(upper_steps) for i in emp):
                for mu in upper_steps:
                    rates = []
                    for i, pr in enumerate(profiles):
                        if i not in unb:
                            rates.append(Fraction(0))
                            continue
                        s = per_unit[i]
                        if i in emp and emp[i] > s:
                            s = emp[i] + (1 - emp[i]) * mu
                        elif pr.exit_unbounded and s < 1:
                            s = s + (1 - s) * mu
                        rates.append(nice_rate(s ** pr.period, up=True) if abs(s - 1) > 1e-9 else Fraction(1))
                    upper.append(TemplateShape(rates, list(periods_up), list(tail_up), reach_box=up_reach,
                                               group=group))
    info = dict(iterations=explo.iterations, states=len(explo.occupation), remaining=explo.remaining,
                leaked=explo.leaked, exit_mass=sum(explo.exit.values()),
                tail_rates={variables[i]: s for i, s in per_unit.items()} if per_unit else None)
    return ParameterPlan(lower=lower, upper=upper, profiles=profiles, exploration=info)
