# pylint: disable=invalid-name,import-outside-toplevel,use-dict-literal
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
r"""
Sandwich bounds for the output distribution of a loop: an exactly verified finite-mass EVT sub-invariant (lower
induction) bounds the output distribution from below, an exactly verified EVT super-invariant (Park induction)
from above.  Both certificates are synthesised by linear programming over eventually geometric templates
(:mod:`prodigy.analysis.lowerbounds.synthesis`) with automatically chosen or user-given template parameters.

Soundness does not depend on termination: for a lower certificate ``nu <= Phi_g(nu)`` with ``|nu| < inf`` we have
``[not guard] nu <= Post`` (lower induction), and for an upper certificate ``Phi_g(I) <= I`` we have
``Post <= [not guard] I`` (Park).  The loop-free code after the loop is a positive linear operator and maps lower
(upper) bounds to lower (upper) bounds.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Union

from prodigy.analysis.lowerbounds import explicit
from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus, Num, directed_decimal, pmul
from prodigy.analysis.lowerbounds.ir import IR, UnsupportedProgram
from prodigy.analysis.lowerbounds.parameters import ParameterPlan, TemplateShape, plan_parameters
from prodigy.analysis.lowerbounds.problem import LoopProblem, Query
from prodigy.analysis.lowerbounds.synthesis import Certificate, TemplateLP, verify_certificate
from prodigy.util.color import Style

INF = math.inf


@dataclass
class QueryBound:
    query: Query
    lower: object
    upper: object

    def to_json(self) -> dict:
        out = dict(query=f"{self.query.kind}[{self.query.text}]", lower=str(self.lower), upper=str(self.upper),
                   lower_dec=directed_decimal(self.lower, False), upper_dec=directed_decimal(self.upper, True))
        return out


@dataclass
class SandwichResult:
    problem: LoopProblem
    lower: Optional[Certificate]
    upper: Optional[Certificate]
    bounds: List[QueryBound]
    mass_bounds: tuple
    time_total: float
    plan: Optional[ParameterPlan] = None
    attempts: List[dict] = field(default_factory=list)
    per_query: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(variables=self.problem.variables,
                    lower=self.lower.to_json() if self.lower else None,
                    upper=self.upper.to_json() if self.upper else None,
                    termination_mass=dict(lower=str(self.mass_bounds[0]), upper=str(self.mass_bounds[1]),
                                          lower_dec=directed_decimal(self.mass_bounds[0], False),
                                          upper_dec=directed_decimal(self.mass_bounds[1], True)),
                    bounds=[b.to_json() for b in self.bounds], time_s=round(self.time_total, 3),
                    attempts=self.attempts,
                    profiles=[p.__dict__ for p in self.plan.profiles] if self.plan else None)


# ----------------------------------------------------------------------------------------------- helpers
def _with_forced(shape: TemplateShape, problem: LoopProblem) -> TemplateShape:
    rates, periods, tail = list(shape.rates), list(shape.periods), list(shape.tail_start)
    for v, (r, p) in problem.forced_rates.items():
        i = problem.variables.index(v)
        if rates[i] == 0:
            tail[i] = max(tail[i] - tail[i] % p, p)
        rates[i], periods[i] = Fraction(r), p
    box = shape.reach_box
    if box is not None:
        box = [max(b, t + p + 1) if r != 0 else b for b, t, r, p in zip(box, tail, rates, periods)]
    return TemplateShape(rates, periods, tail, box, shape.group)


def _upper_support(problem: LoopProblem, shape: TemplateShape, rounds: int = 6) -> set:
    """States to which an upper template may give mass.  Unlike for lower templates, the states reachable from ``g``
    within the box do not suffice: the geometric tails of the template copy the values of the last layer of the box
    beyond the box, and the mass of these copies flows back into the box.  We therefore close the reachable states
    under these copies and under the transitions of the body, within a box twice as large."""
    assert shape.reach_box is not None
    halo = [2 * b + 2 if r != 0 else b for b, r in zip(shape.reach_box, shape.rates)]
    seeds = set(problem.initial_measure().keys())
    reach: set = set()
    for _ in range(rounds):
        reach = explicit.reachable(list(seeds), problem.guard, problem.body, problem.variables, halo)
        copies = set()
        for st in reach:
            for i, (r, p, t) in enumerate(zip(shape.rates, shape.periods, shape.tail_start)):
                if r != 0 and t <= st[i] < t + p:
                    v = st[i] + p
                    while v <= halo[i]:
                        cp = st[:i] + (v,) + st[i + 1:]
                        if cp not in reach:
                            copies.add(cp)
                        v += p
        if not copies:
            break
        seeds = reach | copies
    return reach


def _initial_numerator(calc: GFCalculus, problem: LoopProblem) -> Num:
    """Numerator over the template denominator of the initial measure ``g`` of the loop."""
    if problem.init_form is not None:
        numer, den = problem.init_form
        out = dict(numer)
        for i in range(calc.d):
            if den.r[i] != 0:
                if (calc.den.r[i], calc.den.p[i]) != (den.r[i], den.p[i]):
                    raise UnsupportedProgram("the template denominator does not contain the denominator of the input")
            elif calc.den.r[i] != 0:
                out = pmul(out, calc.den.factor(i))
        return calc.run(problem.prefix, out)
    start = problem.initial_state or tuple([0] * problem.d)
    one = {}
    for e, c in calc.den.poly().items():
        one[tuple(a + b for a, b in zip(e, start))] = c
    return calc.run(problem.prefix, one)


def _post(calc: GFCalculus, problem: LoopProblem, cert_num: Num) -> Num:
    """Exit part of a certificate followed by the loop-free epilogue."""
    return calc.run(problem.epilogue, calc.restrict(cert_num, ("not", problem.guard)))


def _moment_bound(calc: GFCalculus, num: Num, expr: IR):
    """sum_n e(n) * coeff_n for an affine expression e = a0 + sum a_i x_i with nonnegative coefficients."""
    from prodigy.analysis.lowerbounds.egcalc import _affine
    aff = _affine(expr)
    if aff is None:
        raise UnsupportedProgram(f"unsupported expectation {expr}: only affine expressions in one variable")
    a, var, b = aff
    if a < 0 or b < 0:
        raise UnsupportedProgram("expectations with negative coefficients are not supported")
    total: Union[Fraction, float] = Fraction(0)
    if b != 0:
        m = calc.mass(num)
        if m == INF:
            return INF
        total += b * m
    if var is not None and a != 0:
        mo = calc.moment(num, calc.idx[var])
        if mo == INF:
            return INF
        total += a * mo
    return total


def _event_mass(calc: GFCalculus, num: Num, event: IR):
    return calc.mass(calc.restrict(num, event))


# ----------------------------------------------------------------------------------------------- synthesis
def synthesize_lower(problem: LoopProblem, shapes: Sequence[TemplateShape], verbose: bool = False,
                     attempts: Optional[list] = None) -> Optional[Certificate]:
    best = None
    init_states = list(problem.initial_measure().keys())
    for shape in shapes:
        shape = _with_forced(shape, problem)
        calc = GFCalculus(problem.variables, Denominator(shape.rates, shape.periods))
        try:
            g = _initial_numerator(calc, problem)
            support = None
            if shape.reach_box is not None:
                reach = explicit.reachable(init_states, problem.guard, problem.body, problem.variables,
                                           shape.reach_box)
                support = reach.__contains__
            lp = TemplateLP(calc, g, problem.guard, problem.body, shape.tail_start, support=support)
            h = lp.solve("lower")
        except UnsupportedProgram as exc:
            if attempts is not None:
                attempts.append(dict(kind="lower", shape=shape.describe(), error=str(exc)))
            continue
        rec: Dict[str, Any] = dict(kind="lower", shape=shape.describe(), size=lp.size,
                                   time_build=round(lp.time_build, 3), time_lp=round(lp.time_lp, 3),
                                   success=h is not None)
        if h is not None:
            cert = lp.certificate(h, "lower")
            rec["exit_mass"] = directed_decimal(cert.exit_mass, False)
            if best is None or cert.exit_mass > best.exit_mass:
                best = cert
        if attempts is not None:
            attempts.append(rec)
        if verbose:
            print(f"{Style.YELLOW}  lower candidate {shape.describe()}: "
                  f"{'exit mass >= ' + rec['exit_mass'] if h is not None else 'infeasible'}{Style.RESET}")
    return best


def upper_objective(lp: TemplateLP, problem: LoopProblem, objective: str):
    if objective == "mass":
        return lp.exit_functional()
    base = lp.exit_functional()
    total = list(base)
    for q in problem.queries:
        if q.kind == "?Ex":
            from prodigy.analysis.lowerbounds.egcalc import _affine
            aff = _affine(q.expr)
            if aff is not None and aff[1] is not None:
                mom = lp.exit_functional(moment=lp.calc.idx[aff[1]])
                total = [t + m for t, m in zip(total, mom)]
    return total


def synthesize_upper(problem: LoopProblem, shapes: Sequence[TemplateShape], objective: str = "auto",
                     verbose: bool = False, attempts: Optional[list] = None,
                     stop_after_first: bool = False) -> Optional[Certificate]:
    best, best_score = None, None
    done_groups = set()
    for shape in shapes:
        if shape.group in done_groups:
            continue
        shape = _with_forced(shape, problem)
        calc = GFCalculus(problem.variables, Denominator(shape.rates, shape.periods))
        try:
            g = _initial_numerator(calc, problem)
            support = None
            if shape.reach_box is not None:
                support = _upper_support(problem, shape).__contains__
            lp = TemplateLP(calc, g, problem.guard, problem.body, shape.tail_start, support=support)
            obj = upper_objective(lp, problem, objective)
            if any(v == INF for v in obj):
                obj = lp.exit_box_functional()
            h = lp.solve("upper", objective=obj)
        except UnsupportedProgram as exc:
            if attempts is not None:
                attempts.append(dict(kind="upper", shape=shape.describe(), error=str(exc)))
            continue
        rec: Dict[str, Any] = dict(kind="upper", shape=shape.describe(), size=lp.size,
                                   time_build=round(lp.time_build, 3), time_lp=round(lp.time_lp, 3),
                                   success=h is not None)
        if h is not None:
            done_groups.add(shape.group)
            cert = lp.certificate(h, "upper")
            cert.extra["lp"] = lp
            score = sum(o * v for o, v in zip(obj, h))
            rec["exit_mass"] = directed_decimal(cert.exit_mass, True)
            if best is None or score < best_score:
                best, best_score = cert, score
        if attempts is not None:
            attempts.append(rec)
        if verbose:
            print(f"{Style.YELLOW}  upper candidate {shape.describe()}: "
                  f"{'exit mass <= ' + rec['exit_mass'] if h is not None else 'infeasible'}{Style.RESET}")
        if best is not None and stop_after_first:
            break
    return best


# ----------------------------------------------------------------------------------------------- bounds
def refine_upper_per_query(problem: LoopProblem, upper: Certificate, verbose: bool = False) -> Dict[tuple, Certificate]:
    """Re-solve the LP of the best upper certificate once per event ``A`` and ``not A`` of every ``?Pr`` query,
    minimising ``U(A)`` (useful when the total exit mass of the upper certificate is infinite)."""
    out: Dict[tuple, Certificate] = {}
    lp = upper.extra.get("lp")
    if lp is None or problem.epilogue != ("seq", []):
        return out
    for k, q in enumerate(problem.queries):
        if q.kind != "?Pr":
            continue
        for pol, event in ((True, q.expr), (False, ("not", q.expr))):
            obj = lp.exit_functional(query=event)
            if any(v == INF for v in obj):
                continue
            h = lp.solve("upper", objective=obj)
            if h is not None:
                out[(k, pol)] = lp.certificate(h, "upper")
                if verbose:
                    print(f"{Style.YELLOW}  per-query upper certificate for {'' if pol else 'not '}"
                          f"{q.text}{Style.RESET}")
    return out


def query_bounds(problem: LoopProblem, lower: Optional[Certificate], upper: Optional[Certificate],
                 per_query: Optional[Dict[tuple, Certificate]] = None):
    """Bounds on the queries of the program from the two certificates (after the epilogue).

    For an event ``A`` with ``L(A) <= Post(A) <= U(A)`` (and likewise for the complement) and ``|Post| <= |g|``:
    ``Post(A) >= max(L(A), |L| - U(not A))`` and ``Post(A) <= min(U(A), min(|U|, |g|) - L(not A))``.  Normalised
    probabilities ``Post(A) / |Post|`` are increasing in ``Post(A)`` and decreasing in ``Post(not A)``."""
    any_cert = lower or upper
    g_mass = any_cert.calc.mass(_initial_numerator(any_cert.calc, problem)) if any_cert else Fraction(1)
    lpost = _post(lower.calc, problem, lower.numerator) if lower else {}
    upost = _post(upper.calc, problem, upper.numerator) if upper else None
    l_mass = lower.calc.mass(lpost) if lower else Fraction(0)
    u_mass = upper.calc.mass(upost) if upper is not None and upost is not None else INF
    mass_hi = min(u_mass, g_mass)
    mass_lo = l_mass

    per_query = per_query or {}

    def upper_mass(event, key):
        vals = []
        if upper is not None:
            vals.append(_event_mass(upper.calc, upost, event))
        cert = per_query.get(key)
        if cert is not None:
            vals.append(_event_mass(cert.calc, _post(cert.calc, problem, cert.numerator), event))
        return min(vals) if vals else INF

    def event_bounds(event, k=None, flip=False):
        la = _event_mass(lower.calc, lpost, event) if lower else Fraction(0)
        lac = _event_mass(lower.calc, lpost, ("not", event)) if lower else Fraction(0)
        ua = upper_mass(event, (k, not flip))
        uac = upper_mass(("not", event), (k, flip))
        lo = max(la, l_mass - uac) if uac != INF else la
        hi = min(ua, mass_hi - lac)
        return lo, hi

    out = []
    for k, q in enumerate(problem.queries):
        if q.kind == "?Pr":
            lo, hi = event_bounds(q.expr, k)
            out.append(QueryBound(q, lo, hi))
        else:
            lo = _moment_bound(lower.calc, lpost, q.expr) if lower else Fraction(0)
            hi = _moment_bound(upper.calc, upost, q.expr) if upper is not None and upost is not None else INF
            out.append(QueryBound(q, lo, hi))
    return out, (mass_lo, mass_hi)


def evt_sandwich(problem: LoopProblem, lower_shapes: Optional[Sequence[TemplateShape]] = None,
                 upper_shapes: Optional[Sequence[TemplateShape]] = None, size: int = 1200,
                 objective: str = "auto", with_upper: bool = True, verbose: bool = True,
                 per_query: Optional[bool] = None) -> SandwichResult:
    """Synthesise and verify lower and upper certificates and compute bounds for the queries."""
    t0 = time.perf_counter()
    plan = None
    if lower_shapes is None or (with_upper and upper_shapes is None):
        if verbose:
            print(f"{Style.YELLOW}Exploring the loop to choose template parameters...{Style.RESET}")
        plan = plan_parameters(problem.initial_measure(), problem.guard, problem.body, problem.variables, size=size,
                               forced=problem.forced_rates)
        if verbose:
            for p in plan.profiles:
                print(f"{Style.CYAN}  {p.name}: {p.kind}, period {p.period}"
                      f"{', decay ' + format(p.rate, '.4f') if p.kind == 'decaying' else ''}{Style.RESET}")
        lower_shapes = lower_shapes or plan.lower
        upper_shapes = upper_shapes or plan.upper
    attempts: list = []
    if verbose:
        print(f"{Style.YELLOW}Synthesising a finite-mass EVT sub-invariant (lower bound)...{Style.RESET}")
    lower = synthesize_lower(problem, lower_shapes, verbose=verbose, attempts=attempts)
    upper = None
    if with_upper:
        if verbose:
            print(f"{Style.YELLOW}Synthesising an EVT super-invariant (upper bound)...{Style.RESET}")
        assert upper_shapes is not None
        upper = synthesize_upper(problem, upper_shapes, objective=objective, verbose=verbose, attempts=attempts)
    # independent re-check of the final certificates
    for cert in (lower, upper):
        if cert is None:
            continue
        g = _initial_numerator(cert.calc, problem)
        ok, msg = verify_certificate(cert.calc, g, problem.guard, problem.body, cert.numerator, cert.kind)
        if not ok:
            raise AssertionError(f"certificate failed the independent check: {msg}")
        cert.extra["check"] = msg
    refined = {}
    if upper is not None and (per_query or (per_query is None and upper.exit_mass == INF)):
        if verbose:
            print(f"{Style.YELLOW}Refining the upper certificate per query...{Style.RESET}")
        refined = refine_upper_per_query(problem, upper, verbose=verbose)
        for cert in refined.values():
            g = _initial_numerator(cert.calc, problem)
            ok, msg = verify_certificate(cert.calc, g, problem.guard, problem.body, cert.numerator, cert.kind)
            if not ok:
                raise AssertionError(f"certificate failed the independent check: {msg}")
    for cert in [upper] + list(refined.values()):
        if cert is not None:
            cert.extra.pop("lp", None)
    bounds, mass_bounds = query_bounds(problem, lower, upper, refined)
    res = SandwichResult(problem, lower, upper, bounds, mass_bounds, time.perf_counter() - t0, plan, attempts)
    res.per_query = refined
    return res


def print_result(res: SandwichResult, show_certificates: bool = False):
    if res.lower is None:
        print(f"{Style.OKRED}No lower certificate found.{Style.RESET}")
    else:
        c = res.lower
        print(f"{Style.OKGREEN}Lower certificate{Style.RESET} (finite-mass EVT sub-invariant, {c.template_size} "
              f"template values, rates {[str(r) for r in c.den.r]}, periods {c.den.p}): "
              f"exit mass >= {directed_decimal(c.exit_mass, False)}; {c.extra.get('check', '')}")
        if show_certificates:
            print(f"  nu = {c.calc.to_expr(c.numerator)}")
    if res.upper is not None:
        c = res.upper
        print(f"{Style.OKGREEN}Upper certificate{Style.RESET} (EVT super-invariant, {c.template_size} template values, "
              f"rates {[str(r) for r in c.den.r]}, periods {c.den.p}): exit mass <= "
              f"{directed_decimal(c.exit_mass, True)}; {c.extra.get('check', '')}")
        if show_certificates:
            print(f"  I = {c.calc.to_expr(c.numerator)}")
    elif res.attempts and any(a["kind"] == "upper" for a in res.attempts):
        print(f"{Style.OKRED}No upper certificate found.{Style.RESET}")
    lo, hi = res.mass_bounds
    print(f"{Style.OKBLUE}Probability of termination (after the epilogue):{Style.RESET} "
          f"[{directed_decimal(lo, False)}, {directed_decimal(hi, True)}]")
    for b in res.bounds:
        line = f"{Style.OKBLUE}{b.query.kind}[{b.query.text}]:{Style.RESET} " \
               f"[{directed_decimal(b.lower, False)}, {directed_decimal(b.upper, True)}]"
        print(line)
    print(f"CPU-time elapsed: {res.time_total:04f} seconds")


def write_json(res: SandwichResult, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res.to_json(), f, indent=1)
