"""
Tests for the lower-bound modes (occupation subinvariants of finite mass and the synthesis of two-sided bounds).
"""
import glob
import math
import os
import random
from fractions import Fraction

import numpy as np
import pytest
import sympy
from prodigy.pgcl import compiler

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.lowerbounds import explicit
from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus, padd, pscale
from prodigy.analysis.lowerbounds.ir import UnsupportedProgram, split_program
from prodigy.analysis.lowerbounds.parameters import TemplateShape
from prodigy.analysis.lowerbounds.problem import load_problem
from prodigy.analysis.lowerbounds.sandwich import evt_sandwich
from prodigy.analysis.lowerbounds.subinvariant import evt_subinvariant_verification, to_product_form
from prodigy.analysis.lowerbounds.synthesis import verify_certificate

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "pgfexamples", "lower_bounds")
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "sandwich", "*.pgcl")) + glob.glob(os.path.join(ROOT, "exact", "*.pgcl")))


def _load(name):
    path = name if os.path.exists(name) else os.path.join(ROOT, name)
    with open(path, encoding="utf-8") as f:
        prog = compiler.parse_pgcl(f.read())
    return prog, load_problem(prog)


def _random_numerator(d, rng, positive=False):
    poly = {}
    for _ in range(5):
        e = tuple(rng.randint(0, 3) for _ in range(d))
        poly[e] = Fraction(rng.randint(1 if positive else -3, 5), rng.randint(1, 4))
    return {e: c for e, c in poly.items() if c != 0}


def _constant_targets(stmt, acc):
    if isinstance(stmt, tuple) and stmt:
        if stmt[0] == "assign" and stmt[2][0] in ("num", "call"):
            acc.add(stmt[1])
        for sub in stmt[1:]:
            if isinstance(sub, tuple):
                _constant_targets(sub, acc)
            elif isinstance(sub, list):
                for s in sub:
                    _constant_targets(s, acc)
    return acc


@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_calculus_matches_explicit_semantics(path):
    """The generating-function calculus agrees with an explicit-state semantics on random measures."""
    _, pb = _load(path)
    rng = random.Random(hash(os.path.basename(path)) % 1000)
    constant = _constant_targets(pb.body, set())
    d = pb.d
    for _ in range(3):
        rates = [Fraction(0) if v in constant else rng.choice([Fraction(0), Fraction(1, 2), Fraction(2, 3)])
                 for v in pb.variables]
        periods = [rng.choice([1, 2]) for _ in pb.variables]
        if "modulo" in path:
            periods[0] = 2
        if "thirds" in path:
            periods[0] = 3
        calc = GFCalculus(pb.variables, Denominator(rates, periods))
        poly = _random_numerator(d, rng)
        try:
            image = calc.run(pb.body, calc.restrict(poly, pb.guard))
        except UnsupportedProgram:
            pytest.skip("outside the fragment of the calculus")
        arr = calc.expand(poly, [12] * d)
        mu = {tuple(int(i) for i in idx): arr[idx] for idx in np.ndindex(*arr.shape) if arr[idx] != 0}
        kernel = explicit.Kernel(pb.guard, pb.body, pb.variables, exact=True)
        inside = {s: w for s, w in mu.items() if kernel.step(s) is not None}
        expected = explicit.apply(pb.body, inside, pb.variables)
        got = calc.expand(image, [8] * d)
        for idx in np.ndindex(*got.shape):
            assert got[idx] == expected.get(tuple(int(i) for i in idx), 0)


@pytest.mark.parametrize("name", ["sandwich/quadrant_timeout.pgcl", "sandwich/escape_noisy.pgcl", "exact/lossy_retransmission.pgcl",
                                  "exact/modulo_geometric.pgcl", "exact/dueling_cowboys.pgcl"])
def test_calculus_matches_prodigy(name):
    """The calculus agrees with Prodigy's own forward semantics (GiNaC engine) on rational measures."""
    prog, pb = _load(name)
    _, loop, _ = split_program(prog)
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC)
    rng = random.Random(11)
    syms = {v: sympy.Symbol(v) for v in pb.variables}
    for _ in range(2):
        rates = [rng.choice([Fraction(0), Fraction(1, 2), Fraction(1, 3)]) for _ in pb.variables]
        periods = [2 if "modulo" in name and i == 0 else 1 for i in range(pb.d)]
        calc = GFCalculus(pb.variables, Denominator(rates, periods))
        poly = _random_numerator(pb.d, rng, positive=True)
        ours = calc.run(pb.body, calc.restrict(poly, pb.guard))
        dist = config.factory.from_expr(calc.to_expr(poly), *pb.variables)
        zero = config.factory.from_expr("0", *pb.variables)
        out, _ = compute_semantics(loop.body, ProgramInfo(prog), dist.filter(loop.cond), zero, config)
        diff = sympy.sympify(str(out).replace("^", "**"), locals=syms) - \
            sympy.sympify(calc.to_expr(ours).replace("^", "**"), locals=syms)
        point = {s: sympy.Rational(rng.randint(1, 9), 23) for s in syms.values()}
        assert abs(complex(diff.subs(point).evalf(30))) < 1e-20


def test_product_form_roundtrip():
    numer, den = to_product_form("(1 + d/3) / (1 - 3/5*c)", ["d", "c"])
    assert den.r == [0, Fraction(3, 5)] and den.p == [1, 1]
    assert GFCalculus(["d", "c"], den).mass(numer) == Fraction(10, 3)
    assert to_product_form("1/(1 - x)^2", ["x"]) is None


def test_subinvariant_lossy_retransmission():
    """The exact solution of Phi(nu) = nu has finite mass, hence is a lower bound (non-AST loop with abort)."""
    prog, _ = _load("exact/lossy_retransmission.pgcl")
    prefix, loop, _ = split_program(prog)
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC)
    variables = list(prog.variables.keys())
    zero = config.factory.from_expr("0", *variables)
    dist, _ = compute_semantics(prefix, ProgramInfo(prog), config.factory.one(*variables), zero, config)
    nu = config.factory.from_expr("(1 + d/3) / (1 - 3/5*c)", *variables)
    lower = evt_subinvariant_verification(loop, ProgramInfo(prog), dist, nu, config, compute_semantics)
    assert lower == config.factory.from_expr("(d/3) / (1 - 3/5*c)", *variables)
    # twice the solution is not a sub-invariant
    with pytest.raises(VerificationError):
        evt_subinvariant_verification(loop, ProgramInfo(prog), dist, nu * "2", config, compute_semantics)


def test_sandwich_escape_walk():
    """Non-AST walk: termination probability 1/2 is enclosed; the certificates pass the independent check."""
    _, pb = _load("sandwich/escape_walk.pgcl")
    lower = [TemplateShape([Fraction(0)], [1], [30], reach_box=[30])]
    upper = [TemplateShape([Fraction(1)], [1], [4])]
    res = evt_sandwich(pb, lower_shapes=lower, upper_shapes=upper, verbose=False)
    lo, hi = res.mass_bounds
    assert Fraction(49, 100) < lo <= Fraction(1, 2) <= hi
    for cert in (res.lower, res.upper):
        g = cert.calc.run(pb.prefix, cert.calc.den.poly())
        ok, _ = verify_certificate(cert.calc, g, pb.guard, pb.body, cert.numerator, cert.kind)
        assert ok
    # a perturbed certificate must be rejected
    bad = padd(res.lower.numerator, pscale(res.lower.numerator, Fraction(1, 100)))
    g = res.lower.calc.run(pb.prefix, res.lower.calc.den.poly())
    assert not verify_certificate(res.lower.calc, g, pb.guard, pb.body, bad, "lower")[0]


def test_sandwich_rw_counter_query_bounds():
    _, pb = _load("sandwich/rw_counter_23.pgcl")
    lower = [TemplateShape([Fraction(0), Fraction(17, 20)], [1, 2], [16, 32], reach_box=[16, 35])]
    upper = [TemplateShape([Fraction(1, 2), Fraction(9, 10)], [2, 2], [8, 16])]
    res = evt_sandwich(pb, lower_shapes=lower, upper_shapes=upper, verbose=False)
    truth = {"c >= 20": None, "c = 11": Fraction(42) * Fraction(2, 3) ** 6 * Fraction(1, 3) ** 5, "c": Fraction(3)}
    for b in res.bounds:
        t = truth[b.query.text]
        if t is not None:
            assert b.lower <= t <= b.upper


def _tail_rates(program: str, seeds: bool = True):
    """Per-period tail rates of the upper templates, as chosen by the parameter heuristic."""
    from prodigy.analysis.lowerbounds.parameters import profile_loop, tail_kernel, upper_tail_rates
    pb = load_problem(compiler.parse_pgcl(program))
    profiles, explo = profile_loop(pb.initial_measure(), pb.guard, pb.body, pb.variables)
    kernel = tail_kernel(pb.guard, pb.body, pb.variables, profiles, seeds=list(explo.occupation) if seeds else None)
    fixed = {pb.variables.index(v): float(r) ** (1.0 / p) for v, (r, p) in pb.forced_rates.items()}
    rates = upper_tail_rates(kernel, profiles, fixed=fixed)
    return {pb.variables[i]: s ** profiles[i].period for i, s in rates.items()}


def test_upper_tail_rates_account_for_phases():
    """The tail of the geometric loop decays with rate 1/2 because the guard variable (a bounded *phase*) is resampled;
    a per-phase increment condition would wrongly demand rate 1 (no finite moment bound)."""
    geo = """
    nat y;
    nat e;
    e := bernoulli(1/2);
    while (e = 1) { e := 0; y := y + 1; e := bernoulli(1/2) }
    ?Ex[y]
    """
    assert _tail_rates(geo)["y"] == pytest.approx(0.5, abs=1e-3)
    walk = """
    nat x;
    nat c;
    x := 1;
    while (x > 0) { {x := x - 1} [2/3] {x := x + 1}; c := c + 1 }
    ?Ex[c]
    """
    assert _tail_rates(walk)["c"] == pytest.approx(8 / 9, abs=1e-3)


def test_upper_tail_rates_only_use_reachable_phases():
    """The phase i = 0 is not reachable; as a phase of the tail it would loop forever and demand rate 1."""
    stages = """
    nat i;
    nat c;
    i := 1;
    while (i < 3) {
        c := c + 1;
        if (i = 1) { {i := i + 1} [1/2] {skip} } else { if (i = 2) { {i := i + 1} [1/4] {skip} } else { skip } }
    }
    ?Ex[c]
    """
    assert _tail_rates(stages)["c"] == pytest.approx(0.75, abs=1e-3)
    assert _tail_rates(stages, seeds=False)["c"] == pytest.approx(1.0, abs=1e-3)


def test_upper_tail_rates_keep_the_rate_of_a_geometric_input():
    """c is sampled from geometric(1/2) before the loop, so every template has rate 1/2 in c; the rates of m and n
    must then satisfy (1/2)(1/2)/s_m + (1/2)(1/2)/s_n <= 1, i.e. s_m = s_n = 1/2 at the optimum."""
    split = """
    nat c;
    nat m;
    nat n;
    c := geometric(1/2);
    while (c > 0) { {m := m + 1} [1/2] {n := n + 1}; c := c - 1 }
    ?Ex[m]
    """
    rates = _tail_rates(split)
    assert rates["c"] == pytest.approx(0.5, abs=1e-9)
    assert rates["m"] == pytest.approx(0.5, abs=1e-3) and rates["n"] == pytest.approx(0.5, abs=1e-3)


def test_upper_certificate_with_values_over_many_orders_of_magnitude():
    """The occupation measure 2^-(x+j+1) at (x, 2j) spans many orders of magnitude on the template box; without scaling
    the unknowns of the LP, small values are lost in the solver's tolerances and the certificate is far from tight."""
    double_geo = """
    nat x;
    nat y;
    x := geometric(1/2);
    while (x > 0) { x := x - 1; y := y + 2 }
    ?Ex[y]
    """
    pb = load_problem(compiler.parse_pgcl(double_geo))
    res = evt_sandwich(pb, size=300, verbose=False)
    (bound,) = res.bounds
    assert bound.lower <= 2 <= bound.upper < 2 + 1e-4


@pytest.mark.parametrize("name", ["conditioning/rw_counter_observe.pgcl", "conditioning/noisy_geometric.pgcl"])
def test_observe_is_rejected(name):
    """Conditioning is out of scope: the modes must not silently treat violated observations as non-termination."""
    path = os.path.join(ROOT, name)
    with open(path, encoding="utf-8") as f:
        prog = compiler.parse_pgcl(f.read())
    with pytest.raises(UnsupportedProgram, match="observe"):
        load_problem(prog)


def test_rational_mass_is_sound_outside_the_radius_of_convergence():
    """The mass of a closed form must not be read off by evaluating it at 1 (Pringsheim)."""
    from prodigy.analysis.lowerbounds.subinvariant import rational_mass
    assert rational_mass("(1+x)/(1-2*x)", ["x"]) == math.inf          # = 1 + 3x + 6x^2 + ..., evaluates to -2 at 1
    assert rational_mass("1/(1-x-x**2)", ["x"]) == math.inf           # Fibonacci, evaluates to -1 at 1
    assert rational_mass("(1-x**4)/(4*(1-x))", ["x"]) == 1            # uniform on {0,..,3}: 0/0 at 1
    assert rational_mass("(1+y/3)/(1-3*x/5)", ["x", "y"]) == Fraction(10, 3)
    assert rational_mass("(1-sqrt(1-x**2))/x", ["x"]) is None         # not rational: no claim
