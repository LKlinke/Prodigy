"""
Marginals evaluate the closed form at 1; this is the sum of the series only if the series converges there (#393).
`1/((1 - 2x)(1 - y))` has the finite closed form `-1/(1 - y)` at x = 1, but its series in x diverges (radius
1/2). `checked_marginal` proves convergence (:mod:`prodigy.distribution.convergence`) and raises
`NonFiniteMarginalError` otherwise, except in the analysis of a program from a validated input, whose
distributions are finite measures (`known_convergent_series`).
"""
import random

import numpy
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution import NonFiniteMarginalError, known_convergent_series
from prodigy.distribution.convergence import _roots_outside_unit_disc, marginal_obstacle
from prodigy.pgcl.parser import parse_expr

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC]

x, y, z, p = sympy.symbols("x y z p")
ISSUE = "1/((1 - 2*x)*(1 - y))"


def _config(engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine)


def _sympy(dist) -> sympy.Expr:
    return sympy.sympify(str(dist).replace("^", "**"))


@pytest.mark.parametrize("function, summed, kept, proven", [
    # rational: the zeros of the denominator at y = 0 in the closed unit disc of x
    (1 / ((1 - 2 * x) * (1 - y)), "x", "y", False),
    (1 / ((1 - x / 2) * (1 - y)), "x", "y", True),
    (1 / ((1 - x / 2) * (1 - y)), "y", "x", False),
    (1 / (1 - 2 * x * y), "x", "y", True),  # the coefficient of y^j is the monomial (2x)^j
    (1 / (1 + x / 2), "x", "", True),  # signed coefficients, absolutely convergent
    ((1 - x) / (1 - x ** 2), "x", "", False),  # 1/(1 + x): alternating, diverges at 1
    (1 / (1 + x ** 2 / 4), "x", "", True),  # complex poles of modulus 2
    (1 / (1 - x + x ** 2), "x", "", False),  # complex poles of modulus 1
    (1 / (3 - 4 * x + 2 * x ** 2), "x", "", True),  # complex poles of modulus sqrt(3/2)
    (1 / (1 - x * y / 2 - x / 3), "xy", "", True),  # dominant constant term
    (1 / (1 - x * y / 2 - x / 2), "xy", "", False),  # zero at x = y = 1
    (1 / (2 - x - y), "x", "y", True),
    (x ** 3 / (1 - y), "x", "y", True),  # a polynomial in x
    (1 / x, "x", "", False),  # no power series
    # parameters: the zeros are not located
    (p / (1 - (1 - p) * x), "x", "", False),
    (1 / ((1 - x / 2) * (1 - p * y)), "x", "y", True),
    # exp, log and powers
    (sympy.exp(3 * (x - 1)) * y / (2 - y), "x", "y", True),
    (sympy.log(1 - x / 2) / sympy.log(sympy.Rational(1, 2)), "x", "", True),
    (sympy.log(1 - x), "x", "", False),
    (sympy.sqrt(4 - x * y) / (2 - x + sympy.sqrt(4 - x * y)), "x", "y", True),
    (1 / (sympy.sqrt(1 - x / 4) - sympy.Rational(1, 2)), "x", "", False),
    (sympy.sqrt(1 - 2 * x), "x", "", False),
    (2 ** x, "x", "", True),
    (sympy.sin(x), "x", "", False),
])
def test_marginal_obstacle(function, summed, kept, proven):
    assert (marginal_obstacle(function, list(summed), list(kept)) is None) is proven


def test_schur_cohn_agrees_with_numerical_roots():
    rng = random.Random(393)
    checked = 0
    for _ in range(300):
        coefficients = [sympy.Rational(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(rng.randint(2, 5))]
        if coefficients[0] == 0:
            continue
        moduli = [abs(r) for r in numpy.roots([float(c) for c in coefficients])]
        if any(abs(m - 1) < 1e-9 for m in moduli):
            continue
        assert _roots_outside_unit_disc(coefficients) is all(m > 1 for m in moduli), coefficients
        checked += 1
    assert checked > 200


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_example_is_rejected(engine):
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(ISSUE, "x", "y")
        with pytest.raises(NonFiniteMarginalError, match="not proven to be a finite measure"):
            dist.checked_marginal("y")
        # summing out y diverges as well: the pole at y = 1 gives a non-finite value
        with pytest.raises(NonFiniteMarginalError):
            dist.checked_marginal("x")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function, variable, expected", [
    ("1/((2 - x)*(1 - y/2))", "y", 1 / (1 - y / 2)),
    ("1/((2 - x)*(2 - y))", "x", 1 / (2 - x)),
    # a signed measure that converges absolutely at 1
    ("1/((1 + x/2)*(1 - y))", "y", sympy.Rational(2, 3) / (1 - y)),
    ("exp(x - 1)/(2 - y)", "y", 1 / (2 - y)),
])
def test_convergent_marginals_are_computed(engine, function, variable, expected):
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(function, "x", "y")
        marginal = dist.checked_marginal(variable)
    assert sympy.simplify(_sympy(marginal) - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_signed_measure_with_a_divergent_part_is_rejected(engine):
    # the difference of two candidates, as in a check Φ(I) - I: one part diverges at x = 1
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/((1 - 2*x)*(1 - y)) - 1/((1 - x/2)*(1 - y))", "x", "y")
        with pytest.raises(NonFiniteMarginalError, match="not proven"):
            dist.checked_marginal("y")


@pytest.mark.parametrize("engine", ENGINES)
def test_case_split_update_with_approximation_is_rejected(engine):
    # before #393, z := y*w approximated the meaningless marginals of y and w (the closed form at x = 1 is
    # positive, the series diverges) and returned a result
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/((1 - 2*x)^2*(2 - y)*(2 - w))", "x", "y", "w", "z")
        with pytest.raises(NonFiniteMarginalError, match="not proven"):
            dist.update(parse_expr("z = y*w"), approximate="0.5")


@pytest.mark.parametrize("engine", ENGINES)
def test_query_marginal_is_rejected(engine):
    # a query that is not computed on the joint distribution falls back to the marginal of its variables;
    # summing out x diverges, so the query keeps its error (before #393 it went on with -1/((2 - y)(2 - w)))
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/((1 - 2*x)*(2 - y)*(2 - w))", "x", "y", "w")
        error = NotImplementedError("not computed on the joint distribution")
        with pytest.raises(NotImplementedError) as raised:
            dist.query_marginal(parse_expr("y < 2*w"), error)
    assert raised.value is error


@pytest.mark.parametrize("engine", ENGINES)
def test_program_distributions_skip_the_proof(engine):
    # geometric(p) with a parameter p: the pole 1/(1 - p) is not located without a range for p, but the
    # distributions of a program analysis are finite measures for every admissible p
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("p/(1 - (1 - p)*x) * 1/(2 - y)", "x", "y")
        dist = dist.set_parameters("p")
        with pytest.raises(NonFiniteMarginalError, match="parameters p"):
            dist.checked_marginal("y")
        with known_convergent_series(True):
            marginal = dist.checked_marginal("y")
            with known_convergent_series(False):
                with pytest.raises(NonFiniteMarginalError):
                    dist.checked_marginal("y")
    assert sympy.simplify(_sympy(marginal) - 1 / (2 - y)) == 0
