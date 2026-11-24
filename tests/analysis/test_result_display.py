"""The display forms in the result (marginal rows, tail mass, bounds) are bounded in effort and exact."""
import math
import random

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.result import sym, tidy
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

ENGINES = list(ForwardAnalysisConfig.Engine)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.name.lower())
@pytest.mark.parametrize("p", [sympy.Rational(1, 2), sympy.Rational(1, 3)], ids=["half", "third"])
def test_tail_mass_of_a_logarithmic_distribution_is_computed_quickly(engine, p):
    # #252, found by the fuzzing (#183): simplify's logcombine did not finish on the tail mass
    source = f"z: nat;\nz = logdist({p});\nprint;\n"
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy",
                              config=ForwardAnalysisConfig(engine=engine), capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok"
    marginal = result["distribution"]["marginals"]["z"]
    norm = -math.log(1 - float(p))
    exact = sum(float(p) ** k / (k * norm) for k in range(30, 400))  # the rows show the values 0, ..., 29
    assert marginal["tail_mass_float"] == pytest.approx(exact, rel=1e-6)
    assert sym(marginal["rows"][1]["prob"]).equals(-p / sympy.log(1 - p))


def test_tidy_keeps_the_value():
    x = sympy.Symbol("x")
    assert tidy(sympy.Rational(1, 2) + sympy.Rational(1, 3)) == sympy.Rational(5, 6)
    assert tidy((x**2 - 1) / (x - 1)) == x + 1
    large = sum(sympy.Rational(1, 3**k * k) for k in range(1, 31)) / sympy.log(sympy.Rational(3, 2))
    assert tidy(1 - large) == 1 - large  # with log: left as it is
    big = sum(x**k / 3**k for k in range(60))
    assert sympy.expand(tidy(big) - big) == 0


def _random_polynomial(rng, symbols, degree):
    return sum(sympy.Rational(rng.randint(-9, 9), rng.randint(1, 9)) * sympy.Mul(*(s**rng.randint(0, degree) for s in symbols))
               for _ in range(rng.randint(1, 4)))


def test_tidy_simplifies_rational_multiples_like_simplify():
    """#469: a small value is simplified as q * P with the simplified primitive part P kept; the form is that of
    sympy.simplify for generic multiples: of constants (Poisson factors exp(-λ), roots), of polynomials in
    parameters (product forms) and of sums of both."""
    rng = random.Random(469)
    p, q = sympy.symbols("p q")
    for _ in range(30):
        factors = [sympy.exp(-sympy.Rational(rng.randint(1, 9), rng.randint(1, 4))),
                   sympy.sqrt(rng.choice([2, 3, 5, 6])),
                   _random_polynomial(rng, [p], 4), _random_polynomial(rng, [p, q], 2) * (1 - p) ** rng.randint(1, 3),
                   sympy.exp(-2) + sympy.Rational(rng.randint(1, 9), 7)]
        for factor in factors:
            value = sympy.Rational(rng.randint(-10**6, 10**6) or 1, rng.randint(1, 10**6)) * factor
            if sympy.count_ops(value) <= 40:
                assert tidy(value) == sympy.simplify(value), value
                assert tidy(value) == value or sympy.simplify(tidy(value) - value) == 0


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.name.lower())
def test_marginal_lists_the_mass_it_shows(engine):
    # #288: the mass line of the Result tab names the mass shown and the mass beyond it
    source = "y: nat;\ny = geometric(1/3);\n"
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy",
                              config=ForwardAnalysisConfig(engine=engine), capture_output=True)
    marginal = session.run().to_json()["distribution"]["marginals"]["y"]
    assert marginal["truncated"]
    shown = sum(sym(row["prob"]) for row in marginal["rows"])
    assert sympy.simplify(sym(marginal["shown_mass"]) - shown) == 0
    assert sympy.simplify(sym(marginal["shown_mass"]) + sym(marginal["tail_mass"]) - 1) == 0
    assert marginal["shown_mass_float"] + marginal["tail_mass_float"] == pytest.approx(1)
