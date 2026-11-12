"""
Regression tests for engine-specific slowdowns.

#224: unrolling was up to 100x slower on one engine than on another. SymPy factored the sparse polynomials of
degree 2^k of a doubled variable and read their coefficients from a dense list; SymEngine cancelled its
growing unsimplified closed forms in SymPy (at removable poles, to decide finiteness and equality) and let the
coefficients extracted by derivatives grow exponentially.

#274: ``GeneratingFunction.marginal`` called SymPy's ``limit``, which hangs on some removable singularities.
"""
from pathlib import Path

import pytest
import symengine as se
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.generating_function import (GeneratingFunction, _limit_at_one,
                                                      _polynomial_coefficients)
from prodigy.distribution.symengine_distribution import (_differ_at_a_point, _is_polynomial_in,
                                                         _not_polynomial_after_specialising,
                                                         _value_at_removable_pole)
from prodigy.pgcl import parse_pgcl

ROOT = Path(__file__).resolve().parents[2]
HARD = ROOT / "pgfexamples/inference/loopy/hard_distributions"
ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _analyse(source, engine, decisions, simplification=True):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine),
                                   use_simplification=simplification)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              decisions=decisions, capture_output=True, coefficient_limit=0)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    return result


def _values(result):
    return [sym(query["value"]) for query in result["queries"]]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_doubling_is_fast_on_every_engine(engine):
    # payout = 2 * payout: after 30 iterations the polynomial has degree 2^30 (more than 300 s on SymPy)
    source = (HARD / "non_d_finite/st_petersburg.ppy").read_text(encoding="utf-8")
    result = _analyse(source, engine, {"0": {"strategy": "unroll", "params": {"iterations": "30"}}})
    assert _values(result)[:4] == [sympy.Rational(1, 2), sympy.Rational(1, 4), 0, sympy.Rational(1, 8)]
    assert result["bounds"]["kind"] == Bound.LOWER


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_geometric_inside_a_loop_is_fast_on_every_engine(engine):
    # more than 300 s on SymEngine with 30 iterations
    source = (HARD / "algebraic/galton_watson_critical.ppy").read_text(encoding="utf-8")
    result = _analyse(source, engine, {"0": {"strategy": "unroll", "params": {"iterations": "20"}}})
    assert _values(result)[:3] == [sympy.Rational(1, 2), sympy.Rational(1, 8), sympy.Rational(1, 16)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_nested_loops_are_fast_on_every_engine(engine):
    # more than 60 s on SymEngine with 3 outer and 10 inner iterations
    source = (HARD / "d_finite/polya_urn_coins.ppy").read_text(encoding="utf-8")
    result = _analyse(source, engine, {"0": {"strategy": "unroll", "params": {"iterations": "3"}},
                                       "1": {"strategy": "unroll", "params": {"iterations": "10"}}})
    assert _values(result)[:2] == [sympy.Rational(44711, 65536), sympy.Rational(11943, 65536)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_marginal_with_removable_singularity(engine, simplification):
    # The else branch gets the unsimplified zero -z**2/2 - z/2 + z*(z**2 - 1)/(2*(z - 1)) (times the
    # geometric part); SymPy's limit in marginal did not finish on it without simplification (#274).
    source = ("z: nat = unif(1, 2);\nw: nat = geometric(1/2);\n"
              "if (z < 3) {\n    w = w + 1;\n} else {\n    w = w + 2;\n}\nquery Pr[w == 1];\n")
    result = _analyse(source, engine, {}, simplification)
    assert result["bounds"]["kind"] == Bound.EXACT
    assert _values(result) == [sympy.Rational(1, 2)]


@pytest.mark.timeout(60)
def test_marginal_at_one_stays_small():
    z, w = sympy.symbols("z w", real=True)
    with GeneratingFunction.configured(simplification=False, latex_output=False):
        cases = {
            # removable singularity, the value is 0 (limit did not finish)
            (-z ** 2 / 2 - z / 2 + z * (z ** 2 - 1) / (2 * (z - 1))) / (2 - w): sympy.Integer(0),
            (z * (z ** 3 - 1) / (3 * (z - 1)) - z / 3) * w / (2 - w): 2 * w / (3 * (2 - w)),
            # continuous at 1: the plain value, cancelled where that is smaller
            z * w / (2 - w) / 2 + w / (2 - w) / 2: w / (2 - w),
        }
        for function, expected in cases.items():
            value = _limit_at_one(function, z)
            assert sympy.simplify(value - expected) == 0
            # up to the sign in front, e.g. -2*w/(3*w - 6) instead of 2*w/(6 - 3*w)
            assert sympy.count_ops(value) <= sympy.count_ops(expected) + 1
        gf = GeneratingFunction("(-z**2/2 - z/2 + z*(z**2 - 1)/(2*(z - 1)))/(2 - w)", "z", "w")
        assert str(gf.marginal("w")) == "0"


def test_polynomial_coefficients_are_read_sparsely():
    x, y = sympy.symbols("x y")
    function = x ** (2 ** 40) / 4 + y * x ** 2 / 2 + x / 8 + sympy.Rational(1, 8)
    assert _polynomial_coefficients(function, x, 4) == [sympy.Rational(1, 8), sympy.Rational(1, 8), y / 2, 0]
    assert _polynomial_coefficients(x * (1 + x), x, 3) == [0, 1, 1]


def test_symengine_value_at_removable_pole():
    x, y = se.symbols("x y")
    # (x^2 - x)/x at 0 is -1; the cancelled form is x - 1
    assert _value_at_removable_pole((x ** 2 - x) / x, x, se.Integer(0)) == -1
    # at 1, with another symbol: y*(x^2 - 1)/(x - 1) -> 2*y
    assert _value_at_removable_pole(y * (x ** 2 - 1) / (x - 1), x, se.Integer(1)) == 2 * y
    # numerator vanishes to a higher order: 0
    assert _value_at_removable_pole(x ** 3 / x ** 2, x, se.Integer(0)) == 0
    # a real pole and non-rational functions are left to the caller
    assert _value_at_removable_pole((1 + x) / x, x, se.Integer(0)) is None
    assert _value_at_removable_pole(se.log(1 + x) / x, x, se.Integer(0)) is None
    # the value may not contain the variable
    assert _value_at_removable_pole(1 / (x - y), x, y + x) is None


def test_symengine_polynomial_test():
    x, y, p = se.symbols("x y p")
    variables = frozenset({x, y})
    # disproved cheaply
    assert _not_polynomial_after_specialising(y / (1 - x / 2), variables)
    assert not _is_polynomial_in(y / (1 - x / 2), variables)
    # a polynomial in disguise is not disproved, and cancelling shows it
    disguised = (x ** 2 - 1) / (x - 1) * y
    assert not _not_polynomial_after_specialising(disguised, variables)
    assert _is_polynomial_in(disguised, variables)
    # parameters may occur in any form
    assert _is_polynomial_in(x / (1 - p), variables)
    assert not _not_polynomial_after_specialising(x / (1 - p), variables)


def test_symengine_differ_at_a_point():
    x = se.Symbol("x")
    assert _differ_at_a_point(1 / (1 - x), 1 + x)
    assert not _differ_at_a_point(1 / (1 - x), 1 + x / (1 - x))
    # irrational values are not compared (their form need not be canonical)
    assert not _differ_at_a_point(se.log(4 * x), 2 * se.log(2) + se.log(x))
    # nor values at a pole
    assert not _differ_at_a_point(1 / (x - se.Rational(3, 11)), x)
