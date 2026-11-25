"""
Regression tests for engine-specific slowdowns.

#224: unrolling was up to 100x slower on one engine than on another. SymPy factored the sparse polynomials of
degree 2^k of a doubled variable and read their coefficients from a dense list. (The SymEngine engine, removed
in v0.6.0, #416, had slowdowns of its own.)

#274: ``GeneratingFunction.marginal`` called SymPy's ``limit``, which hangs on some removable singularities.
"""
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.generating_function import (GeneratingFunction, _limit_at_one,
                                                      _polynomial_coefficients)
from prodigy.language import parse_program

ROOT = Path(__file__).resolve().parents[2]
HARD = ROOT / "pgfexamples/inference/loopy/hard_distributions"
ENGINES = ["SYMPY", "GINAC"]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _analyse(source, engine, decisions, simplification=True):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine),
                                   use_simplification=simplification)
    session = AnalysisSession(parse_program(source, "ppy"), source=source, syntax="ppy", config=config,
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


@pytest.mark.slow
@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_geometric_inside_a_loop_is_fast_on_every_engine(engine):
    source = (HARD / "algebraic/galton_watson_critical.ppy").read_text(encoding="utf-8")
    result = _analyse(source, engine, {"0": {"strategy": "unroll", "params": {"iterations": "20"}}})
    assert _values(result)[:3] == [sympy.Rational(1, 2), sympy.Rational(1, 8), sympy.Rational(1, 16)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_nested_loops_are_fast_on_every_engine(engine):
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
    with GeneratingFunction.configured(latex_output=False):
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

