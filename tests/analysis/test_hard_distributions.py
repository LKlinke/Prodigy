"""
The examples in ``pgfexamples/inference/loopy/hard_distributions`` build distributions with heavy
tails or non-rational generating functions. Unrolling gives sound intervals; they must contain the
exact values stated in the files.
"""
import math
from fractions import Fraction
from pathlib import Path

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

EXAMPLES = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "hard_distributions"


def _queries(name: str, decisions: dict, config=None) -> list:
    source = (EXAMPLES / name).read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions=decisions, config=config)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "lower"
    return result["queries"]


def _contains(query: dict, value) -> bool:
    upper = query["upper_float"]
    return query["lower_float"] <= float(value) and (upper is None or float(value) <= upper)


@pytest.mark.parametrize("name, decisions, expected", [
    pytest.param("d_finite/yule_simon_coins.ppy", {0: "unroll:iterations=4", 1: "unroll:iterations=20"},
                 [Fraction(1, 2), Fraction(1, 6), Fraction(1, 12)], id="yule_simon"),
    pytest.param("d_finite/yule_simon_rho2_coins.ppy", {0: "unroll:iterations=4", 1: "unroll:iterations=20"},
                 [Fraction(2, 3), Fraction(1, 6), Fraction(1, 15)], id="yule_simon_rho2"),
    pytest.param("algebraic/galton_watson_critical.ppy", {0: "unroll:iterations=10"},
                 [Fraction(1, 2), Fraction(1, 8), Fraction(1, 16), Fraction(5, 128)], id="galton_watson"),
])
def test_unrolled_intervals_contain_exact_values(name, decisions, expected):
    queries = _queries(name, decisions)
    for query, value in zip(queries, expected):
        assert _contains(query, value), (query, value)
    assert queries[len(expected)]["upper"] is None  # Ex = ∞: no upper bound


def test_supercritical_galton_watson_terminates_with_probability_one_half():
    queries = _queries("algebraic/galton_watson_supercritical.ppy", {0: "unroll:iterations=10"})
    assert _contains(queries[0], Fraction(1, 2))
    assert Fraction(queries[1]["value"]) == Fraction(1, 3)
    assert Fraction(queries[2]["value"]) == Fraction(2, 27)


ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]


@pytest.mark.parametrize("engine", ENGINES)
def test_state_dependent_urn_is_sound(engine):
    """The Yule–Simon urn chooses with probability red/(red + 1) (#213): a case split over red."""
    queries = _queries("d_finite/yule_simon_urn.ppy", {0: "unroll:iterations=5"},
                       ForwardAnalysisConfig(engine=engine))
    for query, value in zip(queries, [Fraction(1, 2), Fraction(1, 6), Fraction(1, 12)]):
        assert _contains(query, value), (query, value)
    assert queries[3]["upper"] is None  # Ex = ∞: no upper bound


def test_polya_urn_contains_logarithms():
    """The Pólya urn chooses with probability red/(red + black) (#213): Pr[red == 1] = ln 2."""
    queries = _queries("d_finite/polya_urn.ppy", {0: "unroll:iterations=10"})
    assert _contains(queries[0], math.log(2)), queries[0]
    assert _contains(queries[1], math.log(2) - 0.5), queries[1]
    assert Fraction(queries[2]["value"]) <= Fraction(3, 2)
