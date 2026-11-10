"""
The examples in ``pgfexamples/inference/loopy/hard_distributions`` build distributions with heavy
tails or non-rational generating functions. Unrolling gives sound intervals; they must contain the
exact values stated in the files.
"""
from fractions import Fraction
from pathlib import Path

import pytest

from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

EXAMPLES = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "hard_distributions"


def _queries(name: str, decisions: dict) -> list:
    source = (EXAMPLES / name).read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions=decisions)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "lower"
    return result["queries"]


def _contains(query: dict, value: Fraction) -> bool:
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


@pytest.mark.xfail(strict=True, reason="#213: choice probabilities that depend on program variables")
def test_state_dependent_urn_is_sound():
    """The Yule–Simon urn chooses with probability red/(red + 1); the result must be correct or an error."""
    source = (EXAMPLES / "d_finite" / "yule_simon_urn.ppy").read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={0: "unroll:iterations=5"})
    result = session.run().to_json()
    if result["status"] == "ok":
        expected = [Fraction(1, 2), Fraction(1, 6), Fraction(1, 12)]
        for query, value in zip(result["queries"], expected):
            assert _contains(query, value), (query, value)
