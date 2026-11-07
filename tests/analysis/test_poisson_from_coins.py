"""
``pgfexamples/inference/loopy/conditioning/poisson_from_coins.ppy`` builds Poisson(1/2) from coin
flips (von Neumann's schema). Unrolling both loops gives sound intervals; they must contain the
Poisson probabilities, and the ratios of the first coefficients are exact.
"""
import math
from fractions import Fraction
from pathlib import Path

from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

EXAMPLE = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "conditioning" / \
    "poisson_from_coins.ppy"


def test_unrolled_intervals_contain_poisson():
    source = EXAMPLE.read_text()
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={0: "unroll:iterations=8", 1: "unroll:iterations=30"})
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "lower"
    assert result["bounds"]["residual_mass_float"] < 1e-5
    queries = result["queries"]
    for k in range(3):
        exact = math.exp(-0.5) * 0.5 ** k / math.factorial(k)
        assert queries[k]["lower_float"] <= exact <= queries[k]["upper_float"]
    values = [Fraction(query["value"]) for query in queries[:3]]
    assert values[1] / values[0] == Fraction(1, 2)
    assert values[2] / values[0] == Fraction(1, 8)
    assert queries[3]["lower_float"] <= 0.5
