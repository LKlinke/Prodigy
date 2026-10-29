"""The lower-bound modes as loop strategies: sessions, query intervals, and the GUI's JSON."""
from fractions import Fraction
from pathlib import Path

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

EXAMPLES = Path(__file__).resolve().parents[2] / "pgfexamples" / "lower_bounds"


def analyse(source, decision):
    session = AnalysisSession(parse_pgcl(source), source=source, config=ForwardAnalysisConfig(),
                              decisions={"0": decision}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result.get("error")
    return result


def query(result, expr):
    return next(q for q in result["queries"] if q["expr"] == expr)


def exact(value):
    return Fraction(value.replace("^", "**")) if value is not None else None


def test_sandwich_bounds_the_queries_after_the_loop_from_both_sides():
    result = analyse((EXAMPLES / "sandwich" / "escape_walk.pgcl").read_text(), "evt_sandwich")
    loop = result["loops"][0]
    assert loop["strategy"] == "evt_sandwich" and loop["kind"] == "lower"
    total = query(result, "true")
    # terminates with probability 1/2; the super-invariant gives the upper bound exactly
    assert Fraction("0.4999999") < exact(total["lower"]) <= Fraction(1, 2)
    assert exact(total["upper"]) == Fraction(1, 2)


def test_sandwich_gives_finite_upper_bounds_for_expectations():
    result = analyse((EXAMPLES / "sandwich" / "rw_counter_23.pgcl").read_text(), "evt_sandwich")
    steps = query(result, "c")
    assert steps["lower"] is not None and steps["upper"] is not None
    assert 2.999 < steps["lower_float"] < steps["upper_float"] < 3.01


def test_query_bounds_of_the_loop_end_at_the_next_instruction():
    source = (EXAMPLES / "sandwich" / "rw_counter_23.pgcl").read_text().replace("?Pr[c >= 20]", "skip;\n?Pr[c >= 20]")
    steps = query(analyse(source, "evt_sandwich"), "c")
    # without the super-invariant's bound, an expectation only has a lower bound
    assert steps["lower"] is not None and steps["upper"] is None


def test_sandwich_rejects_conditioning():
    source = """nat x; nat y;
{y := 1} [1/2] {y := 0}
observe(y = 1);
x := 1;
while (x > 0) { {x := x + 1} [2/3] {x := x - 1} }
?Pr[true]
"""
    session = AnalysisSession(parse_pgcl(source), source=source, config=ForwardAnalysisConfig(),
                              decisions={"0": "evt_sandwich"}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] != "ok"
    assert "conditioning" in (result.get("error") or {}).get("message", "")


def test_subinvariant_that_is_the_occupation_measure_is_exact():
    result = analyse((EXAMPLES / "exact" / "lossy_retransmission.pgcl").read_text(),
                     "evt_subinvariant:subinvariant=(1 + d/3) / (1 - 3/5*c)")
    assert result["loops"][0]["kind"] == "exact"
    assert exact(query(result, "c")["value"]) == Fraction(5, 4)
    assert query(result, "c = 2")["bound"] == "exact"


def test_subinvariant_below_the_occupation_measure_gives_intervals():
    result = analyse((EXAMPLES / "exact" / "lossy_retransmission.pgcl").read_text(),
                     "evt_subinvariant:subinvariant=(1 + d/3) / (1 - 3/5*c) / 2")
    probability = query(result, "c = 2")
    assert result["loops"][0]["kind"] == "lower"
    assert exact(probability["lower"]) < Fraction(3, 25) < exact(probability["upper"])


def test_subinvariant_that_is_too_large_is_refuted():
    session = AnalysisSession(parse_pgcl((EXAMPLES / "exact" / "lossy_retransmission.pgcl").read_text()),
                              config=ForwardAnalysisConfig(),
                              decisions={"0": "evt_subinvariant:subinvariant=2 * (1 + d/3) / (1 - 3/5*c)"},
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] != "ok"
