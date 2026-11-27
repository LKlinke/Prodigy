"""The fixed-point stop of ``unroll`` and ``mass_threshold`` (#508) on both engines."""
import pytest

from prodigy.analysis.certificate_checks import certificate_checks
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.loops.unrolling import _fixed_point
from prodigy.analysis.session import AnalysisSession
from prodigy.language import parse_program

ENGINES = ["SYMPY", "GINAC"]
STRATEGIES = ["unroll:iterations=10", "mass_threshold:threshold=1"]


def config_for(engine):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine), use_simplification=True)


def analyse(source, decision, engine):
    session = AnalysisSession(parse_program(source), source=source, config=config_for(engine),
                              decisions={"*": decision}, capture_output=True)
    return session.run().to_json()


def value(result, index=0):
    return result["queries"][index]["value"]


# A loop whose body keeps every state inside the loop: half of the mass diverges, the rest never enters it.
STUCK = """x: nat;
c: nat;
{ x = 0; } [1/2] { x = 1; }
while (x > 0) {
    c = c + 1;
    c = c - 1;
}
query Pr[x == 0];
"""

COUNTDOWN = """x: nat;
y: nat;
x = 3;
while (x > 0) {
    x = x - 1;
    y = y + 2;
}
query Ex[y];
"""


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("decision", STRATEGIES)
def test_fixed_point_stops_exactly_with_divergence(engine, decision):
    result = analyse(STUCK, decision, engine)
    assert result["status"] == "ok"
    assert result["bounds"]["kind"] == Bound.EXACT
    assert value(result) == "1/2"
    loop = result["loops"][0]
    assert loop["kind"] == Bound.EXACT
    assert "Exact: the iteration reached a fixed point after 1 step(s)." in loop["messages"]
    assert "Mass 1/2 stays inside the loop forever (divergence)." in loop["messages"]
    assert any("one more iteration changes nothing" in message for message in loop["messages"])


@pytest.mark.parametrize("engine", ENGINES)
def test_skip_loop_reaches_a_fixed_point(engine):
    source = "x: nat;\nx = 2;\nwhile (x > 0) { skip; }\nquery Pr[x == 0];\n"
    result = analyse(source, "unroll:iterations=10", engine)
    assert result["bounds"]["kind"] == Bound.EXACT and value(result) == "0"


@pytest.mark.parametrize("engine", ENGINES)
def test_terminated_loop_stops_early(engine):
    result = analyse(COUNTDOWN, "unroll:iterations=50", engine)
    assert result["bounds"]["kind"] == Bound.EXACT and value(result) == "6"
    messages = result["loops"][0]["messages"]
    assert "Exact: no mass is left inside the loop after 3 iterations." in messages
    assert "Stopped after 3 of 50 iterations: no mass is left inside the loop." in messages


@pytest.mark.parametrize("engine", ENGINES)
def test_mass_leaving_the_loop_is_not_a_fixed_point(engine):
    # The geometric loop keeps half of its mass inside each round: a lower bound, never a fixed point.
    source = "x: nat;\nc: nat;\nx = 1;\nwhile (x == 1) { { x = 0; } [1/2] { c = c + 1; } }\nquery Pr[c == 0];\n"
    result = analyse(source, "unroll:iterations=5", engine)
    assert result["bounds"]["kind"] == Bound.LOWER
    assert not any("fixed point" in message for message in result["loops"][0]["messages"])


@pytest.mark.parametrize("engine", ENGINES)
def test_fixed_point_needs_unchanged_violation_and_finite_mass(engine):
    factory = config_for(engine).factory
    finite, zero = factory.from_expr("x/2", "x"), factory.from_expr("0", "x")
    assert _fixed_point(finite, zero, finite, zero, zero)
    # Mass lost to an observation in this iteration: the next one loses it again.
    assert not _fixed_point(finite, zero, finite, zero, factory.from_expr("1/2", "x"))
    # Mass leaving the loop.
    assert not _fixed_point(finite, zero, finite, factory.from_expr("1/4", "x"), zero)
    # Infinite mass (the test distribution of the second-order check): no conclusion from the equality.
    infinite = factory.from_expr("1/(1-x)", "x")
    assert not _fixed_point(infinite, zero, infinite, zero, zero)


def test_mass_threshold_certificate_shows_the_fixed_point():
    checks = certificate_checks({"type": "mass_threshold", "threshold": "1", "iterations": 1,
                                 "captured_mass": "1/2", "remaining_mass": "1/2", "fixpoint": True})
    assert [(check["label"], check["status"]) for check in checks] == [("Fixed point", "passed"),
                                                                        ("Divergence", "info")]
