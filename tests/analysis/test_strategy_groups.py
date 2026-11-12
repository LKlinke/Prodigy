"""The loop strategies by what the user wants, the strategy ``auto`` and the exploration of
``evt_iterate`` (#324)."""
from fractions import Fraction

import pytest
import sympy
from click.testing import CliRunner
from prodigy.pgcl import parse_pgcl

from prodigy.analysis import evt_exploration
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import STRATEGY_GROUPS, get_loop_strategy, loop_strategies, strategy_groups
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
"""

OBSERVED = """nat x; nat c; nat y;
x := 1
{y := 1} [1/2] {y := 0}
observe(y = 0)
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
"""

# terminates with probability 1/2 only: the synthesised invariant has infinite mass (upper bound)
BIASED_WALK = """nat x;
x := 1
while (x > 0) { {x := x - 1} [1/3] {x := x + 1} }
?Pr[x = 0]
"""


def analyse(source, decisions):
    session = AnalysisSession(parse_pgcl(source), source=source, config=ForwardAnalysisConfig(), decisions=decisions,
                              capture_output=True)
    return session.run().to_json()


def test_every_builtin_strategy_has_a_group_summary_and_outcome():
    groups = {key for key, _, _ in STRATEGY_GROUPS}
    by_group = {key: [s.name for s in members] for key, _, _, members in strategy_groups()}
    assert by_group == {
        "automatic": ["auto"],
        "exact": ["invariant", "evt_invariant", "evt_synthesis", "ai_invariant"],
        "bounds": ["unroll", "mass_threshold", "evt_subinvariant", "evt_sandwich"],
        "explore": ["evt_iterate"],
    }
    for strategy in loop_strategies():
        assert strategy.group in groups and strategy.summary and strategy.outcome, strategy.name
        assert "you provide" in strategy.summary, strategy.name
        assert strategy.to_json()["menu_text"] == f"{strategy.title}: {strategy.summary}"
        assert strategy.to_json()["group"] == strategy.group


def test_loop_strategies_command_lists_the_groups():
    output = CliRunner().invoke(cli, ["loop_strategies"]).output
    positions = [output.index(f"{title}: ") for _, title, _, _ in strategy_groups()]
    assert positions == sorted(positions)
    assert get_loop_strategy("evt_iterate").summary in output


def test_auto_is_exact_when_the_synthesis_succeeds():
    result = analyse(GEOMETRIC, {"*": "auto"})
    loop = result["loops"][0]
    assert result["bounds"]["kind"] == "exact" and loop["strategy"] == "auto"
    assert loop["certificates"][0]["automatic"] == {"chosen": "evt_synthesis"}
    assert Fraction(result["queries"][0]["value"]) == Fraction(1, 2)


@pytest.mark.parametrize("source", [OBSERVED, BIASED_WALK])
def test_auto_falls_back_to_unrolling(source):
    """After an observe the synthesis cannot run; for the biased walk it only finds an upper bound. Both
    times auto unrolls, which gives a sound lower bound."""
    result = analyse(source, {"*": "auto:iterations=6"})
    loop = result["loops"][0]
    cert = loop["certificates"][0]
    assert cert["type"] == "unrolling" and cert["automatic"]["chosen"] == "unroll" and cert["automatic"]["synthesis"]
    assert result["bounds"]["kind"] == "lower" and loop["kind"] == "lower"
    assert loop["messages"][0].startswith("Automatic: unrolled 6 iterations")


def test_sample_points():
    assert evt_exploration.sample_points(0) == []
    assert evt_exploration.sample_points(5) == [1, 2, 3, 4, 5]
    points = evt_exploration.sample_points(100)
    assert len(points) == evt_exploration.MAX_SAMPLES and points[0] == 1 and points[-1] == 100


def test_rational_guess_and_extrapolation():
    x, c = sympy.symbols("x c")
    table = {(0, k): f"1/{2 ** (k + 1)}" for k in range(10)}
    table.update({(1, k): f"1/{2 ** k}" for k in range(10)})
    guess = evt_exploration.rational_guess(table, ["x", "c"])
    assert sympy.simplify(guess - (x + 1 / sympy.Integer(2)) / (1 - c / 2)) == 0
    # coefficients that converge geometrically in N are extrapolated exactly (Aitken)
    limits = evt_exploration.extrapolate([{(0,): "7/4"}, {(0,): "15/8"}, {(0,): "31/16"}])
    assert limits == {(0,): 2}
    # parameters: no guess
    assert evt_exploration.rational_guess({(0,): "p", (1,): "p^2"}, ["x"]) is None


def test_evt_iterate_shows_an_exploration_with_a_checkable_candidate():
    result = analyse(GEOMETRIC, {"*": "evt_iterate:iterations=12"})
    loop = result["loops"][0]
    cert = loop["certificates"][0]
    exploration = cert["exploration"]
    # still a sound lower bound: the terminating part of H_N
    assert result["bounds"]["kind"] == "lower" and loop["kind"] == "lower"
    assert any("not the output distribution" in m for m in loop["messages"])
    assert [i["n"] for i in exploration["iterates"]] == list(range(1, 13))
    assert exploration["variables"] == ["x", "c"]
    first = exploration["states"][0]
    assert first["state"] == {"x": 0, "c": 0} and len(first["values"]) == 12 and first["floats"][-1] == 0.5
    candidate = exploration["guess"]["candidate"]
    # the guess is only a proposal; Check EVT invariant proves this one exact
    checked = analyse(GEOMETRIC, {"*": {"strategy": "evt_invariant", "params": {"invariant": candidate}}})
    assert checked["status"] == "ok" and checked["bounds"]["kind"] == "exact"
