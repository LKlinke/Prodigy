"""The simulator (prodigy.simulator) on programs with known distributions."""
import math
from fractions import Fraction

import pytest

from prodigy.pgcl.parser import parse_pgcl
from prodigy.simulator import (Estimate, Mismatch, SimulationError, UnsupportedConstruct, compare, simulate,
                               proportion_interval)


def run(source, runs=4000, **kwargs):
    return simulate(parse_pgcl(source, "ppy"), runs, seed=1, **kwargs)


def frequency(simulation, var, value):
    return simulation.probability(lambda s: s[var] == value).value


@pytest.mark.parametrize("source, var, expected", [
    ("x: nat; x = unif(1, 4);", "x", {1: 1 / 4, 2: 1 / 4, 3: 1 / 4, 4: 1 / 4}),
    ("x: nat; x = bernoulli(1/3);", "x", {0: 2 / 3, 1: 1 / 3}),
    ("x: nat; x = binomial(2, 1/2);", "x", {0: 1 / 4, 1: 1 / 2, 2: 1 / 4}),
    ("x: nat; x = geometric(1/2);", "x", {0: 1 / 2, 1: 1 / 4, 2: 1 / 8}),
    ("x: nat; x = negative_binomial(2, 1/2);", "x", {0: 1 / 4, 1: 1 / 4, 2: 3 / 16}),
    ("x: nat; x = poisson(1);", "x", {0: math.exp(-1), 1: math.exp(-1), 2: math.exp(-1) / 2}),
    ("x: nat; x = logdist(1/2);", "x", {1: 0.5 / math.log(2), 2: 0.125 / math.log(2)}),
    ("x: nat; x = categorical({1: 1/6, 4: 5/6});", "x", {1: 1 / 6, 4: 5 / 6}),
    ("x: nat; x = iid(bernoulli(1/2), 2);", "x", {0: 1 / 4, 1: 1 / 2, 2: 1 / 4}),
    ("x: nat; {x = 1;} [0.25] {x = 2;}", "x", {1: 1 / 4, 2: 3 / 4}),
])
def test_distributions(source, var, expected):
    simulation = run(source)
    for value, prob in expected.items():
        assert simulation.probability(lambda s, value=value: s[var] == value).contains(prob)


def test_observe_conditions_and_counts_violations():
    simulation = run("x: nat; x = unif(1, 3); observe(x < 3);")
    assert simulation.violation().contains(1 / 3)
    assert simulation.probability(lambda s: s["x"] == 1).contains(1 / 2)
    assert simulation.mass().contains(1)


def test_abort_is_lost_mass():
    simulation = run("x: nat; {abort;} [1/2] {x = 1;}")
    assert simulation.aborted > 0
    assert simulation.mass().contains(1 / 2)
    assert simulation.probability(lambda s: s["x"] == 1).contains(1 / 2)


def test_loops_and_unfinished_runs():
    geometric = run("x: nat = 1; c: nat; while (x == 1) { {x = 0;} [1/2] {c = c + 1;} }")
    assert geometric.unfinished == 0
    assert geometric.expectation(lambda s: s["c"]).contains(1)
    # a loop that never terminates: every run is unfinished, the intervals stay wide open
    endless = run("x: nat; while (x == 0) { skip; }", runs=10, loop_budget=50)
    assert endless.unfinished == 10
    assert endless.mass().lower == 0 and endless.mass().upper == 1
    assert run("x: nat; loop(3) { x = x + 2; }").samples[0] == {"x": 6}


def test_arithmetic_semantics():
    sim = run("x: nat; y: nat; z: nat; x = 7 // 2; y = 7 % 3; z = 2 - 5;", runs=1)
    assert sim.samples == [{"x": 3, "y": 1, "z": 0}]  # z: monus (#220)
    assert run("x: nat; x = 4; x = x / 2;", runs=1).samples == [{"x": 2}]
    with pytest.raises(SimulationError, match="not a natural number"):
        run("x: nat; x = 3; x = x / 2;", runs=1)
    # comparisons are over the rationals
    assert run("x: nat; y: nat; x = 3; if (x / 2 <= 1) { y = 1; }", runs=1).samples == [{"x": 3, "y": 0}]


def test_block_variables_are_hidden_and_state_probabilities_are_evaluated():
    # the probability 1/k depends on the state (#213); the simulator evaluates it per run
    simulation = run("k: nat = unif(1, 3); ok: nat = 1; if (k > 0) { t: nat = 1; } {ok = 1;} [1/k] {ok = 0;}",
                     runs=6000)
    assert set(simulation.samples[0]) == {"k", "ok"}
    assert simulation.probability(lambda s: s["ok"] == 1).contains(Fraction(11, 18))


def test_functions_and_nested_inference():
    source = """
def f(a: nat) -> nat { b: nat = unif(0, a); observe(b > 0); return b; }
d: dist = infer(f, 2);
x: nat = sample(d);
y: nat = f(1);
"""
    simulation = run(source)
    assert set(simulation.samples[0]) == {"x", "y"}
    assert simulation.violation().contains(1 / 2)  # the plain call conditions the program
    assert simulation.probability(lambda s: s["x"] == 1).contains(1 / 2)


@pytest.mark.parametrize("source", [
    "param p: real; x: nat; x = bernoulli(p);",
    "x: nat; x = sample_pgf(1/(2-x), x);",
])
def test_unsupported(source):
    with pytest.raises(UnsupportedConstruct):
        run(source, runs=1)


def test_seeded_runs_are_reproducible():
    program = parse_pgcl("x: nat; x = geometric(1/3);", "ppy")
    assert simulate(program, 200, seed=5).samples == simulate(program, 200, seed=5).samples


def test_intervals():
    low, high = proportion_interval(50, 100, 2)
    assert low < 0.5 < high
    assert proportion_interval(0, 100, 5)[0] == 0 and proportion_interval(100, 100, 5)[1] == 1
    # one hit in 3570 runs of an event with probability 5.5e-6: not refuted (Wilson would refute it)
    assert proportion_interval(1, 3570, 5)[0] < 5.5e-6
    estimate = Estimate.proportion(10, 100, 5, 5)
    assert estimate.lower < 0.1 and estimate.upper > 0.15
    assert estimate.contains(0.0, 1.0) and not estimate.contains(0.9)
    assert not estimate.contains(math.nan)


def test_compare_finds_a_wrong_result():
    program = parse_pgcl("x: nat; x = bernoulli(1/2); query Pr[x == 1]; query Ex[x];", "ppy")
    simulation = simulate(program, 4000, seed=2)
    good = {"bounds": {"kind": "exact"},
            "queries": [{"index": 0, "kind": "probability", "expr": "x == 1", "value_float": 0.5},
                        {"index": 1, "kind": "expectation", "expr": "x", "lower_float": 0.5, "upper_float": 0.5}],
            "distribution": {"mass_float": 1.0,
                             "marginals": {"x": {"rows": [{"value": 0, "p": 0.5}, {"value": 1, "p": 0.5}]}}}}
    assert not compare(program, good, simulation)
    bad = {**good, "queries": [{"index": 0, "kind": "probability", "expr": "x == 1", "value_float": 0.6}]}
    mismatches = compare(program, bad, simulation)
    assert [m.what for m in mismatches] == ["Pr[x == 1]"]
    assert isinstance(mismatches[0], Mismatch) and "analysis 0.6" in str(mismatches[0])


def test_query_block_conditions_locally():
    # Monty Hall: the host's door is drawn again until it shows neither the prize nor the player's
    # door; the runs are not rejected, so switching wins with 2/3 (global rejection would give 1/2)
    source = """prize: nat = unif(1, 3);
player: nat = unif(1, 3);
door: nat;
other: nat;
query { door = unif(1, 3); observe(not (prize == door or player == door)); }
query { other = unif(1, 3); observe(not (other == player or other == door)); }
"""
    simulation = run(source)
    assert simulation.violated == 0
    assert simulation.probability(lambda s: s["other"] == s["prize"]).contains(2 / 3)
