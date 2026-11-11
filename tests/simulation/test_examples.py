"""
Differential testing: the examples in ``pgfexamples/`` against the simulator (#182).

Every example the simulator can run is analysed (loops are unrolled, which gives a sound interval
with the residual mass, or an exact result if the loop terminates during unrolling) and simulated
with a fixed seed; every reported probability, expected value, marginal row and the mass must meet
the simulation's confidence interval (``z = 5``, see :func:`prodigy.simulator.compare`). This is
statistical, not a proof: it finds errors larger than the width of the intervals.

Left out: examples with parameters or constructs the simulator does not implement (``sample_pgf``,
``Pr``/``Ex`` of ``dist`` values), the examples in :data:`SKIPPED` (with the reason), and the slow
ones in :data:`SLOW` and :data:`SLOW_DIRECTORIES` unless ``PRODIGY_FUZZ=nightly`` (the weekly
workflow ``.github/workflows/fuzzing.yml``). Each comparison runs in a child process with a time
budget (:data:`BUDGET`) and a memory limit (:func:`tests.fuzzing.oracles.isolated`), so a slow or
memory-hungry example fails on its own instead of hanging or killing the whole run.
"""
import os
from pathlib import Path

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl.parser import parse_pgcl
from prodigy.simulator import UnsupportedConstruct, compare, simulate
from tests.fuzzing.oracles import ProgramCrash, ProgramError, ProgramTimeout, isolated

EXAMPLES = Path(__file__).parents[2] / "pgfexamples"
RUNS = 3000
UNROLL = 30
LOOP_BUDGET = 1000
pytestmark = pytest.mark.fuzz
NIGHTLY = os.environ.get("PRODIGY_FUZZ") == "nightly"
BUDGET = 900.0 if NIGHTLY else 100.0
"""Seconds for the analysis and simulation of one example (the slowest in :data:`SLOW` need about 80 s)."""

SKIPPED = {
    "independence/infinite_multiplication.ppy": "not implemented: product of two variables with infinite range",
    "independence/many_variables.ppy": "slower than 90 s with unrolling",
    "inference/loop_free/conditioning/lucky_throw.ppy": "slower than 90 s with unrolling",
    "inference/loop_free/conditioning/skip_lin_regression_biased.ppy": "slower than 90 s with unrolling",
    "inference/loop_free/scalability.ppy": "slower than 90 s with unrolling",
    "lower_bounds/exact/hark_counterexample.ppy": "slower than 90 s with unrolling",
    "lower_bounds/sandwich/escape_noisy.ppy": "slower than 90 s with unrolling",
    "lower_bounds/sandwich/quadrant_drift.ppy": "slower than 90 s with unrolling",
    "lower_bounds/sandwich/quadrant_timeout.ppy": "slower than 90 s with unrolling",
    "lower_bounds/sandwich/symmetric_noisy.ppy": "slower than 90 s with unrolling",
}
"""Examples that are not compared, with the reason."""

SLOW = {
    # seconds for analysis and simulation (one run, Apple M-series)
    "independence/fuzzy_or.ppy": 6,
    "independence/oblivious_transfer.ppy": 6,
    "inference/loop_free/conditioning/digitRecognition.ppy": 82,
    "inference/loop_free/conditioning/monty_hall_nested.ppy": 4,
    "inference/loop_free/fuzzy_or.ppy": 8,
    "inference/loop_free/pi.ppy": 9,
    "inference/loop_free/poor_scale.ppy": 21,
    "inference/loop_free/prob_collatz.ppy": 3,
    "inference/loopy/conditioning/poisson_from_coins.ppy": 37,
    "inference/loopy/skip_exceeding_sum.ppy": 47,
    "inference/loopy/skip_random_walk.ppy": 10,
    "inference/loopy/uniform_sampling.ppy": 5,
    "invariant_synthesis/faulty_decrement.ppy": 7,
    "invariant_synthesis/modulo_geometric.ppy": 4,
    "invariant_synthesis/nontermination.ppy": 13,
    "invariant_synthesis/random_walk.ppy": 6,
    "invariant_synthesis/random_walk_counter.ppy": 9,
    "invariant_synthesis/skip_sequential_loops.ppy": 80,
    "lower_bounds/conditioning/rw_counter_observe.ppy": 5,
    "lower_bounds/exact/escape_counter.ppy": 52,
    "lower_bounds/exact/faulty_decrement.ppy": 6,
    "lower_bounds/exact/modulo_geometric.ppy": 3,
    "lower_bounds/exact/nontermination.ppy": 11,
    "lower_bounds/exact/random_walk.ppy": 6,
    "lower_bounds/exact/random_walk_counter.ppy": 10,
    "lower_bounds/sandwich/escape_counter.ppy": 45,
    "lower_bounds/sandwich/escape_walk.ppy": 31,
    "lower_bounds/sandwich/jump_counter.ppy": 8,
    "lower_bounds/sandwich/rw_counter_1120.ppy": 8,
    "lower_bounds/sandwich/rw_counter_23.ppy": 7,
    "lower_bounds/sandwich/rw_counter_35.ppy": 7,
}
"""Examples that take longer than a few seconds; compared only in the nightly profile."""

SLOW_DIRECTORIES = ("inference/loopy/hard_distributions/",)


def _examples():
    for path in sorted(EXAMPLES.rglob("*.ppy")):
        name = path.relative_to(EXAMPLES).as_posix()
        if name in SKIPPED:
            yield pytest.param(name, marks=pytest.mark.skip(reason=SKIPPED[name]))
        elif not NIGHTLY and (name in SLOW or name.startswith(SLOW_DIRECTORIES)):
            yield pytest.param(name, marks=pytest.mark.skip(reason="slow; only with PRODIGY_FUZZ=nightly"))
        else:
            yield name


def _compare(name: str) -> str:
    """Analyses and simulates the example ``name``; the mismatches, or ``skip: <reason>``."""
    source = (EXAMPLES / name).read_text(encoding="utf-8")
    program = parse_pgcl(source, "ppy")
    try:
        simulation = simulate(program, RUNS, seed=name, loop_budget=LOOP_BUDGET)
    except UnsupportedConstruct as reason:
        return f"skip: the simulator does not implement {reason}"
    session = AnalysisSession(program, source=source, syntax="ppy", config=ForwardAnalysisConfig(),
                              decisions={"*": f"unroll:iterations={UNROLL}"}, capture_output=True,
                              coefficient_limit=12)
    result = session.run().to_json()
    if result["status"] == "error" and "observing a valid run is 0" in result["error"]["message"]:
        # no run terminates and passes the observations; the simulation must not see one either
        return "" if not simulation.samples else "the analysis finds no valid run, the simulation does"
    if result["status"] != "ok":
        return f"status {result['status']}: {result['error']}"
    return "\n".join(map(str, compare(program, result, simulation)))


@pytest.mark.parametrize("name", list(_examples()))
def test_example_agrees_with_simulation(name):
    try:
        mismatches = isolated(_compare, name, seconds=BUDGET, seed=name)
    except (ProgramTimeout, ProgramCrash) as reason:
        pytest.fail(f"{name}: {reason}")
    except ProgramError as error:
        pytest.fail(f"{name}:\n{error}")
    if mismatches.startswith("skip: "):
        pytest.skip(mismatches.removeprefix("skip: "))
    assert not mismatches, mismatches


def test_most_examples_are_compared():
    compared = [p for p in _examples() if isinstance(p, str)]
    assert len(compared) >= 60
