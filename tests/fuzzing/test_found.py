"""
Regression tests for the programs the fuzzing found (shrunk by Hypothesis, then by hand).

A case of an open issue is ``xfail(strict=True)``: once the issue is fixed, the test passes, fails
as XPASS, and the marker (and the switch in :class:`~tests.fuzzing.programs.Features`) can go.
"""
import os
import signal
import sys
import time

import pytest

from prodigy.distribution.equality import decide_equal_strings

from .oracles import (SETUPS, ProgramCrash, ProgramError, ProgramTimeout, analyse, check_program, isolated,
                      time_budget)

ENGINES = {setup.name: setup for setup in SETUPS}


def _ginac_xfail(issue):
    return pytest.mark.xfail(strict=True, reason=f"open issue #{issue}")


def _per_engine(issue_engines, issue):
    return [pytest.param(name, marks=_ginac_xfail(issue)) if name in issue_engines else name for name in ENGINES]


# #232: GiNaC moved the old distribution of y onto x
VARIABLE_COPY = "x: nat;\ny: nat;\ny = unif(0, 2);\ny = x;\nprint;\n"


@pytest.mark.parametrize("engine", ENGINES)
def test_copy_of_a_variable_that_does_not_occur(engine):
    result = analyse(VARIABLE_COPY, "ppy", ENGINES[engine], {})
    assert result["status"] == "ok"
    assert decide_equal_strings(result["distribution"]["closed_form"], "1", ["x", "y"]) is True


# #233: categorical(…) was rejected by the analysis
CATEGORICAL = "x: nat;\nx = categorical({1: 1/2, 2: 1/2});\nprint;\n"


@pytest.mark.parametrize("engine", ENGINES)
def test_categorical_is_computed(engine):
    assert not check_program(CATEGORICAL)
    result = analyse(CATEGORICAL, "ppy", ENGINES[engine], {})
    assert result["status"] == "ok"
    assert decide_equal_strings(result["distribution"]["closed_form"], "x/2 + x^2/2", ["x"]) is True


# #248: the guard x % 3 gives SymPy and GiNaC a distribution with the parameters I and pi (roots of unity), the
# error probability has none. SymPy's equality compares them, so the check for probability 0 failed and the
# division ended in nan ("Undefined value") instead of the error of the other engines.
OBSERVE_ZERO_AFTER_MODULO = """x: nat;
y: nat;
x = poisson(2);
if ((x % 3) == 0 and (x % 3) == 1) {
    skip;
} else {
    y = bernoulli(0.1);
}
observe((y % 3) == 2);
query Ex[x];
"""
# #248: here the error probability is an unsimplified sum of roots of unity that is 1; SymPy and GiNaC could not
# show it and reported E[x] = 0 as an exact result for a program without semantics
OBSERVE_ZERO_UNPROVEN = """x: nat;
y: nat;
x = poisson(2);
y = x % 3;
observe(y == 5);
query Ex[x];
"""


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", [OBSERVE_ZERO_AFTER_MODULO, OBSERVE_ZERO_UNPROVEN])
def test_observation_with_probability_zero_is_an_error_on_all_engines(engine, source):
    result = analyse(source, "ppy", ENGINES[engine], {})
    assert result["status"] == "error"
    assert result["error"]["type"] == "ObserveZeroEventError", result["error"]
    assert result["error"]["message"].startswith("Undefined semantics: Probability of observing a valid run is 0")
    if source == OBSERVE_ZERO_AFTER_MODULO:
        assert result["error"]["message"] == "Undefined semantics: Probability of observing a valid run is 0."


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_result_with_positive_probability_of_a_valid_run(engine):
    # the loop never terminates and nothing is observed: the zero measure is the (exact) result, not an error
    result = analyse("x: nat;\nwhile (x == 0) {\n    skip;\n}\nquery Pr[x == 0];\n", "ppy", ENGINES[engine],
                     {"0": "unroll:iterations=3"})
    assert result["status"] == "ok", result.get("error")


def test_time_budget_interrupts_a_slow_analysis_and_restores_the_outer_timer():
    outer_handler = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 50)
    try:
        started = time.monotonic()
        with pytest.raises(ProgramTimeout):
            with time_budget(0.2):
                while True:  # e.g. an analysis that does not finish
                    pass
        assert time.monotonic() - started < 5
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert 40 < remaining <= 50
        assert signal.getsignal(signal.SIGALRM) is outer_handler
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _spin():
    while True:  # e.g. an analysis that does not finish
        pass


def _die():
    os.kill(os.getpid(), signal.SIGKILL)


def _allocate(size):
    return len(bytearray(size))


def _fail():
    raise ValueError("broken")


def test_isolated_returns_the_value_and_reports_errors():
    assert isolated(check_program, CATEGORICAL, seconds=60) == []
    with pytest.raises(ProgramError, match="ValueError: broken"):
        isolated(_fail, seconds=60)


def test_isolated_stops_a_program_over_its_budget_or_dying():
    started = time.monotonic()
    with pytest.raises(ProgramTimeout):
        isolated(_spin, seconds=0.5)
    assert time.monotonic() - started < 30
    with pytest.raises(ProgramCrash):
        isolated(_die, seconds=60)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_DATA is enforced on Linux only")
def test_isolated_stops_a_program_that_eats_all_memory():
    with pytest.raises(ProgramCrash, match="out of memory"):
        isolated(_allocate, 2**31, seconds=60, memory=2**30)
