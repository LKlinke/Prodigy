"""
Regression tests for the programs the fuzzing found (shrunk by Hypothesis, then by hand).

A case of an open issue is ``xfail(strict=True)``: once the issue is fixed, the test passes, fails
as XPASS, and the marker (and the switch in :class:`~tests.fuzzing.programs.Features`) can go.
"""
import pytest

from prodigy.distribution.equality import decide_equal_strings

from .oracles import SETUPS, analyse, check_program

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


@pytest.mark.xfail(strict=True, reason="open issue #233")
def test_categorical_is_computed():
    source = "x: nat;\nx = categorical({1: 1/2, 2: 1/2});\nprint;\n"
    assert not check_program(source)
    assert analyse(source, "ppy", ENGINES["sympy"], {})["status"] == "ok"
