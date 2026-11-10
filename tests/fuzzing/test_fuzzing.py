"""
Property-based fuzzing (#183): random programs (:mod:`tests.fuzzing.programs`) must give the same
result on all engines, with simplification on and off, and agree with the simulator.

The number of programs comes from a Hypothesis profile, chosen by the environment variable
``PRODIGY_FUZZ``:

* ``ci`` (default): a few programs, derandomized, so every run of the test suite checks the same
  programs and the CI is reproducible;
* ``nightly``: many programs with a fresh seed (``.github/workflows/fuzzing.yml``).

Both print the falsifying program and a blob to reproduce it (``@reproduce_failure``). Every
program has a time budget (:data:`BUDGET`). In the ``ci`` profile a program over budget is
rejected and counted (``--hypothesis-show-statistics`` shows the count), so a slow program never
hangs the test suite; in the ``nightly`` profile it is a failure, reported with the program.

A failure is shrunk to a small program. It becomes an issue and a regression test in
``test_found.py``; if it is a known open bug class, a switch in :class:`~tests.fuzzing.programs.Features`
leaves it out until it is fixed.
"""
import os

import pytest
from hypothesis import HealthCheck, event, given, reject, settings

from prodigy.pgcl.parser import parse_pgcl

from .oracles import ProgramTimeout, check_program, time_budget
from .programs import Features, GeneratedProgram, programs

_HEALTH = [HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large]
# deadline=None: Hypothesis cannot interrupt a slow example; the time budget below does
settings.register_profile("fuzz-ci", max_examples=20, derandomize=True, deadline=None, database=None,
                          print_blob=True, suppress_health_check=_HEALTH)
settings.register_profile("fuzz-nightly", max_examples=400, derandomize=False, deadline=None, database=None,
                          print_blob=True, suppress_health_check=_HEALTH)
# the strategies are built recursively; Hypothesis warns about the size of their repr only
pytestmark = [pytest.mark.fuzz, pytest.mark.filterwarnings("ignore:Generating overly large repr")]
NIGHTLY = os.environ.get("PRODIGY_FUZZ", "ci") == "nightly"
PROFILE = settings.get_profile("fuzz-nightly" if NIGHTLY else "fuzz-ci")
BUDGET = 60.0 if NIGHTLY else 5.0
"""Seconds per program for all analyses and the simulation (a typical program needs well below 1 s)."""


@settings(PROFILE)
@given(programs())
def test_engines_and_simulation_agree(program: GeneratedProgram):
    source = program.source("ppy")
    try:
        with time_budget(BUDGET):
            problems = check_program(source, "ppy", program.decisions())
    except ProgramTimeout:
        if NIGHTLY:
            raise AssertionError(f"slower than {BUDGET} s:\n{source}") from None
        event("rejected: over the time budget")
        reject()
    assert not problems, source + "\n" + "\n".join(problems)


@settings(PROFILE, max_examples=PROFILE.max_examples * 5)
@given(programs(Features(floor_division=False)))
def test_both_syntaxes_give_the_same_program(program: GeneratedProgram):
    ppy, pgcl = program.source("ppy"), program.source("pgcl")
    assert parse_pgcl(ppy, "ppy") == parse_pgcl(pgcl, "pgcl"), ppy + "\n" + pgcl
