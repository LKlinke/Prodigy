"""
Property-based fuzzing (#183): random programs (:mod:`tests.fuzzing.programs`) must give the same
result on all engines, with simplification on and off, and agree with the simulator.

The number of programs comes from a Hypothesis profile, chosen by the environment variable
``PRODIGY_FUZZ``:

* ``ci`` (default): a few programs, derandomized, so every run of the test suite checks the same
  programs and the CI is reproducible;
* ``nightly``: many programs with a fresh seed (``.github/workflows/fuzzing.yml``); a failure prints
  the seed and the shrunk program, run it again with ``@reproduce_failure`` or ``--hypothesis-seed``.

A failure is shrunk to a small program. It becomes an issue and a regression test in
``test_found.py``; if it is a known open bug class, a switch in :class:`~tests.fuzzing.programs.Features`
leaves it out until it is fixed.
"""
import os
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, given, settings

from prodigy.pgcl.parser import parse_pgcl

from .oracles import check_program
from .programs import Features, GeneratedProgram, programs

settings.register_profile(
    "fuzz-ci", max_examples=40, derandomize=True, deadline=timedelta(seconds=60), database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large])
settings.register_profile(
    "fuzz-nightly", max_examples=400, derandomize=False, deadline=None, database=None, print_blob=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large])
# the strategies are built recursively; Hypothesis warns about the size of their repr only
pytestmark = [pytest.mark.fuzz, pytest.mark.filterwarnings("ignore:Generating overly large repr")]
PROFILE = settings.get_profile(f"fuzz-{os.environ.get('PRODIGY_FUZZ', 'ci')}")


@settings(PROFILE)
@given(programs())
def test_engines_and_simulation_agree(program: GeneratedProgram):
    source = program.source("ppy")
    problems = check_program(source, "ppy", program.decisions())
    assert not problems, source + "\n" + "\n".join(problems)


@settings(PROFILE, max_examples=PROFILE.max_examples * 5)
@given(programs(Features(floor_division=False)))
def test_both_syntaxes_give_the_same_program(program: GeneratedProgram):
    ppy, pgcl = program.source("ppy"), program.source("pgcl")
    assert parse_pgcl(ppy, "ppy") == parse_pgcl(pgcl, "pgcl"), ppy + "\n" + pgcl
