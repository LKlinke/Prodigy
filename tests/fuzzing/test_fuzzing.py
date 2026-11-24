"""
Property-based fuzzing (#183): random programs (:mod:`tests.fuzzing.programs`) must give the same
result on all engines, with simplification on and off, and agree with the simulator.

The number of programs comes from a Hypothesis profile, chosen by the environment variable
``PRODIGY_FUZZ``:

* ``ci`` (default): a few programs, derandomized, so every run of the test suite checks the same
  programs and the CI is reproducible;
* ``nightly``: many programs with a fresh seed (``.github/workflows/fuzzing.yml``).

Both print the falsifying program and a blob to reproduce it (``@reproduce_failure``). Every
program runs in a child process (:func:`~tests.fuzzing.oracles.isolated`) with a time budget
(:data:`BUDGET`) and a memory limit, so neither a computation that never returns to the interpreter
nor one that eats all memory can hang or kill the test run. In the ``ci`` profile a program over
budget is rejected and counted (``--hypothesis-show-statistics`` shows the count); in the
``nightly`` profile it is a failure, reported with the program.

``test_parametric_programs_agree`` does the same for programs with parameters (#453): SymPy and GiNaC on the
parametric program, the parametric result instantiated with random admissible values against the analysis
and a simulation of the instantiated program (:func:`~tests.fuzzing.oracles.check_parametric`).

A failure is shrunk to a small program. It becomes an issue and a regression test in
``test_found.py``; if it is a known open bug class, a switch in :class:`~tests.fuzzing.programs.Features`
leaves it out until it is fixed.
"""
import os

import pytest
from hypothesis import HealthCheck, event, given, reject, settings


from .oracles import ProgramCrash, ProgramError, ProgramTimeout, check_parametric, check_program, isolated
from .programs import Features, GeneratedProgram, parameter_values, programs

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
PARAMETRIC = Features(parameters=True)
"""The constructs of the parametric programs: those of the constant programs and parametric probabilities."""
PARAMETRIC_PROFILE = settings(PROFILE, max_examples=150 if NIGHTLY else 8)
"""Fewer parametric programs: each is two programs (parametric and instantiated) on every engine."""


def _run(function, *args, source: str, seconds: float):
    """``function(*args)`` in a child process (:func:`isolated`); a program over budget is rejected in the
    ``ci`` profile and fails in the ``nightly`` profile."""
    try:
        return isolated(function, *args, seconds=seconds, seed=source)
    except (ProgramTimeout, ProgramCrash) as reason:
        if NIGHTLY:
            raise AssertionError(f"{reason}:\n{source}") from None
        event(f"rejected: {type(reason).__name__}")
        reject()
    except ProgramError as error:
        raise AssertionError(f"{source}\n{error}") from None
    return None  # unreachable: reject() raises


@settings(PROFILE)
@given(programs())
def test_engines_and_simulation_agree(program: GeneratedProgram):
    source = program.source("ppy")
    problems = _run(check_program, source, "ppy", dict(program.decisions()), source=source, seconds=BUDGET)
    assert not problems, source + "\n" + "\n".join(problems)


@PARAMETRIC_PROFILE
@given(programs(PARAMETRIC).filter(lambda program: program.parameters), parameter_values())
def test_parametric_programs_agree(program: GeneratedProgram, values):
    source = program.source("ppy")
    used = {name: values[name] for name in program.parameters}
    instantiated = program.instantiate(used).source("ppy")
    # two programs: the parametric one and the instantiated one, each on every engine
    problems = _run(check_parametric, source, instantiated, used, "ppy", dict(program.decisions()),
                    source=source, seconds=2 * BUDGET)
    assert not problems, f"{source}\nwith {used}:\n{instantiated}\n" + "\n".join(problems)
