"""
The EVT invariant synthesis is bounded by a maximal template degree and a time limit (#226): for
loops without a rational invariant it ends with a clear message and the verdict ``unknown``.
"""
import signal
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from sympy.polys.polyerrors import HeuristicGCDFailed

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants import invariant_synthesis
from prodigy.analysis.evtinvariants.heuristics.strategies import DEFAULT_MAX_DEGREE
from prodigy.analysis.evtinvariants.invariant_synthesis import DEFAULT_SYNTHESIS_TIME_LIMIT
from prodigy.analysis.loop_strategies import get_loop_strategy
from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.solver.sympy_solver import SympySolver
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

EXAMPLES = Path(__file__).parents[2] / "pgfexamples"
HARD = EXAMPLES / "inference" / "loopy" / "hard_distributions"

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]


def _run(path: Path, decision: str, engine) -> dict:
    source = path.read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"0": decision}, config=ForwardAnalysisConfig(engine=engine))
    return session.run().to_json()


def _assert_no_invariant(result: dict, reason: str) -> dict:
    assert result["status"] == "error"
    assert result["bounds"]["kind"] == "unknown"
    error = result["error"]
    assert error["type"] == "SynthesisBoundReached"
    assert error["verdict"] == "unknown"
    assert error["details"]["reason"] == reason
    return error


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["non_d_finite/st_petersburg.ppy", "algebraic/galton_watson_critical.ppy"])
def test_degree_bound_ends_without_invariant(engine, name):
    error = _assert_no_invariant(_run(HARD / name, "evt_synthesis:max_degree=0", engine), "degree")
    assert error["message"] == "No rational EVT invariant up to degree 0 found (1 template tried)."
    assert error["details"]["max_degree"] == 0 and error["details"]["templates_tried"] == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_time_limit_interrupts_a_running_solver_call(engine):
    # with degree 1 SymPy's Gröbner basis for one of the templates runs for minutes
    start = time.monotonic()
    result = _run(HARD / "non_d_finite" / "st_petersburg.ppy", "evt_synthesis:max_degree=1,time_limit=1", engine)
    assert time.monotonic() - start < 30
    error = _assert_no_invariant(result, "time_limit")
    assert error["message"].startswith("Stopped after the time limit of 1 s: no rational EVT invariant up to "
                                       "degree 1 found (")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name, invariant", [
    ("faulty_decrement.ppy", "-1 - 4/(x - 2)"),
    ("geometric_counter.ppy", "(-2*x - 1)/(c - 2)"),
])
def test_rational_invariant_is_found_with_the_defaults(engine, name, invariant):
    result = _run(EXAMPLES / "invariant_synthesis" / name, "evt_synthesis", engine)
    assert result["status"] == "ok", result["error"]
    assert result["loops"][0]["kind"] == "exact"
    certificate = result["loops"][0]["certificates"][0]
    assert certificate["max_degree"] == DEFAULT_MAX_DEGREE
    if engine == ForwardAnalysisConfig.Engine.SYMPY:
        assert certificate["invariant"] == invariant


def test_parameters_are_registered():
    params = {spec.name: spec for spec in get_loop_strategy("evt_synthesis").params}
    assert params["max_degree"].default == DEFAULT_MAX_DEGREE and params["max_degree"].minimum == 0
    assert params["time_limit"].default == DEFAULT_SYNTHESIS_TIME_LIMIT and params["time_limit"].minimum == 1
    assert params["max_degree"].optional and params["time_limit"].optional
    with pytest.raises(ValueError):
        params["time_limit"].parse("0")


def test_sympy_failure_on_a_template_is_skipped(monkeypatch):
    """SymPy's HeuristicGCDFailed('no luck') ended the analysis before; now the template is skipped."""
    def fail(self, f, g):
        raise HeuristicGCDFailed("no luck")
    monkeypatch.setattr(SympySolver, "solve", fail)
    result = _run(HARD / "non_d_finite" / "st_petersburg.ppy", "evt_synthesis:max_degree=0",
                  ForwardAnalysisConfig.Engine.SYMPY)
    error = _assert_no_invariant(result, "degree")
    assert error["message"].endswith("SymPy could not solve 1 template(s).")
    assert error["details"]["templates_failed"] == 1


def test_cli_reports_the_bound_without_traceback():
    runner = CliRunner()
    result = runner.invoke(cli, ["main", str(HARD / "non_d_finite" / "st_petersburg.ppy"),
                                 "--loop", "0=evt_synthesis:max_degree=0"])
    assert result.exit_code == 1
    assert "Error: No rational EVT invariant up to degree 0 found" in result.output
    assert "Traceback" not in result.output
    result = runner.invoke(cli, ["invariant_synthesis", "--max-degree", "0",
                                 str(HARD / "non_d_finite" / "st_petersburg.ppy")])
    assert result.exit_code == 1
    assert "Error: No rational EVT invariant up to degree 0 found" in result.output


# -- the time limit itself -------------------------------------------------------------------------

def test_time_limit_interrupts_a_block():
    with pytest.raises(invariant_synthesis._TimeLimitReached):  # pylint: disable=protected-access
        with invariant_synthesis._time_limit(0.2):  # pylint: disable=protected-access
            while True:
                pass


def test_time_limit_restores_a_later_alarm():
    previous = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    try:
        signal.setitimer(signal.ITIMER_REAL, 500)
        with invariant_synthesis._time_limit(5) as expired:  # pylint: disable=protected-access
            assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 5
            assert not expired()
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 400
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN  # pylint: disable=comparison-with-callable
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_time_limit_keeps_an_earlier_alarm():
    previous = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    try:
        signal.setitimer(signal.ITIMER_REAL, 2)
        with invariant_synthesis._time_limit(50) as expired:  # pylint: disable=protected-access
            assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN  # pylint: disable=comparison-with-callable
            assert signal.getitimer(signal.ITIMER_REAL)[0] <= 2
            assert not expired()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_no_time_limit_never_expires():
    with invariant_synthesis._time_limit(None) as expired:  # pylint: disable=protected-access
        assert not expired()
