"""
The EVT certificate rests on the fixed point I = Φ(I) (#117). Parameter values from a solver are only
proposals: an invariant is certified only if it is a proven fixed point with these values substituted.
"""
from types import SimpleNamespace

import pytest
import sympy
from prodigy.pgcl import parse_pgcl

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants import invariant_synthesis, invariant_verification
from prodigy.analysis.instructionhandler import while_handler
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Ex[c]
"""

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]


def analyse(decision, engine=ForwardAnalysisConfig.Engine.SYMPY):
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(GEOMETRIC), source=GEOMETRIC, config=config,
                              decisions={"0": decision}, capture_output=True)
    return session.run().to_json()


def evt_invariant(invariant):
    return {"strategy": "evt_invariant", "params": {"invariant": invariant}}


@pytest.mark.parametrize("engine", ENGINES)
def test_exact_invariant_is_certified(engine):
    result = analyse(evt_invariant("(x + 1/2)/(1 - c/2)"), engine)
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT


@pytest.mark.parametrize("engine", ENGINES)
def test_nearly_exact_invariant_is_refuted(engine):
    result = analyse(evt_invariant("(x + 1/2)/(1 - 50000000000000000001/100000000000000000000*c)"), engine)
    assert result["status"] == "error"
    assert result["error"]["verdict"] == "refuted"


# Refuted vs. not proven (#123)

@pytest.mark.parametrize("engine", ENGINES)
def test_wrong_rational_invariant_is_refuted(engine):
    result = analyse(evt_invariant("(x + 1/3)/(1 - c/2)"), engine)
    assert result["status"] == "error"
    assert result["error"]["verdict"] == "refuted"
    assert result["error"]["details"]["verdict"] == "refuted"
    assert result["error"]["details"]["difference"]
    assert result["loops"][0]["certificates"][0]["verdict"] == "refuted"


@pytest.mark.parametrize("engine", ENGINES)
def test_non_rational_difference_is_not_refuted(engine):
    # wrong (Φ(I) - I = 1 + c - exp(c)), but exact cancellation cannot show that exp(c) - 1 - c is nonzero
    result = analyse(evt_invariant("(x + 1/2)/(1 - c/2) + exp(c) - 1 - c"), engine)
    assert result["status"] == "error"
    assert result["error"]["verdict"] == "unknown"
    assert "not proven" in result["error"]["message"]


@pytest.mark.parametrize("engine, status, verdict", [
    (ForwardAnalysisConfig.Engine.SYMPY, "ok", None),
    (ForwardAnalysisConfig.Engine.SYMENGINE, "error", "unknown"),
    (ForwardAnalysisConfig.Engine.GINAC, "error", "unknown"),
])
def test_correct_but_unproven_invariant_is_not_refuted(engine, status, verdict):
    # sin(c)^2 + cos(c)^2 - 1 = 0: SymPy's simplify proves it, SymEngine and GiNaC cannot
    result = analyse(evt_invariant("(x + 1/2)/(1 - c/2) + sin(c)^2 + cos(c)^2 - 1"), engine)
    assert result["status"] == status
    assert (result["error"] or {}).get("verdict") == verdict


@pytest.mark.parametrize("engine", ENGINES)
def test_template_without_solution_is_not_refuted(engine):
    # (x + a)/(1 - c/3) is wrong for every a, but the solver's search is incomplete
    result = analyse(evt_invariant("(x + a)/(1 - c/3)"), engine)
    assert result["status"] == "error"
    assert result["error"]["verdict"] == "unknown"


def test_solved_parameter_is_certified():
    result = analyse(evt_invariant("(x + a)/(1 - c/2)"))
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT
    assert result["loops"][0]["certificates"][0]["solutions"] == [{"a": "1/2"}]


def test_wrong_solution_of_the_solver_is_not_certified(monkeypatch):
    a = sympy.Symbol("a")
    monkeypatch.setattr(invariant_verification.sympy, "solve",
                        lambda *args, **kwargs: [{a: sympy.Rational(1, 2) + sympy.Rational(1, 10 ** 20)}])
    result = analyse(evt_invariant("(x + a)/(1 - c/2)"))
    assert result["status"] == "error"


def wrong_solutions(_equation, numerator_unknowns, denominator_unknowns, *_args, **_kwargs):
    """Proposes a = 1/2 + 10^-20 for every template; the right value would be 1/2."""
    return [{p: sympy.Rational(1, 2) + sympy.Rational(1, 10 ** 20)
             for p in [*numerator_unknowns, *denominator_unknowns]}]


def one_template(_strategy, variables, factory, **_heuristics):
    """A synthesis strategy with the single template (x + a)/(1 - c/2) (the enumeration is endless)."""
    template = factory.from_expr("(x + a)/(1 - c/2)", *variables)
    return SimpleNamespace(name="one template",
                           template_heuristics=SimpleNamespace(generate=lambda: iter([template])),
                           positivity_heuristics=SimpleNamespace(is_positive=lambda _: True))


@pytest.mark.parametrize("engine", ENGINES)
def test_synthesised_parameter_is_certified(monkeypatch, engine):
    monkeypatch.setattr(while_handler.SynthesisStrategies, "make", one_template)
    result = analyse("evt_synthesis", engine)
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT


@pytest.mark.parametrize("engine", ENGINES)
def test_wrong_solution_of_the_synthesis_solver_is_not_certified(monkeypatch, engine):
    monkeypatch.setattr(while_handler.SynthesisStrategies, "make", one_template)
    monkeypatch.setattr(invariant_synthesis, "solve_template", wrong_solutions)
    result = analyse("evt_synthesis", engine)
    assert result["status"] == "error"


@pytest.mark.parametrize("invariant, verdict", [
    ("(x + 1/3)/(1 - c/2)", "Verdict: refuted"),
    ("(x + 1/2)/(1 - c/2) + exp(c) - 1 - c", "Verdict: not proven"),
])
def test_cli_shows_the_verdict(tmp_path, invariant, verdict):
    from click.testing import CliRunner  # pylint: disable=import-outside-toplevel
    from prodigy.cli import cli  # pylint: disable=import-outside-toplevel
    program = tmp_path / "geometric.pgcl"
    program.write_text(GEOMETRIC)
    result = CliRunner().invoke(cli, ["analyze", str(program), "--loop", f"0=evt_invariant:invariant={invariant}"])
    assert verdict in result.output
