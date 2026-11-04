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
def test_nearly_exact_invariant_is_not_certified(engine):
    result = analyse(evt_invariant("(x + 1/2)/(1 - 50000000000000000001/100000000000000000000*c)"), engine)
    assert result["status"] == "error"


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


class WrongSolver:
    """Proposes a = 1/2 + 10^-20 for every template; the right value would be 1/2."""

    def solve(self, f, g):  # pylint: disable=unused-argument
        params = [sympy.Symbol(p) for p in sorted(f.get_parameters() | g.get_parameters())]
        return True, [{p: sympy.Rational(1, 2) + sympy.Rational(1, 10 ** 20) for p in params}]


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
    monkeypatch.setattr(invariant_synthesis.SolverType, "make", lambda *args, **kwargs: WrongSolver())
    result = analyse("evt_synthesis", engine)
    assert result["status"] == "error"
