"""Variables named like constants or builtins (`pi`, `I`, `pow`) are not misread by the solvers and engine
fallbacks (#344): they were read back from strings by `sympy.S`."""
import pytest
import sympy

from prodigy.analysis.solver.sympy_solver import SympySolver
from prodigy.distribution.generating_function import GeneratingFunction

NAMES = ["pi", "I", "pow", "E"]


@pytest.mark.parametrize("name", NAMES)
def test_sympy_solver_keeps_variable_names(name):
    f = GeneratingFunction(f"p * {name}", name)
    g = GeneratingFunction(f"(1/2) * {name}", name)
    ok, solutions = SympySolver().solve(f, g)
    assert ok is True
    assert solutions == [{sympy.Symbol("p"): sympy.Rational(1, 2)}]


# SymEngine rejects pi, I and E as names (#204)
@pytest.mark.parametrize("name", ["pow", "max", "S"])
def test_engine_fallbacks_keep_variable_names(name):
    from prodigy.distribution.symengine_distribution import SymengineDist  # pylint: disable=import-outside-toplevel
    dist = SymengineDist(f"(1/2) * {name} / (1 - (1/2) * {name})", name)
    terms = [(str(prob), state[name]) for (prob, state), _ in zip(dist, range(3))]
    assert terms == [("0", 0), ("1/2", 1), ("1/4", 2)]
    assert dist.coefficient_sum() == 1
