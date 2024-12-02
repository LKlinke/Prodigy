import pytest

from prodigy.analysis.solver.solver_type import SolverType
from prodigy.distribution import CommonDistributionsFactory
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.symengine_distribution import SymenginePGF


@pytest.mark.parametrize('factory', [SympyPGF, ProdigyPGF, SymenginePGF])
def test_always_true_solver(factory: CommonDistributionsFactory):
    f = factory.undefined("dummy_var", "another_var")
    g = factory.log("x", "1/2")
    assert SolverType.make(SolverType.TRUE).solve(f, g) == (True, [])


@pytest.mark.parametrize('factory', [SympyPGF, ProdigyPGF, SymenginePGF])
def test_always_false_solver(factory: CommonDistributionsFactory):
    f = factory.undefined("dummy_var", "another_var")
    g = factory.log("x", "1/2")
    assert SolverType.make(SolverType.FALSE).solve(f, g) == (False, [])


@pytest.mark.parametrize('factory', [SympyPGF, ProdigyPGF, SymenginePGF])
def test_always_none_solver(factory: CommonDistributionsFactory):
    f = factory.undefined("dummy_var", "another_var")
    g = factory.log("x", "1/2")
    assert SolverType.make(SolverType.NONE).solve(f, g) == (None, [])
