import pytest
from prodigy.language.compiler import compile_program

from prodigy.analysis.session import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_basic_function(engine, factory):
    prog = compile_program("""
def f(y: nat) -> nat {
    y = 42;
    return y;
}
x: nat;
x = f(y = 5);""")
    res, error_prob = compute_discrete_distribution(
        prog, factory.one(), ForwardAnalysisConfig(engine=engine))
    assert res == factory.from_expr("x^42")


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_parameter(engine, factory):
    prog = compile_program("""
def f(y: nat) -> nat {
    y = geometric(p);
    return y;
}
x: nat;
param p: real;
x = f();""")
    res, error_prob = compute_discrete_distribution(
        prog, factory.one(), ForwardAnalysisConfig(engine=engine))
    assert res == factory.geometric('x', 'p')
