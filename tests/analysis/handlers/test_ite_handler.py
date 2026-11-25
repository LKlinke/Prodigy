import pytest
from prodigy import language

from prodigy.analysis.session import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_ite_statement(engine, factory):
    result, error_prob = compute_discrete_distribution(
        language.parse_program("""
x: nat;
y: nat;

if (x > y) {
    x = x + 1;
} else {
    y = y + 1;
}"""), factory.from_expr("x*y"),
        ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr("x*y^2", "x", "y")
