import probably.pgcl as pgcl
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF


# todo hier symengine einbauen
@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_context_injection(engine, factory):
    program = pgcl.parse_pgcl("""
        nat x;
        nat y;
        rparam p;
        nparam n;
        """)
    gf = factory.from_expr("z^3")
    result, error_prob = compute_discrete_distribution(
        program, gf, ForwardAnalysisConfig(engine=engine))
    assert result.get_variables() == {
        "x", "y", "z"
    } and result.get_parameters() == {"p", "n"}


