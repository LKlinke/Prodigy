import pytest
from probably import pgcl as pgcl

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.symengine_distribution import SymenginePGF

@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF),
                          (ForwardAnalysisConfig.Engine.SYMENGINE, SymenginePGF)])
def test_iid_predefined_distributions(engine, factory):
    distributions = {
        "geometric(p)": factory.geometric("x", "p").set_parameters("p", "n"),
        "binomial(n,p)": factory.binomial("x", "n", "p"),
        "poisson(n)": factory.poisson("x", "n").set_parameters("p", "n"),
        "bernoulli(p)": factory.bernoulli("x", "p").set_parameters("p", "n"),
        "unif(n, n+10)": factory.uniform("x", "n",
                                         "n+10").set_parameters("p", "n"),
        "logdist(p)": factory.log("x", "p").set_parameters("p", "n")
    }

    for distribution in distributions.keys():
        result, error_prob = compute_discrete_distribution(
            pgcl.parse_pgcl("""
                    nat x;
                    nparam n;
                    rparam p;
                    
                    x := iid(%s, x);
                """ % distribution), factory.from_expr('x', 'x'),
            ForwardAnalysisConfig(engine=engine))
        assert result == distributions[distribution], distribution


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF),
                          (ForwardAnalysisConfig.Engine.SYMENGINE, SymenginePGF)])
def test_iid_update(engine, factory):
    result, error_prob = compute_discrete_distribution(
        pgcl.parse_pgcl("""
            nat x;
            rparam p;
            
            x := binomial(5,p);
            x := iid(geometric(p), x);
        """), factory.one('x'),
        ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr("(p^2/(1-(1-p)*x) + 1 - p)^5", "x")
