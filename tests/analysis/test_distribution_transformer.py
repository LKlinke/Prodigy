import random

import probably.pgcl as pgcl
import pytest
from pytest import raises

import prodigy.analysis
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF


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
    result = prodigy.analysis.compute_discrete_distribution(
        program, gf, prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result.get_variables() == {
        "x", "y", "z"
    } and result.get_parameters() == {"p", "n"}


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
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
        result = prodigy.analysis.compute_discrete_distribution(
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
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_iid_update(engine, factory):
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
            nat x;
            rparam p;
            
            x := binomial(5,p);
            x := iid(geometric(p), x);
        """), factory.one('x'),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr("(p^2/(1-(1-p)*x) + 1 - p)^5", "x")


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_subtraction_when_already_zero(engine, factory):
    with raises(ValueError, match='negative') as e:
        result = prodigy.analysis.compute_discrete_distribution(
            pgcl.parse_pgcl("""
        nat x;
        x := 0;
        x := x-1;
        """), factory.from_expr("x^5"),
            prodigy.analysis.ForwardAnalysisConfig(engine=engine))

    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
    nat x;
    x := 0;
    x := x - 2*x;
    """), factory.from_expr("x^5"),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr("1", "x")

    with raises(ValueError) as e:
        result = prodigy.analysis.compute_discrete_distribution(
            pgcl.parse_pgcl("""
            nat x
            x := x - 2*x
        """), factory.poisson("x", "3"),
            prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert 'Cannot assign' in str(e)


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_addition_assignment(engine, factory):
    rand_inc = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x+{rand_inc};
        """), factory.from_expr(f"x^{rand_init}"),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr(f"x^{rand_init + rand_inc}", "x")


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_multiplication_assignment(engine, factory):
    rand_factor = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x*{rand_factor};
        """), factory.from_expr(f"x^{rand_init}"),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr(f"x^{rand_init * rand_factor}", "x")

    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x*0.5;
        """), factory.from_expr(f"x^4"),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr(f"x^2", "x")


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
def test_ite_statement(engine, factory):
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
        nat x;
        nat y;
        
        if ( x > y){
            x := x + 1
        } else {
            y := y + 1
        }
        """), factory.from_expr("x*y"),
        prodigy.analysis.ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr("x*y^2", "x", "y")
