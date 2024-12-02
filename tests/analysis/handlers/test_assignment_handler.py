import random

import pytest
from _pytest.python_api import raises
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
def test_subtraction_when_already_zero(engine, factory):
    with raises(ValueError, match='negative') as e:
        result, error_prob = compute_discrete_distribution(
            pgcl.parse_pgcl("""
        nat x;
        x := 0;
        x := x-1;
        """), factory.from_expr("x^5"),
            ForwardAnalysisConfig(engine=engine))
        print(result)

    result, error_prob = compute_discrete_distribution(
        pgcl.parse_pgcl("""
    nat x;
    x := 0;
    x := x - 2*x;
    """), factory.from_expr("x^5"),
        ForwardAnalysisConfig(engine=engine))
    print(result)
    assert result == factory.from_expr("1", "x")

    if engine != ForwardAnalysisConfig.Engine.SYMENGINE:
        with raises(ValueError) as e:
            result, error_prob = compute_discrete_distribution(
                pgcl.parse_pgcl("""
                nat x
                x := x - 2*x
            """), factory.poisson("x", "3"),
                ForwardAnalysisConfig(engine=engine))
            print(result)
        assert 'Cannot assign' in str(e)
    else:
        pytest.xfail("Symengine doesnt like complex infinity in exponential functions.")
        result, error_prob = compute_discrete_distribution(
            pgcl.parse_pgcl("""
            nat x
            x := x - 2*x
        """), factory.poisson("x", "3"),
            ForwardAnalysisConfig(engine=engine))



@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF),
                          (ForwardAnalysisConfig.Engine.SYMENGINE, SymenginePGF)])
def test_addition_assignment(engine, factory):
    rand_inc = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result, error_prob = compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x+{rand_inc};
        """), factory.from_expr(f"x^{rand_init}"),
        ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr(f"x^{rand_init + rand_inc}", "x")


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF),
                          (ForwardAnalysisConfig.Engine.SYMENGINE, SymenginePGF)])
def test_multiplication_assignment(engine, factory):
    rand_factor = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result, error_prob = compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x*{rand_factor};
        """), factory.from_expr(f"x^{rand_init}"),
        ForwardAnalysisConfig(engine=engine))

    assert result == factory.from_expr(f"x^{rand_init * rand_factor}", "x")

    result, error_prob = compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat x;
        x := x*1/2;
        """), factory.from_expr(f"x^4"),
        ForwardAnalysisConfig(engine=engine))
    assert result == factory.from_expr(f"x^2", "x")
