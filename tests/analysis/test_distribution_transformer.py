import random

import probably.pgcl as pgcl
import sympy
from pytest import raises

import prodigy.analysis
from prodigy.distribution.generating_function import GeneratingFunction as GF
from prodigy.distribution.generating_function import SympyPGF


def test_context_injection():
    program = pgcl.parse_pgcl("""
        nat x;
        nat y;
        rparam p;
        nparam n;
        """)
    gf = GF("z^3")
    result = prodigy.analysis.compute_discrete_distribution(
        program, gf, prodigy.analysis.ForwardAnalysisConfig())
    assert result.get_variables() == {
        "x", "y", "z"
    } and result.get_parameters() == {"p", "n"}


def test_iid_predefined_distributions():
    distributions = {
        "geometric(p)":
        SympyPGF.geometric("x", "p").set_parameters("p", "n"),
        "binomial(n,p)":
        SympyPGF.binomial("x", "n", "p"),
        "poisson(n)":
        SympyPGF.poisson("x", "n").set_parameters("p", "n"),
        "bernoulli(p)":
        SympyPGF.bernoulli("x", "p").set_parameters("p", "n"),
        "unif(n, n+10)":
        SympyPGF.uniform("x", "n", "n+10").set_parameters("p", "n"),
        "logdist(p)":
        SympyPGF.log("x", "p").set_parameters("p", "n")
    }

    for distribution in distributions.keys():
        result = prodigy.analysis.compute_discrete_distribution(
            pgcl.parse_pgcl("""
                    nat x;
                    nparam n;
                    rparam p;
                    
                    x := iid(%s, x);
                """ % distribution), GF("x"),
            prodigy.analysis.ForwardAnalysisConfig())
        assert result == distributions[distribution], distribution


def test_iid_update():
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
            nat x;
            rparam p;
            
            x := binomial(5,p);
            x := iid(geometric(p), x);
        """), GF("1"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF("(p^2/(1-(1-p)*x) + 1 - p)^5", "x")


def test_subtraction_when_already_zero():
    with raises(ValueError) as e:
        result = prodigy.analysis.compute_discrete_distribution(
            pgcl.parse_pgcl("""
        nat n;
        n := 0;
        n := n-1;
        """), GF("n^5"), prodigy.analysis.ForwardAnalysisConfig())
    assert "Cannot assign '_0 - 1' to 'n' because it can be negative" in str(e)

    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
    nat n;
    n := 0;
    n := n - 2*n;
    """), GF("n^5"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF("1", "n")

    with raises(ValueError) as e:
        result: GF = prodigy.analysis.compute_discrete_distribution(
            pgcl.parse_pgcl("""
            nat x
            x := x - 2*x
        """), SympyPGF.poisson("x", "3"),
            prodigy.analysis.ForwardAnalysisConfig())
    assert 'Cannot assign' in str(e)


def test_addition_assignment():
    rand_inc = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat n;
        n := n+{rand_inc};
        """), GF(f"n^{rand_init}"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF(f"n^{rand_init + rand_inc}", "n")


def test_multiplication_assignment():
    rand_factor = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat n;
        n := n*{rand_factor};
        """), GF(f"n^{rand_init}"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF(f"n^{rand_init * rand_factor}", "n")

    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl(f"""
        nat n;
        n := n*0.5;
        """), GF(f"n^4"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF(f"n^2", "n")


def test_ite_statement():
    result = prodigy.analysis.compute_discrete_distribution(
        pgcl.parse_pgcl("""
        nat x;
        nat y;
        
        if ( x > y){
            x := x + 1
        } else {
            y := y + 1
        }
        """), GF("x*y"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF("x*y^2", "x", "y")
