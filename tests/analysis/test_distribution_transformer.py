import random

import probably.pgcl as pgcl

import prodigy.analysis
from prodigy.distribution.generating_function import GeneratingFunction as GF


def test_subtraction_when_already_zero():
    result = prodigy.analysis.compute_discrete_distribution(pgcl.parse_pgcl("""
    nat n;
    n := 0;
    n := n-1;
    """), GF("n^5"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF("1", "n")


def test_addition_assignment():
    rand_inc = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(pgcl.parse_pgcl(f"""
        nat n;
        n := n+{rand_inc};
        """), GF(f"n^{rand_init}"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF(f"n^{rand_init + rand_inc}", "n")


def test_multiplication_assignment():
    rand_factor = random.randint(0, 100)
    rand_init = random.randint(0, 100)
    result = prodigy.analysis.compute_discrete_distribution(pgcl.parse_pgcl(f"""
        nat n;
        n := n*{rand_factor};
        """), GF(f"n^{rand_init}"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF(f"n^{rand_init * rand_factor}", "n")


def test_ite_statement():
    result = prodigy.analysis.compute_discrete_distribution(pgcl.parse_pgcl("""
        nat x;
        nat y;
        
        if ( x > y){
            x := x + 1
        } else {
            y := y + 1
        }
        """), GF("x*y"), prodigy.analysis.ForwardAnalysisConfig())
    assert result == GF("x*y^2", "x", "y")
