from probably.pgcl.ast import Program
from probably.pgcl.compiler import compile_pgcl

from prodigy.analysis.static import independent_vars


def test_independence_fully_isolated():
    prog = compile_pgcl("""
        nat x;
        nat y;

        x := geometric(1/2)
        y := 0
    """)
    assert isinstance(prog, Program)

    res = independent_vars(prog)
    assert res == {frozenset({'x', 'y'})}


def test_independence_tri_copy():
    prog = compile_pgcl("""
        nat x;
        nat y;
        nat z;

        x := geometric(1/2)
        y := x
        z := 2*x
    """)
    assert isinstance(prog, Program)

    res = independent_vars(prog)
    assert res == set()


def test_independence_dist_param():
    prog = compile_pgcl("""
        nat x;
        nat y;

        x := geometric(1/2)
        y := unif(0, x)
    """)
    assert isinstance(prog, Program)

    res = independent_vars(prog)
    assert res == set()
