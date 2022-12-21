from probably.pgcl.compiler import CheckFail, compile_pgcl

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instruction_handler import compute_discrete_distribution
from prodigy.distribution.fast_generating_function import FPS, ProdigyPGF


def test_basic_function():
    prog = compile_pgcl("""
        fun f := {
            nat y;
            y := 42;
            return y;
        }
        nat x;
        x := f(x := 5);
    """)
    res = compute_discrete_distribution(
        prog, FPS('1'),
        ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC))
    assert res == FPS("x^42")


def test_parameter():
    prog = compile_pgcl("""
        fun f := {
            nat y;
            y := geometric(p);
            return y;
        }
        nat x;
        rparam p;
        x := f();
    """)
    res = compute_discrete_distribution(
        prog, FPS('1'),
        ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC))
    assert res == ProdigyPGF.geometric('x', 'p')


def test_outside_variables():
    prog = compile_pgcl("""
        fun f := {
            y := geometric(p);
            return y;
        }
        nat x;
        nat y;
        x := f();
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == "y is not a variable."
