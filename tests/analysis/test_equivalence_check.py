import pytest
import sympy
from probably.pgcl.ast import Program
from probably.pgcl.compiler import compile_pgcl

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence


@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY])
def test_equivalence_check(engine):
    prog = compile_pgcl("""
        nat x;
        nat c;
        nat temp;

        while (x >= 1){
        {x := 0 } [1/2] {c := c+1}
        temp :=0
        }
    """)
    assert isinstance(prog, Program)

    inv = compile_pgcl("""
        nat x;
        nat c;
        nat temp;

        if(x >= 1){
            temp := geometric(1/2)
            c := c + temp
            x := 0
            temp := 0
        } else {skip}
    """)
    assert isinstance(inv, Program)
    res, subs = check_equivalence(prog, inv, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert subs == []


@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY])
def test_equivalence_check_parameter(engine):
    prog = compile_pgcl("""
        nat x;
        nat c;
        nat temp;
        rparam p;

        while (x >= 1){
        {x := 0 } [1.0-p] {c := c+1}
        temp :=0
        }
    """)
    assert isinstance(prog, Program)

    inv = compile_pgcl("""
        nat x;
        nat c;
        nat temp;
        rparam p;

        if(x >= 1){
            temp := geometric(p)
            c := c + temp
            x := 0
            temp := 0
        } else {skip}
    """)
    assert isinstance(inv, Program)
    res, subs = check_equivalence(prog, inv, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert len(subs) == 1
    assert sympy.S(subs[0]['p']) == sympy.S('1/2')
