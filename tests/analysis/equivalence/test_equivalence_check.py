import builtins

import pytest
import sympy
from probably.pgcl.ast import Program
from probably.pgcl.compiler import compile_pgcl

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.pgcl.pgcl_operations import cav_phi


@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE])
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

    phi_inv = cav_phi(prog, inv)
    res, subs = check_equivalence(phi_inv, inv, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert subs == []


@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE])
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

    phi_inv = cav_phi(prog, inv)
    res, subs = check_equivalence(phi_inv, inv, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert len(subs) == 1
    assert sympy.S(subs[0][sympy.S('p')]) == sympy.S('0.5') or sympy.S(subs[0][sympy.S('p')]) == sympy.S('1/2')

@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY,
     ForwardAnalysisConfig.Engine.SYMENGINE])
def test_equivalence_file(monkeypatch, engine):
    # Read the body of the program files
    with open("pgfexamples/equivalence/loop_free/bernoulli.pgcl", "r") as f:
        file = "\n".join(f.readlines())

    with open("pgfexamples/equivalence/loop_free/bernoulli2.pgcl", "r") as f:
        inv = "\n".join(f.readlines())

    # Compile them to a program object
    prog1 = compile_pgcl(file)
    prog2 = compile_pgcl(inv)

    assert(isinstance(prog1, Program))
    assert(isinstance(prog2, Program))

    # Simulate the input for the invariant files
    monkeypatch.setattr(builtins, "input", lambda _: "pgfexamples/equivalence/loop_free/bernoulli2.pgcl")

    # Run the main program
    res, subs = check_equivalence(prog1, prog2, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert subs == []

