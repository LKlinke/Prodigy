import builtins
import os
from glob import glob
from os.path import isfile

import pytest
import sympy
from probably.pgcl import parse_pgcl
from probably.pgcl.ast import Program
from probably.pgcl.compiler import compile_pgcl

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.pgcl.pgcl_operations import cav_phi


@pytest.mark.parametrize(
    'engine',
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE])
def test_equivalence_check_geometric_sampler(engine):
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
    "engine",
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY,
     ForwardAnalysisConfig.Engine.SYMENGINE]
)
@pytest.mark.parametrize(
    "file_path",
    [y for x in os.walk("pgfexamples/equivalence/loop_free") for y in glob(os.path.join(x[0], '*.pgcl')) if
     not y.endswith('2.pgcl')]
)
def test_equivalence_loop_free_benchmarks(engine, file_path):
    # Read the body of the program files
    with open(file_path, "r") as f:
        file = "\n".join(f.readlines())
    file2_path = file_path.replace(".pgcl", "2.pgcl")
    with open(file2_path, "r") as f:
        inv = "\n".join(f.readlines())

    # Compile them to a program object
    prog1 = compile_pgcl(file)
    prog2 = compile_pgcl(inv)

    assert (isinstance(prog1, Program))
    assert (isinstance(prog2, Program))

    # Run the main program
    res, subs = check_equivalence(prog1, prog2, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert res
    assert subs == []

@pytest.mark.parametrize(
    "engine",
    [ ForwardAnalysisConfig.Engine.GINAC,ForwardAnalysisConfig.Engine.SYMPY,
     # ForwardAnalysisConfig.Engine.SYMENGINE
     ]
)
@pytest.mark.parametrize(
    "file_path",
    [y for x in os.walk("pgfexamples/equivalence/loopy") for y in glob(os.path.join(x[0], '*.pgcl')) if
     not "invariants" in y]
)
# This test apparently has some side-effect which fails other tests (cf. #64), if it is executed last
# this problem does not occur
# FIXME this test has some effect on GINAC, when removing GINAC from the engine list, all tests pass
def test_equivalence_loopy_benchmarks(monkeypatch, engine, file_path):
    # Read the body of the program files
    with open(file_path, "r") as f:
        lines = f.readlines()
        if "skip" in lines[0]:
            pytest.skip("File marked as skipped")
        file = "\n".join(lines)

    invariant_path = file_path.replace("loopy/", "loopy/invariants/").replace(".pgcl", "_invariant.pgcl")
    if not isfile(invariant_path):
        pytest.skip("File has multiple invariants")

    with open(invariant_path, "r") as f:
        inv = "\n".join(f.readlines())

    # Compile them to a program object
    prog1 = parse_pgcl(file)
    prog2 = parse_pgcl(inv)

    assert (isinstance(prog1, Program))
    assert (isinstance(prog2, Program))

    inputs = iter(["1", invariant_path])
    # Simulate the input for the invariant files
    with monkeypatch.context() as m:
        m.setattr("builtins.input", lambda _: next(inputs))  # Select invariant file1
        # Run the main program
        res, subs = check_equivalence(prog1, prog2, ForwardAnalysisConfig(engine=engine), compute_semantics)

    assert res
    assert subs == []


@pytest.mark.parametrize(
    "engine",
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY,
     # ForwardAnalysisConfig.Engine.SYMENGINE
     ]
)
def test_equivalence_fail(engine):
    prog1 = compile_pgcl("""
    nat x;
    x := geometric(1/2);
    """)

    prog2 = compile_pgcl("""
    nat x;
    x := unif(1,6);
    """)

    assert(isinstance(prog1, Program))
    assert(isinstance(prog2, Program))

    res, subs = check_equivalence(prog1, prog2, ForwardAnalysisConfig(engine=engine), compute_semantics)
    assert not res
