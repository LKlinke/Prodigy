import builtins
import os
from glob import glob
from os.path import isfile

import pytest
import sympy
from prodigy.pgcl.ast import Program
from prodigy.pgcl.compiler import compile_pgcl

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

LOOPY_DIR = "pgfexamples/equivalence/loopy"

# Programs marked `# skip` in their first line, with the reason. A new skip needs an entry here.
LOOPY_SKIPPED = {
    "condand": "the loop has no closed-form invariant (see invariants/condand_invariant.pgcl), "
               "so there is nothing to check equivalence against",
}


def _loopy_invariants(file_path: str) -> tuple[str, list[str]]:
    """The loop-free counterpart of a loopy example and the invariants for its loops.

    The counterpart is ``invariants/<name>_invariant.pgcl``. A program with several loops has one
    invariant per loop, ``invariants/<name>_invariant1.pgcl``, ``..._invariant2.pgcl`` and so on, in
    the order the analysis asks for them (pgfexamples/README.md); for nested loops the invariant of
    the outermost loop is the counterpart, so ``<name>_invariant.pgcl`` may be missing then.
    """
    stem = os.path.join(os.path.dirname(file_path), "invariants",
                        os.path.basename(file_path).removesuffix(".pgcl"))
    numbered = []
    while isfile(f"{stem}_invariant{len(numbered) + 1}.pgcl"):
        numbered.append(f"{stem}_invariant{len(numbered) + 1}.pgcl")
    counterpart = f"{stem}_invariant.pgcl"
    if not isfile(counterpart):
        assert numbered, f"no invariant for {file_path}"
        counterpart = numbered[0]
    return counterpart, numbered or [counterpart]


@pytest.mark.parametrize(
    "engine",
    [ForwardAnalysisConfig.Engine.GINAC, ForwardAnalysisConfig.Engine.SYMPY,
     # ForwardAnalysisConfig.Engine.SYMENGINE
     ]
)
@pytest.mark.parametrize(
    "file_path", sorted(glob(os.path.join(LOOPY_DIR, "*.pgcl"))), ids=lambda path: os.path.basename(path)
)
def test_equivalence_loopy_benchmarks(monkeypatch, engine, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    name = os.path.basename(file_path).removesuffix(".pgcl")
    if source.split("\n", 1)[0].lstrip("#").strip() == "skip":
        assert name in LOOPY_SKIPPED, f"{file_path} is marked as skipped without a reason in LOOPY_SKIPPED"
        pytest.skip(LOOPY_SKIPPED[name])
    assert name not in LOOPY_SKIPPED, f"{file_path} is in LOOPY_SKIPPED but not marked as skipped"

    counterpart, invariants = _loopy_invariants(file_path)
    with open(counterpart, "r", encoding="utf-8") as f:
        other_source = f.read()

    prog1 = compile_pgcl(source)
    prog2 = compile_pgcl(other_source)
    assert isinstance(prog1, Program)
    assert isinstance(prog2, Program)

    # The analysis asks for every loop: strategy 1 (invariant), then the invariant file.
    inputs = iter([answer for invariant in invariants for answer in ("1", invariant)])
    with monkeypatch.context() as m:
        m.setattr("builtins.input", lambda _: next(inputs))
        res, subs = check_equivalence(prog1, prog2, ForwardAnalysisConfig(engine=engine), compute_semantics)

    assert next(inputs, None) is None, "not every invariant was used"
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
