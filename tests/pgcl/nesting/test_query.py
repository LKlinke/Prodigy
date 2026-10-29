from prodigy.pgcl.ast.instructions import QueryInstr
from prodigy.pgcl.compiler import compile_pgcl


def test_basic_query():
    prog = compile_pgcl("""
        nat x
        query {
            x := unif(0,10)
            observe(x > 5)
        }
    """)

    assert isinstance(prog.instructions[0], QueryInstr)
    assert len(prog.instructions[0].instrs) == 2
