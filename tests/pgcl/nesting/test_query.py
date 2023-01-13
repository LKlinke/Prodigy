from probably.pgcl.ast.instructions import QueryInstr
from probably.pgcl.parser import parse_pgcl


def test_basic_query():
    prog = parse_pgcl("""
        nat x
        query {
            x := unif(0,10)
            observe(x > 5)
        }
    """)

    assert isinstance(prog.instructions[0], QueryInstr)
    assert len(prog.instructions[0].instrs) == 2