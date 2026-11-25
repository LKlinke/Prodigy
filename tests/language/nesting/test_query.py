from prodigy.language.ast.instructions import QueryInstr
from prodigy.language.compiler import compile_program


def test_basic_query():
    prog = compile_program("""
x: nat;
query {
    x = unif(0, 10);
    observe(x > 5);
}""")

    assert isinstance(prog.instructions[0], QueryInstr)
    assert len(prog.instructions[0].instrs) == 2
