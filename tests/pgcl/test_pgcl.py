# pylint: disable=wildcard-import
from probably.pgcl import *


def test_chain():
    code = """
        # this is a comment
        # A program starts with the variable declarations.
        # Every variable must be declared.
        # A variable is either of type nat or of type bool.
        bool f;
        nat c;  # optional: provide bounds for the variables. The declaration then looks as follows: nat c [0,100]

        # Logical operators: & (and),  || (or), not (negation)
        # Operator precedences: not > & > ||
        # Comparing arithmetic expressions: <=, <, ==

        skip;

        while (c < 10 & f=false) {
            {c := c+1} [0.8] {f:=true}
        }
    """

    program = parse_pgcl(code)

    # assert declaration values
    decls = program.variables
    assert decls["f"] == BoolType()

    # assert instruction types
    instrs = program.instructions
    assert isinstance(instrs[0], SkipInstr)
    loop = instrs[1]
    assert isinstance(loop, WhileInstr)
    assert isinstance(loop.cond, BinopExpr)
    assert isinstance(loop.body[0], ChoiceInstr)


def test_comments():
    program = parse_pgcl("""
        # hello world
        // another comment
    """)

    assert len(program.instructions) == 0


def test_params():
    program = parse_pgcl("""
        nparam x
        rparam y
    """)
    assert program.parameters["x"] == NatType(None)
    assert program.parameters["y"] == RealType()
