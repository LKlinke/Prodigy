# pylint: disable=wildcard-import
import pytest

from probably.pgcl import *
from probably.pgcl.analyzer.syntax import check_for_nested_loops, get_nested_while_loops


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


def test_parameter_writing():
    with pytest.raises(SyntaxError, match="Parameters must not be assigned a new value."):
        program = parse_pgcl("""
        # get a parameter
        nparam n;
        
        // try to set it to a new value
        n := 12 // <- this should give an error!
        """)


def test_variable_in_distribution_parameter():
    with pytest.raises(SyntaxError, match="In distribution parameter expressions, no variables are allowed."):
        program = parse_pgcl(
            """
        nparam n;
        nat x;
        
        x := unif(x, n)
        """
        )


def test_check_for_nested_loops():
    program = parse_pgcl(
        """
        nat x;
        nat y;
        nat s;
        
        s := 0;
        while( x > 0){
            while (y > 0){
               s := s + x;
               y := y - 1;
            }
            x := x - 1;
        }
        """
    )
    assert len(get_nested_while_loops(program.instructions)) == 2
    print(get_nested_while_loops(program.instructions))

    program = program = parse_pgcl(
        """
        nat x;
        nat y;
        nat s;
        
        s := 0;
        while( x > 0){
            x := x-1;
        }
        while (y > 0){
          s := s + x;
          y := y - 1;
        }
        """
    )
    assert len(get_nested_while_loops(program.instructions)) == 1
