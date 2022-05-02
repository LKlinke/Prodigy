from lark import Lark

from probably.util.lark_expr_parser import (atom, build_expr_parser, infixl,
                                            prefix)


def test_build_expr_parser():
    table = [[infixl("plus", "+"), infixl("minus", "-")],
             [
                 prefix("neg", "-"),
                 atom("parens", '"(" test ")"'),
                 atom("int", "/[0-9]+/")
             ]]

    grammar = build_expr_parser(table, "test")
    parser = Lark(grammar, start="test")

    # with parens, plus is done last
    t = parser.parse("900+(500-600)")
    assert t.data == "plus"

    # without parens, minus is done last
    t = parser.parse("900+500-600")
    assert t.data == "minus"
