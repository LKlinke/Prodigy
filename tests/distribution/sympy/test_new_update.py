import sympy
from probably.pgcl.parser import parse_expr

from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      SympyPGF)


# TODO eventually replace all private function calls by update()
def test_sum_update():
    gf = GeneratingFunction("x^3")
    assert gf.update(parse_expr("x = 3 + 5 + 8")) == GeneratingFunction("x^16")
    assert gf.update(parse_expr("x = x + x")) == GeneratingFunction("x^6")

    gf: GeneratingFunction = SympyPGF.poisson('x', 16)
    assert gf.update(parse_expr("x = 3 + 5 + 8")) == GeneratingFunction("x^16")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 5")) == SympyPGF.zero("x")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x % 2 = 1")) == SympyPGF.zero("x")
    assert gf.update(
        parse_expr("x = x + x")).filter(parse_expr("x = 30"))._function.subs(
            sympy.S("x"), 1) == sympy.S("(16^15 * exp(-16)) / 15!")


def test_var_assignment():
    gf = GeneratingFunction("n^5*m^42")
    assert gf.update(parse_expr("n = m"))._function.subs(sympy.S("m"),
                                                         1) == sympy.S("n^42")
    assert gf.update(parse_expr("n = n"))._function.subs(sympy.S("m"),
                                                         1) == sympy.S("n^5")
