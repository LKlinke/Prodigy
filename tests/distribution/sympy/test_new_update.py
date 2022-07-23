import sympy
from probably.pgcl.parser import parse_expr
from pytest import raises, xfail

from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      SympyPGF)


def test_literal_assignment():
    gf = GeneratingFunction("1", "x")
    assert gf.update(parse_expr("x = 1")) == GeneratingFunction("x")

    gf = SympyPGF.poisson('x', 30)
    assert gf.update(parse_expr("x = 1")) == GeneratingFunction("x")


def test_addition():
    gf = GeneratingFunction("x^3")
    assert gf.update(parse_expr("x = 3 + 4 + 8")) == GeneratingFunction("x^15")
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

    gf = GeneratingFunction('x^8', 'x', 'y')
    assert gf.update(parse_expr('y = x')) == GeneratingFunction('y^8 * x^8')

    gf = GeneratingFunction('p*x + (1-p) * x^2', 'x')
    assert gf.update(parse_expr('x = p')) == GeneratingFunction('x^p', 'x')


def test_multiplication():
    gf = GeneratingFunction("n^5*m^42*o^8")
    assert gf.update(
        parse_expr("n = 3 * 8")) == GeneratingFunction("n^24*m^42*o^8")
    assert gf.update(
        parse_expr("n = 5 * o")) == GeneratingFunction("n^40*m^42*o^8")
    assert gf.update(
        parse_expr("n = m * o")) == GeneratingFunction("n^336*m^42*o^8")
    assert gf.update(
        parse_expr("n = n * o")) == GeneratingFunction("n^40*m^42*o^8")

    gf = GeneratingFunction("0.5 * (n^1 + n^2)")
    assert gf.update(
        parse_expr("n = n * 3")) == GeneratingFunction("0.5*n^3+0.5*n^6")
    assert gf.update(
        parse_expr("n = n * n")) == GeneratingFunction("0.5*n^1+0.5*n^4")

    gf = SympyPGF.poisson('x', 3) * GeneratingFunction('y^3')
    assert gf.update(parse_expr('x = x * y'))._function == gf._function.subs(
        sympy.S('x'),
        sympy.S('x')**3)

    gf = SympyPGF.poisson('x', 3) * SympyPGF.poisson('y', 5)
    assert gf.update(parse_expr('x = 3*7')) == SympyPGF.poisson(
        'y', 5) * GeneratingFunction('x**21')
    assert gf.update(parse_expr('x = x * 3')) == GeneratingFunction(
        gf._function.subs(sympy.S('x'),
                          sympy.S('x')**3))
    assert gf.update(parse_expr('x = y * 4')) == GeneratingFunction(
        gf._function.subs([(sympy.S('x'), 1),
                           (sympy.S('y'), sympy.S('y * x**4'))]))


def test_subtraction():
    gf = GeneratingFunction("n^5*m^42*l^8")
    assert gf.update(
        parse_expr("m = m - 8")) == GeneratingFunction("n^5*m^34*l^8")
    assert gf.update(
        parse_expr("l = l - n")) == GeneratingFunction("n^5*m^42*l^3")
    assert gf.update(
        parse_expr("m = l - n")) == GeneratingFunction("n^5*m^3*l^8")
    with raises(ValueError) as e:
        gf.update(parse_expr("m = n - l"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    gf = GeneratingFunction('p*x', 'x')
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 20"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)
    assert gf.update(parse_expr("x = x - 1")) == GeneratingFunction('p', 'x')

    gf = GeneratingFunction('(1/p)*x', 'x')
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    xfail('known bug / unclear behavior')
    gf = GeneratingFunction('x^p', 'x')
    # TODO this fails because sympy can't assume that p is >= 0 and thus doesn simplify 'x**p*(1/x)**p' to '1'
    # how should we handle parameters here, can we always assume that they are >= 0?
    assert gf.update(parse_expr('x = x - p')) == SympyPGF.one('x')
    assert gf.update(parse_expr('x = x - (p + 1)')) == GeneratingFunction('x')


def test_fresh_variables():
    gf = GeneratingFunction('x*xl')
    assert gf.update(parse_expr('x = 2*3')) == GeneratingFunction('x^6*xl')

    gf = GeneratingFunction('x*_0')
    assert gf.update(parse_expr('x = 2*3')) == GeneratingFunction('x^6*_0')


def test_modulo():
    gf = GeneratingFunction('x * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.update(parse_expr('x = 5 % 3')) == GeneratingFunction(
        'x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.update(parse_expr('x = 5 % (1+1+1)')) == GeneratingFunction(
        'x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    xfail('sympy madness / unclear behavior')
    # TODO should this hold? I think sympy computes the correct solution, it just can't simplify it to 0 (missing assumptions?)
    assert gf.update(parse_expr('x = y % (3+2)'))._function == sympy.S(
        '0.3*y^4*x^4 + 0.3*y^7*x^2 + 0.4*y^8*x^3')
