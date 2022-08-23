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

    gf = GeneratingFunction('n')
    assert gf.update(parse_expr("n = 2")) == GeneratingFunction("n^2")


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

    gf = SympyPGF.poisson('x', 3)
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    gf = GeneratingFunction('0.4*tmp^5*c^13*n^4 + 0.6*tmp^7*c^28*n^77')
    assert gf.update(parse_expr('n = n + (c-tmp)')) == GeneratingFunction(
        '0.4*tmp^5*c^13*n^12 + 0.6*tmp^7*c^28*n^98')

    xfail('known bug / unclear behavior')
    gf = GeneratingFunction('x^p', 'x')
    # TODO this fails because sympy can't assume that p is real and x >= 0 and thus doesn simplify 'x**p*(1/x)**p' to '1'
    # how should we handle parameters and variables here, can we always assume that they are >= 0 and real?
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
    assert gf.update(parse_expr('x = y % (3+2)'))._function == sympy.S(
        '(3/10)*y^4*x^4 + (3/10)*y^7*x^2 + (4/10)*y^8*x^3')

    gf = GeneratingFunction(
        '0.3*a**3*b**5*c**2 + 0.2*a**6*b**9*c**55 + 0.5*a**5*b**346*c**34')
    assert gf.update(parse_expr('a = b % c')) == GeneratingFunction(
        '0.3*a**1*b**5*c**2 + 0.2*a**9*b**9*c**55 + 0.5*a**6*b**346*c**34')


def test_division():
    gf = GeneratingFunction('n')
    # probably simplifies 4 / 2 to a RealLitExpr
    assert gf.update(parse_expr('n = 4 / 2')) == GeneratingFunction('n^2')
    assert gf.update(parse_expr('n = n / 1')) == GeneratingFunction('n')
    with raises(ValueError) as e:
        gf.update(parse_expr('n = n / 2'))
    assert 'because it is not an integer' in str(e)

    gf = GeneratingFunction('n^2*m^6', 'n', 'm', 'o')
    assert gf.update(
        parse_expr('o = m / n')) == GeneratingFunction('n^2*m^6*o^3')
    with raises(ValueError) as e:
        gf.update(parse_expr('o = n / m'))
    assert 'because it is not an integer' in str(e)

    gf = SympyPGF.poisson('x', 3) * GeneratingFunction('y*z')
    assert gf.update(parse_expr('x = z / y')) == GeneratingFunction('x*y*z')


def test_unilateral_approximation():
    gf = GeneratingFunction(
        '0.7*x**3*y**5 + 0.1*x**13*y**17 + 0.15*x**21*y**25 + 0.05*x**33*y**4')
    *_, res = gf.approximate_unilaterally('x', '0.9')
    assert res == GeneratingFunction(
        '0.7*x**3*y**5 + 0.1*x**13*y**17 + 0.15*x**21*y**25 + 0.05*y**4')

    gf = SympyPGF.poisson('x', 7).set_variables('x', 'y')
    gf._function = gf._function.subs(sympy.S('x'), sympy.S('x*y'))
    *_, res = gf.approximate_unilaterally('x', '0.9')
    assert res.filter(parse_expr('x = 100'))._function == 0
    assert res.filter(parse_expr('y = 100'))._function == sympy.S(
        'y**100*(7**100*exp(-7))/100!')
    assert res.filter(parse_expr('x = 5')) == res.filter(parse_expr('y = 5'))
