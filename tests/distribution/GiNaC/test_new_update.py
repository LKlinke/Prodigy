import pygin
import sympy
from probably.pgcl.parser import parse_expr
from pytest import raises, xfail

from prodigy.distribution.distribution import MarginalType
from prodigy.distribution.fast_generating_function import FPS, ProdigyPGF


def test_literal_assignment():
    gf = FPS("1", "x")
    assert gf.new_update(parse_expr("x = 1")) == FPS("x")

    gf = ProdigyPGF.poisson('x', 30)
    assert gf.new_update(parse_expr("x = 5")) == FPS("x^5")
    assert gf.new_update(parse_expr("x = 1")) == FPS("x")

    gf = FPS('n')
    assert gf.new_update(parse_expr("n = 2")) == FPS("n^2")


def test_addition():
    gf = FPS("x^3")
    assert gf.new_update(parse_expr("x = 3 + 4 + 8")) == FPS("x^15")
    assert gf.new_update(parse_expr("x = x + x")) == FPS("x^6")
    assert gf.new_update(parse_expr("x = x + 1")) == FPS("x^4")

    gf: FPS = ProdigyPGF.poisson('x', 16)
    assert gf.new_update(parse_expr("x = 3 + 5 + 8")) == FPS("x^16")
    assert gf.new_update(parse_expr("x = x + x")).filter(
        parse_expr("x = 5")) == ProdigyPGF.zero("x")
    assert gf.new_update(parse_expr("x = x + x")).filter(
        parse_expr("x % 2 = 1")) == ProdigyPGF.zero("x")
    assert gf.new_update(parse_expr("x = x + x")).filter(
        parse_expr("x = 30")).marginal(
            'x', method=MarginalType.EXCLUDE)._dist == pygin.Dist(
                "(16^15 * exp(-16)) / factorial(15)")

    gf = FPS("x^3 * y^5")
    assert gf.new_update(parse_expr("x = x + y")) == FPS("x^8 * y^5")


def test_var_assignment():
    gf = FPS("n^5*m^42")
    assert gf.new_update(parse_expr("n = m")) == FPS("n^42*m^42")
    assert gf.new_update(parse_expr("n = n")) == gf

    gf = FPS('x^8', 'x', 'y')
    assert gf.new_update(parse_expr('y = x')) == FPS('y^8 * x^8')

    with raises(ValueError) as e:
        gf = FPS('p*x + (1-p) * x^2', 'x')
        gf.new_update(parse_expr('x = p'))
    assert 'given rhs is a parameter' in str(e)

    with raises(ValueError) as e:
        gf = FPS('x')
        gf.new_update(parse_expr('x = 0.5'))
    assert 'Cannot assign the real value 0.5 to x' in str(e)


def test_multiplication():
    xfail("not yet implemented")
    gf = FPS("n^5*m^42*o^8")
    assert gf.new_update(parse_expr("n = 3 * 8")) == FPS("n^24*m^42*o^8")
    assert gf.new_update(parse_expr("n = 5 * o")) == FPS("n^40*m^42*o^8")
    assert gf.new_update(parse_expr("n = m * o")) == FPS("n^336*m^42*o^8")
    assert gf.new_update(parse_expr("n = n * o")) == FPS("n^40*m^42*o^8")

    gf = FPS("0.5 * (n^1 + n^2)")
    assert gf.new_update(parse_expr("n = n * 3")) == FPS("0.5*n^3+0.5*n^6")
    assert gf.new_update(parse_expr("n = n * n")) == FPS("0.5*n^1+0.5*n^4")

    gf = ProdigyPGF.poisson('x', 3) * FPS('y^3')
    assert gf.new_update(
        parse_expr('x = x * y'))._function == gf._function.subs(
            sympy.S('x'),
            sympy.S('x') ^ 3)

    gf = ProdigyPGF.poisson('x', 3) * ProdigyPGF.poisson('y', 5)
    assert gf.new_update(
        parse_expr('x = 3*7')) == ProdigyPGF.poisson('y', 5) * FPS('x^21')
    assert gf.new_update(parse_expr('x = x * 3')) == FPS(
        gf._function.subs(sympy.S('x'),
                          sympy.S('x') ^ 3))
    assert gf.new_update(parse_expr('x = y * 4')) == FPS(
        gf._function.subs([(sympy.S('x'), 1),
                           (sympy.S('y'), sympy.S('y * x^4'))]))

    gf = FPS('x*y^4')
    assert gf.new_update(parse_expr('x = 0.5*y')) == FPS('x^2*y^4')

    gf = ProdigyPGF.poisson('x', 5) * FPS('0.6*y^3*z^5 + 0.4*y^6*z^6')
    assert gf.new_update(
        parse_expr('x = z*y')) == FPS('0.6*x^15*y^3*z^5 + 0.4*x^36*y^6*z^6')
    assert gf.new_update(parse_expr('z = x*y'))._function == sympy.S(
        "y^3*(2*y^3*exp(5*x*z^6) + 3*exp(5*x*z^3))*exp(-5)/5")


def test_subtraction():
    xfail("not yet implemented")
    gf = FPS("n^5*m^42*l^8")
    assert gf.new_update(parse_expr("m = m - 8")) == FPS("n^5*m^34*l^8")
    assert gf.new_update(parse_expr("l = l - n")) == FPS("n^5*m^42*l^3")
    assert gf.new_update(parse_expr("m = l - n")) == FPS("n^5*m^3*l^8")
    with raises(ValueError) as e:
        gf.new_update(parse_expr("m = n - l"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    gf = FPS('p*x', 'x')
    with raises(ValueError) as e:
        gf.new_update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)
    with raises(ValueError) as e:
        gf.new_update(parse_expr("x = x - 20"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)
    assert gf.new_update(parse_expr("x = x - 1")) == FPS('p', 'x')

    gf = FPS('(1/p)*x', 'x')
    with raises(ValueError) as e:
        gf.new_update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    gf = ProdigyPGF.poisson('x', 3)
    with raises(ValueError) as e:
        gf.new_update(parse_expr("x = x - 2"))
    assert "Cannot assign '" in str(e) and "because it can be negative" in str(
        e)

    gf = FPS('0.4*tmp^5*c^13*n^4 + 0.6*tmp^7*c^28*n^77')
    assert gf.new_update(parse_expr('n = n + (c-tmp)')) == FPS(
        '0.4*tmp^5*c^13*n^12 + 0.6*tmp^7*c^28*n^98')


def test_fresh_variables():
    xfail("not yet implemented")
    gf = FPS('x*xl')
    assert gf.new_update(parse_expr('x = 2*3')) == FPS('x^6*xl')

    gf = FPS('x*sym_0')
    assert gf.new_update(parse_expr('x = 2*3')) == FPS('x^6*sym_0')


def test_modulo():
    xfail("not yet implemented")
    gf = FPS('x * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.new_update(
        parse_expr('x = 5 % 3')) == FPS('x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.new_update(parse_expr('x = 5 % (1+1+1)')) == FPS(
        'x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.new_update(parse_expr('x = y % (3+2)'))._function == sympy.S(
        '(3/10)*y^4*x^4 + (3/10)*y^7*x^2 + (4/10)*y^8*x^3')

    gf = FPS('0.3*a^3*b^5*c^2 + 0.2*a^6*b^9*c^55 + 0.5*a^5*b^346*c^34')
    assert gf.new_update(parse_expr('a = b % c')) == FPS(
        '0.3*a^1*b^5*c^2 + 0.2*a^9*b^9*c^55 + 0.5*a^6*b^346*c^34')

    gf = FPS('0.4*x^3*y^5 + 0.6*x^7*y^18') * ProdigyPGF.poisson('z', 5)
    assert gf.new_update(
        parse_expr('z = y % x')) == FPS('0.4*x^3*y^5*z^2 + 0.6*x^7*y^18*z^4')


def test_division():
    xfail("not yet implemented")
    gf = FPS('n')
    # probably simplifies 4 / 2 to a RealLitExpr
    assert gf.new_update(parse_expr('n = 4 / 2')) == FPS('n^2')
    assert gf.new_update(parse_expr('n = n / 1')) == FPS('n')
    with raises(ValueError) as e:
        gf.new_update(parse_expr('n = n / 2'))
    assert 'because it is not always an integer' in str(e)

    gf = FPS('n^2*m^6', 'n', 'm', 'o')
    assert gf.new_update(parse_expr('o = m / n')) == FPS('n^2*m^6*o^3')
    with raises(ValueError) as e:
        gf.new_update(parse_expr('o = n / m'))
    assert 'because it is not always an integer' in str(e)

    gf = ProdigyPGF.poisson('x', 3) * FPS('y*z')
    assert gf.new_update(parse_expr('x = z / y')) == FPS('x*y*z')

    gf = ProdigyPGF.poisson('x', 3) * FPS('0.4*y^3*z^12 + 0.6*y^7*z^42')
    assert gf.new_update(parse_expr('y = z/y')) == ProdigyPGF.poisson(
        'x', 3) * FPS('0.4*y^4*z^12 + 0.6*y^6*z^42')
    assert gf.new_update(
        parse_expr('x = z/y')) == FPS('0.4*y^3*z^12*x^4 + 0.6*y^7*z^42*x^6')


def test_unilateral_approximation():
    xfail("not yet implemented")
    gf = FPS('0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25 + 0.05*x^33*y^4')
    *_, res = gf.approximate_unilaterally('x', '0.9')
    assert res == FPS(
        '0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25 + 0.05*y^4')

    gf = ProdigyPGF.poisson('x', 7).set_variables('x', 'y')
    gf._function = gf._function.subs(sympy.S('x'), sympy.S('x*y'))
    *_, res = gf.approximate_unilaterally('x', '0.9')
    assert res.filter(parse_expr('x = 100'))._function == 0
    assert res.filter(parse_expr('y = 100'))._function == sympy.S(
        'y^100*(7^100*exp(-7))/100!')
    assert res.filter(parse_expr('x = 5')) == res.filter(parse_expr('y = 5'))


def test_new_update():
    xfail("not yet implemented")
    gf = FPS('x')
    res = gf.new_update(parse_expr('x = 9/2*1*2'))
    assert res == FPS('x^9')
    res = res.new_update(parse_expr('x = x*(9/2*1*2)'))
    assert res == FPS('x^81')
    assert res._function.subs(sympy.Symbol('x'), 1) == sympy.S(1)
    assert res.marginal('x', method=MarginalType.EXCLUDE) == FPS('1')
    assert res.new_update(parse_expr('x = x/9')) == FPS('x^9')
    assert res.new_update(parse_expr('x = 3')) == FPS('x^3')

    assert FPS('x').new_update(parse_expr('x = x*x'))._function.subs(
        sympy.Symbol('x'), 1) == sympy.S(1)
    assert FPS('x')._new_update_product('x', 'x', 'x')._function.subs(
        sympy.Symbol('x'), 1) == sympy.S(1)
    assert FPS('x').new_update(
        parse_expr('x = x*x'))._variables == {sympy.Symbol('x')}
