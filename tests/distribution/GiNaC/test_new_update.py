import pygin
import sympy
from probably.pgcl.parser import parse_expr
from pytest import raises, xfail

from prodigy.distribution.distribution import MarginalType
from prodigy.distribution.fast_generating_function import FPS, ProdigyPGF


def test_literal_assignment():
    gf = FPS("1", "x")
    assert gf.update(parse_expr("x = 1")) == FPS("x")

    gf = ProdigyPGF.poisson('x', 30)
    assert gf.update(parse_expr("x = 5")) == FPS("x^5")
    assert gf.update(parse_expr("x = 1")) == FPS("x")

    gf = FPS('n')
    assert gf.update(parse_expr("n = 2")) == FPS("n^2")


def test_addition():
    gf = FPS("x^3")
    assert gf.update(parse_expr("x = 3 + 4 + 8")) == FPS("x^15")
    assert gf.update(parse_expr("x = x + x")) == FPS("x^6")
    assert gf.update(parse_expr("x = x + 1")) == FPS("x^4")

    gf: FPS = ProdigyPGF.poisson('x', 16)
    assert gf.update(parse_expr("x = 3 + 5 + 8")) == FPS("x^16")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 5")) == ProdigyPGF.zero("x")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x % 2 = 1")) == ProdigyPGF.zero("x")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 30")).marginal(
            'x', method=MarginalType.EXCLUDE)._dist == pygin.Dist(
                "(16^15 * exp(-16)) / factorial(15)")

    gf = FPS("x^3 * y^5")
    assert gf.update(parse_expr("x = x + y")) == FPS("x^8 * y^5")


def test_var_assignment():
    gf = FPS("n^5*m^42")
    assert gf.update(parse_expr("n = m")) == FPS("n^42*m^42")
    assert gf.update(parse_expr("n = n")) == gf

    gf = FPS('x^8', 'x', 'y')
    assert gf.update(parse_expr('y = x')) == FPS('y^8 * x^8')

    with raises(ValueError) as e:
        gf = FPS('p*x + (1-p) * x^2', 'x')
        gf.update(parse_expr('x = p'))
    assert 'given rhs is a parameter' in str(e)

    with raises(ValueError) as e:
        gf = FPS('x')
        gf.update(parse_expr('x = 0.5'))
    assert 'Cannot assign the real value 0.5 to x' in str(e)


def test_multiplication():
    gf = FPS("n^5*m^42*o^8")
    assert gf.update(parse_expr("n = 3 * 8")) == FPS("n^24*m^42*o^8")
    assert gf.update(parse_expr("n = 5 * o")) == FPS("n^40*m^42*o^8")
    assert gf.update(parse_expr("n = m * o")) == FPS("n^336*m^42*o^8")
    assert gf.update(parse_expr("n = n * o")) == FPS("n^40*m^42*o^8")

    gf = FPS("0.5 * (n^1 + n^2)")
    assert gf.update(parse_expr("n = n * 3")) == FPS("0.5*n^3+0.5*n^6")
    assert gf.update(parse_expr("n = n * n")) == FPS("0.5*n^1+0.5*n^4")

    gf = ProdigyPGF.poisson('x', 3) * FPS('y^3')
    assert gf.update(
        parse_expr('x = x * y')) == FPS("exp(3 * (x^3 - 1))") * FPS('y^3')

    gf = ProdigyPGF.poisson('x', 3) * ProdigyPGF.poisson('y', 5)
    assert gf.update(
        parse_expr('x = 3*7')) == ProdigyPGF.poisson('y', 5) * FPS('x^21')
    assert gf.update(
        parse_expr('x = x * 3')
    ) == FPS("exp(3 * (x^3 - 1))") * ProdigyPGF.poisson('y', 5)
    assert gf.update(parse_expr('x = y * 4')) == FPS("exp(5 * (x^4*y - 1))")

    gf = FPS('x*y^4')
    assert gf.update(parse_expr('x = 0.5*y')) == FPS('x^2*y^4')

    gf = ProdigyPGF.poisson('x', 5) * FPS('0.6*y^3*z^5 + 0.4*y^6*z^6')
    assert gf.update(
        parse_expr('x = z*y')) == FPS('0.6*x^15*y^3*z^5 + 0.4*x^36*y^6*z^6')
    assert gf.update(parse_expr('z = x*y')) == FPS(
        '0.6*y^3*exp(5 * (x*z^3 - 1)) + 0.4*y^6*exp(5 * (x*z^6 - 1))')


def test_subtraction():
    gf = FPS("n^5*m^42*l^8")
    assert gf.update(parse_expr("m = m - 8")) == FPS("n^5*m^34*l^8")
    assert gf.update(parse_expr("l = l - n")) == FPS("n^5*m^42*l^3")
    assert gf.update(parse_expr("m = l - n")) == FPS("n^5*m^3*l^8")
    with raises(ValueError) as e:
        gf.update(parse_expr("m = n - l"))
    assert "division by zero" in str(e)

    gf = FPS('p*x', 'x')
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "division by zero" in str(e)
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 20"))
    assert "division by zero" in str(e)
    assert gf.update(parse_expr("x = x - 1")) == FPS('p', 'x')

    gf = FPS('(1/p)*x', 'x')
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "division by zero" in str(e)

    gf = ProdigyPGF.poisson('x', 3)
    with raises(ValueError) as e:
        gf.update(parse_expr("x = x - 2"))
    assert "division by zero" in str(e)

    gf = FPS('0.4*tmp^5*c^13*n^4 + 0.6*tmp^7*c^28*n^77')
    assert gf.update(parse_expr('n = n + (c-tmp)')) == FPS(
        '0.4*tmp^5*c^13*n^12 + 0.6*tmp^7*c^28*n^98')


def test_fresh_variables():
    gf = FPS('x*xl')
    assert gf.update(parse_expr('x = 2*3')) == FPS('x^6*xl')

    gf = FPS('x*sym_0')
    assert gf.update(parse_expr('x = 2*3')) == FPS('x^6*sym_0')


def test_modulo():
    gf = FPS('x * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.update(
        parse_expr('x = 5 % 3')) == FPS('x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.update(parse_expr('x = 5 % (1+1+1)')) == FPS(
        'x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)')
    assert gf.update(parse_expr('x = y % (3+2)'))._dist == pygin.Dist(
        '(3/10)*y^4*x^4 + (3/10)*y^7*x^2 + (4/10)*y^8*x^3')

    gf = FPS('0.3*x^3*y^5*z^2 + 0.2*x^6*y^9*z^55 + 0.5*x^5*y^346*z^34')
    assert gf.update(parse_expr('x = y % z')) == FPS(
        '0.3*x^1*y^5*z^2 + 0.2*x^9*y^9*z^55 + 0.5*x^6*y^346*z^34')

    gf = FPS('0.4*x^3*y^5 + 0.6*x^7*y^18') * ProdigyPGF.poisson('z', 5)
    assert gf.update(
        parse_expr('z = y % x')) == FPS('0.4*x^3*y^5*z^2 + 0.6*x^7*y^18*z^4')


def test_division():
    gf = FPS('n')
    # probably simplifies 4 / 2 to a RealLitExpr
    assert gf.update(parse_expr('n = 4 / 2')) == FPS('n^2')
    assert gf.update(parse_expr('n = n / 1')) == FPS('n')
    with raises(ValueError) as e:
        gf.update(parse_expr('n = n / 2'))
    assert 'numerator is not always divisible by denominator' in str(e)

    gf = FPS('n^2*m^6', 'n', 'm', 'o')
    assert gf.update(parse_expr('o = m / n')) == FPS('n^2*m^6*o^3')
    with raises(ValueError) as e:
        gf.update(parse_expr('o = n / m'))
    assert 'numerator is not always divisible by denominator' in str(e)

    gf = ProdigyPGF.poisson('x', 3) * FPS('y*z')
    assert gf.update(parse_expr('x = z / y')) == FPS('x*y*z')

    gf = ProdigyPGF.poisson('x', 3) * FPS('0.4*y^3*z^12 + 0.6*y^7*z^42')
    assert gf.update(parse_expr('y = z/y')) == ProdigyPGF.poisson(
        'x', 3) * FPS('0.4*y^4*z^12 + 0.6*y^6*z^42')
    assert gf.update(
        parse_expr('x = z/y')) == FPS('0.4*y^3*z^12*x^4 + 0.6*y^7*z^42*x^6')


def test_unilateral_approximation():
    gf = FPS('0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25 + 0.05*x^33*y^4')
    assert gf.approximate_unilaterally(
        'x', '0.9') == FPS('0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25')

    gf = ProdigyPGF.poisson('x', 7) * ProdigyPGF.poisson('y', 81)
    res = gf.approximate_unilaterally('x', 0.9)
    assert res.filter(parse_expr('x = 4')) == gf.filter(parse_expr('x = 4'))
    assert res.filter(parse_expr('x = 400'))._dist == pygin.Dist('0')
    assert gf.filter(parse_expr('x = 400'))._dist != pygin.Dist('0')

    gf = FPS('exp(7*x*y - 7)')
    assert gf.filter(parse_expr('x = 100'))._dist != pygin.Dist('0')
    res = gf.approximate_unilaterally('x', '0.9')
    assert res.filter(parse_expr('x = 100'))._dist == pygin.Dist('0')
    assert res.filter(parse_expr('x = 5')) == res.filter(parse_expr('y = 5'))


def test_update():
    gf = FPS('x')
    res = gf.update(parse_expr('x = 9/2*1*2'))
    assert res == FPS('x^9')
    res = res.update(parse_expr('x = x*(9/2*1*2)'))
    assert res == FPS('x^81')
    assert res._update_var('x', '0')._dist == pygin.Dist('1')
    assert res.marginal('x', method=MarginalType.EXCLUDE) == FPS('1')
    assert res.update(parse_expr('x = x/9')) == FPS('x^9')
    assert res.update(parse_expr('x = 3')) == FPS('x^3')

    assert FPS('x').update(parse_expr('x = x*x'))._update_var('x', '0')._dist == pygin.Dist('1')
    assert FPS('x')._update_product('x', 'x', 'x', None)._update_var('x', '0')._dist == pygin.Dist('1')
    assert FPS('x').update(parse_expr('x = x*x')).get_variables() == {'x'}


def test_power():
    gf = FPS('x^3*y^5*z^9') * ProdigyPGF.poisson('w', 3)
    assert gf.update(
        parse_expr('z=y^x')) == FPS("x^3*y^5*z^125") * ProdigyPGF.poisson(
            'w', 3)
    assert gf.update(
        parse_expr('y=3^5')) == FPS("x^3*y^243*z^9") * ProdigyPGF.poisson(
            'w', 3)
    assert gf.update(
        parse_expr('z=x^5')) == FPS("x^3*y^5*z^243") * ProdigyPGF.poisson(
            'w', 3)
    assert gf.update(
        parse_expr('z=5^y')) == FPS("x^3*y^5*z^3125") * ProdigyPGF.poisson(
            'w', 3)
    assert gf.update(parse_expr('w=3^5')) == FPS("x^3*y^5*z^9*w^243")

    with raises(ValueError):
        gf.update(parse_expr('z=w^x'))
