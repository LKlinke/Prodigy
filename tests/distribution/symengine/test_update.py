import pytest
import symengine as se
from probably.pgcl.parser import parse_expr
import sympy as sp

from prodigy.distribution import MarginalType
from prodigy.distribution.symengine_distribution import SymengineDist, SymenginePGF

# TODO
#   remove dependencies to sympy and replace equality checks by symengine (once implemented)
#   also add variables and parameters (as they are ignored for now)


def test_literal_assignment():
    gf = SymengineDist("1", "x")
    assert gf.update(parse_expr("x = 1")) == SymengineDist("x")    

    gf = SymenginePGF.poisson('x', "30")
    assert gf.update(parse_expr("x = 1")) == SymengineDist("x")

    gf = SymengineDist('n')
    assert gf.update(parse_expr("n = 2")) == SymengineDist("n^2")


def test_addition():
    gf = SymengineDist("x**3", "x")
    assert gf.update(parse_expr("x = 3 + 4 + 8")) == SymengineDist("x**15", "x")
    assert gf.update(parse_expr("x = x + x")) == SymengineDist("x**6", "x")
    assert gf.update(parse_expr("x = x + 1")) == SymengineDist("x**4", "x")

    gf = SymenginePGF.poisson('x', "16")
    assert gf.update(parse_expr("x = 3 + 5 + 8")) == SymengineDist("x**16", "x")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 5")) == SymenginePGF.undefined("x")
    # FIXME fails because of hadamard product
    #   assert gf.update(parse_expr("x = x + x")).filter(
    #   parse_expr("x % 2 = 1")) == SymenginePGF.undefined("x")
    assert sp.S(gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 30")).marginal(
            'x', method=MarginalType.EXCLUDE)._s_func).equals(sp.S(
                "(16^15 * exp(-16)) / factorial(15)"))


def test_modulo():
    gf = SymengineDist('x * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)', "x", "y")

    res = gf.update(parse_expr('x = 5 % 3'))
    expected = SymengineDist('x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)', "x", "y")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('x = 5 % (1+1+1)'))
    expected = SymengineDist('x^2 * (0.3*y^4 + 0.3*y^7 + 0.4*y^8)', "x", "y")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('x = y % (3+2)'))._s_func
    expected = se.S(
        '(3/10)*y^4*x^4 + (3/10)*y^7*x^2 + (4/10)*y^8*x^3')
    assert sp.S(res).equals(sp.S(expected))

    gf = SymengineDist('0.3*x^3*y^5*z^2 + 0.2*x^6*y^9*z^55 + 0.5*x^5*y^346*z^34', "x", "y", "z")

    res = gf.update(parse_expr('x = y % z'))
    expected = SymengineDist('0.3*x^1*y^5*z^2 + 0.2*x^9*y^9*z^55 + 0.5*x^6*y^346*z^34', "x", "y")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    gf = SymengineDist('0.4*x^3*y^5 + 0.6*x^7*y^18', "x", "y") * SymenginePGF.poisson('z', "5")
    res = gf.update(parse_expr('z = y % x'))
    expected = SymengineDist('0.4*x^3*y^5*z^2 + 0.6*x^7*y^18*z^4', "x", "y", "z")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))


def test_division():
    gf = SymengineDist('n', "n")

    res = gf.update(parse_expr('n = 4 / 2'))
    expected = SymengineDist('n^2')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('n = n / 1'))
    expected = SymengineDist('n')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    with pytest.raises(ValueError, match='Cannot assign x_0 / 2 to n because it is not always an integer'):
        gf.update(parse_expr('n = n / 2'))

    gf = SymengineDist('n^2*m^6', 'n', 'm', 'o')


    res = gf.update(parse_expr('o = m / n'))
    # FIXME somehow a ".0" is introduced and breaks equality check
    expected = SymengineDist('n^2*m^6*o^3.0')
    assert sp.S(res._s_func).simplify().equals(sp.S(expected._s_func))

    with pytest.raises(ValueError, match='Cannot assign x_0 / x_1 to o because it is not always an integer'):
        gf.update(parse_expr('o = n / m'))

    gf = SymenginePGF.poisson('x', "3") * SymengineDist('y*z')
    res = gf.update(parse_expr('x = z / y'))
    # FIXME somehow a ".0" is introduced and breaks equality check
    expected = SymengineDist('x**1.0*y*z')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    gf = SymenginePGF.poisson('x', "3") * SymengineDist('0.4*y^3*z^12 + 0.6*y^7*z^42')

    res = gf.update(parse_expr('y = z/y'))
    # FIXME somehow a ".0" is introduced and breaks equality check
    expected = SymenginePGF.poisson(
        'x', "3") * SymengineDist('0.4*y^4.0*z^12 + 0.6*y^6.0*z^42')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(
        parse_expr('x = z/y'))
    # FIXME somehow a ".0" is introduced and breaks equality check
    expected = SymengineDist('0.4*y^3*z^12*x^4.0 + 0.6*y^7*z^42*x^6.0')
    print(res)
    print(expected)
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))


def test_unilateral_approximation():
    gf = SymengineDist('0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25 + 0.05*x^33*y^4')
    assert gf.approximate_unilaterally(
        'x', '0.9') == SymengineDist('0.7*x^3*y^5 + 0.1*x^13*y^17 + 0.15*x^21*y^25')

    gf = SymenginePGF.poisson('x', "7") * SymenginePGF.poisson('y', "81")
    res = gf.approximate_unilaterally('x', 0.9)
    assert sp.S(res.filter(parse_expr('x = 4'))._s_func).equals(sp.S(gf.filter(parse_expr('x = 4'))._s_func))
    assert res.filter(parse_expr('x = 400'))._s_func == se.S('0')
    assert gf.filter(parse_expr('x = 400'))._s_func != se.S('0')

    gf = SymengineDist('exp(7*x*y - 7)')
    assert gf.filter(parse_expr('x = 100'))._s_func != se.S('0')
    res = gf.approximate_unilaterally('x', '0.9')
    assert res.filter(parse_expr('x = 100'))._s_func == se.S('0')
    assert res.filter(parse_expr('x = 5')) == res.filter(parse_expr('y = 5'))


def test_update():
    # FIXME
    #   i think this has to do with the assumptions that are not yet implemented in symengine
    # gf = SymengineDist('x')
    # res = gf.update(parse_expr('x = 9/2*1*2'))
    # expected = SymengineDist('x**9')
    # print(f"Result: {res}")
    #
    # assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    # updated_gf = res.update(parse_expr('x = x*(9/2*1*2)'))
    # print(updated_gf)
    # expected = SymengineDist('x^81')
    # assert sp.S(updated_gf._s_func).equals(sp.S(expected._s_func))

    # todo remove this line
    updated_gf = SymengineDist("x**81")

    res = updated_gf._update_var('x', '0')
    expected = SymengineDist("1")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = updated_gf.marginal('x', method=MarginalType.EXCLUDE)
    expected = SymengineDist('1')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = updated_gf.update(parse_expr('x = x/9'))
    expected = SymengineDist('x^9')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = updated_gf.update(parse_expr('x = 3'))
    expected = SymengineDist('x^3')
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    gf = SymengineDist("x")

    res = gf.update(parse_expr('x = x*x'))._update_var(
        'x', '0')
    expected = SymengineDist("1")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf._update_product('x', 'x', 'x', None)._update_var(
        'x', '0')
    assert sp.S(res._s_func).equals(sp.S("1"))

    res = gf.update(parse_expr('x = x*x'))
    assert res.get_variables() == {'x'}

    gf = SymengineDist('x').set_parameters('p')
    assert gf.update(parse_expr('x = 4')).get_parameters() == {'p'}


def test_power():
    gf = SymengineDist('x^3*y^5*z^9') * SymenginePGF.poisson('w', "3")

    res = gf.update(
        parse_expr('z=y^x'))
    expected = SymengineDist("x^3*y^5*z^125") * SymenginePGF.poisson(
            'w', "3")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('y=3^5'))
    expected = SymengineDist("x^3*y^243*z^9") * SymenginePGF.poisson(
            'w', "3")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('z=x^5'))
    expected = SymengineDist("x^3*y^5*z^243") * SymenginePGF.poisson(
            'w', "3")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('z=5^y'))
    expected = SymengineDist("x^3*y^5*z^3125") * SymenginePGF.poisson(
            'w', "3")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    res = gf.update(parse_expr('w=3^5'))
    expected = SymengineDist("x^3*y^5*z^9*w^243")
    assert sp.S(res._s_func).equals(sp.S(expected._s_func))

    with pytest.raises(ValueError):
        gf.update(parse_expr('z=w^x'))
