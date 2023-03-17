from fractions import Fraction

import sympy
from probably.pgcl.parser import parse_expr
from pytest import raises

from prodigy.distribution.distribution import MarginalType
from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      SympyPGF,
                                                      _parse_to_sympy,
                                                      _sympy_symbol)


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
    assert gf.update(parse_expr("x = x + 1")) == GeneratingFunction("x^4")

    gf: GeneratingFunction = SympyPGF.poisson('x', 16)
    assert gf.update(parse_expr("x = 3 + 5 + 8")) == GeneratingFunction("x^16")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x = 5")) == SympyPGF.zero("x")
    assert gf.update(parse_expr("x = x + x")).filter(
        parse_expr("x % 2 = 1")) == SympyPGF.zero("x")
    assert gf.update(
        parse_expr("x = x + x")).filter(parse_expr("x = 30"))._function.subs(
            _sympy_symbol('x'), 1) == sympy.S("(16^15 * exp(-16)) / 15!")

    gf = GeneratingFunction("x^3 * y^5")
    assert gf.update(
        parse_expr("x = x + y")) == GeneratingFunction("x^8 * y^5")
    assert gf._update_sum('x', 'x', 'y') == GeneratingFunction("x^8 * y^5")

    assert gf.update(
        parse_expr('x = 1/3 + 1/3 + 1/3')) == GeneratingFunction('x*y^5')
    assert gf.update(
        parse_expr('x = 1/2 + 1/2')) == GeneratingFunction('x * y^5')


def test_var_assignment():
    gf = GeneratingFunction("n^5*m^42")
    assert gf.update(parse_expr("n = m"))._function.subs(
        _sympy_symbol("m"), 1) == _parse_to_sympy("n^42")
    assert gf.update(parse_expr("n = n"))._function.subs(
        _sympy_symbol("m"), 1) == _parse_to_sympy("n^5")

    gf = GeneratingFunction('x^8', 'x', 'y')
    assert gf.update(parse_expr('y = x')) == GeneratingFunction('y^8 * x^8')

    with raises(ValueError) as e:
        gf = GeneratingFunction('p*x + (1-p) * x^2', 'x')
        gf.update(parse_expr('x = p'))
    assert 'Assignment to parameters is not allowed' in str(e)

    with raises(ValueError, match='Result of update is not an integer'):
        gf = GeneratingFunction('x')
        gf.update(parse_expr('x = 0.5'))


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
        _sympy_symbol('x'),
        _sympy_symbol('x')**3)

    gf = SympyPGF.poisson('x', 3) * SympyPGF.poisson('y', 5)
    assert gf.update(parse_expr('x = 3*7')) == SympyPGF.poisson(
        'y', 5) * GeneratingFunction('x**21')
    assert gf.update(parse_expr('x = x * 3')) == GeneratingFunction(
        gf._function.subs(_sympy_symbol('x'),
                          _sympy_symbol('x')**3))
    assert gf.update(parse_expr('x = y * 4')) == GeneratingFunction(
        gf._function.subs([(_sympy_symbol('x'), 1),
                           (_sympy_symbol('y'), _parse_to_sympy('y * x**4'))]))

    gf = GeneratingFunction('x*y**4')
    assert gf.update(
        parse_expr('x = 0.5*y')) == GeneratingFunction('x**2*y**4')

    gf = SympyPGF.poisson(
        'x', 5) * GeneratingFunction('0.6*y**3*z**5 + 0.4*y**6*z**6')
    assert gf.update(parse_expr('x = z*y')) == GeneratingFunction(
        '0.6*x**15*y**3*z**5 + 0.4*x**36*y**6*z**6')
    assert gf.update(parse_expr('z = x*y'))._function == _parse_to_sympy(
        "0.6*y**3*exp(5 * (x*z**3-1)) + 0.4*y**6*exp(5*(x*z**6-1))")
    
    gf = GeneratingFunction('x**6')
    assert gf.update(parse_expr('x = x * (1/3)')) == GeneratingFunction('x**2')
    assert gf.update(parse_expr('x = (1/2) * x')) == GeneratingFunction('x**3')
    with raises(ValueError):
        gf.update(parse_expr('x = x*(1/4)'))
    assert gf.update(parse_expr('x = 6 * (1/3)')) == GeneratingFunction('x^2')


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
    with raises(ValueError, match='because it can be negative'):
        gf.update(parse_expr("x = x - 2"))
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

    gf = GeneratingFunction('x^4')
    assert gf.update(parse_expr('x = x - 2.0')) == GeneratingFunction('x^2')
    assert gf.update(parse_expr('x = x - (2/1)')) == GeneratingFunction('x^2')
    with raises(ValueError):
        gf.update(parse_expr('x = x - (1/2)'))
    assert gf.update(parse_expr('x = 9/6 - 1/2')) == GeneratingFunction('x')

    with raises(ValueError, match='Intermediate result 5 - 7 is negative'):
        gf.update(parse_expr('x = 5 - 7 + 3'))


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
    assert gf.update(parse_expr('x = y % (3+2)'))._function == _parse_to_sympy(
        '(3/10)*y^4*x^4 + (3/10)*y^7*x^2 + (4/10)*y^8*x^3')

    gf = GeneratingFunction(
        '0.3*a**3*b**5*c**2 + 0.2*a**6*b**9*c**55 + 0.5*a**5*b**346*c**34')
    assert gf.update(parse_expr('a = b % c')) == GeneratingFunction(
        '0.3*a**1*b**5*c**2 + 0.2*a**9*b**9*c**55 + 0.5*a**6*b**346*c**34')

    gf = GeneratingFunction(
        '0.4*x**3*y**5 + 0.6*x**7*y**18') * SympyPGF.poisson('z', 5)
    assert gf.update(parse_expr('z = y % x')) == GeneratingFunction(
        '0.4*x**3*y**5*z**2 + 0.6*x**7*y**18*z**4')

    gf = GeneratingFunction('x^5')
    assert gf.update(parse_expr('x = (1/2) % (1/2)')) == GeneratingFunction(
        '1', 'x')
    assert gf.update(parse_expr('x = x % (25/3)')) == gf
    assert gf.update(parse_expr('x = x % (1/2)')) == GeneratingFunction(
        '1', 'x')
    with raises(
            ValueError,
            match=
            "Modulo of fraction and integer / variable doesn't result in an integer"
    ):
        gf.update(parse_expr('x = (2/3) % x'))
    with raises(ValueError, match='is not an integer'):
        gf.update(parse_expr('x = (2/3) % 4'))
    with raises(ValueError, match='is not an integer'):
        gf._update_modulo_with_fraction('x', Fraction(2, 3), 4, None)

    gf = SympyPGF.poisson('x', 3) * GeneratingFunction('y^5')
    assert gf.update(parse_expr('y = y % (25/3)')) == gf

    gf = GeneratingFunction('x^9')
    assert gf.update(parse_expr('x = x % (7/2)')) == GeneratingFunction('x^2')


def test_division():
    gf = GeneratingFunction('n')
    # probably simplifies 4 / 2 to a RealLitExpr
    assert gf.update(parse_expr('n = 4 / 2')) == GeneratingFunction('n^2')
    assert gf.update(parse_expr('n = n / 1')) == GeneratingFunction('n')
    with raises(ValueError) as e:
        gf.update(parse_expr('n = n / 2'))
    assert 'because it is not always an integer' in str(e)

    gf = GeneratingFunction('n^2*m^6', 'n', 'm', 'o')
    assert gf.update(
        parse_expr('o = m / n')) == GeneratingFunction('n^2*m^6*o^3')
    with raises(ValueError) as e:
        gf.update(parse_expr('o = n / m'))
    assert 'because it is not always an integer' in str(e)

    gf = SympyPGF.poisson('x', 3) * GeneratingFunction('y*z')
    assert gf.update(parse_expr('x = z / y')) == GeneratingFunction('x*y*z')

    gf = SympyPGF.poisson(
        'x', 3) * GeneratingFunction('0.4*y**3*z**12 + 0.6*y**7*z**42')
    assert gf.update(parse_expr('y = z/y')) == SympyPGF.poisson(
        'x', 3) * GeneratingFunction('0.4*y**4*z**12 + 0.6*y**6*z**42')
    assert gf.update(parse_expr('x = z/y')) == GeneratingFunction(
        '0.4*y**3*z**12*x**4 + 0.6*y**7*z**42*x**6')

    gf = GeneratingFunction('x^5')
    assert gf.update(parse_expr('x = (5/3)/(5/3)')) == GeneratingFunction('x')
    assert gf.update(parse_expr('x = x / (1/5)')) == GeneratingFunction('x^25')
    with raises(ValueError, match='not always an integer'):
        gf.update(parse_expr('x = x / (2/5)'))
    gf = GeneratingFunction('x^4')
    assert gf.update(parse_expr('x = x / (2/5)')) == GeneratingFunction('x^10')
    assert gf.update(parse_expr('x = (20/5) / x')) == GeneratingFunction('x')
    gf = SympyPGF.geometric('x', '0.5')
    with raises(ZeroDivisionError):
        gf.update(parse_expr('x = (1/2)/x'))


def test_unilateral_approximation():
    gf = GeneratingFunction(
        '0.7*x**3*y**5 + 0.1*x**13*y**17 + 0.15*x**21*y**25 + 0.05*x**33*y**4')
    assert gf.approximate_unilaterally('x', '0.9') == GeneratingFunction(
        '0.7*x**3*y**5 + 0.1*x**13*y**17 + 0.15*x**21*y**25')

    gf = SympyPGF.poisson('x', 7).set_variables('x', 'y')
    gf._function = gf._function.subs(_sympy_symbol('x'),
                                     _parse_to_sympy('x*y'))
    assert gf.filter(parse_expr('x = 100'))._function != 0
    res = gf.approximate_unilaterally('x', '0.9')
    assert res.filter(parse_expr('x = 100'))._function == 0
    assert res.filter(parse_expr('x = 5')) == res.filter(parse_expr('y = 5'))

    gf = SympyPGF.poisson('x', 7) * SympyPGF.poisson('y', 81)
    res = gf.approximate_unilaterally('x', 0.9)
    assert res.filter(parse_expr('x = 4')) == gf.filter(parse_expr('x = 4'))
    assert res.filter(parse_expr('x = 400'))._function == 0
    assert gf.filter(parse_expr('x = 400'))._function != 0


def test_update():
    gf = GeneratingFunction('x')
    res = gf.update(parse_expr('x = 9/2*1*2'))
    assert res == GeneratingFunction('x**9')
    res = res.update(parse_expr('x = x*(9/2*1*2)'))
    assert res == GeneratingFunction('x**81')
    assert res._function.subs(_sympy_symbol('x'), 1) == sympy.S(1)
    assert res.marginal('x',
                        method=MarginalType.EXCLUDE) == GeneratingFunction('1')
    assert res.update(parse_expr('x = x/9')) == GeneratingFunction('x**9')
    assert res.update(parse_expr('x = 3')) == GeneratingFunction('x**3')

    assert GeneratingFunction('x').update(
        parse_expr('x = x*x'))._function.subs(_sympy_symbol('x'),
                                              1) == sympy.S(1)
    assert GeneratingFunction('x')._update_product('x', 'x', 'x',
                                                   None)._function.subs(
                                                       _sympy_symbol('x'),
                                                       1) == sympy.S(1)
    assert GeneratingFunction('x').update(
        parse_expr('x = x*x'))._variables == {_sympy_symbol('x')}

    gf = GeneratingFunction('x').set_parameters('p')
    assert gf.update(parse_expr('x = 4')).get_parameters() == {'p'}


def test_power():
    gf = GeneratingFunction('x^3*y^5*z^9') * SympyPGF.poisson('a', 3)
    assert gf.update(
        parse_expr('z=y^x')
    ) == GeneratingFunction("x^3*y^5*z^125") * SympyPGF.poisson('a', 3)
    assert gf.update(
        parse_expr('y=3^5')
    ) == GeneratingFunction("x^3*y^243*z^9") * SympyPGF.poisson('a', 3)
    assert gf.update(
        parse_expr('z=x^5')
    ) == GeneratingFunction("x^3*y^5*z^243") * SympyPGF.poisson('a', 3)
    assert gf.update(
        parse_expr('z=5^y')
    ) == GeneratingFunction("x^3*y^5*z^3125") * SympyPGF.poisson('a', 3)
    assert gf.update(
        parse_expr('a=3^5')) == GeneratingFunction("x^3*y^5*z^9*a^243")

    gf = GeneratingFunction('1/3*x^4+1/3*x^9+1/3*x^16')
    assert gf.update(parse_expr('x = x^(1/2)')) == GeneratingFunction(
        '1/3*x^2+1/3*x^3+1/3*x^4')
    assert gf.update(parse_expr('x = 125^(1/3)')) == GeneratingFunction('x^5')
    gf = GeneratingFunction('x^125')
    assert gf.update(parse_expr('x = x^(1/3)')) == GeneratingFunction('x^5')
