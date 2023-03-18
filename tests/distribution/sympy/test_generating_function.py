import random

import pytest
import sympy
from probably.pgcl import Binop, BinopExpr, NatLitExpr, VarExpr
from probably.pgcl.parser import parse_expr

from prodigy.distribution.distribution import MarginalType, State
from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      SympyPGF)


def create_random_gf(number_of_variables: int = 1, terms: int = 1):
    # This does most likely does not create a PGF!
    symbols = [sympy.S("x" + str(i)) for i in range(number_of_variables)]
    values = [
        sympy.S(random.randint(0, 100)) for _ in range(len(symbols) * terms)
    ]
    coeffs = [
        sympy.S(str(random.uniform(0, 1)), rational=True) for _ in range(terms)
    ]

    result = sympy.S(0)
    for i in range(terms):
        monomial = sympy.S(1)
        for var in symbols:
            monomial *= var**values[i]
        result += monomial * coeffs[i]
    return GeneratingFunction(result, *symbols)


class TestDistributionInterface:
    def test_arithmetic(self):
        g = GeneratingFunction("x", "x").set_parameters('p')
        h = GeneratingFunction("y", "y")
        summe = g + h
        produkt = g * h
        assert summe.get_parameters() == {'p'}
        assert summe == GeneratingFunction("x + y", "x",
                                           "y").set_parameters('p')
        assert produkt.get_parameters() == {'p'}
        assert produkt == GeneratingFunction("x*y", "x",
                                             "y").set_parameters('p')

        f = GeneratingFunction("x*y", "y")
        with pytest.raises(ArithmeticError):
            f + g

    def test_finite_leq(self):
        gf1 = GeneratingFunction("x**2*y**3")
        gf2 = GeneratingFunction("1/2*x**2*y**3")
        assert gf2 <= gf1

    def test_infinite_leq(self):
        gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
        gf2 = GeneratingFunction("2/(2-x)-1")
        with pytest.raises(RuntimeError):
            assert gf1 <= gf2

    def test_finite_le(self):
        gf1 = GeneratingFunction("x**2*y**3")
        gf2 = GeneratingFunction("1/2*x**2*y**3")
        assert gf2 < gf1

        gf1 = GeneratingFunction("(1/2) * x**2*y**3 + (1/2) * x**3*y**2")
        gf2 = GeneratingFunction("(1/3) * x**2*y**3 + (2/3) * x**3*y**2")
        assert not gf1 < gf2

    def test_infinite_le(self):
        gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
        gf2 = GeneratingFunction("2/(2-x) - 1")
        with pytest.raises(RuntimeError):
            assert gf1 < gf2

    def test_equality_param_fail(self):
        gf1 = GeneratingFunction("y*x", "x")
        gf2 = GeneratingFunction("x*y", "y", "x")
        assert gf1 != gf2

    def test_equality_variable_fail(self):
        gf1 = GeneratingFunction("y*x")
        gf2 = GeneratingFunction("x*y")
        gf1 = gf1.set_variables("x").set_parameters()
        gf2 = gf2.set_variables("y").set_parameters()
        assert gf1 != gf2

    def test_iteration(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        expected_terms = [("1/2", {
            "x": 1
        }), ("1/8", State({"x": 3})), ("1/16", {
            "x": 5
        }), ("5/128", {
            "x": 7
        }), ("7/256", {
            "x": 9
        }), ("21/1024", {
            "x": 11
        }), ("33/2048", {
            "x": 13
        }), ("429/32768", {
            "x": 15
        }), ("715/65536", {
            "x": 17
        }), ("2431/262144", {
            "x": 19
        })]
        i = 0
        for prob, state in gf:
            if i >= 4:
                break
            if prob == "0":
                continue
            else:
                assert (prob, state) == expected_terms[i]
                i += 1

    def test_copy(self):
        gf = create_random_gf(3, 5)
        assert gf.copy() == gf

    def test_get_probability_of(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_of(parse_expr("x <= 3")) == "5/8"

        gf = SympyPGF.zero("z", "y")
        assert gf.get_probability_of(parse_expr("not (z*y < 12)")) == "0"

    def test_get_probability_mass(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_mass() == "1"

        gf = SympyPGF.zero("x")
        assert gf.get_probability_mass() == "0"

        gf = SympyPGF.uniform("x", "3", "10")
        assert gf.get_probability_mass() == "1"

    def test_expected_value_of(self):
        gf: GeneratingFunction = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_expected_value_of("x") == "Infinity"

        gf = SympyPGF.zero("x")
        assert gf.get_expected_value_of("x") == "0"

        gf = SympyPGF.uniform("x", "3", "10")
        with pytest.raises(Exception) as e:
            gf.get_expected_value_of("x**2+y")
        assert "Cannot compute expected value" in str(e)

        gf = GeneratingFunction("(1-p) + p*x", 'x')
        assert gf.get_expected_value_of('x') == "p"
        assert gf.get_expected_value_of('p*x') == "p**2"

        gf = SympyPGF.poisson("x", "p")
        assert gf.get_expected_value_of("x") == "p"
        assert sympy.S(gf.get_expected_value_of("p*x")) == sympy.S("p^2")
        assert gf.get_expected_value_of("p") == "p"

        gf = GeneratingFunction("n^5")
        # Linearity breaks if intermediate results are negative.
        with pytest.raises(ValueError):
            gf.get_expected_value_of("n - 7 + 1")
        pytest.xfail(
            'this is simplified to "n" by sympy, meaning we cannot detect that the intermediate results are negative'
        )
        with pytest.raises(ValueError):
            gf.get_expected_value_of("n - 7 + 7")

    def test_normalize(self):
        assert create_random_gf().normalize().coefficient_sum() == 1

    def test_get_variables(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)
        assert len(gf.get_variables()) == variable_count, \
            f"Amount of variables does not match. Should be {variable_count}, is {len(gf.get_variables())}."

        assert all(map(lambda x: x in ["x" + str(i) for i in range(variable_count)], gf.get_variables())), \
            f"The variables do not coincide with the actual names." \
            f"Should be {['x' + str(i) for i in range(variable_count)]}, is {gf.get_variables()}."

        gf = GeneratingFunction("a*x + (1-a)", "x")
        print(gf.get_variables(), gf.get_parameters())
        assert len(gf.get_variables()) == 1
        assert gf.get_variables() == {"x"}

    def test_get_parameters(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)

        gf *= GeneratingFunction("p", "")
        assert len(gf.get_parameters()) == 1, \
            f"Amount of variables does not match. Should be {2}, is {len(gf.get_parameters())}."

        assert all(map(lambda x: x in {"p"}, gf.get_parameters())), \
            f"The variables do not coincide with the actual names." \
            f"Should be {{'a', 'b'}}, is {gf.get_parameters()}."

    def test_filter(self):
        gf = SympyPGF.zero("x", "y")
        assert gf.filter(parse_expr("x*3 < 25*y")) == gf

        # check filter on infinite GF
        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.filter(parse_expr("c <= 5")) == GeneratingFunction(
            "c/2 + c**3/8 + c**5/16")

        # check filter on finite GF
        gf = GeneratingFunction("1/2*x*c + 1/4 * x**2 + 1/4")
        assert gf.filter(parse_expr("x*c < 123")) == gf

        gf = GeneratingFunction("(1-sqrt(1-c**2))/c", "c", "x", "z")
        assert gf.filter(parse_expr("x*z <= 10")) == gf

        gf = GeneratingFunction("(c/2 + c^3/2) * (1-sqrt(1-x**2))/x")
        assert gf.filter(parse_expr("c*c <= 5")) == GeneratingFunction(
            "c/2 * (1-sqrt(1-x**2))/x")

    def test_is_zero_dist(self):
        gf = create_random_gf(4, 10)
        assert (gf == SympyPGF.zero(*gf.get_variables())) == gf.is_zero_dist()

        gf = SympyPGF.zero("x")
        assert (gf == SympyPGF.zero(*gf.get_variables())) == gf.is_zero_dist()

    def test_is_finite(self):
        gf = create_random_gf(10, 10)
        assert gf.is_finite()

        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert not gf.is_finite()

        gf = GeneratingFunction("1", "x")
        assert gf.is_finite()

    def test_update(self):
        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.update(BinopExpr(Binop.EQ, VarExpr('c'), BinopExpr(Binop.PLUS, VarExpr('c'), NatLitExpr(1)))) == \
               GeneratingFunction("c*(1-sqrt(1-c**2))/c")

        gf = SympyPGF.zero("x")
        expr = BinopExpr(
            Binop.EQ, VarExpr('x'),
            BinopExpr(Binop.PLUS,
                      BinopExpr(Binop.TIMES, VarExpr('x'), NatLitExpr(5)),
                      NatLitExpr(1)))
        assert gf.update(expr) == SympyPGF.zero("x")

        gf = SympyPGF.uniform("x", "0", "5")
        expr = BinopExpr(Binop.EQ, VarExpr('x'),
                         BinopExpr(Binop.TIMES, VarExpr('x'), VarExpr('x')))
        assert gf.update(expr) == GeneratingFunction(
            "1/6 * (1 + x + x**4 + x**9 + x**16 + x**25)")

    def test_marginal(self):
        gf = SympyPGF.uniform("x", '0', '10') * SympyPGF.binomial(
            'y', n='10', p='1/2')
        assert gf.marginal('x') == SympyPGF.uniform("x", '0', '10')
        assert gf.marginal(
            'x', method=MarginalType.EXCLUDE) == SympyPGF.binomial('y',
                                                                   n='10',
                                                                   p='1/2')
        assert gf.marginal('x', 'y') == gf

        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        with pytest.raises(ValueError) as e:
            gf.marginal('x', method=MarginalType.INCLUDE)
        assert "Unknown variable(s): {x}" in str(e)

        with pytest.raises(ValueError) as e:
            gf.marginal('x', method=MarginalType.EXCLUDE)
        assert "Unknown variable(s): {x}" in str(e)

    def test_set_variables(self):
        gf = create_random_gf(3, 5)
        gf = gf.set_variables("a", "b", "c")
        assert all([x in gf.get_variables() for x in {'a', 'b', 'c'}])

    def test_approximate(self):
        gf = GeneratingFunction("2/(2-x) - 1")
        assert list(gf.approximate("0.99"))[-1] == GeneratingFunction(
            "1/2*x + 1/4*x**2 + 1/8 * x**3 + 1/16 * x**4"
            "+ 1/32 * x**5 + 1/64 * x**6 + 1/128 * x**7")

        gf = SympyPGF.zero("x", 'y')
        for prob, state in gf.approximate(10):
            assert prob == "0" and state == dict()


def test_split_addend():
    probability = sympy.S(random.random())
    number_of_vars = random.randint(1, 10)
    values = [random.randint(1, 5000) for _ in range(number_of_vars)]
    monomial = sympy.S(1)
    for i in range(number_of_vars):
        monomial *= sympy.S("x" + str(i))**values[i]
    addend = probability * monomial
    assert GeneratingFunction._split_addend(addend) == (probability, monomial)


def test_predefined_variable_names():
    gf = GeneratingFunction('sum^3*x^5', 'sum', 'x')
    assert len(gf.get_variables()) == 2
    assert gf.marginal('sum') == GeneratingFunction('sum^3', 'sum')
    assert gf.update(parse_expr('sum = sum + 1')) == GeneratingFunction(
        'sum^4*x^5', 'sum', 'x')

    assert SympyPGF.bernoulli('sum', '1/2') == GeneratingFunction(
        '1/2+1/2*sum', 'sum')
    gf = GeneratingFunction('sum*x', 'x')
    assert len(gf.get_parameters()) == 1
    assert gf.update(parse_expr('x = 2')) == GeneratingFunction('sum*x^2', 'x')
