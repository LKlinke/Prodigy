import random

import pytest
from probably.pgcl.ast import Binop, BinopExpr, NatLitExpr, VarExpr
from probably.pgcl.parser import parse_expr

from prodigy.distribution.distribution import MarginalType, State
from prodigy.distribution.fast_generating_function import FPS, ProdigyPGF


def test_geometric():
    dist = ProdigyPGF.geometric("x", "1/2")
    print(dist)
    assert True


def test_conditions():
    dist = ProdigyPGF.from_expr("1/2 * x*y + 1/2*y", "x", "y")
    assert dist.filter(parse_expr("y*x < 5")) == dist


def create_random_gf(number_of_variables: int = 1, terms: int = 1):
    # This does most likely does not create a PGF!
    symbols = ["x" + str(i) for i in range(number_of_variables)]
    values = [random.randint(0, 100) for _ in range(len(symbols) * terms)]
    coeffs = [str(random.uniform(0, 1)) for _ in range(terms)]

    result = ''
    for i in range(terms):
        monomial = '1'
        for var in symbols:
            monomial = f'{monomial}*{var}^{values[i]}'
        result = f'{result}+{monomial} * {coeffs[i]}'
    return FPS(result, *symbols)


class TestDistributionInterface:
    def test_arithmetic(self):
        g = FPS("x", "x").set_parameters('p')
        h = FPS("y", "y")
        summe = g + h
        produkt = g * h
        assert summe.get_parameters() == {'p'}
        assert summe.get_variables() == {'x', 'y'}
        assert summe == FPS("x + y", "x", "y").set_parameters('p')
        assert produkt.get_parameters() == {'p'}
        assert produkt == FPS("x*y", "x", "y").set_parameters('p')

    def test_iteration(self):
        # we need to filter here because the order in FPS iteration is chaotic
        gf = FPS("(1-sqrt(1-x^2))/x").filter(parse_expr('x <= 7'))
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
                assert (prob, state) in expected_terms
                i += 1

    def test_copy(self):
        gf = create_random_gf(3, 5)
        assert gf.copy() == gf

    def test_get_probability_mass(self):
        gf = FPS("(1-sqrt(1-x^2))/x")
        assert gf.get_probability_mass() == "1"

        gf = ProdigyPGF.zero("x")
        assert gf.get_probability_mass() == "0"

        gf = ProdigyPGF.uniform("x", "3", "10")
        assert gf.get_probability_mass() == "1"

    def test_expected_value_of(self):
        # TODO this doesn't work as well as the sympy version
        gf = ProdigyPGF.zero("x")
        assert gf.get_expected_value_of("x") == "0"

        gf = ProdigyPGF.uniform("x", "3", "10")
        with pytest.raises(Exception) as e:
            gf.get_expected_value_of("x^2+y")

        gf = FPS("(1-p) + p*x", 'x')
        assert gf.get_expected_value_of('x') == "p"

        gf = ProdigyPGF.poisson("x", "p")
        assert gf.get_expected_value_of("x") == "p"

    def test_get_variables(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)
        assert len(gf.get_variables()) == variable_count, \
            f"Amount of variables does not match. Should be {variable_count}, is {len(gf.get_variables())}."

        assert all(map(lambda x: x in ["x" + str(i) for i in range(variable_count)], gf.get_variables())), \
            f"The variables do not coincide with the actual names." \
            f"Should be {['x' + str(i) for i in range(variable_count)]}, is {gf.get_variables()}."

        gf = FPS("a*x + (1-a)", "x")
        print(gf.get_variables(), gf.get_parameters())
        assert len(gf.get_variables()) == 1
        assert gf.get_variables() == {"x"}

    def test_get_parameters(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)

        gf *= FPS("p", "")
        assert len(gf.get_parameters()) == 1, \
            f"Amount of variables does not match. Should be {2}, is {len(gf.get_parameters())}."

        assert all(map(lambda x: x in {"p"}, gf.get_parameters())), \
            f"The variables do not coincide with the actual names." \
            f"Should be {{'a', 'b'}}, is {gf.get_parameters()}."

    def test_filter(self):
        gf = ProdigyPGF.zero("x", "y")
        assert gf.filter(parse_expr("x*3 < 25*y")) == gf

        # check filter on infinite GF
        gf = FPS("(1-sqrt(1-c^2))/c")
        assert gf.filter(parse_expr("c <= 5")) == FPS("c/2 + c^3/8 + c^5/16")
        assert gf.filter(parse_expr('3+2<10')) == gf

        # check filter on finite GF
        gf = FPS("1/2*x*c + 1/4 * x^2 + 1/4")
        assert gf.filter(parse_expr("x*c < 123")) == gf

        gf = FPS("0.1*x^5+0.2*x^6+0.3*x^7+0.4*x^8")
        # TODO this also results in huge complex expressions which are likely correct but cannot be simplified
        #assert gf.filter(parse_expr("x % 3 = 2")) == FPS('0.1*x^5 + 0.4*x^8')

        gf = FPS("(1-sqrt(1-c^2))/c", "c", "x", "z")
        assert gf.filter(parse_expr("x*z <= 10")) == gf

        gf = FPS("(c/2 + c^3/2) * (1-sqrt(1-x^2))/x")
        assert gf.filter(
            parse_expr("c*c <= 5")) == FPS("c/2 * (1-sqrt(1-x^2))/x")

    def test_is_zero_dist(self):
        gf = create_random_gf(4, 10)
        assert (gf == ProdigyPGF.zero(
            *gf.get_variables())) == gf.is_zero_dist()

        gf = ProdigyPGF.zero("x")
        assert (gf == ProdigyPGF.zero(
            *gf.get_variables())) == gf.is_zero_dist()

    def test_is_finite(self):
        gf = create_random_gf(10, 10)
        assert gf.is_finite()
        assert not (gf * ProdigyPGF.poisson('x', 5)).is_finite()
        assert (gf * ProdigyPGF.poisson('x', 5) /
                ProdigyPGF.poisson('x', 5)).is_finite()

        gf = FPS("(1-sqrt(1-c^2))/c")
        assert not gf.is_finite()

        gf = FPS("1", "x")
        assert gf.is_finite()

    def test_update(self):
        gf = FPS("(1-sqrt(1-c^2))/c")
        assert gf.update(BinopExpr(Binop.EQ, VarExpr('c'), BinopExpr(Binop.PLUS, VarExpr('c'), NatLitExpr(1)))) == \
               FPS("c*(1-sqrt(1-c^2))/c")

        gf = ProdigyPGF.zero("x")
        expr = BinopExpr(
            Binop.EQ, VarExpr('x'),
            BinopExpr(Binop.PLUS,
                      BinopExpr(Binop.TIMES, VarExpr('x'), NatLitExpr(5)),
                      NatLitExpr(1)))
        assert gf.update(expr) == ProdigyPGF.zero("x")

        gf = ProdigyPGF.uniform("x", "0", "5")
        expr = BinopExpr(Binop.EQ, VarExpr('x'),
                         BinopExpr(Binop.TIMES, VarExpr('x'), VarExpr('x')))
        assert gf.update(expr) == FPS(
            "1/6 * (1 + x + x^4 + x^9 + x^16 + x^25)")

    def test_marginal(self):
        gf = FPS('x*y')
        assert gf.marginal('x', method=MarginalType.EXCLUDE) == FPS('y')

        gf = ProdigyPGF.uniform("x", '0', '10') * ProdigyPGF.binomial(
            'y', n='10', p='1/2')
        assert gf.marginal('x') == ProdigyPGF.uniform("x", '0', '10')
        assert gf.marginal('x', 'y') == gf
        assert gf.marginal('y') == ProdigyPGF.binomial('y', n='10', p='1/2')
        assert gf.marginal(
            'x', method=MarginalType.EXCLUDE) == ProdigyPGF.binomial('y',
                                                                     n='10',
                                                                     p='1/2')

    def test_set_variables(self):
        gf = create_random_gf(3, 5)
        gf = gf.set_variables("a", "b", "c")
        assert all([x in gf.get_variables() for x in {'a', 'b', 'c'}])
