from fractions import Fraction

from itertools import islice

from prodigy.distribution.symengine_distribution import *
import pytest
import symengine as se
import random
from probably.pgcl import NatLitExpr, BoolLitExpr

import re

# TODO
#   At workarounds with sympy.equals(), add checks for variables / parameters (once symengine supports .equals())


def create_random_gf(number_of_variables: int = 1, terms: int = 1):
    # This does most likely does not create a PGF!
    symbols = ["x" + str(i) for i in range(number_of_variables)]
    values = [
        se.S(random.randint(0, 100)) for _ in range(len(symbols) * terms)
    ]
    coeffs = [
        se.S(str(random.uniform(0, 1))) for _ in range(terms)
    ]

    result = se.S(0)
    for i in range(terms):
        monomial = se.S(1)
        for var in symbols:
            monomial *= se.S(var) ** values[i]
        result += monomial * coeffs[i]
    return SymengineDist(str(result), *symbols)


class TestDistributionInterface:
    def test_addition(self):
        # Two Symengine Dists
        g = SymengineDist("x", "x").set_parameters('p')
        h = SymengineDist("y", "y")
        gf_sum = g + h
        assert gf_sum.get_parameters() == {'p'}
        assert gf_sum == SymengineDist("x + y", "x", "y").set_parameters("p")

        # SymengineDist + int
        gf_sum = g + 2
        assert gf_sum == SymengineDist("x + 2").set_parameters("p")

        # SymengineDist + float
        gf_sum = g + 2.5
        assert gf_sum == SymengineDist("x + 2.5").set_parameters("p")

        # SymengineDist + string
        h = "x"
        gf_sum = g + h
        assert gf_sum == SymengineDist("2 * x").set_parameters("p")

        # No other types allowed
        with pytest.raises(SyntaxError, match=re.escape("You cannot add")):
            g + []

    def test_product(self):
        # Two SymengineDists
        g = SymengineDist("x", "x").set_parameters('p')
        h = SymengineDist("y", "y")
        gf_prod = g * h
        assert gf_prod.get_parameters() == {'p'}
        assert gf_prod == SymengineDist("x * y", "x", "y").set_parameters("p")

        g = SymengineDist("x")

        # SymengineDist * int
        assert g * 0 == SymenginePGF.undefined("x")

        # SymengineDist * float
        # TODO ".0" is addes and breaks equality
        assert g * 0.0 == SymenginePGF.from_expr("0.0", "x")

        # SymengineDist * string
        assert g * "0" == SymenginePGF.undefined("x")

        # No other types allowed
        with pytest.raises(SyntaxError, match=re.escape("You cannot multiply")):
            g * []

    def test_subtraction(self):
        # Two SymengineDists
        g = SymengineDist("x", "x")
        h = SymengineDist("y", "y")
        assert g - h == SymengineDist("x - y", "x", "y")

        # x - x should be 0 (with variable "x")
        h = SymengineDist("x", "x")
        assert g - h == SymenginePGF.undefined("x")

        g = SymengineDist("x + 2", "x")
        res = SymengineDist("x", "x")

        # SymengineDist - int
        assert g - 2 == res

        # SymengineDist - float
        assert g - 2.0 == res

        # SymengineDist - string
        assert g - "2" == res

        # No other types allowed
        with pytest.raises(SyntaxError, match=re.escape("You cannot subtract")):
            g - []

    def test_division(self):
        # Two SymengineDists
        g = SymengineDist("x**2", "x")
        h = SymengineDist("x", "x")
        assert g / h == h

        # Division by zero results in se.zoo
        # TODO is this intended? Should a div by zero check be implemented?
        h = SymenginePGF.undefined("x")
        assert g / h == SymenginePGF.from_expr("zoo", "x")

        g = SymengineDist("2 * x", "x")
        res = SymengineDist("x", "x")

        # SymengineDist divided by int
        assert g / 2 == res

        # SymengineDist divided by float
        # TODO should one use .simplify() to achieve g / 2.0 == res?
        assert g / 2.0 == SymenginePGF.from_expr("1.0*x", "x")

        # SymengineDist divided by string
        assert g / "2" == res

        # No other types allowed
        with pytest.raises(SyntaxError, match=re.escape("You cannot divide")):
            g / []

    def test_conflict_parameter_variable(self):
        g = SymengineDist("x", "x")
        h = SymengineDist("y", "y").set_parameters("x")
        with pytest.raises(SyntaxError, match=re.escape("Name clash: {x} for x and y.")):
            g + h

    def test_arithmetic_prelims(self):
        # When applying an operator to a SymengineDist and a string where the string contains a symbol
        # that is known as a parameter in the Dist, the symbol should be interpreted
        # as a parameter in the result
        g = SymengineDist("x", "x").set_parameters('p')
        h = "p"
        res = SymengineDist("x + p", "x").set_parameters("p")

        assert g + h == res

    def test_le_fail(self):
        gf = create_random_gf(3, 5)
        with pytest.raises(TypeError, match=re.escape("Incomparable types")):
            assert gf < 1

    def test_finite_leq(self):
        gf1 = SymengineDist("x**2*y**3")
        gf2 = SymengineDist("1/2*x**2*y**3")
        assert gf2 <= gf1

    def test_infinite_leq(self):
        gf1 = SymengineDist("(1-sqrt(1-x**2))/x")
        gf2 = SymengineDist("2/(2-x)-1")
        with pytest.raises(RuntimeError):
            assert gf1 <= gf2

    def test_finite_le(self):
        gf1 = SymengineDist("x**2*y**3")
        gf2 = SymengineDist("1/2*x**2*y**3")
        assert gf2 < gf1

        gf1 = SymengineDist("(1/2) * x**2*y**3 + (1/2) * x**3*y**2")
        gf2 = SymengineDist("(1/3) * x**2*y**3 + (2/3) * x**3*y**2")
        assert not gf1 < gf2

    def test_infinite_le(self):
        gf1 = SymengineDist("(1-sqrt(1-x**2))/x")
        gf2 = SymengineDist("2/(2-x) - 1")
        with pytest.raises(RuntimeError):
            assert gf1 < gf2

    def test_equality(self):
        gf = create_random_gf(3,5)
        assert gf == gf

        # Different variables
        gf1 = SymengineDist("x", "x")
        gf2 = SymengineDist("x", "x", "y")
        assert gf1 != gf2

        # Different parameters
        gf1 = SymengineDist("x", "x").set_parameters("p")
        gf2 = SymengineDist("x", "x").set_parameters("q")
        assert gf1 != gf2

        # Different types
        assert gf != 1

    def test_variable_parameter_clash(self):
        gf = SymengineDist("y*x")
        with pytest.raises(ValueError, match=re.escape("There are unknown symbols which are neither variables nor parameters")):
            gf.set_variables("x")

        with pytest.raises(ValueError, match=re.escape("At least one parameter is already known as a variable")):
            gf.set_parameters("x")

    def test_iteration(self):
        gf = SymengineDist("(1-sqrt(1-x**2))/x")
        expected_terms = [
            ("1/2", State({"x": 1})),
            ("1/8", State({"x": 3})),
            ("1/16", State({"x": 5})),
            ("5/128", State({"x": 7})),
            ("7/256", State({"x": 9})),
            ("21/1024", State({"x": 11})),
            ("33/2048", State({"x": 13})),
            ("429/32768", State({"x": 15})),
            ("715/65536", State({"x": 17})),
            ("2431/262144", State({"x": 19}))
        ]
        i = 0
        for prob, state in gf:
            if i >= 4:
                break
            if prob == "0":
                continue
            else:
                assert (prob, state) == expected_terms[i]
                i += 1


        gf = SymengineDist("x + (1/2)*x**2 + (1/8)*x**3")
        expected_terms = [
            ("1", State({"x": 1})),
            ("1/2", State({"x": 2})),
            ("1/8", State({"x": 3}))
        ]
        i = 0
        for prob, state in gf:
            if i > 3:
                break
            if prob == "0":
                continue
            # Sometimes a different order is returned
            expected_prob = next((p for p, s in expected_terms if s == state), None)
            assert prob == expected_prob
            i += 1

    def test_iteration_multivariate(self):
        pass    # TODO


    def test_iteration_finite(self):
        gf = SymengineDist("(1/2) * x**2 + (1/2) * x**3")
        expected_terms = [
            ("1/2", State({"x": 2})),
            ("1/2", State({"x": 3}))
        ]
        assert list(iter(gf)) == expected_terms

    def test_iter_with(self):
        # Infinite distributions
        # When using the default monomial iterator, the results should be the same
        gf = SymengineDist("(1-sqrt(1-x**2))/x")
        i = 10
        iterator = list(islice(gf, i))
        iterator_with = list(islice(gf.iter_with(default_monomial_iterator(len(gf._variables))), i))
        assert iterator == iterator_with

        # "Reversed" iteration
        def custom_iterator() -> Iterator[List[int]]:
            count = i - 1
            while count >= 0:
                yield [count]
                count -= 1

        iterator_with = list(gf.iter_with(custom_iterator()))
        assert iterator == list(reversed(iterator_with))

        # Finite distributions
        gf = SymengineDist("x + x**2", "x")
        iterator = list(islice(gf, i))
        iterator_with = list(islice(gf.iter_with(default_monomial_iterator(len(gf._variables))), i))
        assert iterator == iterator_with

    def test_get_prob_by_diff(self):
        gf = SymenginePGF.geometric("x", "1/2")
        for i in range(10):
            assert gf.get_prob_by_diff(State({"x": i})) == se.S(f"(1/2)**({i}+1)")

    def test_get_prob_by_series(self):
        gf = SymenginePGF.geometric("x", "1/2")
        for i in range(10):
            assert gf.get_prob_by_series(State({"x": i})) == se.S(f"(1/2)**({i}+1)")

    def test_copy(self):
        gf = create_random_gf(3, 5)
        assert gf.copy() == gf

    def test_get_probability_of(self):
        gf = SymengineDist("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_of(parse_expr("x <= 3")) == "5/8"

        gf = SymenginePGF.undefined("z", "y")
        assert gf.get_probability_of(parse_expr("not (z*y < 12)")) == "0"

    def test_get_probability_mass(self):
        gf = SymengineDist("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_mass() == "1"

        gf = SymenginePGF.undefined("x")
        assert gf.get_probability_mass() == "0"

        gf = SymenginePGF.uniform("x", "3", "10")
        assert gf.get_probability_mass() == "1"

    def test_expected_value_of(self):
        gf = SymengineDist("(1-sqrt(1-x**2))/x")
        assert gf.get_expected_value_of("x") == "Infinity"

        gf = SymenginePGF.undefined("x")
        assert gf.get_expected_value_of("x") == "0"

        gf = SymenginePGF.uniform("x", "3", "10")
        with pytest.raises(ValueError, match=re.escape("Cannot compute expected value")):
            gf.get_expected_value_of("x**2+y")

        gf = SymengineDist("(1-p) + p*x", 'x')
        assert gf.get_expected_value_of('x') == "p"
        assert gf.get_expected_value_of('p*x') == "p**2"

        gf = SymenginePGF.poisson("x", "p")
        assert gf.get_expected_value_of("x") == "p"
        assert se.S(gf.get_expected_value_of("p*x")) == se.S("p^2")
        assert gf.get_expected_value_of("p") == "p"

        gf = SymengineDist("x", "x")
        with pytest.raises(NotImplementedError, match="Expected Value only computable for polynomial expressions."):
            gf.get_expected_value_of("1 / (x + 1)")

        gf = SymengineDist("n^5")

        # Linearity breaks if intermediate results are negative.
        with pytest.raises(ValueError):
            gf.get_expected_value_of("n - 7 + 1")
        pytest.xfail(
            'this is simplified to "n" by se, meaning we cannot detect that the intermediate results are negative'
        )

    def test_normalize(self):
        assert create_random_gf().normalize().coefficient_sum() == 1

        gf = SymengineDist("1 - x", "x")
        with pytest.raises(ZeroDivisionError):
            gf.normalize()

    def test_get_variables(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)
        assert len(gf.get_variables()) == variable_count, \
            f"Amount of variables does not match. Should be {variable_count}, is {len(gf.get_variables())}."

        assert all(map(lambda x: x in ["x" + str(i) for i in range(variable_count)], gf.get_variables())), \
            f"The variables do not coincide with the actual names." \
            f"Should be {['x' + str(i) for i in range(variable_count)]}, is {gf.get_variables()}."

        gf = SymengineDist("a*x + (1-a)", "x")
        assert len(gf.get_variables()) == 1
        assert gf.get_variables() == {"x"}

    def test_get_parameters(self):
        variable_count = random.randint(1, 10)
        gf = create_random_gf(variable_count, 5)
        gf *= SymengineDist("p", "")
        assert len(gf.get_parameters()) == 1, \
            f"Amount of parameters does not match. Should be {1}, is {len(gf.get_parameters())}."

        assert all(map(lambda x: x in {"p"}, gf.get_parameters())), \
            f"The parameters do not coincide with the actual names." \
            f"Should be {{'a', 'b'}}, is {gf.get_parameters()}."

    def test_filter(self):
        gf = SymenginePGF.undefined("x", "y")
        assert gf.filter(parse_expr("x*3 < 25*y")) == gf

        # check filter on infinite GF
        gf = SymengineDist("(1-sqrt(1-c**2))/c")
        assert gf.filter(parse_expr("c <= 5")) == SymengineDist(
            "c/2 + c**3/8 + c**5/16")

        # check filter on finite GF
        gf = SymengineDist("1/2*x*c + 1/4 * x**2 + 1/4")
        assert gf.filter(parse_expr("x*c < 123")) == gf

        gf = SymengineDist("(1-sqrt(1-c**2))/c", "c", "x", "z")
        assert gf.filter(parse_expr("x*z <= 10")) == gf

        gf = SymengineDist("(c/2 + c^3/2) * (1-sqrt(1-x**2))/x")

        # TODO how to check for equality with symengine?
        #   cf. https://github.com/symengine/symengine.py/issues/496
        assert sp.S(str(gf.filter(parse_expr("c*c <= 5"))._s_func)).equals(sp.S("c/2 * (1-sqrt(1-x**2))/x"))

        # todo "m>0" from sequential_loops_second_inv.pgcl

    def test_is_zero_dist(self):
        gf = create_random_gf(4, 10)
        # Sometimes Symengine is not sure whether it's a zero dist
        assert (gf == SymenginePGF.undefined(*gf.get_variables())) == gf.is_zero_dist() or gf.is_zero_dist() is None

        gf = SymenginePGF.undefined("x")
        assert (gf == SymenginePGF.undefined(*gf.get_variables())) == gf.is_zero_dist()

        gf = SymenginePGF.from_expr("x - x", "x")
        assert (gf == SymenginePGF.undefined(*gf.get_variables())) == gf.is_zero_dist()

        gf = SymenginePGF.one("")
        assert not gf.is_zero_dist()

    def test_is_finite(self):
        gf = create_random_gf(10, 10)
        assert gf.is_finite()

        gf = SymengineDist("(1-sqrt(1-c**2))/c")
        assert not gf.is_finite()

        gf = SymenginePGF.one("")
        assert gf.is_finite()

        gf = SymengineDist("(1/6)*(-1 + x_1**6)/(-1 + x_1)")
        assert gf.is_finite()

    def test_get_fresh_variables(self):
        gf = SymengineDist("x_0 * x_1")
        assert "x_2" == gf.get_fresh_variable()
        assert "x_3" == gf.get_fresh_variable(exclude={"x_2"})

    def test_update_sum(self):
        # Two variables are added
        gf = SymengineDist("x*y*z")
        assert gf._update_sum("x", "x", "z") == SymengineDist("x**2 * y * z")
        assert gf._update_sum("x", "x", "x") == SymengineDist("x**2 * y * z")

        assert gf._update_sum("x", "y", "x") == SymengineDist("x**2 * y * z")
        assert gf._update_sum("x", "y", "z") == SymengineDist("x**2 * y * z")

        gf = SymengineDist("x*y")
        assert gf._update_sum("x", "y", 1) == SymengineDist("x**2 * y")
        assert gf._update_sum("x", "x", 5) == SymengineDist("x**6 * y")
        assert gf._update_sum("x", 5, "x") == SymengineDist("x**6 * y")

        gf = SymengineDist("x")
        assert gf._update_sum("x", 1, 1) == SymengineDist("x ** 2")

    def test_update_var(self):
        # Parameter assignment is not allowed
        gf = SymengineDist("x").set_parameters("y")
        with pytest.raises(ValueError, match=re.escape("Assignment to parameters is not allowed")):
            gf._update_var("x", "y")

        # Variables should all be known
        gf = SymengineDist("x")
        with pytest.raises(ValueError, match=re.escape("Unknown symbol: y")):
            gf._update_var("x", "y")

        # _update_var with assign_var is integer
        gf = SymengineDist("x")
        assert gf._update_var("x", 5) == SymengineDist("x ** 5")

        # _update_var with updated_var == assign_var
        assert gf._update_var("x", "x") == gf

        # _update_var with updated_var != assign_var, assign_var is symbol
        gf = SymengineDist("x * y")
        assert gf._update_var("x", "y") == SymengineDist("x*y")

    def test_update_product(self):
        gf = SymengineDist("x", "x").set_parameters("p")

        with pytest.raises(ValueError, match=re.escape("Assignment of parameters is not allowed")):
            gf._update_product("x", "x", "p", approximate=None)

        gf = SymengineDist("x / (x*y - 1)", "x", "y")
        with pytest.raises(ValueError, match=re.escape("Cannot perform the multiplication x * y because both variables have infinite range")):
            gf._update_product("x", "x", "y", approximate=None)

        # TODO with approximate = True

        gf = SymengineDist("x + y", "x", "y")

        # Two variables
        assert gf._update_product("x", "x", "y", approximate=None) == SymengineDist("1 + y", "x", "y")

        # One variable and one literal
        assert gf._update_product("x", "x", "2", approximate=None) == SymengineDist("x**2 + y", "x", "y")
        assert gf._update_product("x", "2", "x", approximate=None) == SymengineDist("x**2 + y", "x", "y")
        assert gf._update_product("x", "y", "2", approximate=None) == SymengineDist("1 + x**2*y", "x", "y")

        # Two literals
        assert gf._update_product("x", "2", "2", approximate=None) == SymengineDist("x**4*(1 + y)", "x", "y")

    def test_update_subtraction(self):
        gf = SymengineDist("x + y", "x", "y")

        # Two variables
        assert gf._update_subtraction("x", "x", "x") == SymengineDist("1 + y", "x", "y")
        assert gf._update_subtraction("x", "x", "y") == SymengineDist("x + y/x", "x", "y")
        assert gf._update_subtraction("x", "y", "x") == SymengineDist("x*y + 1/x", "x", "y")
        assert gf._update_subtraction("x", "y", "y") == SymengineDist("1 + y", "x", "y")

        # Variable - literal
        assert gf._update_subtraction("x", "x", 1) == SymengineDist("1 + y/x", "x", "y")
        assert gf._update_subtraction("x", "y", 1) == SymengineDist("y + 1/x", "x", "y")

        # Literal - variable
        assert gf._update_subtraction("x", 1, "x") == SymengineDist("1 + x*y", "x", "y")
        assert gf._update_subtraction("x", 1, "y") == SymengineDist("x + y", "x", "y")

        # Literal - literal
        assert gf._update_subtraction("x", 2, 1) == SymengineDist("x + x*y", "x", "y")

        with pytest.raises(ValueError, match=re.escape("Cannot assign '1 - 2' to 'x' because it is negative")):
            gf._update_subtraction("x", 1, 2)

        # TODO test ValueError

    def test_update_division(self):
        gf = SymengineDist("x", "x").set_parameters("p")

        with pytest.raises(ValueError, match=re.escape("Division containing parameters is not allowed")):
            gf._update_division("x", "x", "p", approximate=None)

        gf = SymengineDist("x * y", "x", "y")

        # FIXME somehow a ".0" is introduced and breaks equality check
        assert gf._update_division("x", "x", "y", approximate=None) == SymengineDist("x**1.0*y", "x", "y")

        with pytest.raises(ValueError, match=re.escape("Cannot assign x / 2 to x because it is not always an integer")):
            gf._update_division("x", "x", 2, approximate=None)

        assert gf._update_division("x", 2, "y", approximate=None) == SymengineDist("x**2*y", "x", "y")

        assert gf._update_division("x", 2, 2, approximate=None) == gf

        with pytest.raises(ValueError, match=re.escape("Cannot assign 1 / 2 to x because it is not always an integer")):
            gf._update_division("x", 1, 2, approximate=None)


    def test_safe_subs(self):
        gf = SymengineDist("x*y")

        # Illegal number of arguments
        with pytest.raises(ValueError, match=re.escape("There has to be an equal amount of variables and values")):
            gf.safe_subs("x", 1, "y")

        # safe_subs should behave exactly the same as subs
        assert (gf.safe_subs("x", 1, "y", "x") == gf.safe_subs(("x", 1), ("y", "x"))
                == gf._s_func.subs("x", 1).subs("y", "x"))

        # Direct substitution leads to nan, safe_subs should not
        gf = SymengineDist("(1/11)*(1/2 + (1/2)*y)**10*(-1 + x**11)/(-1 + x)")
        assert gf._s_func.subs("x", 1).simplify() == se.nan
        assert gf.safe_subs("x", 1) == se.S("(1/1024)*(1 + y)**10")
        # TODO add test for limit

    def test_update(self):
        gf = SymengineDist("(1-sqrt(1-c**2))/c")
        # c -> c + 1
        updated_gf = gf.update(BinopExpr(Binop.EQ, VarExpr('c'), BinopExpr(Binop.PLUS, VarExpr('c'), NatLitExpr(1))))
        assert updated_gf == SymengineDist("c*(1-sqrt(1-c**2))/c")

        gf = SymenginePGF.undefined("x")
        expr = BinopExpr(
            Binop.EQ, VarExpr('x'),
            BinopExpr(Binop.PLUS,
                      BinopExpr(Binop.TIMES, VarExpr('x'), NatLitExpr(5)),
                      NatLitExpr(1)))
        assert gf.update(expr) == SymenginePGF.undefined("x")

        gf = SymenginePGF.uniform("x", "0", "5")
        expr = BinopExpr(Binop.EQ, VarExpr('x'),
                         BinopExpr(Binop.TIMES, VarExpr('x'), VarExpr('x')))
        updated_gf = gf.update(expr)
        # TODO how to check for equality with symengine?
        #   cf. https://github.com/symengine/symengine.py/issues/496
        assert sp.S(updated_gf._s_func).equals(sp.S("1/6 * (1 + x + x**4 + x**9 + x**16 + x**25)"))

    def test_marginal(self):
        gf = SymenginePGF.uniform("x", '0', '10') * SymenginePGF.binomial(
            'y', n='10', p='1/2')
        assert gf.marginal('x') == SymenginePGF.uniform("x", '0', '10')
        # TODO how to check for equality with symengine?
        #   cf. https://github.com/symengine/symengine.py/issues/496
        assert sp.S(gf.marginal(
            'x', method=MarginalType.EXCLUDE)._s_func).equals(sp.S(SymenginePGF.binomial('y', n='10', p='1/2')._s_func))
        assert sp.S(gf.marginal('x', 'y')._s_func).equals(sp.S(gf._s_func))

        gf = SymengineDist("(1-sqrt(1-c**2))/c")
        with pytest.raises(ValueError, match=re.escape("Unknown variable(s): {x}")):
            gf.marginal('x', method=MarginalType.INCLUDE)

        with pytest.raises(ValueError, match=re.escape("No variables were provided")):
            gf.marginal()

    def test_set_variables(self):
        gf = create_random_gf(3, 5)

        gf = gf.set_variables("a", "b", "c", *gf.get_variables())
        assert all([x in gf.get_variables() for x in {'a', 'b', 'c'}])

        gf = SymengineDist("x * p", "x")
        with pytest.raises(ValueError, match=re.escape(f"At least one variable is already known as a parameter.")):
            gf.set_variables("p")

    def test_set_parameters(self):
        gf = SymengineDist("x*p_1*p_2", "x").set_parameters("p_1", "p_2")
        assert gf.get_parameters() == {"p_1", "p_2"}

        with pytest.raises(ValueError, match=re.escape("There are unknown symbols which are neither variables nor parameters.")):
            gf.set_parameters("p_1")

    def test_set_variables_and_parameters(self):
        gf = SymengineDist("x * p", "x").set_parameters("p")

        # Swap variables and parameters
        updated_gf = gf.set_variables_and_parameters({"p"}, {"x"})
        assert updated_gf.get_variables() == {"p"}
        assert updated_gf.get_parameters() == {"x"}

        with pytest.raises(ValueError, match=re.escape("There are unknown symbols which are neither variables nor parameters:")):
            gf.set_variables_and_parameters({"x"}, set())

    def test_approximate(self):
        gf = SymengineDist("2/(2-x) - 1")

        assert list(gf.approximate("0.99"))[-1] == SymengineDist(
            "1/2*x + 1/4*x**2 + 1/8 * x**3 + 1/16 * x**4"
            "+ 1/32 * x**5 + 1/64 * x**6 + 1/128 * x**7")

        gf = SymenginePGF.undefined("x", 'y')
        for dist in gf.approximate(10):
            assert dist.is_zero_dist()

        with pytest.raises(TypeError, match=re.escape("Parameter threshold can only be of type str or int")):
            for _ in gf.approximate(1.23):
                pass

    def test_approximate_unilaterally(self):
        pass    # TODO

    def test_arithmetic_progression(self):
        pass    # TODO

    def test_find_symbols(self):
        gf = create_random_gf(3, 5)
        assert gf._find_symbols("x ** 2 + y ** 2") == {"x", "y"}
        # Symengine does not recognize "%", test workaround
        assert gf._find_symbols("c % d") == {"c", "d"}

    def test_parse_to_symengine(self):
        from prodigy.distribution.symengine_distribution import _parse_to_symengine
        from probably.pgcl import NatLitExpr, VarExpr

        # Base case
        expr = "x**2 + y**2"
        assert _parse_to_symengine(expr) == se.S(expr)

        # Probably: NatLitExpr
        expr = NatLitExpr(1)
        assert _parse_to_symengine(expr) == se.S("1")

        # Probably: VarExpr
        expr = VarExpr("x")
        assert _parse_to_symengine(expr) == se.Symbol("x")

        # Probably: RealLitExpr
        expr = RealLitExpr(Fraction(2, 4))
        assert _parse_to_symengine(expr) == se.S("1/2")

        # Probably: BinopExpr
        expr = BinopExpr(
            lhs=VarExpr("x"),
            operator=Binop.PLUS,
            rhs=VarExpr("y")
        )
        assert _parse_to_symengine(expr) == se.S("x + y")

        # Probably: Nested BinopExpr
        expr = BinopExpr(
            lhs=BinopExpr(
                lhs=VarExpr("x"),
                operator=Binop.POWER,
                rhs=NatLitExpr(2)
            ),
            operator=Binop.DIVIDE,
            rhs=BinopExpr(
                lhs=VarExpr("y"),
                operator=Binop.MINUS,
                rhs=BinopExpr(
                    lhs=VarExpr("x"),
                    operator=Binop.TIMES,
                    rhs=RealLitExpr(Fraction(1,3))
                )
            )
        )
        assert _parse_to_symengine(expr) == se.S("(x**2) / (y - x * 1/3)")

        # Probably: Not supported operator
        expr = BinopExpr(
            lhs=VarExpr("x"),
            operator=Binop.LEQ,
            rhs=NatLitExpr(5)
        )

        with pytest.raises(ValueError, match=re.escape(f"Unsupported operand: {Binop.LEQ}")):
            _parse_to_symengine(expr)

        # Probably: Not supported type
        expr = BoolLitExpr(True)

        with pytest.raises(ValueError, match=re.escape(f"Unsupported type: {type(expr)}")):
            _parse_to_symengine(expr)



    def test_hadamard_product(self):
        gf1, gf2 = create_random_gf(3, 5), create_random_gf(3, 5)

        # lol
        with pytest.raises(NotImplementedError):
            gf1.hadamard_product(gf2)

