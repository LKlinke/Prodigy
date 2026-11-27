"""
Conformance suite of the engines (#361): one set of contract tests of the ``Distribution`` interface, run on
every engine through the fixture ``factory`` (``conftest.py``). The sections follow #361:

1. ``copy()``, the getters and setters, and the contract of ``set_variables``/``set_parameters``;
2. equality and ``<=``/``<``, including undecidable cases (an engine may answer "cannot decide", never wrong);
3. ``finite_support``, filters, marginals, probabilities and expected values;
4. updates ``+``, ``*``, monus, ``/``, ``%``, ``**``;
5. ``iid`` updates for every distribution of the language, and the factory constructors;
6. iteration of distributions, with the same stable order of the states of finite ones;
7. non-finite values, unknown names and variable/parameter collisions;
8. the same error messages on all engines (#349).

Expected differences between the engines are declared in the section "Declared differences" at the end and
tested as such; nothing is skipped silently. The FLINT rational kernel (``prodigy.distribution.kernel``) is not
an engine, but its operations overlap with the engines' on rational generating functions; the last section
checks that they agree there.
"""
import itertools
import operator

import re

import pytest
import sympy

from prodigy.distribution import known_convergent_series
from prodigy.distribution.distribution import (MarginalType, NonFiniteMarginalError, State,
                                               UndecidedComparisonError, ZeroDivisorError)
from prodigy.distribution.kernel.rational import RationalGF
from prodigy.language.ppy_parser import parse_ppy_expr
from tests.distribution.engines import ENGINES, in_common_words, same, symbolic

P = parse_ppy_expr  # pylint: disable=invalid-name


def _names(dist):
    return dist.get_variables(), dist.get_parameters()


# --- 1. copy, getters, setters -------------------------------------------------------------------------------


@pytest.fixture(name="dist")
def _dist(factory):
    return factory.from_expr("p * x * y / 2 + (1 - p) * x*x", "x", "y")


def test_getters_return_copies(dist):
    before = _names(dist)
    dist.get_variables().add("zzz")
    dist.get_parameters().add("zzz")
    dist.get_variables().clear()
    assert _names(dist) == before


def test_copy_is_equal_and_independent(dist):
    before = _names(dist)
    clone = dist.copy()
    assert clone == dist
    assert _names(clone) == before
    clone.get_variables().add("zzz")
    clone.get_parameters().add("zzz")
    assert _names(dist) == before
    assert _names(clone) == before


def test_copy_of_a_distribution_without_variables_keeps_its_parameters(factory):
    # SymPy's copy read "no variables" as "every symbol is a variable" (#361)
    dist = factory.from_expr("p", "")
    assert _names(dist.copy()) == (set(), {"p"})


def test_setters_do_not_change_the_receiver(dist):
    before = _names(dist)
    result = dist.set_variables("x", "y")
    result.get_variables().add("zzz")
    result.get_parameters().add("zzz")
    assert _names(dist.set_parameters("p")) == ({"x", "y"}, {"p"})
    assert _names(dist) == before


@pytest.mark.parametrize("function, variables, new_variables, expected", [
    # a variable that is not named becomes a parameter if it occurs (SymPy dropped it, #369)
    ("x*y", ("x", "y"), ("x",), ({"x"}, {"y"})),
    # ... and is dropped if it does not occur
    ("x", ("x", "y"), ("x",), ({"x"}, set())),
    # a parameter that is named becomes a variable (GiNaC raised)
    ("p*x", ("x",), ("x", "p"), ({"x", "p"}, set())),
    # no variables at all (SymPy raised)
    ("x", ("x",), (), (set(), {"x"})),
    # new variables that do not occur
    ("x", ("x",), ("x", "z"), ({"x", "z"}, set())),
])
def test_set_variables_keeps_every_occurring_name(factory, function, variables, new_variables, expected):
    result = factory.from_expr(function, *variables).set_variables(*new_variables)
    assert _names(result) == expected
    assert same(result, function)


def test_set_parameters_rejects_a_variable(factory):
    # SymPy compared names with symbols and never raised (#369)
    with pytest.raises(ValueError, match="cannot be parameters and variables at the same time"):
        factory.from_expr("x*y", "x", "y").set_parameters("x")


def test_set_parameters_keeps_every_occurring_name(factory):
    dist = factory.from_expr("p*x + q*(1-p)", "x")
    assert _names(dist.set_parameters("p")) == ({"x", "q"}, {"p"})
    assert _names(dist.set_parameters()) == ({"x", "p", "q"}, set())
    assert _names(dist.set_parameters("p", "q", "r")) == ({"x"}, {"p", "q", "r"})


def test_variables_and_parameters_of_an_expression(factory):
    assert _names(factory.from_expr("a*x + (1-a)", "x")) == ({"x"}, {"a"})
    assert _names(factory.from_expr("x0^3*x1^5/2 + x0*x1/2")) == ({"x0", "x1"}, set())


@pytest.mark.parametrize("name", ["sum", "exp", "sign", "ln", "log", "sqrt"])
def test_names_of_functions_are_variable_names(factory, name):
    dist = factory.from_expr(f"{name}^3*x^5")
    assert len(dist.get_variables()) == 2
    assert dist.marginal(name) == factory.from_expr(f"{name}^3")
    assert dist.update(P(f"{name} == {name} + 1")) == factory.from_expr(f"{name}^4*x^5")
    assert factory.bernoulli(name, "1/2") == factory.from_expr(f"1/2 + 1/2*{name}")
    dist = factory.from_expr(f"{name}*x", "x")
    assert dist.get_parameters() == {name}
    assert dist.update(P("x == 2")) == factory.from_expr(f"{name}*x^2", "x")


# --- 2. equality and order -----------------------------------------------------------------------------------


def test_finite_order(factory):
    whole, half = factory.from_expr("x^2*y^3"), factory.from_expr("1/2*x^2*y^3")
    assert half <= whole
    assert half < whole
    assert not whole <= half
    first = factory.from_expr("1/2*x^2*y^3 + 1/2*x^3*y^2")
    second = factory.from_expr("1/3*x^2*y^3 + 2/3*x^3*y^2")
    assert not first < second


def test_infinite_order_refuted_by_a_coefficient(factory):
    # the coefficient of x^5 of the difference is 1/32 - 1/16 < 0 (#441)
    walk, geometric = factory.from_expr("(1-sqrt(1-x^2))/x"), factory.from_expr("2/(2-x) - 1")
    assert not walk <= geometric
    assert not walk < geometric


def test_undecidable_order_is_not_answered(factory):
    # p*x <= (1-p)*x holds exactly for p <= 1/2: neither proven nor refuted
    smaller, larger = factory.from_expr("p*x", "x"), factory.from_expr("(1-p)*x", "x")
    assert smaller.decide_le(larger) is None
    assert smaller.decide_le_verdict(larger).holds is None
    with pytest.raises(UndecidedComparisonError):
        _ = smaller <= larger
    assert smaller.decide_le(factory.from_expr("p*x + x/2", "x")) is True  # the difference x/2 is non-negative
    assert factory.from_expr("x", "x").decide_le(factory.from_expr("x/2", "x")) is False


def test_equality(factory):
    assert factory.from_expr("exp(x - 1)", "x") == factory.poisson("x", 1)
    assert factory.from_expr("p*x", "x") != factory.from_expr("(1-p)*x", "x")
    # different variables, or a different split into variables and parameters
    assert factory.from_expr("y*x", "x") != factory.from_expr("x*y", "y", "x")
    assert factory.from_expr("x*y").set_variables("x") != factory.from_expr("x*y").set_variables("y")


def test_comparing_different_parameters_is_an_error(factory):
    with pytest.raises(TypeError, match="Incomparable distributions"):
        factory.from_expr("p*x", "x").decide_le(factory.from_expr("q*x", "x"))


# --- 3. finite support, filters, marginals, queries ----------------------------------------------------------


def test_finite_support(factory):
    dist = factory.from_expr("1/(2-x) * (y/2 + 1/2)", "x", "y")
    # the other variables stay symbols in the coefficients (parameters), so it needs no marginal (#249)
    support = dist.finite_support("y")
    assert _names(support) == ({"y"}, {"x"})
    assert same(support, dist)
    assert dist.finite_support("x") is None
    assert factory.from_expr("x/2 + x^3/2", "x").is_finite()
    assert not dist.is_finite()
    assert not (factory.from_expr("x^3", "x") * factory.poisson("x", 5)).is_finite()


@pytest.mark.parametrize("function, variables, condition, expected", [
    ("0", ("x", "y"), "x * 3 < 25 * y", "0"),
    ("(1-sqrt(1-c^2))/c", ("c",), "c <= 5", "c/2 + c^3/8 + c^5/16"),
    ("(1-sqrt(1-c^2))/c", ("c",), "3 + 2 < 10", "(1-sqrt(1-c^2))/c"),
    ("1/2*x*c + 1/4*x^2 + 1/4", ("x", "c"), "x * c < 123", "1/2*x*c + 1/4*x^2 + 1/4"),
    ("(1-sqrt(1-c^2))/c", ("c", "x", "z"), "x * z <= 10", "(1-sqrt(1-c^2))/c"),
    ("(c/2 + c^3/2) * (1-sqrt(1-x^2))/x", ("c", "x"), "c * c <= 5", "c/2 * (1-sqrt(1-x^2))/x"),
    ("1/2*x*y + 1/2*y", ("x", "y"), "y * x < 5", "1/2*x*y + 1/2*y"),
    ("1/10*x^5 + 2/10*x^6 + 3/10*x^7 + 4/10*x^8", ("x",), "x % 3 == 2", "1/10*x^5 + 4/10*x^8"),
    ("p*x + (1-p)", ("x",), "x == 1", "p*x"),
    ("p*x + (1-p)/2*x^2 + (1-p)/2", ("x",), "not x == 1", "(1-p)/2*x^2 + (1-p)/2"),
])
def test_filter(factory, function, variables, condition, expected):
    assert factory.from_expr(function, *variables).filter(P(condition)) == factory.from_expr(expected, *variables)


def test_filter_state(factory):
    dist = factory.from_expr("1/2*x*y + 1/2", "x", "y")
    assert dist.filter_state(State({"x": 1})) == factory.from_expr("1/2*x*y", "x", "y")


def test_marginal(factory):
    dist = factory.uniform("x", "0", "10") * factory.binomial("y", n="10", p="1/2")
    assert dist.marginal("x") == factory.uniform("x", "0", "10")
    assert dist.marginal("y") == factory.binomial("y", n="10", p="1/2")
    assert dist.marginal("x", method=MarginalType.EXCLUDE) == factory.binomial("y", n="10", p="1/2")
    assert dist.marginal("x", "y") == dist
    assert factory.from_expr("x*y").marginal("x", method=MarginalType.EXCLUDE) == factory.from_expr("y")
    product = factory.geometric("x", "1/3") * factory.geometric("y", "4/5")
    assert product.marginal("x") == factory.geometric("x", "1/3")


@pytest.mark.parametrize("method", [MarginalType.INCLUDE, MarginalType.EXCLUDE])
def test_marginal_of_an_unknown_variable(factory, method):
    # GiNaC returned the mass as the marginal of a name that is not a variable (#361)
    with pytest.raises(ValueError, match=r"Unknown variable\(s\): \{x\}"):
        factory.from_expr("(1-sqrt(1-c^2))/c").marginal("x", method=method)
    with pytest.raises(ValueError, match="No variables were provided"):
        factory.from_expr("c", "c").marginal()


def test_probability_of_a_condition(factory):
    assert factory.from_expr("(1-sqrt(1-x^2))/x").get_probability_of(P("x <= 3")) == "5/8"
    assert factory.zero("z", "y").get_probability_of(P("not z * y < 12")) == "0"
    assert same(factory.from_expr("p*x + (1-p)", "x").get_probability_of(P("x == 1")), "p")


def test_probability_mass(factory):
    # the walk's series is a finite measure by the shift and the monomial substitution (#431)
    assert factory.from_expr("(1-sqrt(1-x^2))/x").get_probability_mass() == "1"
    dist = factory.from_expr("(1 + x^2 - x^3)*(1-sqrt(1-x^2))/x")
    # no rule proves this series (with radius 1) convergent at 1 (#398); an analysis vouches for it
    with pytest.raises(NonFiniteMarginalError, match="not proven"):
        dist.get_probability_mass()
    with known_convergent_series(True):
        assert dist.get_probability_mass() == "1"
    assert factory.zero("x").get_probability_mass() == "0"
    assert factory.uniform("x", "3", "10").get_probability_mass() == "1"
    assert same(factory.from_expr("p*x^2 + (1-p)", "x").get_probability_mass(), "1")
    with pytest.raises(NonFiniteMarginalError, match="not proven to converge"):
        factory.from_expr("1/(1-x)", "x").get_probability_mass()


@pytest.mark.parametrize("function, variables, expression, expected", [
    ("(1-sqrt(1-x^2))/x", (), "x", "Infinity"),  # proven a finite measure (#431)
    ("0", ("x",), "x", "0"),
    ("(1-p) + p*x", ("x",), "x", "p"),
    ("(1-p) + p*x", ("x",), "p*x", "p^2"),
    ("(1-p) + p*x", ("x",), "p", "p"),
    ("exp(p*(x - 1))", ("x",), "x", "p"),
    ("exp(p*(x - 1))", ("x",), "p*x", "p^2"),
    ("1/(2-x)", ("x",), "x", "1"),
    ("0.5*x + 0.5", ("x",), "x", "1/2"),
    # subtraction is truncated at 0 in every intermediate result (monus, #220), before any simplification
    ("n^5", (), "n - 7 + 1", "1"),
    ("n^5", (), "n - 7 + 7", "7"),
    ("n^5", (), "(n - 7) * (n - 7)", "0"),
    ("n^5", (), "n - 3", "2"),
])
def test_expected_value(factory, function, variables, expression, expected):
    result = factory.from_expr(function, *variables).get_expected_value_of(expression)
    assert result == expected if expected == "Infinity" else same(result, expected)


# x ~ geometric(1/2) on {0, 1, ...} (mean 1, E[x^2] = 3) and y ~ bernoulli(1/3), independent
INDEPENDENT = "1/2/(1 - x/2) * (2/3 + y/3)"


@pytest.mark.parametrize("expression, expected", [
    ("x", "1"), ("y", "1/3"), ("x + y", "4/3"), ("2*x + 3*y + 1", "4"), ("x*y", "1/3"), ("x^2", "3"),
    ("x^2 + x*y", "10/3"),
])
def test_expected_value_of_a_polynomial(factory, expression, expected):
    assert same(factory.from_expr(INDEPENDENT, "x", "y").get_expected_value_of(expression), expected)


@pytest.mark.parametrize("expression", ["x + z", "x*z", "x^2 + z"])
def test_expected_value_with_an_unknown_name(factory, expression):
    # z is neither a variable nor a parameter; the GiNaC bindings would read it as 0
    with pytest.raises(ValueError, match="unknown symbols"):
        factory.from_expr(INDEPENDENT, "x", "y").get_expected_value_of(expression)


@pytest.mark.parametrize("expression, expected", [("x", "0"), ("x + y", "1/2"), ("x*y + 3", "3"), ("x^2 + 2", "2")])
def test_expected_value_of_a_variable_that_does_not_occur(factory, expression, expected):
    # x is a program variable that does not occur in the generating function, so it is 0 in every state
    assert same(factory.from_expr("1/2 + y/2", "x", "y").get_expected_value_of(expression), expected)


def test_expected_value_errors(factory):
    with pytest.raises(NonFiniteMarginalError, match="not proven"):
        factory.from_expr("(1 + x^2 - x^3)*(1-sqrt(1-x^2))/x").get_expected_value_of("x")
    with pytest.raises(ValueError, match="Cannot compute expected value of x\\^2\\+y because it contains unknown"):
        factory.uniform("x", "3", "10").get_expected_value_of("x^2+y")
    assert factory.uniform("x", "3", "10").get_expected_value_of("x*x") == "95/2"
    assert factory.geometric("x", "1/3").get_expected_value_of("x") == "2"


def test_normalize(factory):
    assert factory.from_expr("x/3 + 1/3", "x").normalize() == factory.from_expr("x/2 + 1/2", "x")
    # a declared parameter stays one, also if it does not occur (SymPy dropped it, #361)
    assert _names(factory.from_expr("x/3 + 1/3", "x").set_parameters("q").normalize()) == ({"x"}, {"q"})
    with pytest.raises(ZeroDivisionError, match="Cannot normalize the zero distribution"):
        factory.zero("x").normalize()


def test_zero_and_one(factory):
    assert factory.zero("x").is_zero_dist()
    assert factory.from_expr("x - x", "x").is_zero_dist()
    assert not factory.from_expr("x", "x").is_zero_dist()
    assert factory.one("x", "y") == factory.from_expr("1", "x", "y")
    assert factory.zero("x", "y") == factory.from_expr("0", "x", "y")


def test_unilateral_approximation(factory):
    dist = factory.from_expr("7/10*x^3*y^5 + 1/10*x^13*y^17 + 15/100*x^21*y^25 + 5/100*x^33*y^4")
    assert dist.approximate_unilaterally("x", "0.9") == factory.from_expr(
        "7/10*x^3*y^5 + 1/10*x^13*y^17 + 15/100*x^21*y^25")
    product = factory.poisson("x", 7) * factory.poisson("y", 81)
    approximated = product.approximate_unilaterally("x", 0.9)
    assert approximated.filter(P("x == 4")) == product.filter(P("x == 4"))
    assert approximated.filter(P("x == 400")).is_zero_dist()
    assert not product.filter(P("x == 400")).is_zero_dist()
    diagonal = factory.from_expr("exp(7*x*y - 7)", "x", "y")
    approximated = diagonal.approximate_unilaterally("x", "0.9")
    assert approximated.filter(P("x == 100")).is_zero_dist()
    assert approximated.filter(P("x == 5")) == approximated.filter(P("y == 5"))


# --- 4. updates ----------------------------------------------------------------------------------------------


UPDATES = [
    # literals
    ("1", ("x",), "x == 1", "x"),
    ("n", (), "n == 2", "n^2"),
    ("x^3", (), "x == 3 + 4 + 8", "x^15"),
    # variables
    ("n^5*m^42", (), "n == m", "n^42*m^42"),
    ("n^5*m^42", (), "n == n", "n^5*m^42"),
    ("x^8", ("x", "y"), "y == x", "y^8*x^8"),
    # addition
    ("x^3", (), "x == x + x", "x^6"),
    ("x^3", (), "x == x + 1", "x^4"),
    ("x^3*y^5", (), "x == x + y", "x^8*y^5"),
    ("0", ("x",), "x == x*5 + 1", "0"),
    ("(1-sqrt(1-c^2))/c", (), "c == c + 1", "c*(1-sqrt(1-c^2))/c"),
    # multiplication
    ("n^5*m^42*o^8", (), "n == 3 * 8", "n^24*m^42*o^8"),
    ("n^5*m^42*o^8", (), "n == 5 * o", "n^40*m^42*o^8"),
    ("n^5*m^42*o^8", (), "n == m * o", "n^336*m^42*o^8"),
    ("n^5*m^42*o^8", (), "n == n * o", "n^40*m^42*o^8"),
    ("1/2*(n + n^2)", (), "n == n * 3", "1/2*n^3 + 1/2*n^6"),
    ("1/2*(n + n^2)", (), "n == n * n", "1/2*n + 1/2*n^4"),
    ("x*y^4", (), "x == 0.5 * y", "x^2*y^4"),
    ("1/6*(1 + x + x^2 + x^3 + x^4 + x^5)", (), "x == x * x", "1/6*(1 + x + x^4 + x^9 + x^16 + x^25)"),
    ("exp(3*(x - 1))*y^3", ("x", "y"), "x == x * y", "exp(3*(x^3 - 1))*y^3"),
    ("exp(3*(x - 1))*exp(5*(y - 1))", ("x", "y"), "x == 3 * 7", "x^21*exp(5*(y - 1))"),
    ("exp(3*(x - 1))*exp(5*(y - 1))", ("x", "y"), "x == x * 3", "exp(3*(x^3 - 1))*exp(5*(y - 1))"),
    ("exp(3*(x - 1))*exp(5*(y - 1))", ("x", "y"), "x == y * 4", "exp(5*(x^4*y - 1))"),
    ("exp(5*(x - 1))*(3/5*y^3*z^5 + 2/5*y^6*z^6)", ("x", "y", "z"), "x == z * y",
     "3/5*x^15*y^3*z^5 + 2/5*x^36*y^6*z^6"),
    ("exp(5*(x - 1))*(3/5*y^3*z^5 + 2/5*y^6*z^6)", ("x", "y", "z"), "z == x * y",
     "3/5*y^3*exp(5*(x*z^3 - 1)) + 2/5*y^6*exp(5*(x*z^6 - 1))"),
    # subtraction is truncated at 0 (monus, #220)
    ("n^5*m^42*l^8", (), "m == m - 8", "n^5*m^34*l^8"),
    ("n^5*m^42*l^8", (), "l == l - n", "n^5*m^42*l^3"),
    ("n^5*m^42*l^8", (), "m == l - n", "n^5*m^3*l^8"),
    ("n^5*m^42*l^8", (), "m == n - l", "n^5*l^8"),
    ("p*x", ("x",), "x == x - 2", "p"),
    ("p*x", ("x",), "x == x - 20", "p"),
    ("(1/p)*x", ("x",), "x == x - 2", "1/p"),
    ("exp(3*(x - 1))", ("x",), "x == x - 2", "exp(-3)*(1 + 3) + (exp(3*(x - 1)) - exp(-3) - 3*x*exp(-3))/x^2"),
    ("2/5*t^5*c^13*n^4 + 3/5*t^7*c^28*n^77", (), "n == n + (c - t)", "2/5*t^5*c^13*n^12 + 3/5*t^7*c^28*n^98"),
    ("(-1 + q*c)^(-1)*p*(-1 + p*n)^(-1)*n", ("c", "n"), "n == n - 1", "(-1 + q*c)^(-1)*p*(-1 + p*n)^(-1)"),
    # x < 5 has probability 31/32 and gives 0; x = 5 + k gives k
    ("1/(2 - x)*y^5", ("x", "y"), "x == x - y", "y^5*(31/32 + 1/32 * 1/(2 - x))"),
    # the Laurent terms after a subtraction stay a closed form (#366)
    ("(exp(2*x - 2) - exp(-2))/(2 - y)", ("x", "y"), "x == x - 1", "(exp(2*x - 2) - exp(-2))/((2 - y)*x)"),
    # fresh variables do not clash with the program's names
    ("x*xl", (), "x == 2 * 3", "x^6*xl"),
    ("x*sym_0", (), "x == 2 * 3", "x^6*sym_0"),
    # modulo
    ("x*(3/10*y^4 + 3/10*y^7 + 2/5*y^8)", (), "x == 5 % 3", "x^2*(3/10*y^4 + 3/10*y^7 + 2/5*y^8)"),
    ("x*(3/10*y^4 + 3/10*y^7 + 2/5*y^8)", (), "x == 5 % (1 + 1 + 1)", "x^2*(3/10*y^4 + 3/10*y^7 + 2/5*y^8)"),
    ("x*(3/10*y^4 + 3/10*y^7 + 2/5*y^8)", (), "x == y % (3 + 2)", "3/10*y^4*x^4 + 3/10*y^7*x^2 + 2/5*y^8*x^3"),
    ("3/10*a^3*b^5*c^2 + 1/5*a^6*b^9*c^55 + 1/2*a^5*b^346*c^34", (), "a == b % c",
     "3/10*a*b^5*c^2 + 1/5*a^9*b^9*c^55 + 1/2*a^6*b^346*c^34"),
    ("(2/5*x^3*y^5 + 3/5*x^7*y^18)*exp(5*(z - 1))", ("x", "y", "z"), "z == y % x",
     "2/5*x^3*y^5*z^2 + 3/5*x^7*y^18*z^4"),
    # division
    ("n", (), "n == 2/1", "n^2"),
    ("n", (), "n == n / 1", "n"),
    ("n^2*m^6", ("n", "m", "o"), "o == m / n", "n^2*m^6*o^3"),
    ("exp(3*(x - 1))*y*z", ("x", "y", "z"), "x == z / y", "x*y*z"),
    ("exp(3*(x - 1))*(2/5*y^3*z^12 + 3/5*y^7*z^42)", ("x", "y", "z"), "y == z / y",
     "exp(3*(x - 1))*(2/5*y^4*z^12 + 3/5*y^6*z^42)"),
    ("exp(3*(x - 1))*(2/5*y^3*z^12 + 3/5*y^7*z^42)", ("x", "y", "z"), "x == z / y",
     "2/5*y^3*z^12*x^4 + 3/5*y^7*z^42*x^6"),
    ("x", (), "x == 9/2 * 1 * 2", "x^9"),
    ("x^9", (), "x == x * (9/2 * 1 * 2)", "x^81"),
    ("x^81", (), "x == x / 9", "x^9"),
    # a constant that is not a natural number is a division, never a fractional exponent (#26)
    ("x^3", ("x", "y"), "x == x / (3/2)", "x^2"),
    ("x^3", ("x", "y"), "x == x / (1/2)", "x^6"),
    ("1/2*x^4 + 1/2*x^2*y", ("x", "y"), "x == 1/2 * x", "1/2*x^2 + 1/2*x*y"),
    ("1/2*x^4 + 1/2*x^2*y", ("x", "y"), "x == 3/2 * x", "1/2*x^6 + 1/2*x^3*y"),
    ("1/2*x^4 + 1/2*x^2*y", ("x", "y"), "y == x * 1 / 2", "1/2*x^4*y^2 + 1/2*x^2*y"),
    # with an infinite marginal, the division is exact if all mass provably lies on multiples
    ("1/(2 - x^2)", ("x",), "x == 1/2 * x", "1/(2 - x)"),
    ("p/(1 - (1-p)*x^3)*y", ("x", "y"), "y == x / 3", "p/(1 - (1-p)*x^3*y)"),
    # powers
    ("x^3*y^5*z^9*exp(3*(w - 1))", ("x", "y", "z", "w"), "z == y ** x", "x^3*y^5*z^125*exp(3*(w - 1))"),
    ("x^3*y^5*z^9*exp(3*(w - 1))", ("x", "y", "z", "w"), "y == 3 ** 5", "x^3*y^243*z^9*exp(3*(w - 1))"),
    ("x^3*y^5*z^9*exp(3*(w - 1))", ("x", "y", "z", "w"), "z == x ** 5", "x^3*y^5*z^243*exp(3*(w - 1))"),
    ("x^3*y^5*z^9*exp(3*(w - 1))", ("x", "y", "z", "w"), "z == 5 ** y", "x^3*y^5*z^3125*exp(3*(w - 1))"),
    ("x^3*y^5*z^9*exp(3*(w - 1))", ("x", "y", "z", "w"), "w == 3 ** 5", "x^3*y^5*z^9*w^243"),
]


@pytest.mark.parametrize("function, variables, update, expected", UPDATES)
def test_update(factory, function, variables, update, expected):
    dist = factory.from_expr(function, *variables)
    result = dist.update(P(update))
    assert result == factory.from_expr(expected, *dist.get_variables()), str(result)
    assert _names(result) == _names(dist)


@pytest.mark.parametrize("operation", [
    lambda d: d.update(P("x == x + y")), lambda d: d.update(P("x == x * y")), lambda d: d.update(P("x == x - y")),
    lambda d: d.update(P("x == x / 2")), lambda d: d.update(P("x == x % 3")), lambda d: d.update(P("x == y ** 2")),
    lambda d: d.update(P("x == x // 2")), lambda d: d.update(P("x == 7")), lambda d: d.update(P("x == y")),
    lambda d: d.filter(P("x < 3")), lambda d: d.filter(P("x % 2 == 0")), lambda d: d.filter(P("x < y")),
    lambda d: d.update_iid(P("bernoulli(1/2)"), P("y"), "x"), lambda d: d.normalize(), lambda d: d.copy(),
    lambda d: d + d, lambda d: d * d,
], ids=["sum", "product", "monus", "division", "modulo", "power", "floor", "literal", "variable", "filter",
        "residue", "comparison", "iid", "normalize", "copy", "add", "multiply"])
def test_operations_keep_the_names(factory, operation):
    # parameters that do not occur in the generating function stay parameters (#361)
    dist = factory.from_expr("1/2*x^2*y + 1/2*x^4", "x", "y").set_parameters("q")
    assert _names(operation(dist)) == ({"x", "y"}, {"q"})


def test_update_keeps_the_parameters(factory):
    dist = factory.from_expr("x", "x").set_parameters("p")
    assert dist.update(P("x == 4")).get_parameters() == {"p"}
    assert factory.from_expr("x").update(P("x == x * x")).get_variables() == {"x"}


def test_update_then_marginal(factory):
    result = factory.from_expr("x").update(P("x == x * x")).update(P("x == 3"))
    assert result == factory.from_expr("x^3")
    assert result.marginal("x", method=MarginalType.EXCLUDE) == factory.from_expr("1", "")
    assert same(factory.from_expr("x^81").marginal("x", method=MarginalType.EXCLUDE), "1")


def test_laurent_terms_after_a_subtraction_stay_unexpanded(factory):
    # #366, #464: a pole at 0 of the factor 1/x, not expanded into a sum of terms
    dist = factory.from_expr("(exp(2*x - 2) - exp(-2))/(2 - y)", "x", "y").update(P("x == x - 1"))
    expression = symbolic(dist)
    assert expression.is_Mul or expression.is_Pow
    assert not expression.is_Add


def test_update_of_a_doubled_poisson(factory):
    doubled = factory.poisson("x", 16).update(P("x == x + x"))
    assert doubled.filter(P("x == 5")).is_zero_dist()
    assert doubled.filter(P("x % 2 == 1")) == factory.zero("x")
    assert doubled.filter(P("x == 30")) == factory.from_expr("16^15*exp(-16)/1307674368000*x^30", "x")
    assert factory.poisson("x", 30).update(P("x == 5")) == factory.from_expr("x^5", "x")


def test_odd_part_of_a_doubled_poisson_is_zero(factory):
    # #464, from the SymPy-independent SymEngine tests
    doubled = factory.from_expr("exp(3*(x - 1))", "x").update(P("x == x + x"))
    with known_convergent_series(True):
        assert same(doubled.filter(P("x % 2 == 1")).get_probability_mass(), "0")
    assert same(doubled.filter(P("x % 2 == 0")), doubled)


@pytest.mark.parametrize("modulus, remainder", [(2, 0), (2, 1), (3, 2)])
def test_residue_class_with_a_parameter(factory, modulus, remainder):
    # #464: the sections of p/(1 - (1-p)x) by the residue of x
    dist = factory.from_expr("p/(1 - (1 - p)*x)", "x")
    section = dist.filter(P(f"x % {modulus} == {remainder}"))
    expected = sympy.sympify(f"p*(1-p)**{remainder}*x**{remainder}/(1 - ((1-p)*x)**{modulus})")
    assert same(section, expected)


def test_infinite_even_part_divided_by_two(factory):
    even = factory.geometric("x", "1/2").filter(P("x % 2 == 0"))
    assert not even.is_finite()
    assert even.update(P("x == x / 2")) == factory.from_expr("2/(4 - x)", "x")


# --- 5. iid and the factory constructors ---------------------------------------------------------------------


IID = [
    ("bernoulli(1/3)", "(2/3 + 1/3*y)"),
    ("geometric(1/2)", "(1/(2 - y))"),
    ("poisson(3)", "exp(3*(y - 1))"),
    ("unif(1, 3)", "(1/3*(y + y^2 + y^3))"),
    ("binomial(4, 1/3)", "(2/3 + 1/3*y)^4"),
    ("logdist(1/2)", "(log(1 - y/2)/log(1/2))"),
    ("negative_binomial(2, 1/2)", "(1/(2 - y))^2"),
    ("1/2 + y/2", "(1/2 + y/2)"),
    ("sample_pgf(1/4 + 3/4*y**2, y)", "(1/4 + 3/4*y^2)"),
]


@pytest.mark.parametrize("sampled, pgf", IID)
def test_iid_with_a_constant_count(factory, sampled, pgf):
    result = factory.from_expr("x^2*y^7", "x", "y").update_iid(P(sampled), P("3"), "y")
    assert result == factory.from_expr(f"x^2*({pgf})^3", "x", "y")


@pytest.mark.parametrize("sampled, pgf", IID)
def test_iid_with_a_variable_count(factory, sampled, pgf):
    finite = factory.from_expr("x^2*y^7", "x", "y").update_iid(P(sampled), P("x"), "y")
    assert finite == factory.from_expr(f"x^2*({pgf})^2", "x", "y")
    # an infinite count substitutes x*G(y) for x
    infinite = factory.from_expr("1/(2 - x)", "x", "y").update_iid(P(sampled), P("x"), "y")
    assert infinite == factory.from_expr(f"1/(2 - x*{pgf})", "x", "y")


@pytest.mark.parametrize("sampled, pgf", IID)
def test_iid_with_the_assigned_variable_as_count(factory, sampled, pgf):
    # y := iid(G, y): the count is the old value of y, so y is substituted by G(y)
    result = factory.from_expr("x^2*y^2", "x", "y").update_iid(P(sampled), P("y"), "y")
    assert result == factory.from_expr(f"x^2*({pgf})^2", "x", "y")


@pytest.mark.parametrize("sampled", ["y/2", "2*y", "y**2 + y", "1/(3 - y)"])
def test_iid_of_a_function_of_another_mass_is_refused(factory, sampled):
    """``iid(G, n)`` needs a generating function of mass 1; the type checker rejects every other one before the
    analysis, and at the interface both engines refuse it alike (decided in #511)."""
    with pytest.raises(ValueError, match="mass 1"):
        factory.from_expr("x^2", "x", "y").update_iid(P(sampled), P("x"), "y")


FACTORIES = [
    (lambda f: f.geometric("x", "1/2"), "1/(2 - x)"),
    (lambda f: f.uniform("x", "1", "3"), "(x + x^2 + x^3)/3"),
    (lambda f: f.bernoulli("x", "1/3"), "2/3 + x/3"),
    (lambda f: f.poisson("x", "3"), "exp(3*(x - 1))"),
    (lambda f: f.log("x", "1/2"), "log(1 - x/2)/log(1/2)"),
    (lambda f: f.binomial("x", "4", "1/3"), "(2/3 + x/3)^4"),
    (lambda f: f.negative_binomial("x", "2", "1/2"), "1/(2 - x)^2"),
]


@pytest.mark.parametrize("make, pgf", FACTORIES)
def test_factories_build_the_table(factory, make, pgf):
    built = make(factory)
    assert isinstance(built, factory)
    assert built == factory.from_expr(pgf, "x")
    assert built.get_variables() == {"x"} and not built.get_parameters()


def test_factories_with_parameters(factory):
    built = factory.geometric("x", "p")
    assert built.get_variables() == {"x"} and built.get_parameters() == {"p"}
    assert built == factory.from_expr("p/(1 - (1 - p)*x)", "x")


@pytest.mark.parametrize("make", [
    lambda f: f.geometric("x", "0"), lambda f: f.geometric("x", "3/2"), lambda f: f.uniform("x", "3", "1"),
    lambda f: f.uniform("x", "-1", "1"), lambda f: f.bernoulli("x", "2"), lambda f: f.poisson("x", "-1"),
    lambda f: f.log("x", "2"), lambda f: f.binomial("x", "-1", "1/2"), lambda f: f.binomial("x", "2", "2"),
    lambda f: f.negative_binomial("x", "r", "1/2"), lambda f: f.negative_binomial("x", "2", "0"),
])
def test_factories_validate_constant_parameters_alike(factory, make):
    with pytest.raises(ValueError):
        make(factory)


def test_iid_validates_the_parameters_of_its_distribution(factory):
    with pytest.raises(ValueError, match="0 < p <= 1"):
        factory.from_expr("x^2", "x", "y").update_iid(P("geometric(2)"), P("x"), "y")


def test_iid_in_its_own_count(factory):
    # y := iid(bernoulli(1/2), y): the count is the old value of y
    result = factory.from_expr("y^3", "y").update_iid(P("bernoulli(1/2)"), P("y"), "y")
    assert result == factory.from_expr("(1/2 + y/2)^3", "y")


@pytest.mark.parametrize("probability", ["1/7", "1/2", "5/6", "p"])
def test_factory_bernoulli_and_geometric(factory, probability):
    assert factory.bernoulli("v", probability) == factory.from_expr(f"1 - {probability} + {probability}*v", "v")
    assert factory.geometric("v", probability) == factory.from_expr(f"{probability}/(1 - (1 - {probability})*v)",
                                                                    "v")


@pytest.mark.parametrize("start, end", [(0, 0), (0, 5), (3, 10), (7, 9)])
def test_factory_uniform(factory, start, end):
    assert factory.uniform("v", str(start), str(end)) == factory.from_expr(
        "(" + " + ".join(f"v^{i}" for i in range(start, end + 1)) + f")/{end - start + 1}", "v")


@pytest.mark.parametrize("probability", ["1/7", "1/2", "5/6"])
def test_factory_log(factory, probability):
    assert factory.log("v", probability) == factory.from_expr(f"log(1 - {probability}*v)/log(1 - {probability})",
                                                              "v")


@pytest.mark.parametrize("rate", ["0", "1", "7", "1/3", "l"])
def test_factory_poisson(factory, rate):
    assert factory.poisson("v", rate) == factory.from_expr(f"exp({rate}*(v - 1))", "v")


@pytest.mark.parametrize("n, p", [("0", "1/2"), ("1", "1/3"), ("7", "2/9"), ("4", "p")])
def test_factory_binomial_and_negative_binomial(factory, n, p):
    assert factory.binomial("v", n, p) == factory.from_expr(f"(1 - {p} + {p}*v)^{n}", "v")
    assert factory.negative_binomial("v", n, p) == factory.from_expr(f"({p}/(1 - (1 - {p})*v))^{n}", "v")


# --- 6. iteration --------------------------------------------------------------------------------------------


# polynomials whose coefficients are sums with parameters (#444)
POLYNOMIALS = [
    ("p*x + (1-p)", ("x",)),
    ("p*x*y + (1-p)/2*x^2*y^4 + (1-p)/2", ("x", "y")),
    ("(p*x + 1 - p)*(q*y + 1 - q)", ("x", "y")),
    ("p*q*x + p*(1-q) + (1-p)*x", ("x",)),
    ("p*x + (1-p)*x^3", ("x",)),
    ("p*x*w + (1-p)*w^2", ("x", "w")),
    ("1/3*x + 1/3*x^2 + 1/3", ("x",)),
    ("1/4*x^2 + 1/8*x + 1/2*y + 1/8*x*y", ("x", "y")),
]


def _monomial(state):
    return sympy.Mul(*(sympy.Symbol(var) ** value for var, value in state.items()))


@pytest.mark.parametrize("function, variables", POLYNOMIALS)
def test_one_item_per_state_with_its_full_coefficient(factory, function, variables):
    items = list(factory.from_expr(function, *variables))
    keys = [frozenset(state.items()) for _, state in items]
    assert len(keys) == len(set(keys)), items
    assert not any(same(prob, "0") for prob, _ in items), items
    rebuilt = sympy.Add(*(symbolic(prob) * _monomial(state) for prob, state in items))
    assert same(rebuilt, function), items


@pytest.mark.parametrize("function, variables", POLYNOMIALS)
def test_finite_iteration_order_is_the_same_on_all_engines(factory, function, variables):
    # graded lexicographic in the sorted variable names (#361); GiNaC's order changed between runs (#418)
    states = [state for _, state in factory.from_expr(function, *variables)]
    order = sorted(variables)
    keys = [(sum(state[v] for v in order), tuple(state[v] for v in order)) for state in states]
    assert keys == sorted(keys)


def test_exhaustive_filter_keeps_sum_coefficients_together(factory):
    # the filter rebuilds the polynomial from (probability, state) pairs: a probability `1/2 - p/2` must stay a
    # factor of its monomial (GiNaC wrote `1/2-1/2*p * x^2`)
    dist = factory.from_expr("p*x + (1-p)/2*x^2 + (1-p)/2", "x")
    filtered = dist._exhaustive_search(P("not x == 1"))  # pylint: disable=protected-access
    assert same(filtered, "(1-p)/2*x^2 + (1-p)/2")


def test_infinite_univariate_iteration(factory):
    # SymPy's series expansion grows slowly with the order: four non-zero terms
    items = [(prob, state) for prob, state in itertools.islice(factory.from_expr("(1-sqrt(1-x^2))/x"), 8)
             if prob != "0"]
    assert items == [("1/2", {"x": 1}), ("1/8", {"x": 3}), ("1/16", {"x": 5}), ("5/128", {"x": 7})]


def test_infinite_multivariate_iteration(factory):
    # every state with its probability; the order among states of the same degree is declared below
    items = list(itertools.islice(factory.from_expr("1/(2-x) * 1/(2-y)", "x", "y"), 10))
    assert {(prob, frozenset(state.items())) for prob, state in items} == {
        (f"1/{2 ** (a + b + 2)}", frozenset({("x", a), ("y", b)}))
        for a in range(4) for b in range(4) if a + b <= 3}


def test_get_state(factory):
    assert factory.from_expr("1/2*x^2 + 1/2*x^3", "x").get_state() == (State({"x": 2}), "1/2")


# --- 7. non-finite values, unknown names, collisions ---------------------------------------------------------


def test_update_with_an_unknown_name(factory):
    # GiNaC read the unknown name as a variable with value 0 (#361)
    with pytest.raises(ValueError, match="z is neither a variable nor a parameter"):
        factory.from_expr("x", "x").update(P("x == z + 1"))
    with pytest.raises(ValueError, match="because z does not exist"):
        factory.from_expr("x", "x").update(P("z == x"))


def test_filter_with_an_unknown_name(factory):
    with pytest.raises(ValueError, match="contains unknown variables"):
        factory.from_expr("x", "x").filter(P("z < 1"))


@pytest.mark.parametrize("condition", ["x < p", "x == p", "p < 1", "x + p <= 2", "not x < p",
                                       "x < 1 && x < p"])
@pytest.mark.parametrize("function", ["p*x + 1 - p", "p/(1 - (1 - p)*x)"])
def test_condition_with_a_parameter_is_not_implemented(factory, function, condition):
    # SymPy read the parameter as 0 and reported an exact result, GiNaC did not terminate (#361)
    dist = factory.from_expr(function, "x")
    with pytest.raises(NotImplementedError, match="depends on the parameter"):
        dist.filter(P(condition))
    with pytest.raises(NotImplementedError, match="depends on the parameter"):
        dist.get_probability_of(P(condition))


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_arithmetic_with_a_name_clash(factory, operation):
    left, right = factory.from_expr("x*y", "y"), factory.from_expr("x", "x")
    with pytest.raises(ArithmeticError, match="Name clash: x is a variable of one distribution and a parameter"):
        operation(left, right)


def test_arithmetic(factory):
    left = factory.from_expr("x", "x").set_parameters("p")
    right = factory.from_expr("y", "y")
    assert _names(left + right) == ({"x", "y"}, {"p"})
    assert left + right == factory.from_expr("x + y", "x", "y").set_parameters("p")
    assert left * right == factory.from_expr("x*y", "x", "y").set_parameters("p")
    assert (factory.from_expr("x^3", "x") * factory.poisson("x", 5) / factory.poisson("x", 5)).is_finite()


def test_non_finite_values(factory):
    assert not factory.from_expr("x", "x").has_non_finite_value()
    with pytest.raises(NonFiniteMarginalError, match="not proven to converge"):
        factory.geometric("x", "p").get_expected_value_of("x")


# --- 8. error messages ---------------------------------------------------------------------------------------


ERRORS = [
    # (function, variables, update, exception, words of the message)
    ("n", ("n",), "n == n / 2", ValueError, "because it is not always an integer"),
    ("n^2*m^6", ("n", "m", "o"), "o == n / m", ValueError, "because it is not always an integer"),
    ("x^3", ("x", "y"), "x == 1/2 * x", ValueError, "because it is not always an integer"),
    ("x^3", ("x", "y"), "x == 0.5 * x", ValueError, "because it is not always an integer"),
    ("x^3", ("x", "y"), "x == 1/2 * x * 2", ValueError, "because it is not always an integer"),
    ("1/(2-x)", ("x",), "x == 1/2 * x", ValueError, "infinite range"),
    ("exp(x^2-1)", ("x",), "x == 1/2 * x", ValueError, "infinite range"),
    ("x^3", ("x",), "x == x + 1/2", ValueError, "the constant 1/2 is not a natural number"),
    ("x^3", ("x",), "x == x - 0.5", ValueError, "is not a natural number"),
    ("x^3", ("x",), "x == x % (3/2)", ValueError, "is not a natural number"),
    ("x^3", ("x",), "x == x ** (1/2)", ValueError, "is not a natural number"),
    ("x^3", ("x",), "x == (1/2) ** x", ValueError, "is not a natural number"),
    ("p*x + (1-p) * x^2", ("x",), "x == p", ValueError, "a parameter cannot be the source of an assignment"),
    ("x", ("x",), "x == 0.5", ValueError, "Cannot assign the real value 0.5 to x"),
    ("x^3", ("x",), "x == x % 0", ZeroDivisorError, "The modulus has to be positive, got 0"),
    ("x^3", ("x",), "x == x / 0", ZeroDivisorError, "Division by zero: the divisor is 0"),
    ("x", ("x",), "x == z + 1", ValueError, "z is neither a variable nor a parameter"),
    ("1/(3 - 2*a)*exp(4*(b - 1))*c", ("a", "b", "c"), "c == a * b", ValueError,
     "Cannot compute a * b exactly: a and b have infinite range, and the case split over the values is not "
     "implemented without an approximation"),
]


@pytest.mark.parametrize("function, variables, update, exception, words", ERRORS)
def test_error_message(factory, function, variables, update, exception, words):
    with pytest.raises(exception, match=words.replace("*", r"\*").replace("(", r"\(").replace(")", r"\)")):
        factory.from_expr(function, *variables).update(P(update))


def _message(engine, function, variables, update, exception):
    with pytest.raises(exception) as caught:
        engine().from_expr(function, *variables).update(P(update))
    return in_common_words(str(caught.value))


@pytest.mark.parametrize("function, variables, update, exception, _words", ERRORS)
def test_error_message_is_the_same_on_all_engines(function, variables, update, exception, _words):
    messages = {_message(engine, function, variables, update, exception) for engine in ENGINES}
    assert len(messages) == 1, messages


def test_power_of_infinite_variables(factory):
    dist = factory.from_expr("x^3*y^5*z^9", "x", "y", "z") * factory.poisson("w", 3)
    # x is fixed to its value 3, w takes infinitely many values
    with pytest.raises(ValueError, match=r"Cannot compute w \*\* 3 exactly: w has infinite range"):
        dist.update(P("z == w ** x"))


# --- 9. binary operators: one case analysis on all engines (#511, P4c) ------------------------------------------


@pytest.mark.parametrize("operator", ["*", "/", "%", "**", "//"])
@pytest.mark.parametrize("parameter_left", [True, False])
def test_operators_with_a_parameter_are_not_implemented(factory, operator, parameter_left):
    """B1: an operand that is a parameter has no values to split over; the message names the program's term
    (before, the engines reported an assignment of a temporary variable, ``_1 = n``)."""
    term = f"n {operator} x" if parameter_left else f"x {operator} n"
    with pytest.raises(NotImplementedError, match=f"Cannot compute {re.escape(term)}: .* with the parameter n is "
                                                  f"not implemented"):
        factory.from_expr("n*x + (1 - n)*x^2", "x").update(P(f"x == {term}"))


@pytest.mark.parametrize("update", ["y == x ** z", "y == z ** x", "y == x * z", "y == x % (z + 1)"])
def test_two_infinite_operands_are_approximated_alike(update):
    """(c): both operands take infinitely many values; with ``approximate``, every engine splits over the
    approximated marginals (GiNaC's bindings failed on ``x ** z``)."""
    results = [symbolic(engine().from_expr("1/(2 - x)*1/(2 - z)", "x", "y", "z").update(P(update), approximate="0.9"))
               for engine in ENGINES]
    assert sympy.simplify(results[0] - results[1]) == 0
    assert 0 < results[0].subs({sympy.Symbol(name): 1 for name in "xyz"}) <= 1


@pytest.mark.parametrize("update, error", [("y == x / z", ZeroDivisorError), ("y == x % z", ZeroDivisorError)])
def test_a_divisor_zero_of_an_approximated_split_is_a_zero_divisor_error(factory, update, error):
    """(d): ``z`` takes the value 0 in the approximated marginal; the split reports the divisor 0 alike (SymPy
    raised Python's ``ZeroDivisionError``, GiNaC ``NotAnIntegerError`` or ``RuntimeError``)."""
    with pytest.raises(error, match="0"):
        factory.from_expr("1/(2 - x)*1/(2 - z)", "x", "y", "z").update(P(update), approximate="0.9")


@pytest.mark.parametrize("update", ["y == x ** 2", "y == 2 ** x", "y == x % z", "y == x / z", "y == x * z"])
def test_infinite_operands_without_approximation_have_one_message(update):
    """(e): the message names the program's operands (no temporary variable) and is the same on all engines."""
    messages = set()
    for engine in ENGINES:
        with pytest.raises(ValueError) as caught:
            engine().from_expr("1/(2 - x)*1/(2 - z)", "x", "y", "z").update(P(update))
        messages.add(str(caught.value))
    assert len(messages) == 1, messages
    (message,) = messages
    assert "infinite range" in message and not re.search(r"\b(_\d+|sym_\d+)\b", message), message


# --- Declared differences ------------------------------------------------------------------------------------
# Each difference is a test that states what each engine does; a change of either side fails it.


def test_declared_difference_iid_of_a_parametric_mass(factory):
    """``iid(G, n)`` with a mass that depends on parameters (``p*y + q``): SymPy computes the formula, which holds
    for the parameter values with mass 1; GiNaC's bindings refuse every ``G`` whose mass they cannot prove to be 1.
    Neither result is wrong; the analysis checks the generating function before it runs."""
    dist = factory.from_expr("x^2", "x", "y")
    if factory.__name__ == "SympyPGF":
        assert dist.update_iid(P("p*y + q"), P("x"), "y") == factory.from_expr("x^2*(p*y + q)^2", "x", "y")
    else:
        with pytest.raises(ValueError, match="must not be a sub distribution"):
            dist.update_iid(P("p*y + q"), P("x"), "y")


def test_declared_difference_iid_of_a_function_of_another_state_variable(factory):
    """``iid(G, n)`` whose ``G`` also contains another variable of the state (``c``, as the invariants of
    ``pgfexamples/equivalence/loopy/invariants`` write it): SymPy composes it and keeps ``c`` a variable, GiNaC's
    bindings take only a univariate ``G``."""
    dist = factory.from_expr("x^2*c", "x", "c", "y")
    if factory.__name__ == "SympyPGF":
        result = dist.update_iid(P("1/2 + c*y/2"), P("2"), "y")
        assert result.get_variables() == {"x", "c", "y"} and not result.get_parameters()
        assert result == factory.from_expr("x^2*c*(1/2 + c*y/2)^2", "x", "c", "y")
    else:
        with pytest.raises(ValueError, match="univariate"):
            dist.update_iid(P("1/2 + c*y/2"), P("2"), "y")


def test_declared_difference_fresh_variable_names(factory):
    """The names of fresh variables differ (``_0``, ``sym_0``); the analysis spells them as program terms."""
    fresh = factory.from_expr("x", "x").get_fresh_variable()
    assert fresh == ("_0" if factory.__name__ == "SympyPGF" else "sym_0")
    assert in_common_words(fresh) == "<temporary>"


def test_declared_difference_order_of_infinite_multivariate_iteration(factory):
    """States of the same degree come in a different order: SymPy enumerates the last variable first, GiNaC the
    first. Both orders are graded and stable."""
    states = [state for _, state in itertools.islice(factory.from_expr("1/(2-x) * 1/(2-y)", "x", "y"), 3)]
    first = {"x": 0, "y": 1} if factory.__name__ == "SympyPGF" else {"x": 1, "y": 0}
    assert states == [{"x": 0, "y": 0}, first, {"x": 1 - first["x"], "y": 1 - first["y"]}]


# --- The rational kernel -------------------------------------------------------------------------------------


RATIONAL = [
    ("1/(2-x)", ("x",)),
    ("x*y/((3 - x)*(2 - y))", ("x", "y")),
    ("(1 + x^2)/(4 - x - x^2)", ("x",)),
    ("p/(1 - (1 - p)*x)", ("x",)),
]


def _kernel(text):
    return RationalGF.from_sympy(symbolic(text))


def _agrees(dist, kernel):
    return _kernel(str(symbolic(dist))) == kernel


@pytest.mark.parametrize("function, variables", RATIONAL)
def test_kernel_agrees_with_the_engines(factory, function, variables):
    dist, kernel = factory.from_expr(function, *variables), _kernel(function)
    other_text = "1/2 + x/4 + x^2/4"
    other = factory.from_expr(other_text, *variables)
    assert _agrees(dist + other, kernel + _kernel(other_text))
    assert _agrees(dist * other, kernel * _kernel(other_text))
    # x := x + 1 multiplies by x
    assert _agrees(dist.update(P("x == x + 1")), kernel * RationalGF.variable("x"))
    # the marginal of the other variables substitutes 1
    if len(variables) > 1:
        with known_convergent_series(True):
            assert _agrees(dist.marginal("x"), kernel.subs("y", 1))
    # the filter x <= 4 keeps the first Taylor coefficients in x
    coefficients = kernel.coefficient_list("x", 5)
    truncated = sum((c * RationalGF.variable("x") ** i for i, c in enumerate(coefficients)),
                    RationalGF.constant(0))
    assert _agrees(dist.filter(P("x <= 4")), truncated)


# --- 10. filters of comparisons with constants: one normalisation on all engines (#511, P4d) --------------------

_RELATIONS = ["<", "<=", "=", ">", ">="]
_BOUNDS = ["0", "3", "5/2", "1/2"]
_FILTERED = [("1/(2 - x)", "x"), ("1/4 + x/4 + x^2/4 + x^3/4", "x"), ("1/(2 - x)*(1/2 + y/2)", "x, y")]


def _holds(relation, value, bound):
    return {"<": operator.lt, "<=": operator.le, "=": operator.eq, ">": operator.gt, ">=": operator.ge}[relation](
        value, bound)


@pytest.mark.parametrize("function, variables", _FILTERED)
@pytest.mark.parametrize("relation", _RELATIONS)
@pytest.mark.parametrize("bound", _BOUNDS)
@pytest.mark.parametrize("flipped", [False, True])
def test_filter_of_a_comparison_with_a_constant(factory, function, variables, relation, bound, flipped):
    """`x op c` and `c op' x` keep exactly the terms x^i with `i op c`, for a natural, fractional or zero bound,
    on finite and infinite functions (the engines' own normalisations differed, #511)."""
    names = variables.split(", ")
    written = {"=": "=="}.get(relation, relation)
    mirrored = {"<": ">", "<=": ">=", "=": "==", ">": "<", ">=": "<="}[relation]
    condition = f"{bound} {mirrored} x" if flipped else f"x {written} {bound}"
    filtered = symbolic(factory.from_expr(function, *names).filter(P(condition)))
    x = sympy.Symbol("x")
    expected = sympy.Integer(0)
    series = sympy.series(sympy.sympify(function.replace("^", "**")), x, n=8).removeO()
    limit = sympy.Rational(bound)
    for i in range(8):
        if _holds(relation, i, limit):
            expected += series.coeff(x, i) * x ** i
    # compare the first coefficients in x (the complements of infinite functions stay closed forms)
    difference = sympy.series(filtered - expected, x, n=8).removeO()
    assert sympy.simplify(difference) == 0, (condition, filtered)


def test_filter_of_a_negative_constant_selects_nothing(factory):
    dist = factory.from_expr("1/(2 - x)", "x")
    # 1 - 2 is 0 on natural numbers (monus): x < 0 holds nowhere
    assert symbolic(dist.filter(P("x < 1 - 2"))) == 0
    assert symbolic(dist.filter(P("x <= 1/2 - 1/2"))) == sympy.Rational(1, 2)


def test_probability_splits_over_a_finite_support(factory):
    """`get_probability_of` computes a condition of a variable with a finite support and one with an infinite
    support, by the case split over the finite one where no filter rule applies (#306; before #511 only the query
    handler split)."""
    dist = factory.from_expr("(1/2 + x/2)*1/(2 - y)", "x", "y")
    # x + y < 2: x = 0 and y < 2 (1/2 * 3/4), x = 1 and y < 1 (1/2 * 1/2)
    assert same(dist.get_probability_of("x + y < 2"), "5/8")


# --- 11. approximate and the Hadamard product on all engines (#487) ---------------------------------------------


def test_approximate_reaches_a_probability_mass(factory):
    """The partial sums in the order of the iteration, one more state each, until the mass reaches the threshold
    (formerly SymPy only, a declared difference)."""
    dist = factory.from_expr("2/(2-x) - 1", "x")
    sums = list(dist.approximate("0.99"))
    assert len(sums) == 7
    assert sums[-1] == factory.from_expr(" + ".join(f"1/{2 ** i}*x^{i}" for i in range(1, 8)), "x")
    assert sums[0] == factory.from_expr("x/2", "x")
    assert list(dist.approximate("99/100"))[-1] == sums[-1]


def test_approximate_of_a_finite_distribution_up_to_its_mass(factory):
    dist = factory.from_expr("1/4 + x*y/4 + x^2/2", "x", "y")
    assert list(dist.approximate("1"))[-1] == dist
    assert not list(factory.zero("x", "y").approximate("0"))


@pytest.mark.parametrize("threshold, error", [(10, TypeError), ("2", ValueError), ("p", ValueError)])
def test_approximate_refuses_counts_and_unreachable_thresholds(factory, threshold, error):
    with pytest.raises(error):
        list(factory.from_expr("1/(2-x)", "x").approximate(threshold))


def test_approximate_with_a_parametric_mass_is_refused(factory):
    with pytest.raises(ValueError, match="not a number"):
        list(factory.from_expr("p*x + 1 - p", "x").approximate("1/2"))


@pytest.mark.parametrize("left, right, expected", [
    ("x/2 + x^2/2", "x/3 + 2/3*x^2", "x/6 + x^2/3"),
    ("1/(2-x)", "1/(2-x)", "1/(4-x)"),
    ("1/(2-x)", "1/2 + x/2", "1/4 + x/8"),
    ("1/(2-x)", "1/(3-x)", "1/(6-x)"),
])
def test_hadamard_product(factory, left, right, expected):
    """The termwise product of the coefficients of two univariate rational functions (formerly SymPy only)."""
    result = factory.from_expr(left, "x").hadamard_product(factory.from_expr(right, "x"))
    assert same(symbolic(result), expected)


def test_hadamard_product_needs_one_common_variable(factory):
    with pytest.raises(NotImplementedError, match="univariate"):
        factory.from_expr("x*y", "x", "y").hadamard_product(factory.from_expr("x*y", "x", "y"))
    with pytest.raises(NotImplementedError, match="univariate"):
        factory.from_expr("x", "x").hadamard_product(factory.from_expr("y", "y"))

