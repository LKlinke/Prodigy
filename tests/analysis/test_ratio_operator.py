"""
A choice ``{…} [x/n] {…}`` whose probability is the ratio of two program variables with infinitely many values
(#237), computed with the operator θ_x followed by 1/θ_n
(:func:`prodigy.analysis.instructionhandler.counter_operators.apply_ratio`). The example
``pgfexamples/inference/loop_free/ratio_geometric_bound.ppy`` is also compared against the simulator
(``tests/simulation/test_examples.py``); the gambler's ruin check is in
``tests/analysis/equivalence/test_equivalence_check.py``.
"""
import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution, compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants import exp_polynomial
from prodigy.analysis.evtinvariants.exp_polynomial import analytic_at_zero, integral_from_zero
from prodigy.analysis.instructionhandler import counter_operators
from prodigy.analysis.instructionhandler.counter_operators import RatioProbability, apply_ratio, ratio_probability
from prodigy.distribution import Distribution
from prodigy.distribution.equality import decide_equal, sympify_distribution
from prodigy.language import Program
from prodigy.language.compiler import compile_program
from prodigy.language.ppy_parser import parse_ppy_expr

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

n, x, y, u, v = sympy.symbols("n x y u v")
RATIO = RatioProbability("x", "n")
CONSTRUCT = "The probability x / n of the probabilistic choice"


def _distribution(source: str, engine):
    program = compile_program(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine, normalize=False)
    with config.activate():
        result, _ = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, config


def _coefficients(expr: sympy.Expr, order: int) -> dict:
    """The coefficients of x^a n^b (a, b < order) of the power series of ``expr``."""
    series = sympy.series(expr, n, 0, order).removeO()
    return {(a, b): sympy.nsimplify(sympy.series(series.coeff(n, b), x, 0, order).removeO().coeff(x, a))
            for a in range(order) for b in range(order)}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("prefix", [
    pytest.param("n: nat = unif(1, 3);\nx: nat = unif(0, n);\n", id="uniform"),
    pytest.param("n: nat = binomial(3, 1/2);\nn = n + 1;\nx: nat = binomial(n, 1/3);\ny: nat = bernoulli(1/4);\n",
                 id="binomial-and-other-variable"),
])
def test_finite_values_agree_with_the_case_split(engine, prefix):
    before, config = _distribution(prefix, engine)
    split, _ = _distribution(prefix + "z: nat = 0;\n{ z = 1; } [x / n] { z = 2; }", engine)
    with config.activate():
        left, right = apply_ratio(RATIO, before, CONSTRUCT)
        operator = left * "z" + right * "z^2"
    assert decide_equal(sympify_distribution(operator), sympify_distribution(split)) is True


@pytest.mark.slow
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("generating_function", [
    # the second-order PGF of x and n (the uninitialised input of an equivalence check), with n > 0 and x <= n
    pytest.param("1/((1-x)*(1-n))", id="sum-of-powers"),
    # n geometric, x = iid(bernoulli(1/3), n) on top: x <= n
    pytest.param("1/(2-n*(2+x)/3)", id="geometric"),
])
def test_infinite_values_are_weighted_coefficient_by_coefficient(engine, generating_function):
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        before = config.factory.from_expr(generating_function, "n", "x").filter(parse_ppy_expr("n > 0 and x <= n"))
        left, right = apply_ratio(RATIO, before, CONSTRUCT)
    original = _coefficients(sympify_distribution(before), 5)
    for (a, b), coefficient in _coefficients(sympify_distribution(left), 5).items():
        expected = original[(a, b)] * a / b if b > 0 else 0
        assert coefficient == expected, (a, b)
    for (a, b), coefficient in _coefficients(sympify_distribution(right), 5).items():
        expected = original[(a, b)] * (b - a) / b if b > 0 else 0
        assert coefficient == expected, (a, b)


@pytest.mark.parametrize("engine", ENGINES)
def test_rational_closed_form(engine):
    # n - 1 geometric(1/2), x binomial(n, 1/2): the weight a/b turns (1 + x)^b into x (1 + x)^(b - 1)
    source = ("n: nat = geometric(1/2);\nn = n + 1;\nx: nat = iid(bernoulli(1/2), n);\ny: nat = 0;\n"
              "{ y = 1; } [x / n] { y = 0; }")
    result, _ = _distribution(source, engine)
    assert decide_equal(sympify_distribution(result), n * (1 + x * y) / (4 - n * (1 + x))) is True


@pytest.mark.parametrize("engine", ENGINES)
def test_bernoulli_is_the_choice(engine):
    prefix = "n: nat = geometric(1/2);\nn = n + 1;\nx: nat = iid(bernoulli(1/2), n);\ny: nat = 0;\n"
    choice, _ = _distribution(prefix + "{ y = 1; } [x / n] { y = 0; }", engine)
    sampled, _ = _distribution(prefix + "y = bernoulli(x / n);", engine)
    assert decide_equal(sympify_distribution(sampled), sympify_distribution(choice)) is True


CHOICE = "\ny: nat = 0;\n{ y = 1; } [x / n] { y = 0; }"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, message", [
    # n = 0 has probability 1/2: x/n is undefined there, an error of the whole program (as for finitely many values)
    pytest.param("n: nat = geometric(1/2);\nx: nat = 0;",
                 "is undefined in the state n = 0, x = 0, which has probability 1/2: 0 / 0 has no value",
                 id="n-can-be-zero"),
    # x and n independent: x > n is reachable, x/n is not a probability there
    pytest.param("n: nat = geometric(1/2);\nn = n + 1;\nx: nat = geometric(1/2);",
                 r"is 2 in the state n = 1, x = 2, which has probability 1/16, which is not a probability",
                 id="x-can-exceed-n"),
])
def test_inadmissible_values_are_errors_with_a_witness(engine, source, message):
    with pytest.raises(ValueError, match=message):
        _distribution(source + CHOICE, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_undecided_admissibility_is_not_implemented(engine):
    # x = unif(0, n) has logarithms, on which the comparison x > n is not implemented
    with pytest.raises(NotImplementedError, match="could not decide whether x/n is a probability in every reachable "
                                                  "state: the comparison x > n is not implemented"):
        _distribution("n: nat = geometric(1/2);\nn = n + 1;\nx: nat = unif(0, n);" + CHOICE, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_witness_and_no_proof_is_not_implemented(engine, monkeypatch):
    # the part where x > n is neither shown to be zero nor has a coefficient found to be non-zero: no result
    monkeypatch.setattr(counter_operators, "nonzero_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(Distribution, "is_provably_zero", lambda self: False)
    with pytest.raises(NotImplementedError, match="could not decide whether x/n is a probability"):
        _distribution("n: nat = geometric(1/2);\nn = n + 1;\nx: nat = iid(bernoulli(1/2), n);" + CHOICE, engine)


def _check_on_every_input(guard: str, engine):
    program = compile_program(f"x: nat; n: nat; y: nat; if ({guard}) {{     {{         y = 1;     }} [x / n] {{         y = 0;     }} }} else {{     skip; }}")
    assert isinstance(program, Program)
    return check_equivalence(program, program, ForwardAnalysisConfig(engine=engine), compute_semantics)


@pytest.mark.parametrize("engine", ENGINES)
def test_guarded_ratio_on_every_input_is_well_formed(engine):
    # on the second-order generating function (infinite mass) the guard makes the part n = 0 or x > n the zero series
    assert _check_on_every_input("n > 0 and x <= n", engine) == (True, [])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("guard, message", [
    pytest.param("true", "is undefined in the state n = 0, x = 0: 0 / 0 has no value", id="unguarded"),
    pytest.param("n > 0", "is 2 in the state n = 1, x = 2, which is not a probability", id="x-can-exceed-n"),
])
def test_unguarded_ratio_on_every_input_is_an_error_with_a_witness(engine, guard, message):
    # reachable means a non-zero coefficient of the formal power series, not a positive mass
    with pytest.raises(ValueError, match=message):
        _check_on_every_input(guard, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_uncertified_integral_is_not_implemented(engine, monkeypatch):
    monkeypatch.setattr(counter_operators, "integral_from_zero", lambda *args: None)
    source = "n: nat = geometric(1/2);\nn = n + 1;\nx: nat = iid(bernoulli(1/2), n);\n{ x = 0; } [x / n] { x = 1; }"
    with pytest.raises(NotImplementedError, match="needs the integral from 0"):
        _distribution(source, engine)


@pytest.mark.parametrize("probability, expected", [
    ("x / n", RATIO),
    ("n / x", RatioProbability("n", "x")),
    ("x / x", None),
    ("2 * x / n", None),
    ("(x + 1) / (n + 1)", None),
    ("x / 3", None),
])
def test_ratio_probability_shapes(probability, expected):
    assert ratio_probability(parse_ppy_expr(probability), {"x", "n"}) == expected


def test_ratio_probability_needs_both_variables_in_the_state():
    assert ratio_probability(parse_ppy_expr("x / n"), {"x"}) is None


def test_rational_integrand_with_parameters():
    # SymPy's integrate is not used with parameters; the partial fractions in n give the antiderivative
    integrand = u * v ** 2 * x / ((1 - v * n) * (1 - u * v * n * x) ** 2)
    integral = integral_from_zero(integrand, n, [n, x])
    assert integral is not None
    assert decide_equal(sympy.diff(integral, n), integrand) is True
    assert decide_equal(integral.subs(n, 0), sympy.Integer(0)) is True
    # a denominator without linear factors over the rationals in the other symbols: no proposal
    assert integral_from_zero(u / (1 + u * n ** 2), n, [n]) is None


def test_rational_antiderivative_is_only_a_proposal(monkeypatch):
    monkeypatch.setattr(exp_polynomial, "_cas_antiderivative", lambda *args: None)
    monkeypatch.setattr(exp_polynomial, "_rational_antiderivative", lambda *args: sympy.log(1 - u * n))
    assert integral_from_zero(u / (1 - u * n), n, [n]) is None


def test_logarithm_over_a_monomial_is_analytic():
    assert analytic_at_zero(sympy.log(1 - u * n * x) / x, [n, x])
    assert analytic_at_zero(sympy.log(1 - u * n * x) / (x * n * (2 - x)), [n, x])
    assert not analytic_at_zero(sympy.log(1 - n) / x, [n, x])  # (A - 1)/x = -n/x has a pole
    assert not analytic_at_zero(sympy.log(2 - n * x) / x, [n, x])  # log(2)/x has a pole
    assert not analytic_at_zero(sympy.log(1 - n * x) / x ** 2, [n, x])
