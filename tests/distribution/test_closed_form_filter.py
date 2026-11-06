"""
Regression tests for issue #163: filtering a constant guard on the closed form of a finite distribution
(e.g. `x < 10` on `unif(0, 3)`) did not terminate on the SymPy engine, which computed every coefficient as
the limit of a derivative. All engines must filter such closed forms quickly and exactly.
"""
import signal
from contextlib import contextmanager
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

import prodigy.pgcl as pgcl
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import FPS
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.rational_series import rational_taylor_coefficients
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl.parser import parse_expr

ENGINES = [GeneratingFunction, FPS, SymengineDist]
X, Y, P = sympy.symbols('x y p')
TIME_LIMIT = 30  # seconds; each case takes well below a second


@contextmanager
def _time_limit(seconds: int):
    def _raise(_signum, _frame):
        raise TimeoutError(f"did not finish within {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _to_sympy(text: str) -> sympy.Expr:
    return sympy.parse_expr(text, local_dict={'x': X, 'y': Y, 'p': P},
                            transformations=standard_transformations + (convert_xor,))


def _uniform(low, high):
    return {(i, 0): Fraction(1, high - low + 1) for i in range(low, high + 1)}


def _binomial(n, p):
    return {(k, 0): Fraction(sympy.binomial(n, k)) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)}


def _mix(first, second):
    support = {}
    for state, prob in [*first.items(), *second.items()]:
        support[state] = support.get(state, 0) + prob / 2
    return support


# (name, constructor taking the engine, support {(x, y): probability})
DISTRIBUTIONS = [
    ("unif(0,3)", lambda e: e.factory().uniform('x', '0', '3'), _uniform(0, 3)),
    ("unif(2,7)", lambda e: e.factory().uniform('x', '2', '7'), _uniform(2, 7)),
    ("binomial(5,1/3)", lambda e: e.factory().binomial('x', '5', '1/3'), _binomial(5, Fraction(1, 3))),
    ("closed unif(0,3)", lambda e: e('(x^4-1)/(4*(x-1))', 'x'), _uniform(0, 3)),
    ("mix of closed forms", lambda e: e('1/2*(x^4-1)/(4*(x-1)) + 1/2*x^2*(x^6-1)/(6*(x-1))', 'x'),
     _mix(_uniform(0, 3), _uniform(2, 7))),
    ("product with y", lambda e: e('(x^4-1)/(4*(x-1)) * (y^3-1)/(3*(y-1))', 'x', 'y'),
     {(i, j): Fraction(1, 12) for i in range(4) for j in range(3)}),
    # not a closed form, but each coefficient used to be a limit of a derivative, too (c < 100 took 33 s)
    ("polynomial", lambda e: e('1/2*x + 1/4*x^3*y + 1/4*x^150', 'x', 'y'),
     {(1, 0): Fraction(1, 2), (3, 1): Fraction(1, 4), (150, 0): Fraction(1, 4)}),
]

GUARDS = [
    ("x < 10", lambda x: x < 10),
    ("x < 100", lambda x: x < 100),
    ("x <= 3", lambda x: x <= 3),
    ("x = 5", lambda x: x == 5),
    ("x = 50", lambda x: x == 50),
    ("x > 2", lambda x: x > 2),
    ("x >= 12", lambda x: x >= 12),
    ("x < 0", lambda x: x < 0),
]


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('name,make,support', DISTRIBUTIONS, ids=[d[0] for d in DISTRIBUTIONS])
@pytest.mark.parametrize('guard,holds', GUARDS, ids=[g[0] for g in GUARDS])
def test_filter_finite_closed_form(engine, name, make, support, guard, holds):
    del name
    with _time_limit(TIME_LIMIT):
        result = _to_sympy(str(make(engine).filter(parse_expr(guard))))
    expected = sum((sympy.Rational(prob.numerator, prob.denominator) * X ** x * Y ** y
                    for (x, y), prob in support.items() if holds(x)), sympy.Integer(0))
    assert sympy.cancel(result - expected) == 0


@pytest.mark.parametrize('engine', ENGINES)
def test_probability_of_closed_form(engine):
    with _time_limit(TIME_LIMIT):
        probability = engine('(x^4-1)/(4*(x-1))', 'x').get_probability_of(parse_expr('x = 2'))
    assert sympy.nsimplify(str(probability)) == sympy.Rational(1, 4)


@pytest.mark.parametrize('engine', list(ForwardAnalysisConfig.Engine))
def test_program_guard_on_uniform(engine):
    config = ForwardAnalysisConfig(engine=engine)
    program = pgcl.parse_pgcl("""
        nat x;
        nat y;
        x := unif(0, 3);
        if (x < 10) {y := 1} else {y := 2}
    """)
    with _time_limit(TIME_LIMIT):
        result, error_prob = compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    assert sympy.cancel(_to_sympy(str(result)) - Y * (1 + X + X ** 2 + X ** 3) / 4) == 0
    assert _to_sympy(str(error_prob)) == 0


def test_rational_taylor_coefficients():
    # geometric distribution with a parameter
    coefficients = rational_taylor_coefficients(P / (1 - (1 - P) * X), X, 3)
    assert [sympy.expand(c - e) for c, e in zip(coefficients, [P, P * (1 - P), P * (1 - P) ** 2])] == [0, 0, 0]
    # polynomial after cancelling: the trailing zeros are left out
    assert rational_taylor_coefficients((X ** 4 - 1) / (4 * (X - 1)), X, 100) == [sympy.Rational(1, 4)] * 4
    # non-rational coefficients are kept symbolic
    assert rational_taylor_coefficients(sympy.exp(-P) / (1 - X), X, 2) == [sympy.exp(-P)] * 2
    assert rational_taylor_coefficients(X, X, 0) == []
    # not a power series or not rational in x: left to the caller
    assert rational_taylor_coefficients(1 / X, X, 2) is None
    assert rational_taylor_coefficients(sympy.exp(X), X, 2) is None
