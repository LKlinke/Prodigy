"""
Regression tests for issues #115 and #45: filtering a product of polynomials, e.g. a branch on
`cipher := a + b`, keeps exactly the terms that satisfy the guard on every engine.
"""
from itertools import product

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

import prodigy.pgcl as pgcl
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import FPS
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl.parser import parse_expr

ENGINES = [GeneratingFunction, FPS, SymengineDist]
X, Y = sympy.Symbol('x'), sympy.Symbol('y')

# (guard, the values of x for which it holds)
GUARDS = [
    ("x < 6", lambda x: x < 6),
    ("x <= 5", lambda x: x <= 5),
    ("x > 5", lambda x: x > 5),
    ("x >= 6", lambda x: x >= 6),
    ("x = 6", lambda x: x == 6),
    ("5 < x", lambda x: x > 5),
]


def _to_sympy(text: str, symbols: dict) -> sympy.Expr:
    return sympy.parse_expr(text, local_dict=symbols, transformations=standard_transformations + (convert_xor,))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', GUARDS)
def test_filter_product_of_polynomials(engine, guard, holds):
    # GiNaC does not truncate the series of this product at x^6
    result = sympy.expand(_to_sympy(str(engine('1/4*(1+x*y)*(1+x^5)', 'x', 'y').filter(parse_expr(guard))),
                                    {'x': X, 'y': Y}))
    terms = [(0, 0), (1, 1), (5, 0), (6, 1)]
    assert result == sum((sympy.Rational(1, 4) * X**x * Y**y for x, y in terms if holds(x)), sympy.Integer(0))


@pytest.mark.parametrize('engine', ENGINES)
def test_parameters_survive_filter_and_division(engine):
    # exhaustive search and division used to register the parameter p as a variable on GiNaC
    distribution = engine('p*x*y + (1-p)', 'x', 'y').set_parameters('p')
    filtered = distribution.filter(parse_expr('x*y = 1'))
    symbols = {'x': X, 'y': Y, 'p': sympy.Symbol('p')}
    assert sympy.simplify(_to_sympy(str(filtered + distribution), symbols) - (2 * symbols['p'] * X * Y + 1 - symbols['p'])) == 0
    normalized = filtered / engine('p', 'x', 'y').set_parameters('p')
    assert sympy.simplify(_to_sympy(str(normalized), symbols) - X * Y) == 0
    assert normalized.get_parameters() == {'p'}
    assert sympy.simplify(_to_sympy(str(filtered.normalize()), symbols) - X * Y) == 0


def _run(engine, source: str):
    config = ForwardAnalysisConfig(engine=engine)
    program = pgcl.parse_pgcl(source)
    result, error_prob = compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    symbols = {name: sympy.Symbol(name) for name in [*program.variables, *program.parameters]}
    return _to_sympy(str(result), symbols), _to_sympy(str(error_prob), symbols), symbols


@pytest.mark.parametrize('engine', list(ForwardAnalysisConfig.Engine))
def test_subtraction_in_branch(engine):
    result, error_prob, symbols = _run(engine, """
        nat a;
        nat b;
        nat cipher;
        a := unif(0,5);
        b := unif(0,5);
        cipher := a+b;
        if (5 < cipher) {cipher := cipher - 5} else {skip}
    """)
    a, b, cipher = symbols['a'], symbols['b'], symbols['cipher']
    expected = sum(sympy.Rational(1, 36) * a**i * b**j * cipher**(i + j - 5 if i + j > 5 else i + j)
                   for i, j in product(range(6), repeat=2))
    assert sympy.cancel(result - expected) == 0
    assert error_prob == 0


@pytest.mark.parametrize('engine', [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.GINAC])
def test_caesar(engine):
    with open("pgfexamples/inference/loop_free/conditioning/caesar.pgcl", encoding="utf-8") as file:
        result, error_prob, symbols = _run(engine, file.read())
    p = symbols['p']
    assert sympy.simplify(error_prob - sympy.Rational(29, 36) * (1 - p)) == 0
    # the observation leaves cipher = 4, unless the run failed (fail = 1)
    marginal = result.subs({symbols['key']: 1, symbols['randomChar']: 1})
    cipher, fail = symbols['cipher'], symbols['fail']
    states = [(i + j - 5 if i + j > 5 else i + j) for i, j in product(range(6), repeat=2)]
    failed = sum(sympy.Rational(1, 36) * p * fail * cipher**c for c in states)
    observed = sympy.Rational(states.count(4), 36) * (1 - p) * cipher**4
    expected = (failed + observed) / (1 - error_prob)
    assert sympy.simplify(marginal - expected) == 0
