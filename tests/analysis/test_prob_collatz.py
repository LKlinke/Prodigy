"""
Regression test for issue #26: `x := 1/2 * x` on a variable with infinite marginal is computed
exactly once all mass lies on even values, with the same result on every engine.
"""
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

import prodigy.pgcl as pgcl
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig

PROGRAM = "pgfexamples/inference/loop_free/prob_collatz.pgcl"
DEGREE = 6


def expected_coefficients() -> list[Fraction]:
    """
    The probabilities of x = 0..DEGREE, computed state by state. Every step at least halves x, so
    only initial values up to 8 * DEGREE can end in x <= DEGREE after the three steps.
    """
    states = {k: Fraction(1, 2**(k + 1)) for k in range(8 * DEGREE + 1)}
    for _ in range(3):
        successors: dict[int, Fraction] = {}
        for x, mass in states.items():
            branches = [(3 * x + 1, Fraction(1, 10)), (x // 2, Fraction(9, 10))] if x % 2 == 0 \
                else [(3 * x + 1, Fraction(1))]
            for value, probability in branches:
                successors[value] = successors.get(value, Fraction(0)) + mass * probability
        states = successors
    return [states.get(value, Fraction(0)) for value in range(DEGREE + 1)]


@pytest.mark.parametrize('engine', list(ForwardAnalysisConfig.Engine))
def test_prob_collatz(engine):
    config = ForwardAnalysisConfig(engine=engine)
    with open(PROGRAM, encoding="utf-8") as file:
        program = pgcl.parse_pgcl(file.read())
    result, error_prob = compute_discrete_distribution(
        program, config.factory.one(*program.variables), config)

    text = str(result)
    symbols = {name: sympy.Symbol(name) for name in ("x", "n")}
    function = sympy.parse_expr(text, local_dict=symbols,
                                transformations=standard_transformations + (convert_xor,))
    x = symbols["x"]
    series = sympy.series(function.subs(symbols["n"], 1), x, 0, DEGREE + 1).removeO()
    coefficients = [sympy.nsimplify(sympy.simplify(series.coeff(x, k))) for k in range(DEGREE + 1)]
    assert coefficients == [sympy.Rational(c.numerator, c.denominator) for c in expected_coefficients()]
    assert sympy.simplify(function.subs({x: 1, symbols["n"]: 1})) == 1
    assert sympy.S(str(error_prob)) == 0
