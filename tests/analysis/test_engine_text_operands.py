"""
Compound operands reach the engines intact (P5a, T6). Engine text writes every compound operand in parentheses,
also a rational literal: ``(1/2) / (3/4)`` was written as ``1/2 / 3/4``, which the engines read as
``((1/2)/3)/4 = 1/24``, so the probability of a choice was wrong on both engines. The generating functions of
the GiNaC engine's sampling put a compound parameter in parentheses: ``bernoulli(1/2 - p)`` gave
``1/2 - p * x + (1 - 1/2 - p)``.
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.values import sym
from prodigy.language import parse_program

ENGINES = ["SYMPY", "GINAC"]
p = sympy.Symbol("p", real=True)


def probability_of_one(source: str, engine: str) -> sympy.Expr:
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine))
    result = AnalysisSession(parse_program(source), source=source, config=config, capture_output=True).run().to_json()
    assert result["status"] == "ok", result["error"]
    [query] = result["queries"]
    return sym(query["value"])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("probability, expected", [
    ("(1/2) / (3/4)", sympy.Rational(2, 3)),
    ("(1/3) / (1/2) * (3/4)", sympy.Rational(1, 2)),
    ("1 - (1/2) / (2/3)", sympy.Rational(1, 4)),
    ("p / (3/2)", 2 * p / 3),
    ("(p / (5/4)) ** 2", 16 * p ** 2 / 25),
])
def test_rational_literals_as_operands(engine, probability, expected):
    source = f"x: nat;\nparam p: real;\n{{ x = 1; }} [{probability}] {{ x = 3; }}\nquery Pr[x == 1];\n"
    assert sympy.simplify(probability_of_one(source, engine) - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sampling, expected", [
    ("bernoulli(1/2 - p)", sympy.Rational(1, 2) - p),
    ("bernoulli(p * (1 - p))", p * (1 - p)),
    ("poisson(1/2 + p)", (sympy.Rational(1, 2) + p) * sympy.exp(-(sympy.Rational(1, 2) + p))),
])
def test_compound_parameters_of_sampled_distributions(engine, sampling, expected):
    source = f"x: nat;\nparam p: real;\nx = {sampling};\nquery Pr[x == 1];\n"
    assert sympy.simplify(probability_of_one(source, engine) - expected) == 0
