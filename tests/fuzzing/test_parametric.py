"""
The parametric programs of the fuzzer (#453): the class of parametric probabilities, the instantiation of a
program and of a result, and the oracles of :func:`~tests.fuzzing.oracles.check_parametric`.
"""
import itertools
from fractions import Fraction

import pytest
from hypothesis import given, settings

from .oracles import check_parametric, instantiate_result
from .programs import (PARAMETER_VALUES, PARAMETERS, Assign, Call, Choice, Features, GeneratedProgram, Param,
                       Query, Var, _value, parameter_values, parametric_probabilities, programs)

pytestmark = pytest.mark.filterwarnings("ignore:Generating overly large repr")


@settings(max_examples=50, deadline=None, database=None)
@given(parametric_probabilities())
def test_parametric_probabilities_are_admissible_for_every_value(probability):
    # the class maps (0, 1) into (0, 1): an admissible probability and a positive geometric argument
    for p, q in itertools.product(PARAMETER_VALUES, repeat=2):
        assert 0 < _value(probability, {"p": p, "q": q}) < 1


@settings(max_examples=30, deadline=None, database=None)
@given(programs(Features(parameters=True)), parameter_values())
def test_instantiation_removes_every_parameter(program, values):
    source = program.source()
    for name in program.parameters:
        assert f"param {name}: real;" in source
    instantiated = program.instantiate(values)
    assert not instantiated.parameters
    assert "param " not in instantiated.source()
    assert instantiated.decisions() == program.decisions()


CHOICE = GeneratedProgram(
    (Choice(Param("p"), (Assign("x", Call("bernoulli", (Param("q"),))),),
            (Assign("y", Call("geometric", (Param("p"),))),)),),
    (Query("Pr", Var("x")),),
)


def test_instantiated_source():
    values = {"p": Fraction(1, 3), "q": Fraction(3, 4)}
    assert CHOICE.parameters == PARAMETERS
    text = CHOICE.instantiate(values).source()
    assert "[1/3]" in text and "bernoulli(3/4)" in text and "geometric(1/3)" in text


def test_instantiate_result_computes_the_floats():
    result = {"bounds": {"kind": "exact", "residual_mass": "0"},
              "queries": [{"index": 0, "kind": "probability", "expr": "x", "value": "p*q", "value_float": None}],
              "distribution": {"variables": ["x"], "parameters": ["p", "q"], "mass": "1", "mass_float": None,
                               "marginals": {"x": {"rows": [{"value": 0, "prob": "1 - p^2", "p": None}]}}}}
    instantiated = instantiate_result(result, {"p": Fraction(1, 2), "q": Fraction(1, 3)})
    assert instantiated["queries"][0]["value_float"] == pytest.approx(1 / 6)
    assert instantiated["distribution"]["marginals"]["x"]["rows"][0]["p"] == pytest.approx(3 / 4)
    assert instantiated["distribution"]["mass_float"] == 1.0
    assert result["queries"][0]["value_float"] is None  # the original is not changed


def test_check_parametric_passes_a_correct_program():
    values = {"p": Fraction(1, 3), "q": Fraction(3, 4)}
    assert not check_parametric(CHOICE.source(), CHOICE.instantiate(values).source(), values)


def test_check_parametric_finds_a_wrong_instantiation():
    # the oracles must see a difference between the parametric result and the program it claims to instantiate
    values = {"p": Fraction(1, 3), "q": Fraction(3, 4)}
    wrong = CHOICE.instantiate({"p": Fraction(2, 3), "q": Fraction(3, 4)}).source()
    problems = check_parametric(CHOICE.source(), wrong, values)
    assert any("the instantiated program gives" in problem for problem in problems)
    assert any("vs. simulation" in problem for problem in problems)
