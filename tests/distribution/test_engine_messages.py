"""
The engines report the same problems in the same words, in the terms of the program (#349).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.pgcl import parse_pgcl

ENGINES = ["SYMPY", "GINAC"]


def _error_message(source, engine):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine))
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              capture_output=True, coefficient_limit=0)
    result = session.run().to_json()
    assert result["status"] == "error"
    return result["error"]["message"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_product_of_two_infinite_variables_names_the_program_variables(engine):
    source = "a: nat;\nb: nat;\nc: nat;\na = geometric(1/3);\nb = poisson(4);\nc = a * b;\n"
    message = _error_message(source, engine)
    assert "a * b" in message
    assert not any(name in message for name in ("_0", "_1", "sym_", "x_0"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_guard_with_a_subtraction_names_the_term_as_written(engine):
    source = "a: nat;\nb: nat;\na = geometric(1/3);\nb = poisson(4);\nif (a - 1 < b) { a = 1; } else { skip; }\n"
    message = _error_message(source, engine)
    assert "The guard (a - 1) < b" in message
    assert not any(name in message for name in ("_0", "sym_", "x_0"))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("engine", ENGINES)
def test_iid_of_sample_pgf_is_the_iid_sum_of_the_generating_function(engine):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine))

    def distribution(rhs):
        source = f"x: nat;\ny: nat;\nx = 2;\ny = {rhs};\n"
        session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                                  capture_output=True)
        result = session.run().to_json()
        assert result["status"] == "ok", result
        return sympy.simplify(sympy.sympify(result["distribution"]["closed_form"].replace("^", "**")))

    assert sympy.simplify(distribution("iid(sample_pgf(1/2 + y/2, y), x)") - distribution("iid(1/2 + y/2, x)")) == 0


@pytest.mark.timeout(60)
def test_iid_of_sample_pgf_in_another_variable_is_not_implemented():
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.generating_function import GeneratingFunction
    from prodigy.pgcl.ppy_parser import parse_ppy_expr
    call = parse_ppy_expr("sample_pgf(1/2 + z/2, z)")
    with pytest.raises(NotImplementedError, match="only sampling in the assigned variable y"):
        GeneratingFunction("x^2", "x", "y").update_iid(call, parse_ppy_expr("x"), "y")


@pytest.mark.timeout(60)
def test_convergence_message_substitutes_the_term_for_the_temporary_variable():
    """The convergence message names the closed form in the terms of the program, without the temporary
    variable and its 'where' clause (#221)."""
    dist = GeneratingFunction("1/((1 - 2*_0)*(1 - y))", "_0", "y")
    with pytest.raises(NonFiniteMarginalError) as caught:
        dist._require_convergent_evaluation({"y"}, "The filter", {"_0": "x - 1"})  # pylint: disable=protected-access
    message = str(caught.value)
    assert "sets x - 1 to 1 in" in message
    assert "(x - 1)" in message
    assert "_0" not in message and "stands for" not in message


@pytest.mark.timeout(60)
def test_messages_parenthesise_compound_terms_for_temporary_variables():
    """A compound term of the program replaces the temporary variable in parentheses, so that the closed form
    in a message does not read as `2*x - 1*x - 1` (#454); a name or a number stays bare."""
    # pylint: disable=import-outside-toplevel,protected-access
    from prodigy.distribution.distribution import ComparisonNotImplemented, _in_programs_terms
    names = {"_0": "x - 1", "_1": "y * y", "_2": "z", "_3": "2"}
    error = _in_programs_terms(NonFiniteMarginalError("2*_0 - _1 + _2*_3"), names)
    assert str(error) == "2*(x - 1) - (y * y) + z*2"
    assert isinstance(error, NonFiniteMarginalError)
    comparison = _in_programs_terms(ComparisonNotImplemented("_0 < _1", "reason"), names)
    assert isinstance(comparison, ComparisonNotImplemented)
    assert comparison.condition == "(x - 1) < (y * y)"
    unchanged = ValueError("nothing here")
    assert _in_programs_terms(unchanged, names) is unchanged


@pytest.mark.timeout(60)
def test_guard_message_with_compound_terms_keeps_the_program_text():
    source = ("x: nat;\ny: nat;\nx = geometric(1/3);\ny = poisson(4);\n"
              "if (x - 1 < y * y) { x = 1; } else { skip; }\n")
    message = _error_message(source, "SYMPY")
    assert not any(name in message for name in ("_0", "_1", "sym_"))
