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
    assert "The guard a - 1 < b" in message
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
    from prodigy.pgcl.parser import parse_expr
    call = parse_expr("sample_pgf(1/2 + z/2, z)")
    with pytest.raises(NotImplementedError, match="only sampling in the assigned variable y"):
        GeneratingFunction("x^2", "x", "y").update_iid(call, parse_expr("x"), "y")


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
