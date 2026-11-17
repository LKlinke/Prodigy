"""
The engines report the same problems in the same words, in the terms of the program (#349), and SymEngine's
subtraction does not leave nested fractions behind (#366).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl import parse_pgcl

ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]


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
def test_symengine_subtraction_of_variables_leaves_no_nested_fractions():
    dist = SymengineDist("1/((1 - y/2)*(1 - z/2))", "w", "y", "z")
    result = dist._update_subtraction("w", "y", "z")  # pylint: disable=protected-access
    expression = sympy.sympify(str(result))
    assert not any(power.exp.is_negative and any(sympy.fraction(term)[1].free_symbols for term in power.base.args)
                   for power in expression.atoms(sympy.Pow) if power.base.is_Add)
    expected = sympy.sympify("1/((1 - w*y/2)*(1 - z/(2*w)))")
    assert sympy.simplify(expression - expected) == 0


@pytest.mark.timeout(60)
def test_symengine_subtraction_keeps_laurent_polynomials():
    dist = SymengineDist("(1/2 + y/2)*(1/3 + 2*z/3)", "w", "y", "z")
    result = dist._update_subtraction("y", "y", "z")  # pylint: disable=protected-access
    assert "z/y" in str(result)  # a Laurent term stays as it is, it is not turned into a fraction of sums
    assert "/(" not in str(result)
