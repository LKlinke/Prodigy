"""
Variables whose names SymEngine reads as constants (issue #204): `e` works on every engine, and `E`, `I`,
`pi` are rejected by the SymEngine engine instead of giving a silently wrong result.
"""
import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl.compiler import parse_pgcl

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]

# filter (observe), update (e = e + d) and marginal (Ex[e]) of a variable called e
PROGRAM = """
d: nat;
e: nat;
e = unif(1, 2);
d = unif(1, 2);
observe(e == 2);
e = e + d;
"""


def _config(engine):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_variable_named_e(engine):
    config = _config(engine)
    with config.activate():
        dist, _ = compute_discrete_distribution(parse_pgcl(PROGRAM, "ppy"), config.factory.one("d", "e"), config)
        normalized = dist.normalize()
        # given e = 2: d is 1 or 2, so e is 3 or 4, each with probability 1/2
        assert sympy.S(str(normalized.get_expected_value_of("e"))) == sympy.Rational(7, 2)
        assert sympy.S(str(normalized.get_expected_value_of("d"))) == sympy.Rational(3, 2)


@pytest.mark.parametrize("name", ["E", "I", "pi"])
def test_symengine_rejects_names_of_constants(name):
    config = _config(ForwardAnalysisConfig.Engine.SYMENGINE)
    program = parse_pgcl(f"{name}: nat;\n{name} = unif(1, 2);\n", "ppy")
    with pytest.raises(ValueError, match="reads .* as a constant"):
        compute_discrete_distribution(program, config.factory.one(), config)
