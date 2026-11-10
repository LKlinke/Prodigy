"""
Variables that do not occur in the generating function, and poles at 1, on all engines.

- #232: ``a = b`` where ``b`` does not occur in the function (``b`` is 0 in every state).
- #227: an infinite expected value is ∞ (and the analysis ends with ``NonFiniteResult``), not a crash.
- #242 (GiNaC): marginalising a variable at a pole raises `NonFiniteMarginalError`.
"""
import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.distribution import MarginalType, NonFiniteMarginalError
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.symengine_distribution import SymenginePGF
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.parser import parse_expr


def _ginac_factory():
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


FACTORIES = [
    pytest.param(lambda: SympyPGF, id="sympy"),
    pytest.param(_ginac_factory, id="ginac"),
    pytest.param(lambda: SymenginePGF, id="symengine"),
]


def _engine(name):
    if name == "GINAC":
        pytest.importorskip("prodigy_ginac")
    return getattr(ForwardAnalysisConfig.Engine, name)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("function, assignment, expected", [
    # the old distribution of y must not end up on x (#232)
    ("1/3 + y/3 + y^2/3", "y = x", "1"),
    ("1/3 + y/3 + y^2/3", "x = z", "1/3 + y/3 + y^2/3"),
    ("1/(2 - y)", "y = z", "1"),
    ("1/(2 - y)", "z = x", "1/(2 - y)"),
    # z occurs: a copy
    ("z/(2 - y)", "y = z", "z*y"),
    ("1/(2 - y)", "z = y", "1/(2 - y*z)"),
])
def test_copy_of_a_variable(factory, function, assignment, expected):
    pgf = factory()
    dist = pgf.from_expr(function, "x", "y", "z")
    result = dist.update(parse_expr(assignment))
    assert result == pgf.from_expr(expected, "x", "y", "z")


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("function, expression, expected", [
    ("1/(1 - c)^2", "c", "Infinity"),
    ("(1 - c)/((1 - c)*(2 - c))", "c", "1"),
    ("1/(1 - c)^2", "x", "0"),
    ("(1 - (1 - c^2)^(1/2))/c", "c", "Infinity"),
])
def test_expected_value_at_a_pole(factory, function, expression, expected):
    dist = factory().from_expr(function, "c", "x")
    assert dist.get_expected_value_of(expression) == expected


@pytest.mark.parametrize("variables", [("a",), ("a", "b")])
def test_ginac_marginal_at_a_pole_is_an_error(variables):
    dist = _ginac_factory().from_expr("1/((1 - w)*(2 - b))", "a", "b", "w")
    with pytest.raises(NonFiniteMarginalError, match="pole at w = 1"):
        dist.marginal(*variables)


def test_ginac_marginal_without_the_pole():
    pgf = _ginac_factory()
    dist = pgf.from_expr("1/((1 - w)*(2 - b))", "a", "b", "w")
    # a measure with infinite mass, but no pole in the variables marginalised out
    assert dist.marginal("w") == pgf.from_expr("1/(1 - w)", "w")
    assert dist.marginal("a", "b", method=MarginalType.EXCLUDE) == pgf.from_expr("1/(1 - w)", "w")


RANDOM_WALK = """x: nat;
c: nat;
x = 1;
while (x > 0) {
  {x = x - 1;} [1/2] {x = x + 1;}
  c = c + 1;
}
query Pr[c == 1];
query Ex[c];
"""


@pytest.mark.parametrize("engine", ["SYMPY", "GINAC", "SYMENGINE"])
def test_infinite_expectation_is_non_finite_result(engine):
    decisions = {"0": {"strategy": "evt_invariant",
                       "params": {"invariant": "(1-(1-c^2)^(1/2)+c*x)/(c+(1-c^2)^(1/2)*x-x)"}}}
    config = ForwardAnalysisConfig(engine=_engine(engine))
    session = AnalysisSession(parse_pgcl(RANDOM_WALK, "ppy"), source=RANDOM_WALK, syntax="ppy", config=config,
                              decisions=decisions, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert result["error"]["type"] == "NonFiniteResult"
    expectation = next(q for q in result["queries"] if q["kind"] == "expectation")
    assert expectation["value"] == "Infinity"
    assert expectation["bound"] == "unknown"
