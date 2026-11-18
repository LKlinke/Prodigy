"""
The public entry points of the Python API that evaluate a generating function at 1 (`marginal`,
`get_probability_mass`, `get_probability_of`, `get_expected_value_of`) prove that the series converges there, or
raise `NonFiniteMarginalError` (#398). Before, they evaluated the closed form, which need not be the sum of the
series: `Pr[y < w]` of the issue's example was -1/3. Inside program analyses (`known_convergent_series`), the
distributions are finite measures and the proof is skipped.
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution import NonFiniteMarginalError, known_convergent_series
from prodigy.distribution.distribution import MarginalType
from prodigy.pgcl import parse_pgcl

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC]

# the example of #398: the filter of y < w is right as a formal power series, but the series diverges at x = 1
ISSUE = "1/((1 - 2*x)*(2 - y)*(2 - w)) + (x - 1)*x*y*w"
# the invariant of the formal random walk whose terminating part 2/(1 - 2y) diverges at y = 1: before #393 the
# analysis reported Pr[x = 0] <= -2, the value of the closed form (tests/analysis/test_evt_series_identity.py)
DIVERGENT_EXIT = "(2 + 4*x^2*y)/((1 - x)^2*(1 - 2*y))"


def _factory(engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine).factory


def _sympy(value) -> sympy.Expr:
    return sympy.sympify(str(value).replace("^", "**"))


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_example_is_not_evaluated(engine):
    dist = _factory(engine).from_expr(ISSUE, "x", "y", "w")
    with pytest.raises(NonFiniteMarginalError, match=r"Pr\[y < w\] is not proven to converge"):
        dist.get_probability_of("y < w")
    with pytest.raises(NonFiniteMarginalError, match="probability mass is not proven"):
        dist.get_probability_mass()
    with pytest.raises(NonFiniteMarginalError, match="expected value of y is not proven"):
        dist.get_expected_value_of("y")
    with pytest.raises(NonFiniteMarginalError, match="marginal of y is not proven"):
        dist.marginal("y")
    with pytest.raises(NonFiniteMarginalError, match="marginal of y is not proven"):
        dist.marginal("x", "w", method=MarginalType.EXCLUDE)


@pytest.mark.parametrize("engine", ENGINES)
def test_former_wrong_bound_is_not_evaluated(engine):
    # the closed form gives Pr[x = 0] = -2 and Ex[y] = 4 (as main did inside the analysis before #393)
    dist = _factory(engine).from_expr(DIVERGENT_EXIT, "x", "y")
    with pytest.raises(NonFiniteMarginalError, match="not proven"):
        dist.get_probability_of("x = 0")
    with pytest.raises(NonFiniteMarginalError, match="not proven"):
        dist.get_expected_value_of("y")
    with pytest.raises(NonFiniteMarginalError, match="marginal of x is not proven"):
        dist.marginal("x")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function, check", [
    # rational, analytic on a neighbourhood of the closed unit polydisc (marginal_obstacle)
    ("1/((2 - x)*(2 - y))", lambda d: _sympy(d.get_probability_of("x < y")) == sympy.Rational(1, 3)),
    ("1/((2 - x)*(2 - y))", lambda d: _sympy(d.get_expected_value_of("x + y")) == 2),
    ("1/((2 - x)*(2 - y))", lambda d: sympy.simplify(_sympy(d.marginal("x")) - 1 / (2 - sympy.Symbol("x"))) == 0),
    # a signed measure that converges absolutely at 1
    ("1/(1 + x/2)", lambda d: _sympy(d.get_probability_mass()) == sympy.Rational(2, 3)),
    # exp: the Poisson distribution, also with a parameter
    ("exp(3*(x - 1))", lambda d: _sympy(d.get_expected_value_of("x")) == 3),
    # a finite support needs no proof, whatever its coefficients
    ("1 - 2*x + 2*x^2*y", lambda d: _sympy(d.get_probability_mass()) == 1),
])
def test_proven_evaluations_are_computed(engine, function, check):
    dist = _factory(engine).from_expr(function, "x", "y")
    assert check(dist)


@pytest.mark.parametrize("engine", ENGINES)
def test_validated_distribution_is_computed(engine):
    # the branch point at x = 1 is on the closed unit disc, so marginal_obstacle does not prove it, but the check
    # of input distributions proves non-negative coefficients and finite mass: then the closed form at 1 is the
    # sum of the series (Abel), and an infinite moment is infinite
    dist = _factory(engine).from_expr("(1 - sqrt(1 - x))*exp(y - 1)", "x", "y")
    assert _sympy(dist.get_probability_mass()) == 1
    assert sympy.simplify(_sympy(dist.marginal("y")) - sympy.exp(sympy.Symbol("y") - 1)) == 0
    assert _sympy(dist.get_expected_value_of("x")) in (sympy.oo, sympy.Symbol("Infinity"))


@pytest.mark.parametrize("engine", ENGINES)
def test_parameters_need_an_analysis(engine):
    # geometric(p): the pole 1/(1 - p) is not located without a range for p; an analysis vouches for p
    dist = _factory(engine).from_expr("p/(1 - (1 - p)*x)", "x").set_parameters("p")
    with pytest.raises(NonFiniteMarginalError, match="parameters p"):
        dist.get_expected_value_of("x")
    with known_convergent_series(True):
        assert sympy.simplify(_sympy(dist.get_expected_value_of("x")) - (1 - sympy.Symbol("p")) / sympy.Symbol("p")) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_candidate_checks_keep_their_fast_path(engine):
    # with known_convergent_series(False) (candidate checks), only checked_marginal proves convergence; the other
    # evaluations are guarded by the checks themselves (#276, finite mass of sub-invariants)
    dist = _factory(engine).from_expr(DIVERGENT_EXIT, "x", "y")
    with known_convergent_series(False):
        assert _sympy(dist.get_probability_of("x = 0")) == -2
        with pytest.raises(NonFiniteMarginalError, match="not proven"):
            dist.checked_marginal("x")


@pytest.mark.parametrize("engine", ["SYMPY", "SYMENGINE", "GINAC"])
def test_analysis_results_with_parameters_are_summarised(engine):
    # the summary of an analysis result evaluates the output at 1 after the analysis; it is a finite measure
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    source = "rparam p;\nnat x;\nx := geometric(p);\n"
    config = ForwardAnalysisConfig(engine=getattr(Engine, engine))
    session = AnalysisSession(parse_pgcl(source, "pgcl"), source=source, syntax="pgcl", config=config,
                              decisions={}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result.get("error")
    assert result["distribution"]["expectations"], result["distribution"]


@pytest.mark.parametrize("engine", ENGINES)
def test_analysis_outputs_are_declared_finite_measures(engine):
    # compute_discrete_distribution declares its outputs, so the API queries them as the analysis does
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import compute_discrete_distribution
    factory = _factory(engine)
    source = "rparam p;\nnat x;\nx := geometric(p);\n"
    program = parse_pgcl(source, "pgcl")
    result, error_prob = compute_discrete_distribution(program, factory.one("x"), ForwardAnalysisConfig(engine=engine))
    p = sympy.Symbol("p")
    assert sympy.simplify(_sympy(result.get_expected_value_of("x")) - (1 - p) / p) == 0
    assert sympy.simplify(_sympy(result.get_probability_of("x = 0")) - p) == 0
    assert _sympy(error_prob.get_probability_mass()) == 0
    # the declaration belongs to the object: a distribution computed from it is proven again
    with pytest.raises(NonFiniteMarginalError, match="parameters p"):
        result.copy().get_expected_value_of("x")
