"""
Probabilities and distribution arguments that depend on program variables (issue #213), e.g.
``{ok = 1;} [1/k] {ok = 0;}`` or ``x = bernoulli(1/k)``. They are computed exactly by a case split
over the values of the variables if these take finitely many values; a value that is not a
probability (or not an admissible argument) in a state with positive probability is an error, and
infinitely many values are "not implemented" (except for choices with a probability α + β/(k + a), which act
as operators, #323; see ``tests/analysis/test_counter_operators.py``). The expected generating functions are computed by
hand, state by state.
"""
import re
from fractions import Fraction

import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl import Program
from prodigy.pgcl.compiler import compile_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _run(source: str, engine, simplification: bool):
    program = compile_pgcl(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification, normalize=False)
    result, error_prob = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, error_prob, config.factory


def _sum(terms) -> str:
    return " + ".join(f"({term})" for term in terms)


# k is uniform on {1, 2, 3}: the state k = j has probability 1/3
UNIF_K = "k: nat = unif(1, 3);\n"


def _over_k(term) -> str:
    return _sum(f"k^{j}/3 * ({term(j)})" for j in (1, 2, 3))


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    # the program of the issue: ok = 1 with probability 1/k
    pytest.param(UNIF_K + "ok: nat = 1;\n{ok = 1;} [1/k] {ok = 0;}",
                 "k*ok/3 + k^2*ok/6 + k^2/6 + k^3*ok/9 + 2*k^3/9", id="choice"),
    pytest.param(UNIF_K + "x: nat = 0;\n{x = 1;} [(k - 1)/2] {x = 2;}",
                 _over_k(lambda j: f"({j} - 1)/2*x + (1 - ({j} - 1)/2)*x^2"), id="choice-zero-and-one"),
    pytest.param(UNIF_K + "x: nat = bernoulli(1/k);", _over_k(lambda j: f"1 - 1/{j} + x/{j}"),
                 id="bernoulli"),
    pytest.param(UNIF_K + "x: nat = geometric(1/(k+1));",
                 _over_k(lambda j: f"1/({j}+1) / (1 - {j}/({j}+1)*x)"), id="geometric"),
    pytest.param(UNIF_K + "x: nat = binomial(k, 1/2);", _over_k(lambda j: f"((1 + x)/2)^{j}"),
                 id="binomial"),
    pytest.param(UNIF_K + "x: nat = unif(0, k);",
                 _over_k(lambda j: _sum(f"x^{i}/({j}+1)" for i in range(j + 1))), id="uniform"),
    pytest.param(UNIF_K + "n: nat = 2;\nx: nat = iid(bernoulli(1/k), n);",
                 _over_k(lambda j: f"n^2 * (1 - 1/{j} + x/{j})^2"), id="iid"),
    # the assigned variable in the argument: its value before the assignment counts
    pytest.param("x: nat = unif(1, 2);\nx = binomial(x, 1/2);", "1/2*(1 + x)/2 + 1/2*((1 + x)/2)^2",
                 id="assigned-variable"),
    # a Pólya urn after three draws: red - 1 is uniform on {0, ..., 3}
    pytest.param("red: nat = 1;\nblack: nat = 1;\n"
                 "loop (3) { {red = red + 1;} [red/(red+black)] {black = black + 1;} }",
                 _sum(f"red^{r}*black^{5 - r}/4" for r in range(1, 5)), id="polya-urn"),
    # nested choices: the inner probability depends on the outcome of the outer one
    pytest.param("x: nat = bernoulli(1/2);\ny: nat = 0;\n{y = 1;} [x/3 + 1/3] {skip;}",
                 "1/2*(2/3 + y/3) + 1/2*x*(1/3 + 2/3*y)", id="after-choice"),
])
def test_case_split_is_exact(engine, simplification, source, expected):
    result, error_prob, factory = _run(source, engine, simplification)
    assert result == factory.from_expr(expected, *result.get_variables())
    assert error_prob.is_zero_dist()


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
def test_parameters_stay_symbolic(engine, simplification):
    source = "param p: real;\nk: nat = unif(1, 2);\nx: nat = 0;\n{x = 1;} [p/k] {x = 2;}"
    result, _, factory = _run(source, engine, simplification)
    expected = _sum(f"k^{j}/2 * (p/{j}*x + (1 - p/{j})*x^2)" for j in (1, 2))
    assert result == factory.from_expr(expected, *result.get_variables())


@pytest.mark.parametrize("engine", ENGINES)
def test_observations_in_branches_are_weighted(engine):
    # k = 2 (probability 1/2) takes the left branch with probability 1/2 and fails the observation
    source = "k: nat = unif(1, 2);\n{observe(k == 1);} [1/k] {skip;}"
    result, error_prob, factory = _run(source, engine, False)
    assert result == factory.from_expr("k/2 + k^2/4", *result.get_variables())
    assert Fraction(error_prob.get_probability_mass()) == Fraction(1, 4)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, message", [
    pytest.param("k: nat = unif(0, 2);\nx: nat = 0;\n{x = 1;} [1/k] {x = 2;}",
                 "is undefined in the state k = 0", id="division-by-zero"),
    pytest.param("k: nat = unif(0, 3);\nx: nat = 0;\n{x = 1;} [k/2] {x = 2;}",
                 "is 3/2 in the state k = 3", id="above-one"),
    pytest.param("x: nat = 0;\n{x = 1;} [3/2] {x = 2;}", "is 3/2, which is not a probability",
                 id="constant-above-one"),
    pytest.param("k: nat = unif(0, 2);\nx: nat = bernoulli(1/k);",
                 "bernoulli(1 / k) is not defined in the state k = 0", id="bernoulli"),
    pytest.param("k: nat = unif(2, 3);\nx: nat = bernoulli(k/2);",
                 "bernoulli(k / 2) is not defined in the state k = 3", id="bernoulli-above-one"),
    # constant arguments are checked the same way on every engine
    pytest.param("x: nat = bernoulli(3/2);", "is 3/2; it must be in [0, 1]", id="constant-bernoulli"),
    pytest.param("x: nat = geometric(0);", "is 0; it must be in (0, 1]", id="constant-geometric"),
    pytest.param("x: nat = binomial(2, 3/2);", "is 3/2; it must be in [0, 1]", id="constant-binomial"),
    pytest.param("k: nat = unif(0, 1);\nx: nat = negative_binomial(2, k);", "is 0; it must be in (0, 1]",
                 id="negative-binomial"),
    pytest.param("x: nat = logdist(1);", "is 1; it must be in (0, 1)", id="constant-logdist"),
    pytest.param("x: nat = unif(3, 1);", "the lower bound must not exceed", id="constant-uniform"),
    pytest.param("k: nat = unif(0, 1);\nx: nat = unif(1, k);", "the lower bound must not exceed",
                 id="uniform"),
])
def test_values_that_are_not_probabilities_are_rejected(engine, source, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _run(source, engine, False)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", [
    # not of the form α + β/(k + a) (#323)
    pytest.param("k: nat = geometric(1/2);\nx: nat = 0;\n{x = 1;} [1/(k*k+1)] {x = 2;}", id="choice"),
    pytest.param("k: nat = geometric(1/2);\nx: nat = bernoulli(1/(k+1));", id="bernoulli"),
])
def test_infinitely_many_values_are_not_implemented(engine, source):
    with pytest.raises(NotImplementedError, match="infinitely many values"):
        _run(source, engine, False)


@pytest.mark.parametrize("engine", ENGINES)
def test_states_unreachable_in_a_branch_are_not_checked(engine):
    # k = 0 is excluded by the guard, so 1/k is only evaluated for k = 1 and k = 2
    source = "k: nat = unif(0, 2);\nx: nat = 0;\nif (k > 0) { {x = 1;} [1/k] {x = 2;} }"
    result, _, factory = _run(source, engine, False)
    assert result == factory.from_expr("1/3 + k/3*x + k^2/3*(x/2 + x^2/2)", *result.get_variables())


@pytest.mark.parametrize("engine", ENGINES)
def test_geometric_with_parameter_one_is_a_point_mass(engine):
    result, _, factory = _run("k: nat = unif(1, 2);\nx: nat = geometric(1/k);", engine, False)
    assert result == factory.from_expr("k/2 + k^2/2 * (1/2)/(1 - x/2)", *result.get_variables())


@pytest.mark.parametrize("engine", ENGINES)
def test_admissible_boundary_arguments(engine):
    result, _, factory = _run("x: nat = geometric(1);\ny: nat = unif(2, 2);\nz: nat = bernoulli(0);",
                              engine, False)
    assert result == factory.from_expr("y^2", *result.get_variables())
