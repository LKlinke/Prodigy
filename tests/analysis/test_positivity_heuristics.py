"""
The positivity heuristics are three-valued and sound (#295): ``True`` only if every coefficient of the power
series in the program variables is proven non-negative, ``False`` only if a coefficient is proven negative,
``None`` otherwise. Program parameters are real coefficients, never series variables; ``1/(1 - p)`` is negative
for ``p = 3``. Sums of terms that are non-negative by construction are recognised before they are combined over
a common denominator (#297).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import nonneg_by_construction
from prodigy.analysis.evtinvariants.heuristics.positivity.dominat_root_rejection import DominantRootHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.positivity.individual_sum_terms import SumPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.iterate_terms import IterateTermsHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.logical import AndHeuristic, NotHeuristic, OrHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.rational_function import RationalFunctionDenomSign, \
    SingleRationalFunction
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.lowerbounds import subinvariant
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]
VARIABLES = ["x", "y", "t"]


def _factory(engine):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine).factory


def _leaf_heuristics(factory):
    return [
        RationalFunctionDenomSign(factory, variables=VARIABLES),
        SingleRationalFunction(variables=VARIABLES),
        SumPositivity(SingleRationalFunction(variables=VARIABLES)),
    ]


class _Fixed(PositivityHeuristic):
    """Always gives the same verdict."""

    def __init__(self, verdict):
        super().__init__()
        self.verdict = verdict

    def _is_positive(self, f):
        return self.verdict


# -- parameters are coefficients, not series variables (#295) -----------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function", ["1/(1 - p)", "x/(2 - p)", "p*x/(1 - x)", "1/(1 - p*x)", "x/(1 - x - p)",
                                      "(1 - p)/(1 - x)"])
def test_a_parameter_of_unknown_sign_is_not_decided(engine, function):
    for heuristic in _leaf_heuristics(_factory(engine)):
        assert heuristic.is_positive(function) is None, heuristic
    assert nonneg_by_construction(function, VARIABLES) is False


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function", ["p^2*x/(1 - x)", "(p^2 + 1)/(2 - x)", "x/(1 + p^2 - x)", "p^2 + x*y"])
def test_a_sign_that_holds_for_every_real_parameter_is_used(engine, function):
    for heuristic in _leaf_heuristics(_factory(engine)):
        assert heuristic.is_positive(function) is True, heuristic
    assert nonneg_by_construction(function, VARIABLES) is True


@pytest.mark.parametrize("function, expected", [
    ("0", True), ("3*x^2 + y", True), ("x - 1", False), ("p*x", None), ("p^2*x + 1", True), ("-p^2 - 1", False),
    ("x - p*y", None), ("x - y + p*t", False),
])
def test_polynomial(function, expected):
    assert FinitePolynomialPositivity(variables=VARIABLES).is_positive(function) is expected


def test_single_rational_function_needs_a_proven_numerator():
    # the numerator p*x was only "not negative" (undecided); it was taken as non-negative
    assert SingleRationalFunction(variables=VARIABLES).is_positive("p*x/(1 - x)") is None
    assert SingleRationalFunction(variables=VARIABLES).is_positive("0") is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function, expected", [
    ("1 + x + x^2", True), ("1 - x", False), ("0", True),
    # with x as series variable, 1 - p is a constant of unknown sign; p as a variable gave "negative"
    ("1 - p", None), ("(1 - p)*x + p^2", None), ("1 - p^2 - x", False),
    ("1/(1 - x)", None),
])
def test_iterate_terms(engine, function, expected):
    heuristic = IterateTermsHeuristic(_factory(engine), variables=["x"])
    assert heuristic.is_positive(function) is expected


def test_dominant_root_refutes_only_by_pringsheim():
    heuristic = DominantRootHeuristic(variables=["x"])
    assert heuristic.is_positive("1/(1 + x^2)") is False
    assert heuristic.is_positive("1/(1 - x)") is None
    assert heuristic.is_positive("1/x") is None
    assert heuristic.is_positive("1/(1 - p)") is None  # a parameter: cannot decide (it used to raise)


def test_dominant_root_is_conservative_outside_its_preconditions():
    heuristic = DominantRootHeuristic(variables=["x"])
    assert heuristic.is_positive("1/(1 - x + x^2)") is False      # complex dominant roots: eventually negative
    assert heuristic.is_positive("1/(1 - x/2)") is None           # positive real dominant root
    assert heuristic.is_positive("1/(1 + x)") is None             # real dominant root (negative): not decided
    assert heuristic.is_positive("1/(1 - x^3)") is None           # moduli tie: undecided
    assert heuristic.is_positive("1/(1 - p*x + x^2)") is None     # a parameter
    assert heuristic.is_positive("x*y/(1 - x*y + x^2*y^2)") is None  # two variables
    assert heuristic.is_positive("1/(1.0 - x + x^2)") is None      # floating point
    assert heuristic.is_positive("1") is None or heuristic.is_positive("1") is True  # no variable: no crash


def test_default_strategy_refutes_only_where_the_proof_fails():
    from prodigy.analysis.evtinvariants.heuristics.strategies import DefaultStrategy
    strategy = DefaultStrategy(variables=["x"], dist_factory=_factory("sympy"))
    heuristic = strategy.positivity_heuristics
    assert heuristic.is_positive("1/(1 - x + x^2)") is False
    assert heuristic.is_positive("1/(2 - x)") is True             # proven by the existing heuristic
    assert heuristic.is_positive("1/(1 + x)") is None


# -- combining heuristics ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("verdicts, expected", [
    ((True, True), True), ((True, None), None), ((None, None), None), ((None, False), False), ((True, False), False),
])
def test_and_is_three_valued(verdicts, expected):
    # AndHeuristic returned False ("negative coefficient") when one of them was undecided
    assert AndHeuristic(None, *map(_Fixed, verdicts)).is_positive("x") is expected
    assert PositivityHeuristics.create(PositivityHeuristics.AND, None, *map(_Fixed, verdicts)) \
        .is_positive("x") is expected


@pytest.mark.parametrize("verdicts, expected", [
    ((None, True), True), ((None, None), None), ((False, None), False),
])
def test_or_takes_the_first_decided_verdict(verdicts, expected):
    assert OrHeuristic(None, *map(_Fixed, verdicts)).is_positive("x") is expected


def test_not_refutes_by_the_negation_only():
    polynomial = FinitePolynomialPositivity(variables=VARIABLES)
    heuristic = NotHeuristic(polynomial)
    # inverting the verdict on f called every non-negative function negative
    assert heuristic.is_positive("1 + x") is None
    assert heuristic.is_positive("x - y") is None
    assert heuristic.is_positive("-1 - x") is False
    assert heuristic.is_positive("0") is True
    assert NotHeuristic(_Fixed(True)).is_positive("x") is False


# -- sums that are non-negative by construction (#297) ----------------------------------------------------------

@pytest.mark.parametrize("function", [
    "5/(2 - x) + t*(1/2 + x^5/2)",
    # the same function as GiNaC writes it, over a common denominator
    "-1/2*(-2+x)^(-1)*(10+2*t*x^5-t*x^6+2*t-t*x)",
    "t*(x**5/2 + 1/2) + 5/(2 - x)",
    "(1 - x^2)/(1 - x)",
    "x*y/(1 - 9/50*(x + y))*(1 + t/10)",
    "(50 - 9*x)^(-2)",
    "(1/(2 - x))^3 + y",
    "(2 + 2*x - x^2)/((1 - x/2)*(1 - y/3))",
])
def test_non_negative_by_construction(function):
    assert nonneg_by_construction(function, VARIABLES) is True
    assert subinvariant.nonneg_by_construction(function, VARIABLES) is True
    assert RationalFunctionDenomSign(None, variables=VARIABLES).is_positive(function) is True


@pytest.mark.parametrize("function", [
    "1/(1 + x)", "x - x^2", "1/(2*x - 1)", "1/x", "(1 - x)/(1 - y)", "1/(2 - x) - 1/(3 - y)", "sqrt(1 - x)",
    "5/(2 - x) - t",
])
def test_not_non_negative_by_construction(function):
    assert nonneg_by_construction(function, VARIABLES) is False
    assert RationalFunctionDenomSign(None, variables=VARIABLES).is_positive(function) is None


# -- every strategy hands the variables to its heuristic ---------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("strategy", [s for s in SynthesisStrategies if s != SynthesisStrategies.FROM_HEURISTICS])
def test_strategies_do_not_treat_parameters_as_variables(engine, strategy):
    built = SynthesisStrategies.make(strategy, ["x"], _factory(engine), max_degree=1)
    function = "p*x" if strategy in (SynthesisStrategies.INF_POLY, SynthesisStrategies.FIN_POLY) else "1/(1 - p*x)"
    assert built.positivity_heuristics.is_positive(sympy.S(function)) is None


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("heuristic", [PositivityHeuristics.POLYNOMIAL, PositivityHeuristics.TRUE_RAT_FUNC,
                                       PositivityHeuristics.MIXED_RAT_FUNC, PositivityHeuristics.ITERATE_TERMS])
def test_selectable_heuristics_do_not_treat_parameters_as_variables(engine, heuristic):
    from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
    built = SynthesisStrategies.make(SynthesisStrategies.FROM_HEURISTICS, ["x"], _factory(engine),
                                     templ_heuristic=TemplateHeuristics.DEFAULT, positivity_heuristic=heuristic,
                                     max_degree=1)
    assert built.positivity_heuristics.is_positive(sympy.S("p*x + 1 - p")) is None


# -- end to end ----------------------------------------------------------------------------------------------

# a geometric loop with a parameter: its least fixed point (1 + (1 - p)*t)/(1 - p*x) has non-negative
# coefficients only for 0 <= p <= 1, which the program does not state (parameters have no bounds)
GEOMETRIC_PARAMETER = """nat x; nat t; rparam p;
while (t = 0) { {t := 1} [1 - p] {x := x + 1} }
?Pr[x = 0]
"""


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("strategy", [SynthesisStrategies.DEFAULT, SynthesisStrategies.SUM_THEN_RATIONAL,
                                      SynthesisStrategies.APART_OR_TOGETHER])
def test_parameter_invariant_is_undecided_not_refuted(engine, strategy):
    # apart_or_together reported "The invariant has negative coefficients": it expanded p as a series variable
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=engine, strategy=strategy)
    session = AnalysisSession(parse_pgcl(GEOMETRIC_PARAMETER), source=GEOMETRIC_PARAMETER, config=config,
                              decisions={"0": {"strategy": "evt_invariant",
                                               "params": {"invariant": "(1 + (1 - p)*t)/(1 - p*x)"}}},
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.UNKNOWN
    assert loop["certificates"][0]["nonnegative"] is None
    assert not any("negative coefficients" in message for message in loop["messages"])


def test_parameter_assumptions_of_the_caller_are_kept():
    # input distributions hand a nat parameter over as a non-negative integer symbol
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    x = sympy.Symbol("x")
    assert nonneg_by_construction(n * x / (2 - x), ["x"]) is True
    assert nonneg_by_construction(sympy.Symbol("p") * x / (2 - x), ["x"]) is False
    assert nonneg_by_construction(x / (n + 1 - x), ["x"]) is True
