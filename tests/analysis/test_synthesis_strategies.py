"""Tests for the synthesis strategies, in particular FROM_HEURISTICS (issue #67)."""
import pytest
from click.testing import CliRunner
from prodigy.pgcl import parse_pgcl

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.positivity.iterate_terms import IterateTermsHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.rational_function import RationalFunctionDenomSign, \
    SingleRationalFunction
from prodigy.analysis.evtinvariants.heuristics.strategies import (SELECTABLE_POSITIVITY_HEURISTICS,
                                                                  DefaultStrategy, FinitePolynomial, FromHeuristics,
                                                                  SynthesisStrategies)
from prodigy.analysis.evtinvariants.heuristics.templates.polynomial_function import PolynomialEnumeration
from prodigy.analysis.evtinvariants.heuristics.templates.rational_function import RationalFunctionMaxDeg
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli

# The EVT invariant 2*x + 1 is a polynomial, so the polynomial heuristics find it.
GEOMETRIC = """nat x;
x := 1
while (x = 1) { {x := x - 1} [1/2] {skip} }
"""

FACTORY = ForwardAnalysisConfig().factory


@pytest.mark.parametrize("template, template_class", [
    (TemplateHeuristics.DEFAULT, RationalFunctionMaxDeg),
    (TemplateHeuristics.POLY, PolynomialEnumeration),
])
@pytest.mark.parametrize("positivity, positivity_class", [
    (PositivityHeuristics.DEFAULT, RationalFunctionDenomSign),
    (PositivityHeuristics.MIXED_RAT_FUNC, RationalFunctionDenomSign),
    (PositivityHeuristics.TRUE_RAT_FUNC, SingleRationalFunction),
    (PositivityHeuristics.POLYNOMIAL, FinitePolynomialPositivity),
    (PositivityHeuristics.ITERATE_TERMS, IterateTermsHeuristic),
])
def test_from_heuristics_uses_the_chosen_heuristics(template, template_class, positivity, positivity_class):
    strategy = SynthesisStrategies.make(SynthesisStrategies.FROM_HEURISTICS, ["x"], FACTORY,
                                        templ_heuristic=template, positivity_heuristic=positivity)
    assert isinstance(strategy, FromHeuristics)
    assert isinstance(strategy.template_heuristics, template_class)
    assert isinstance(strategy.positivity_heuristics, positivity_class)
    assert next(iter(strategy.template_heuristics.generate())) is not None


def test_poly_template_respects_the_maximum_degree():
    strategy = FromHeuristics(["x"], FACTORY, TemplateHeuristics.POLY, PositivityHeuristics.POLYNOMIAL, max_degree=2)
    assert len(list(strategy.template_heuristics.generate())) == 3


@pytest.mark.parametrize("missing", [
    {"positivity_heuristic": PositivityHeuristics.DEFAULT},
    {"templ_heuristic": TemplateHeuristics.DEFAULT},
    {},
])
def test_from_heuristics_needs_both_heuristics(missing):
    with pytest.raises(HeuristicsError, match="from_heuristics needs a"):
        SynthesisStrategies.make(SynthesisStrategies.FROM_HEURISTICS, ["x"], FACTORY, **missing)


@pytest.mark.parametrize("combinator", [PositivityHeuristics.OR, PositivityHeuristics.AND,
                                        PositivityHeuristics.NOT, PositivityHeuristics.INDIVIDUAL_SUM])
def test_combining_positivity_heuristics_are_rejected(combinator):
    assert combinator.name.lower() not in SELECTABLE_POSITIVITY_HEURISTICS
    with pytest.raises(HeuristicsError, match="cannot be used on its own"):
        SynthesisStrategies.make(SynthesisStrategies.FROM_HEURISTICS, ["x"], FACTORY,
                                 templ_heuristic=TemplateHeuristics.DEFAULT, positivity_heuristic=combinator)


def test_other_strategies_keep_their_own_heuristics():
    strategy = SynthesisStrategies.make(SynthesisStrategies.FIN_POLY, ["x"], FACTORY,
                                        templ_heuristic=TemplateHeuristics.DEFAULT,
                                        positivity_heuristic=PositivityHeuristics.TRUE_RAT_FUNC)
    assert isinstance(strategy, FinitePolynomial)
    assert isinstance(strategy.template_heuristics, PolynomialEnumeration)
    assert isinstance(strategy.positivity_heuristics, FinitePolynomialPositivity)
    assert isinstance(SynthesisStrategies.make(SynthesisStrategies.DEFAULT, ["x"], FACTORY), DefaultStrategy)


def test_from_config_passes_the_configured_heuristics():
    config = ForwardAnalysisConfig(strategy=SynthesisStrategies.FROM_HEURISTICS,
                                   templ_heuristic=TemplateHeuristics.POLY,
                                   positivity_heuristic=PositivityHeuristics.POLYNOMIAL)
    strategy = SynthesisStrategies.from_config(config, ["x"])
    assert isinstance(strategy.template_heuristics, PolynomialEnumeration)
    assert isinstance(strategy.positivity_heuristics, FinitePolynomialPositivity)


def test_cli_synthesises_with_the_chosen_heuristics(tmp_path):
    program = tmp_path / "geometric.pgcl"
    program.write_text(GEOMETRIC)
    runner = CliRunner()
    poly = runner.invoke(cli, ["--strategy", "from_heuristics", "--template-heuristic", "poly",
                               "--pos-heuristic", "polynomial", "invariant_synthesis", str(program)])
    assert poly.exit_code == 0, poly.output
    assert "Invariant: 2*x + 1" in poly.output
    assert "d_0" not in poly.output  # polynomial templates have no denominator
    rational = runner.invoke(cli, ["--strategy", "from_heuristics", "invariant_synthesis", str(program)])
    assert rational.exit_code == 0, rational.output
    assert "n_0/d_0" in rational.output and "Invariant: 2*x + 1" in rational.output


def test_cli_rejects_heuristics_without_from_heuristics(tmp_path):
    program = tmp_path / "geometric.pgcl"
    program.write_text(GEOMETRIC)
    runner = CliRunner()
    result = runner.invoke(cli, ["--template-heuristic", "poly", "invariant_synthesis", str(program)])
    assert result.exit_code == 2
    assert "only apply to --strategy from_heuristics" in result.output
    result = runner.invoke(cli, ["--strategy", "from_heuristics", "--pos-heuristic", "or",
                                 "invariant_synthesis", str(program)])
    assert result.exit_code == 2
    assert "Invalid value for '--pos-heuristic'" in result.output


def test_session_synthesises_with_the_chosen_heuristics():
    config = ForwardAnalysisConfig(use_simplification=True, strategy=SynthesisStrategies.FROM_HEURISTICS,
                                   templ_heuristic=TemplateHeuristics.POLY,
                                   positivity_heuristic=PositivityHeuristics.POLYNOMIAL)
    session = AnalysisSession(parse_pgcl(GEOMETRIC), source=GEOMETRIC, config=config,
                              decisions={"0": "evt_synthesis"}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok" and result["loops"][0]["kind"] == "exact"
    certificate = result["loops"][0]["certificates"][0]
    assert certificate["synthesis_strategy"] == "FROM_HEURISTICS"
    assert certificate["invariant"] == "2*x + 1"
