"""The command group ``prodigy``: its global options become the :class:`ForwardAnalysisConfig` of the command."""
from typing import Any, Optional, Tuple

import click

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import (ConfigurationError, HeuristicsError, InvalidProbabilityError,
                                         UndefinedDistributionError, UnsupportedFeatureError, VerificationError)
from prodigy.analysis.tools import (SELECTABLE_POSITIVITY_HEURISTICS, InputDistributionError, PositivityHeuristics,
                                    SolverType, SynthesisStrategies, TemplateHeuristics)


class _CliGroup(click.Group):
    """Reports programs the analysis cannot handle as a plain error instead of a traceback."""

    def invoke(self, ctx: click.Context) -> Any:  # click leaves the return value of a command untyped
        try:
            return super().invoke(ctx)
        except (UnsupportedFeatureError, UndefinedDistributionError, InputDistributionError, HeuristicsError,
                VerificationError, InvalidProbabilityError) as exc:
            # e.g. a refuted sub-invariant (#321) or the undecided candidates of a synthesis
            raise click.ClickException("\n".join([str(exc), *getattr(exc, "messages", ())])) from exc


def _synthesis(strategy: str, template_heuristic: Optional[str], pos_heuristic: Optional[str]
               ) -> Tuple[SynthesisStrategies, TemplateHeuristics, PositivityHeuristics]:
    """The strategy and heuristics of EVT invariant synthesis from the options ``--strategy``,
    ``--template-heuristic`` and ``--pos-heuristic``."""
    synthesis_strategy = SynthesisStrategies.__members__[strategy.upper()]
    if synthesis_strategy != SynthesisStrategies.FROM_HEURISTICS and (template_heuristic or pos_heuristic):
        raise click.UsageError("--template-heuristic and --pos-heuristic only apply to --strategy from_heuristics.")
    return (synthesis_strategy, TemplateHeuristics.__members__[(template_heuristic or 'default').upper()],
            PositivityHeuristics.__members__[(pos_heuristic or 'default').upper()])


def _solver(solver: str) -> SolverType:
    try:
        return SolverType.named(solver)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--solver") from exc


def _engine(engine: str) -> ForwardAnalysisConfig.Engine:
    try:
        return ForwardAnalysisConfig.Engine.from_name(engine)
    except ConfigurationError as exc:
        raise click.BadParameter(str(exc), param_hint="--engine") from exc


@click.group(cls=_CliGroup)
@click.pass_context
@click.option('--engine', type=str, required=False, default='')
@click.option("--strategy",
              type=click.Choice([name.lower() for name in SynthesisStrategies.__members__], case_sensitive=False),
              required=False, default='default', show_default=True,
              help="Strategy for EVT invariant synthesis. 'from_heuristics' combines the heuristics given with "
                   "--template-heuristic and --pos-heuristic.")
@click.option("--solver", type=str, required=False, default='sympy')
@click.option("--template-heuristic",
              type=click.Choice([name.lower() for name in TemplateHeuristics.__members__], case_sensitive=False),
              required=False, default=None,
              help="Template heuristic for --strategy from_heuristics (default: default, rational functions).")
@click.option("--pos-heuristic",
              type=click.Choice(SELECTABLE_POSITIVITY_HEURISTICS, case_sensitive=False),
              required=False, default=None,
              help="Positivity heuristic for --strategy from_heuristics (default: default, i.e. mixed_rat_func).")
@click.option('--intermediate-results', is_flag=True, required=False, default=False,
              help="Print every step of the analysis (line, instruction, distribution, guarantee) as it is done. "
                   "'prodigy analyze --steps' records the same steps, with --json as JSON.")
@click.option("--stepwise", is_flag=True, required=False, default=False,
              help="Like --intermediate-results, and wait for Enter after each step.")
@click.option('--no-simplification', is_flag=True, required=False, default=False,
              help="Show the result as computed. By default the shown result (not the computation) is simplified "
                   "with bounded effort: the shortest of cheap exact rewrites (together, cancel, factor) that is "
                   "checked to be equal, within a time limit of a few seconds.")
@click.option('--use-latex', is_flag=True, required=False, default=False)
@click.option("--no-normalize", is_flag=True, required=False, default=False)
@click.option("--show-all-invs", is_flag=True, required=False, default=False)
# one parameter per option, as click passes them; they are most of the locals
def cli(ctx: click.Context, *,  # pylint: disable=too-many-arguments,too-many-locals
        engine: str, strategy: str, solver: str, template_heuristic: Optional[str], pos_heuristic: Optional[str],
        intermediate_results: bool, stepwise: bool, no_simplification: bool, use_latex: bool, no_normalize: bool,
        show_all_invs: bool) -> None:
    ctx.ensure_object(dict)
    # checked in this order: solver, synthesis options, engine
    solver_type = _solver(solver)
    synthesis_strategy, template, positivity = _synthesis(strategy, template_heuristic, pos_heuristic)
    engine_choice = _engine(engine)
    ctx.obj['CONFIG'] = ForwardAnalysisConfig(
        engine=engine_choice,
        show_intermediate_steps=intermediate_results,
        step_wise=stepwise,
        use_simplification=not no_simplification,
        use_latex=use_latex,
        normalize=not no_normalize,
        strategy=synthesis_strategy,
        templ_heuristic=template,
        positivity_heuristic=positivity,
        show_all_invs=show_all_invs,
        solver_type=solver_type
    )
    # The engine settings (simplification, LaTeX output) hold for this command, printing included.
    ctx.with_resource(ctx.obj['CONFIG'].activate())
