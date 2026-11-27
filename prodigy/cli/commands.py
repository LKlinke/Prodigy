"""The analysis commands: ``main`` (interactive), ``analyze`` (decisions given up front), ``outline``,
``loop_strategies``, the lists of synthesis strategies and heuristics, and ``web`` (the GUI)."""
import json
import sys
import time
from typing import IO, Optional, Tuple

import click

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import loop_strategies, strategy_groups
from prodigy.analysis.session import AnalysisSession, compute_discrete_distribution, parse_decisions
from prodigy.analysis.tools import SELECTABLE_POSITIVITY_HEURISTICS, SynthesisStrategies, TemplateHeuristics
from prodigy.cli.common import (LOOP_OPTION_HELP, checked_decisions, checked_program, collect_decisions,
                                initial_distribution, logger, syntax_option)
from prodigy.cli.group import cli
from prodigy.cli.output import AnalysisJson, draw_plots, print_summary
from prodigy.distribution.display import display_form
from prodigy.util.color import Style


@cli.command('main')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('input_dist', type=str, required=False)
@click.option('--show-input-program',
              is_flag=True,
              required=False,
              default=False)
@click.option('--loop', 'loop_specs', multiple=True,
              help=LOOP_OPTION_HELP + " Loops without a decision are asked for.")
@syntax_option
def main(ctx: click.Context, *,  # pylint: disable=too-many-arguments  # one parameter per option, as click passes them
         program_file: IO, input_dist: Optional[str], show_input_program: bool, loop_specs: Tuple[str, ...],
         syntax: Optional[str]) -> None:
    """
    Compile the given program and print some information about it.
    """

    logger.info("Prodigy started.")

    # Parse and the input and do typechecking.
    logger.debug("Read file %s", program_file.name)
    logger.debug("Input distribution: %s", input_dist)
    program_source, program = checked_program(program_file, syntax)

    if show_input_program:
        print("Program source:")
        print(program_source)
        print()
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    dist = initial_distribution(config, input_dist, program)

    logger.debug("Start analysis.")
    decisions = checked_decisions(collect_decisions(loop_specs, None), program)
    session = AnalysisSession.interactive(program, source=program_source, config=config,
                                          decisions=decisions)
    start = time.perf_counter()
    with session.activate():
        dist, error_prob = compute_discrete_distribution(
            program, dist, config)
    seconds = time.perf_counter() - start
    draw_plots(session)
    _print_result(session, display_form(dist).distribution if config.use_simplification else dist, error_prob,
                  seconds)


def _print_result(session: AnalysisSession, shown: object, error_prob: object, seconds: float) -> None:
    """The text output of ``prodigy main``: the distribution as shown and the error probability, warnings, time."""
    print(Style.OKBLUE + "Result: \t" + Style.OKGREEN + "(" + str(shown) +
          ", " + str(error_prob) + ")" + Style.RESET)
    # e.g. a side condition that could not be proven (#424)
    for warning in session.warnings:
        print(f"{Style.YELLOW}Warning:{Style.RESET} {warning}")
    print(f"CPU-time elapsed: {seconds:04f} seconds")


@cli.command()
def print_strategies() -> None:
    print("Currently implemented strategies are: ")
    for strategy in SynthesisStrategies:
        print(strategy.name)


@cli.command()
def print_pos_heuristics() -> None:
    print("Positivity heuristics for --pos-heuristic (with --strategy from_heuristics): ")
    for name in SELECTABLE_POSITIVITY_HEURISTICS:
        print(name.upper())


@cli.command()
def print_template_heuristics() -> None:
    print("Currently implemented template generation heuristics are: ")
    for heuristic in TemplateHeuristics:
        print(heuristic.name)


def _print_steps(result: AnalysisJson) -> None:
    """The step trace of ``prodigy analyze --steps`` as text, with the hints of the trace."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.trace import describe_step
    for step in result["steps"]:
        print(describe_step(step))
    for hint in (result.get("trace") or {}).get("hints", []):
        print(f"{Style.YELLOW}Hint:{Style.RESET} {hint['text']}")


@cli.command('analyze')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('input_dist', type=str, required=False)
@click.option('--loop', 'loop_specs', multiple=True, help=LOOP_OPTION_HELP)
@click.option('--decisions', 'decisions_file', type=click.File('r'), required=False,
              help='JSON file {"0": {"strategy": ..., "params": {...}}, "*": ...}.')
@click.option('--json', 'as_json', is_flag=True, default=False, help="Print the JSON result.")
@click.option('--coefficients', type=int, default=30, show_default=True,
              help="Number of coefficients per marginal in the result (0 to skip).")
@click.option('--steps', is_flag=True, default=False,
              help="Record the step trace: for each instruction its line and column, the distribution before and "
                   "after it with its guarantee, marginals, time and size; loops add their iterates and the checks "
                   "of their certificates. Printed as text, or in the JSON result under 'steps' and 'trace'.")
@click.option('--watch', 'watches', multiple=True, metavar="QUERY",
              help="With --steps: evaluate this query after every top-level step, e.g. 'Pr[x == 3]' or 'Ex[x]' "
                   "(a .ppy expression; repeatable).")
@click.option('--step-iterations', type=click.IntRange(min=1), default=40, show_default=True,
              help="With --steps: iterates kept per loop (spread evenly over the iterations).")
@click.option('--output', type=click.File('w'), required=False, help="Write the JSON result to this file.")
@syntax_option
# one parameter per option, as click passes them; they are most of the locals
def analyze(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: click.Context, *, program_file: IO, input_dist: Optional[str], loop_specs: Tuple[str, ...],
        decisions_file: Optional[IO], as_json: bool, coefficients: int, steps: bool,
        watches: Tuple[str, ...], step_iterations: int, output: Optional[IO], syntax: Optional[str]) -> None:
    """
    Analyse a program without asking questions: every loop needs a strategy (--loop or
    --decisions). Queries are answered with sound intervals.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.trace import InvalidWatchError, TraceLimits
    program_source, program = checked_program(program_file, syntax)
    decisions = parse_decisions(checked_decisions(collect_decisions(loop_specs, decisions_file), program))
    if watches and not steps:
        raise click.UsageError("--watch needs --steps.")
    try:
        session = AnalysisSession(program, source=program_source, config=ctx.obj['CONFIG'], decisions=decisions,
                                  input_distribution=input_dist, capture_output=True,
                                  record_steps=steps, coefficient_limit=coefficients,
                                  trace_limits=TraceLimits(iterates=step_iterations), watches=watches)
    except InvalidWatchError as exc:
        raise click.BadParameter(str(exc), param_hint="--watch") from exc
    result = session.run().to_json()
    if steps and not as_json:
        _print_steps(result)
    if output is not None:
        json.dump(result, output, indent=2)
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
    if result["status"] == "needs_decision":
        sys.exit(2)
    if result["status"] != "ok":
        sys.exit(1)


@cli.command('outline')
@click.argument('program_file', type=click.File('r'))
@click.option('--json', 'as_json', is_flag=True, default=False)
@syntax_option
def outline(*, program_file: IO, as_json: bool, syntax: Optional[str]) -> None:
    """
    List the loops (with the ids used by --loop) and queries of a program.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.language.outline import outline as outline_of
    _, program = checked_program(program_file, syntax)
    info = outline_of(program).to_json()
    if as_json:
        print(json.dumps(info, indent=2))
        return
    print("Variables: " + ", ".join(f"{name}: {typ}" for name, typ in info["variables"].items()))
    if info["parameters"]:
        print("Parameters: " + ", ".join(info["parameters"]))
    for loop in info["loops"]:
        where = f"line {loop['line']}" if loop["line"] else "?"
        print(f"loop {loop['id']} ({where}, depth {loop['depth']}): while ({loop['cond']})")
    for query in info["queries"]:
        print(f"query {query['index']} (line {query['line']}): {query['text']}")


@cli.command('loop_strategies')
@click.option('--json', 'as_json', is_flag=True, default=False)
def print_loop_strategies(*, as_json: bool) -> None:
    """
    List the strategies for while loops and their parameters.
    """
    strategies = loop_strategies()
    if as_json:
        print(json.dumps([s.to_json() for s in strategies], indent=2))
        return
    for _, group_title, group_text, members in strategy_groups():
        print(f"{group_title}: {group_text}\n")
        for strategy in members:
            print(f"  {strategy.name}  ({strategy.outcome or strategy.soundness})  {strategy.title}")
            if strategy.summary:
                print(f"      {strategy.summary}")
            print(f"      {strategy.description}")
            for param in strategy.params:
                default = f", default {param.default}" if param.default is not None else ""
                print(f"      - {param.name} [{param.kind}{default}]: {param.help}")
            print()


@cli.command('web')
@click.pass_context
@click.option('--host', default='127.0.0.1', show_default=True)
@click.option('--port', default=8080, show_default=True, type=int)
@click.option('--no-browser', is_flag=True, default=False, help="Do not open a browser window.")
@click.option('--time-limit', default=120.0, show_default=True, type=float,
              help="Seconds an analysis may run before it is stopped.")
@click.option('--max-jobs', default=2, show_default=True, type=int,
              help="Analyses and checks running at the same time.")
@click.option('--max-background-jobs', default=2, show_default=True, type=int,
              help="Background computations of the GUI (grids of the joint view, suggestions) running at the same "
                   "time, in addition to --max-jobs; a new grid replaces the oldest one.")
@click.option('--memory-limit', default=None, type=int, help="Memory limit per analysis in MB (Linux).")
@click.option('--examples', 'examples_dir', default=None, type=click.Path(exists=True, file_okay=False),
              help="Folder with example programs (default: pgfexamples of this checkout or package).")
@click.option('--allowed-host', 'allowed_hosts', multiple=True,
              help="Additional host name the server may be reached under (when not on localhost).")
def web(  # pylint: disable=too-many-arguments  # one parameter per option, as click passes them
        ctx: click.Context, *, host: str, port: int, no_browser: bool, time_limit: float, max_jobs: int,
        max_background_jobs: int, memory_limit: Optional[int], examples_dir: Optional[str],
        allowed_hosts: Tuple[str, ...]) -> None:
    """
    Start the web GUI (a local web server) and open it in the browser.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.web.app import serve
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    serve(host=host, port=port, open_browser=not no_browser, time_limit=time_limit, max_jobs=max_jobs,
          max_background_jobs=max_background_jobs, memory_limit=memory_limit, examples_dir=examples_dir,
          default_engine=config.engine.name.lower(), allowed_hosts=allowed_hosts)
