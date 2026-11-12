"""
prodigy also offers a simple command-line interface for quick program inspection.

If you do not have prodigy installed globally, you can use `uv run prodigy INPUT` in the project folder.

.. click:: prodigy.cli:cli
   :prog: prodigy
   :show-nested:
"""

import json
import logging
import sys
import time
from itertools import combinations
from typing import IO, Any, Dict, Optional, Set, Tuple

import click
from lark.exceptions import UnexpectedInput
from prodigy.pgcl import Program, Var, compiler, WhileInstr
from prodigy.pgcl.parser import SYNTAXES, select_syntax
from prodigy.pgcl.syntax_errors import ProgramSyntaxError

from prodigy.analysis.analyzer import compute_discrete_distribution, compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import (HeuristicsError, UndefinedDistributionError, UnsupportedFeatureError,
                                        VerificationError)
from prodigy.analysis.input_distribution import InputDistributionError, input_distribution_from_text
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.strategies import (DEFAULT_MAX_DEGREE, SELECTABLE_POSITIVITY_HEURISTICS,
                                                                  SynthesisStrategies)
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.analysis.evtinvariants.invariant_synthesis import (DEFAULT_SYNTHESIS_TIME_LIMIT,
                                                                DEFAULT_TEMPLATE_TIME_LIMIT, evt_invariant_synthesis)
from prodigy.analysis.evtinvariants.invariant_verification import evt_invariant_verification
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.analysis.independence.independence import independent_vars as get_independent_vars_of
from prodigy.analysis.loop_strategies import loop_strategies, strategy_groups
from prodigy.analysis.session import AnalysisSession, DEFAULT_KEY, parse_decisions, validate_decisions, LoopRegistry
from prodigy.distribution.distribution import State
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup("CLI", logging.DEBUG)


class _CliGroup(click.Group):
    """Reports programs the analysis cannot handle as a plain error instead of a traceback."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (UnsupportedFeatureError, UndefinedDistributionError, InputDistributionError, HeuristicsError,
                VerificationError) as exc:
            # e.g. a refuted sub-invariant (#321) or the undecided candidates of a synthesis
            raise click.ClickException("\n".join([str(exc), *getattr(exc, "messages", ())])) from exc


# pylint: disable-msg=too-many-arguments
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
@click.option('--no-simplification', is_flag=True, required=False, default=False)
@click.option('--use-latex', is_flag=True, required=False, default=False)
@click.option("--no-normalize", is_flag=True, required=False, default=False)
@click.option("--show-all-invs", is_flag=True, required=False, default=False)
def cli(ctx,
        engine: str, strategy: str, solver: str, template_heuristic: Optional[str], pos_heuristic: Optional[str],
        intermediate_results: bool, stepwise: bool, no_simplification: bool, use_latex: bool, no_normalize: bool,
        show_all_invs: bool):
    ctx.ensure_object(dict)
    if solver.upper() not in SolverType.__members__:
        raise ValueError(f"Solver {solver} is not known.")
    synthesis_strategy = SynthesisStrategies.__members__[strategy.upper()]
    if synthesis_strategy != SynthesisStrategies.FROM_HEURISTICS and (template_heuristic or pos_heuristic):
        raise click.UsageError("--template-heuristic and --pos-heuristic only apply to --strategy from_heuristics.")
    ctx.obj['CONFIG'] = \
        ForwardAnalysisConfig(
            engine=ForwardAnalysisConfig.Engine.GINAC if engine == 'ginac'
            else ForwardAnalysisConfig.Engine.SYMENGINE if engine == 'symengine' else
            ForwardAnalysisConfig.Engine.SYMPY,
            show_intermediate_steps=intermediate_results,
            step_wise=stepwise,
            use_simplification=not no_simplification,
            use_latex=use_latex,
            normalize=not no_normalize,
            strategy=synthesis_strategy,
            templ_heuristic=TemplateHeuristics.__members__[(template_heuristic or 'default').upper()],
            positivity_heuristic=PositivityHeuristics.__members__[(pos_heuristic or 'default').upper()],
            show_all_invs=show_all_invs,
            solver_type=SolverType.__members__[solver.upper()]
        )
    # The engine settings (simplification, LaTeX output) hold for this command, printing included.
    ctx.with_resource(ctx.obj['CONFIG'].activate())


# pylint: enable-msg=too-many-arguments


_SYNTAX_HELP = ("Syntax of the program files, pgcl or ppy (default: by file extension, '.ppy' is ppy, "
                "everything else pgcl). Invariants given as program text use the program's syntax.")


def _syntax_option(command):
    return click.option('--syntax', type=click.Choice(SYNTAXES), default=None, help=_SYNTAX_HELP)(command)


def _with_source_line(message: str, source: str, line: Optional[int], column: Optional[int]) -> str:
    lines = source.split("\n")
    if line is None or column is None or not 1 <= line <= len(lines):
        return message
    return f"{message}\n    {lines[line - 1]}\n    {' ' * (column - 1)}^"


def _pgcl_syntax_error(name: str, source: str, exc: UnexpectedInput) -> click.ClickException:
    return click.ClickException(f"{name}: syntax error in line {exc.line}, column {exc.column} (pgcl syntax).\n"
                                f"{exc.get_context(source)}")


def _read_program(program_file: IO, syntax: Optional[str]) -> Tuple[str, Program, str]:
    """Read and parse a program file; the syntax is chosen by ``syntax`` or the file extension."""
    source = program_file.read()
    name = getattr(program_file, "name", "<program>")
    chosen = select_syntax(syntax, name)
    try:
        program = compiler.parse_pgcl(source, chosen)
    except ProgramSyntaxError as exc:
        raise click.ClickException(_with_source_line(f"{name}: {exc}", source, exc.line, exc.column)) from exc
    except UnexpectedInput as exc:
        raise _pgcl_syntax_error(name, source, exc) from exc
    return source, program, chosen


_LOOP_OPTION_HELP = ("Strategy for a loop, as ID=STRATEGY[:NAME=VALUE,...], e.g. 0=unroll:iterations=100 or "
                     "'*=auto' for all loops. See `prodigy loop_strategies` for the strategies by group: "
                     "automatic, exact result, bounds, explore.")


def _collect_decisions(loop_specs: Tuple[str, ...], decisions_file: Optional[IO]) -> Dict[Any, Any]:
    decisions: Dict[Any, Any] = {}
    if decisions_file is not None:
        loaded = json.load(decisions_file)
        decisions.update(loaded.get("decisions", loaded) if isinstance(loaded, dict) else {})
    for spec in loop_specs:
        key, sep, rest = spec.partition("=")
        key = key.strip()
        if sep and (key == DEFAULT_KEY or key.isdigit()):
            decisions[key] = rest
        else:
            decisions[DEFAULT_KEY] = spec
    return decisions


def _checked_decisions(decisions: Dict[Any, Any], program: Program, source: str, syntax: str) -> Dict[Any, Any]:
    """``decisions`` (from ``--loop``/``--decisions``) after checking them, in particular their generating
    functions, before any engine sees them; exits with the problems otherwise."""
    try:
        parsed = parse_decisions(decisions)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--loop' / '--decisions'") from exc
    problems = validate_decisions(parsed, LoopRegistry(program, source, syntax), syntax)
    if problems:
        for problem in problems:
            print(f"{Style.OKRED}Error:{Style.RESET} {problem}", file=sys.stderr)
        sys.exit(2)
    return decisions


def _expression_option(text: str, option: str, program: Program, syntax: str) -> str:
    """The generating function ``text`` of the command-line option ``option``, checked before any engine
    sees it (SymPy's parser evaluates Python code): an expression of ``syntax`` over numbers, the
    program's variables and parameters and the functions of
    :data:`~prodigy.analysis.ai_candidates.MATH_FUNCTIONS`. Returned in the engines' spelling."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import MATH_FUNCTIONS, CandidateError, admissible_expression
    names = [str(v) for v in program.variables] + [str(p) for p in program.parameters]
    try:
        return admissible_expression(text.strip(), names=names, syntax=syntax, functions=MATH_FUNCTIONS)
    except CandidateError as exc:
        raise click.BadParameter(str(exc), param_hint=f"'{option}'") from exc


@cli.command('main')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('input_dist', type=str, required=False)
@click.option('--show-input-program',
              is_flag=True,
              required=False,
              default=False)
@click.option('--loop', 'loop_specs', multiple=True,
              help=_LOOP_OPTION_HELP + " Loops without a decision are asked for.")
@_syntax_option
def main(ctx, program_file: IO, input_dist: str,  # pylint: disable=too-many-arguments
         show_input_program: bool, loop_specs: Tuple[str, ...], syntax: Optional[str]) -> None:
    """
    Compile the given program and print some information about it.
    """

    logger.info("Prodigy started.")

    # Parse and the input and do typechecking.
    logger.debug("Read file %s", program_file.name)
    logger.debug("Input distribution: %s", input_dist)
    program_source, program, syntax = _read_program(program_file, syntax)

    if show_input_program:
        print("Program source:")
        print(program_source)
        print()
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    if input_dist is None:
        dist = config.factory.one(*program.variables.keys())
    else:
        dist = input_distribution_from_text(config, input_dist, program, syntax)

    logger.debug("Start analysis.")
    decisions = _checked_decisions(_collect_decisions(loop_specs, None), program, program_source, syntax)
    session = AnalysisSession.interactive(program, source=program_source, syntax=syntax, config=config,
                                          decisions=decisions)
    start = time.perf_counter()
    with session.activate():
        dist, error_prob = compute_discrete_distribution(
            program, dist, config)
    stop = time.perf_counter()

    print(Style.OKBLUE + "Result: \t" + Style.OKGREEN + "(" + str(dist) +
          ", " + str(error_prob) + ")" + Style.RESET)
    print(f"CPU-time elapsed: {stop - start:04f} seconds")


@cli.command('check_equality')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('other_program_file', type=click.File('r'))
@_syntax_option
def check_equality(ctx, program_file: IO, other_program_file: IO, syntax: Optional[str]):
    """
    Checks whether a certain loop-free program is an invariant of a specified while loop.
    Each file is read in the syntax of its extension, unless --syntax is given.
    """
    _, prog, program_syntax = _read_program(program_file, syntax)
    _, other_prog, _ = _read_program(other_program_file, syntax)

    start = time.perf_counter()
    # invariant files asked for on the terminal are read in the syntax of the program
    with AnalysisSession.detached(program_syntax).activate():
        equiv, result = check_equivalence(prog, other_prog, ctx.obj['CONFIG'], compute_semantics)
    stop = time.perf_counter()
    if equiv is True:
        assert isinstance(result, list)
        print(
            f"Program{Style.OKGREEN} is equivalent{Style.RESET} to invariant",
            end="")
        if len(result) == 0:
            print(".")
        else:
            print(
                f" {Style.OKGREEN}under the following constraints:{Style.RESET} {result}."
            )
    elif equiv is False:
        assert isinstance(result, State)
        print(
            f"Program{Style.OKRED} is not equivalent{Style.RESET} to invariant. "
            f"{Style.OKRED}Counterexample:{Style.RESET} {result.valuations}"
        )
    else:
        assert equiv is None
        print(
            f"Program equivalence{Style.OKYELLOW} cannot be determined{Style.RESET}, but depends on {result}."
        )

    print(f"CPU-time elapsed: {stop - start:04f} seconds")
    return equiv, result


@cli.command('independent_vars')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.option('--compute-exact', is_flag=True, required=False, default=False)
@_syntax_option
def independent_vars(ctx, program_file: IO, compute_exact: bool, syntax: Optional[str]):
    """
    Outputs an under-approximation or the actual result of the pairwise stochastic independence relation.

    :param program_file: the program file of interest
    :param compute_exact: if True, also compute and return the actual stochastic independencies
    :return: under-approximation or the actual relation
    """
    _, prog, _ = _read_program(program_file, syntax)

    start = time.perf_counter()
    indep_rel: Set[frozenset[Var]] = get_independent_vars_of(prog) #, program_file, compute_exact)
    stop = time.perf_counter()
    print(Style.OKBLUE + "Under-approximation: \t" + str(indep_rel) +
          Style.RESET)
    print(f"CPU-time elapsed: {stop - start:04f} seconds")

    if compute_exact:
        underapprox_rel = indep_rel
        indep_rel = set()
        start = time.perf_counter()
        config = ctx.obj['CONFIG']
        dist = config.factory.one(*prog.variables.keys())

        dist, _ = compute_discrete_distribution(prog, dist, config)

        marginal_cache = {}
        for var in prog.variables:
            marginal_cache[var] = dist.marginal(var)

        for v_1, v_2 in combinations(prog.variables, 2):
            common = dist.marginal(v_1, v_2)
            if common == marginal_cache[v_1] * marginal_cache[v_2]:
                indep_rel.add(frozenset({v_1, v_2}))
        stop = time.perf_counter()
        difference = indep_rel - underapprox_rel
        print(Style.OKBLUE + "Exact result: \t\t" + str(indep_rel) +
              Style.RESET)
        print(Style.OKBLUE + "Difference: \t\t" + str(difference) +
              Style.RESET)
        print(f"CPU-time elapsed: {stop - start:04f} seconds")

    return indep_rel


@cli.command('invariant_synthesis')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.option('--invariant', is_flag=False, type=str, required=False)
@click.argument('input_dist', type=str, required=False)
@click.option('--max-degree', type=click.IntRange(min=0), default=DEFAULT_MAX_DEGREE, show_default=True,
              help="Maximal degree of the templates (numerator degree in each variable).")
@click.option('--time-limit', type=click.IntRange(min=1), default=DEFAULT_SYNTHESIS_TIME_LIMIT, show_default=True,
              help="Seconds after which the synthesis stops without a result.")
@click.option('--template-time-limit', type=click.IntRange(min=1), default=DEFAULT_TEMPLATE_TIME_LIMIT,
              show_default=True, help="Seconds after which a single template is skipped.")
@_syntax_option
def invariant_synthesis(ctx,  # pylint: disable=too-many-arguments
                        program_file: IO, invariant: str, input_dist: str, max_degree: int, time_limit: int,
                        template_time_limit: int, syntax: Optional[str]):
    """
    Tries to synthesize an EVT Invariant for the given input file.
    Supports a loop-free program to prefix the actual loop describing an initial distribution.
    """

    # read the program source file and parse it into a program object.
    _, prog, syntax = _read_program(program_file, syntax)
    if invariant:
        invariant = _expression_option(invariant, "--invariant", prog, syntax)
    with AnalysisSession.detached(syntax).activate():
        _invariant_synthesis(ctx.obj["CONFIG"], prog, invariant, input_dist, max_degree, time_limit,
                             template_time_limit, syntax)


def _invariant_synthesis(config: ForwardAnalysisConfig, prog: Program,  # pylint: disable=too-many-arguments
                         invariant: str, input_dist: str,
                         max_degree: int = DEFAULT_MAX_DEGREE, time_limit: int = DEFAULT_SYNTHESIS_TIME_LIMIT,
                         template_time_limit: int = DEFAULT_TEMPLATE_TIME_LIMIT, syntax: str = "pgcl") -> None:

    # isolate the initial distribution description from the actual while loop.
    loops = [1 if isinstance(instr, WhileInstr) else 0 for instr in prog.instructions]
    if sum(loops) > 1:
        raise ValueError(f"There are {sum(loops)} loops in the program. Only 1 is supported.")

    # compute the initial distribution with respect to the user specified configuration
    dist = config.factory.one(*prog.variables.keys())
    if input_dist is not None:
        dist = input_distribution_from_text(config, input_dist, prog, syntax)
    dist, _ = compute_semantics(prog.instructions[:loops.index(1)],
                                  ProgramInfo(prog),
                                  dist,
                                  config.factory.from_expr("0"),
                                  config
                                  )

    # Create the strategy to do invariant synthesis and start synthesis.
    strategy = SynthesisStrategies.from_config(config, prog.variables.keys(), max_degree)
    start = time.perf_counter()
    loop = prog.instructions[loops.index(1)]
    assert isinstance(loop, WhileInstr)
    if invariant:
        invariant_dist = config.factory.from_expr(invariant, *prog.variables.keys())
        evt_invariant_verification(loop,
                                   ProgramInfo(prog), dist, invariant_dist, config, compute_semantics)
    else:
        evt_invariant_synthesis(loop, ProgramInfo(prog), dist, config, strategy, compute_semantics,
                                time_limit=time_limit, template_time_limit=template_time_limit)

    stop = time.perf_counter()
    print(f"CPU-time elapsed: {stop - start:04f} seconds")


def _loop_parts(prog):
    loops = [1 if isinstance(instr, WhileInstr) else 0 for instr in prog.instructions]
    if sum(loops) != 1:
        raise ValueError(f"There are {sum(loops)} top-level loops in the program. Exactly 1 is supported.")
    k = loops.index(1)
    return prog.instructions[:k], prog.instructions[k], prog.instructions[k + 1:]


@cli.command('evt_lower_bound')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.option('--invariant', type=str, required=True,
              help="EVT sub-invariant nu (a generating function in the program variables).")
@click.argument('input_dist', type=str, required=False)
@_syntax_option
def evt_lower_bound(ctx, program_file: IO, invariant: str, input_dist: str, syntax: Optional[str]):
    """
    Lower induction: verifies that NU is an EVT sub-invariant of finite mass (NU <= Phi(NU) coefficient-wise and
    NU(1,...,1) < oo) and continues the analysis with the resulting lower bound on the output distribution of the
    loop.  No termination assumption is needed.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.lowerbounds.subinvariant import evt_subinvariant_verification
    _, prog, syntax = _read_program(program_file, syntax)
    prefix, loop, rest = _loop_parts(prog)
    invariant = _expression_option(invariant, "--invariant", prog, syntax)
    config = ctx.obj["CONFIG"]
    variables = list(prog.variables.keys())
    dist = config.factory.one(*variables) if input_dist is None \
        else input_distribution_from_text(config, input_dist, prog, syntax)
    zero = config.factory.from_expr("0", *variables)
    with AnalysisSession.detached(syntax).activate():
        dist, err = compute_semantics(prefix, ProgramInfo(prog), dist, zero, config)
        start = time.perf_counter()
        subinvariant = config.factory.from_expr(invariant, *variables)
        lower = evt_subinvariant_verification(loop, ProgramInfo(prog), dist, subinvariant, config, compute_semantics)
        print(f"{Style.CYAN}Queries below are evaluated on the lower bound (they are lower bounds for the "
              f"unnormalised probabilities and expectations).{Style.RESET}")
        compute_semantics(rest, ProgramInfo(prog), lower, err, config)
    print(f"CPU-time elapsed: {time.perf_counter() - start:04f} seconds")


def _parse_shape(spec: str):
    """'r1,r2;p1,p2;b1,b2' -> TemplateShape"""
    # pylint: disable=import-outside-toplevel
    from fractions import Fraction
    from prodigy.analysis.lowerbounds.parameters import TemplateShape
    parts = spec.split(";")
    if len(parts) != 3:
        raise click.BadParameter("template shapes are given as 'rates;periods;tail starts', e.g. '0,7/8;1,2;24,48'")
    rates = [Fraction(x) for x in parts[0].split(",")]
    periods = [int(x) for x in parts[1].split(",")]
    tails = [int(x) for x in parts[2].split(",")]
    box = [t + p + 1 if r != 0 else t for r, p, t in zip(rates, periods, tails)]
    return TemplateShape(rates, periods, tails, reach_box=box)


@cli.command('evt_sandwich')
@click.argument('program_file', type=click.File('r'))
@click.option('--size', type=int, default=1200, show_default=True,
              help="Budget for the number of template values of the lower certificate.")
@click.option('--lower', 'lower_specs', multiple=True,
              help="Template shape of the lower certificate 'rates;periods;tail starts' (repeatable).")
@click.option('--upper', 'upper_specs', multiple=True,
              help="Template shape of the upper certificate 'rates;periods;tail starts' (repeatable).")
@click.option('--no-upper', is_flag=True, default=False, help="Only synthesise the lower certificate.")
@click.option('--objective', type=click.Choice(['auto', 'mass']), default='auto', show_default=True,
              help="Objective of the upper certificate (auto: exit mass plus the moments of ?Ex queries).")
@click.option('--per-query/--no-per-query', default=None,
              help="Re-optimise the upper certificate for every ?Pr query "
                   "(default: only if its exit mass is infinite).")
@click.option('--show-certificates', is_flag=True, default=False)
@click.option('--json', 'json_path', type=str, default=None, help="Write certificates and bounds to this file.")
@_syntax_option
def evt_sandwich(  # pylint: disable=too-many-arguments
        program_file: IO, size: int, lower_specs, upper_specs, no_upper: bool, objective: str,
        per_query, show_certificates: bool, json_path: str, syntax: Optional[str]):
    # pylint: disable=import-outside-toplevel
    """
    Sandwich bounds for a program with one loop: synthesises an EVT sub-invariant of finite mass (lower bound,
    no termination assumption) and an EVT super-invariant (upper bound) with eventually geometric templates by
    linear programming, verifies both exactly, and prints two-sided bounds for the queries after the loop.
    """
    from prodigy.analysis.lowerbounds.problem import load_problem
    from prodigy.analysis.lowerbounds.sandwich import evt_sandwich as run_sandwich, print_result, write_json
    _, prog, _ = _read_program(program_file, syntax)
    problem = load_problem(prog)
    lower = [_parse_shape(s) for s in lower_specs] or None
    upper = [_parse_shape(s) for s in upper_specs] or None
    res = run_sandwich(problem, lower_shapes=lower, upper_shapes=upper, size=size, objective=objective,
                       with_upper=not no_upper, per_query=per_query)
    print_result(res, show_certificates=show_certificates)
    if json_path:
        write_json(res, json_path)


@cli.command()
def print_strategies():
    print("Currently implemented strategies are: ")
    for strategy in SynthesisStrategies:
        print(strategy.name)


@cli.command()
def print_pos_heuristics():
    print("Positivity heuristics for --pos-heuristic (with --strategy from_heuristics): ")
    for name in SELECTABLE_POSITIVITY_HEURISTICS:
        print(name.upper())


@cli.command()
def print_template_heuristics():
    print("Currently implemented template generation heuristics are: ")
    for heuristic in TemplateHeuristics:
        print(heuristic.name)


def _format_value(exact: Optional[str], number: Optional[float]) -> str:
    if number is not None:
        text = f"{number:.6g}"
        return text if exact is None or exact == text else f"{text}  ({exact})" if len(exact) <= 40 else text
    return exact if exact is not None else "?"


def _print_summary(result: Dict[str, Any]) -> None:
    status = result["status"]
    if status != "ok":
        error = result.get("error") or {}
        print(f"{Style.OKRED}{status}:{Style.RESET} {error.get('message', '')}")
        if error.get("verdict") == "refuted":
            print("Verdict: refuted (the invariant is provably wrong).")
        elif error.get("verdict") == "unknown":
            print("Verdict: not proven (neither proven nor refuted; the invariant may still be correct).")
        if status == "needs_decision":
            print("Choose strategies with --loop ID=STRATEGY[:NAME=VALUE,...]; "
                  "`prodigy outline FILE` lists the loops, `prodigy loop_strategies` the strategies.")
    dist = result.get("distribution")
    if dist:
        print(f"{Style.OKBLUE}Result:{Style.RESET}\t{dist['closed_form']}")
        if dist.get("error_probability") not in (None, "0"):
            print(f"{Style.OKBLUE}Observation violated with probability:{Style.RESET}\t{dist['error_probability']}")
    bounds = result.get("bounds") or {}
    if bounds and status == "ok":
        print(f"{Style.OKBLUE}Guarantee:{Style.RESET}\t{bounds.get('explanation', bounds.get('kind'))}")
    for query in result.get("queries", []):
        where = f"line {query['line']}: " if query.get("line") else ""
        if query.get("error"):
            label = {"probability": "Pr", "expectation": "E"}.get(query["kind"], query["kind"])
            label = f"{label}[{query['expr']}]" if query["kind"] in ("probability", "expectation") else \
                f"{label} of {query['expr']}" if query["expr"] else label
            print(f"  {where}{label}: {Style.OKRED}not computed:{Style.RESET} {query['error']['message']}")
        elif query["kind"] in ("probability", "expectation"):
            label = ("Pr" if query["kind"] == "probability" else "E") + f"[{query['expr']}]"
            value = _format_value(query["value"], query["value_float"])
            interval = ""
            if query["bound"] != "exact":
                low = _format_value(query["lower"], query["lower_float"]) if query["lower"] is not None else "?"
                high = _format_value(query["upper"], query["upper_float"]) if query["upper"] is not None else "∞" \
                    if query["bound"] == "lower" else "?"
                interval = f"\t∈ [{low}, {high}]  ({query['bound']})"
            print(f"  {where}{label} = {value}{interval}")
        elif query["kind"] == "marginal":
            print(f"  {where}marginal of {query['expr']}: {query['value']}")
        elif query["kind"] == "optimization":
            print(f"  {where}optimum for {query['expr']}: {query['value']}")
    for loop in result.get("loops", []):
        where = f" (line {loop['line']})" if loop.get("line") else ""
        line = f"  loop {loop['id']}{where} [{loop['strategy']}] {loop['kind']}"
        if loop["messages"]:
            line += ": " + " ".join(loop["messages"])
        elif loop.get("failed"):
            # the error itself is printed above (#333)
            line += " (the strategy failed, see the error above)"
        print(line)
    for warning in result.get("warnings", []):
        print(f"{Style.YELLOW}Warning:{Style.RESET} {warning}")
    print(f"CPU-time elapsed: {result['timing'].get('analysis', 0):04f} seconds")


@cli.command('analyze')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('input_dist', type=str, required=False)
@click.option('--loop', 'loop_specs', multiple=True, help=_LOOP_OPTION_HELP)
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
                   "(in the program's syntax; repeatable).")
@click.option('--step-iterations', type=click.IntRange(min=1), default=40, show_default=True,
              help="With --steps: iterates kept per loop (spread evenly over the iterations).")
@click.option('--output', type=click.File('w'), required=False, help="Write the JSON result to this file.")
@_syntax_option
def analyze(  # pylint: disable=too-many-arguments
        ctx, program_file: IO, input_dist: Optional[str], loop_specs: Tuple[str, ...],
        decisions_file: Optional[IO], as_json: bool, coefficients: int, steps: bool,
        watches: Tuple[str, ...], step_iterations: int, output: Optional[IO], syntax: Optional[str]) -> None:
    """
    Analyse a program without asking questions: every loop needs a strategy (--loop or
    --decisions). Queries are answered with sound intervals.
    """
    program_source, program, syntax = _read_program(program_file, syntax)
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    decisions = parse_decisions(_checked_decisions(_collect_decisions(loop_specs, decisions_file), program,
                                                   program_source, syntax))
    if watches and not steps:
        raise click.UsageError("--watch needs --steps.")
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.trace import TraceLimits, describe_step, parse_watch
    names = list(program.variables) + list(program.parameters)
    for watch in watches:
        try:
            parse_watch(watch, syntax, names)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--watch") from exc
    session = AnalysisSession(program, source=program_source, syntax=syntax, config=config, decisions=decisions,
                              input_distribution=input_dist, capture_output=True,
                              record_steps=steps, coefficient_limit=coefficients,
                              trace_limits=TraceLimits(iterates=step_iterations), watches=watches)
    result = session.run().to_json()
    if steps and not as_json:
        for step in result["steps"]:
            print(describe_step(step))
        for hint in (result.get("trace") or {}).get("hints", []):
            print(f"{Style.YELLOW}Hint:{Style.RESET} {hint['text']}")
    if output is not None:
        json.dump(result, output, indent=2)
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)
    if result["status"] == "needs_decision":
        sys.exit(2)
    if result["status"] != "ok":
        sys.exit(1)


@cli.command('outline')
@click.argument('program_file', type=click.File('r'))
@click.option('--json', 'as_json', is_flag=True, default=False)
@_syntax_option
def outline(program_file: IO, as_json: bool, syntax: Optional[str]) -> None:
    """
    List the loops (with the ids used by --loop) and queries of a program.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import outline_program
    source, program, syntax = _read_program(program_file, syntax)
    info = outline_program(program, source, syntax)
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


@cli.command('convert')
@click.argument('program_file', type=click.File('r'))
@click.option('-o', '--output', type=click.File('w'), default=None,
              help="Write the .ppy program to this file instead of printing it.")
def convert(program_file: IO, output: Optional[IO]) -> None:
    """
    Convert a program from the .pgcl syntax into the .ppy syntax.

    Names that are keywords in .ppy (e.g. a variable 'switch') are renamed by appending '_'; every
    renaming is reported on stderr. Comments are kept, at approximately their original place.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.converter import convert_pgcl
    from prodigy.pgcl.ppy_printer import PpyPrintError
    name = getattr(program_file, "name", "<program>")
    if select_syntax(path=name) == "ppy":
        raise click.ClickException(f"{name} is already a .ppy program.")
    source = program_file.read()
    try:
        result = convert_pgcl(source)
    except ProgramSyntaxError as exc:
        raise click.ClickException(_with_source_line(f"{name}: {exc}", source, exc.line, exc.column)) from exc
    except UnexpectedInput as exc:
        raise _pgcl_syntax_error(name, source, exc) from exc
    except PpyPrintError as exc:
        raise click.ClickException(f"{name}: {exc}") from exc
    for message in result.messages():
        click.echo(f"{name}: {message}", err=True)
    if output is None:
        click.echo(result.text, nl=False)
    else:
        output.write(result.text)


@cli.command('loop_strategies')
@click.option('--json', 'as_json', is_flag=True, default=False)
def print_loop_strategies(as_json: bool) -> None:
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
@click.option('--max-jobs', default=2, show_default=True, type=int, help="Analyses running at the same time.")
@click.option('--memory-limit', default=None, type=int, help="Memory limit per analysis in MB (Linux).")
@click.option('--examples', 'examples_dir', default=None, type=click.Path(exists=True, file_okay=False),
              help="Folder with example programs (default: pgfexamples of this checkout or package).")
@click.option('--allowed-host', 'allowed_hosts', multiple=True,
              help="Additional host name the server may be reached under (when not on localhost).")
def web(  # pylint: disable=too-many-arguments
        ctx, host: str, port: int, no_browser: bool, time_limit: float, max_jobs: int,
        memory_limit: Optional[int], examples_dir: Optional[str], allowed_hosts: Tuple[str, ...]) -> None:
    """
    Start the web GUI (a local web server) and open it in the browser.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.web.app import serve
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    serve(host=host, port=port, open_browser=not no_browser, time_limit=time_limit, max_jobs=max_jobs,
          memory_limit=memory_limit, examples_dir=examples_dir, default_engine=config.engine.name.lower(),
          allowed_hosts=allowed_hosts)


if __name__ == "__main__":
    # execute only if run as a script
    cli()  # pylint: disable=no-value-for-parameter
