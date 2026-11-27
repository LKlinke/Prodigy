"""Commands for single proof rules: equivalence of two programs, independence of variables, EVT invariants
(synthesis, verification, lower induction) and sandwich bounds."""
import time
from fractions import Fraction
from itertools import combinations
from typing import IO, Optional, Set, Tuple

import click

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession, InteractiveDecisions, compute_discrete_distribution
from prodigy.analysis.tools import (DEFAULT_MAX_DEGREE, DEFAULT_SYNTHESIS_TIME_LIMIT, DEFAULT_TEMPLATE_TIME_LIMIT,
                                    ProgramInfo, SynthesisStrategies, TemplateShape, compare, compute_semantics,
                                    evt_invariant_synthesis, evt_invariant_verification, evt_sandwich as run_sandwich,
                                    evt_subinvariant_verification, load_problem, print_result, write_json)
from prodigy.analysis.tools import independent_vars as get_independent_vars_of
from prodigy.cli.common import checked_program, expression_option, initial_distribution, loop_parts, syntax_option
from prodigy.cli.group import cli
from prodigy.cli.output import print_comparison
from prodigy.language import Program, Var, WhileInstr
from prodigy.util.color import Style

EXIT_EQUIVALENT = 0
"""Exit code of ``prodigy check_equality``: equivalence is proven."""
EXIT_NOT_EQUIVALENT = 3
"""Exit code of ``prodigy check_equality``: a witness proves the programs different."""
EXIT_UNDECIDED = 4
"""Exit code of ``prodigy check_equality``: neither is proven (also if only one direction is)."""

_VERDICT_EXIT = {"equivalent": EXIT_EQUIVALENT, "not_equivalent": EXIT_NOT_EQUIVALENT, "unknown": EXIT_UNDECIDED}


@cli.command('check_equality')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('other_program_file', type=click.File('r'))
@syntax_option
def check_equality(ctx: click.Context, *, program_file: IO, other_program_file: IO, syntax: Optional[str]) -> None:
    """
    Checks whether two programs are equivalent, i.e. have the same output distribution for every input,
    e.g. a while loop and a loop-free invariant program, both .ppy. Loops are decided interactively, as in
    `prodigy main`.

    The verdict is sound: "equivalent" only if the check proves equality and both analyses are exact,
    "not equivalent" only with a witness (an output state whose probabilities provably differ on one
    input), "undecided" otherwise, with the reason (e.g. only one direction is proven because a loop was
    handled by an invariant, which gives an upper bound).

    Exit codes: 0 equivalent, 3 not equivalent, 4 undecided; 1 and 2 are errors (e.g. a syntax error).
    """
    _, prog = checked_program(program_file, syntax)
    _, other_prog = checked_program(other_program_file, syntax)

    start = time.perf_counter()
    # the witness search analyses the programs again and takes the answers given for the check
    session = AnalysisSession(None, config=ctx.obj['CONFIG'],
                              provider=InteractiveDecisions(remember=True), parameter_synthesis="allow")
    session.detached_session = True
    with session.activate():
        comparison = compare(prog, other_prog, session)
    stop = time.perf_counter()
    print_comparison(comparison, program_file.name, other_program_file.name)
    print(f"CPU-time elapsed: {stop - start:04f} seconds")
    ctx.exit(_VERDICT_EXIT[comparison.verdict["verdict"]])


def _exact_independence(config: ForwardAnalysisConfig, prog: Program) -> Set[frozenset[Var]]:
    """The pairs of variables that are independent in the output distribution (from the program's start in 0)."""
    dist, _ = compute_discrete_distribution(prog, config.factory.one(*prog.variables.keys()), config)
    marginals = {var: dist.marginal(var) for var in prog.variables}
    return {frozenset({v_1, v_2}) for v_1, v_2 in combinations(prog.variables, 2)
            if dist.marginal(v_1, v_2) == marginals[v_1] * marginals[v_2]}


@cli.command('independent_vars')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.option('--compute-exact', is_flag=True, required=False, default=False)
@syntax_option
def independent_vars(ctx: click.Context, *, program_file: IO, compute_exact: bool,
                     syntax: Optional[str]) -> Set[frozenset[Var]]:
    """
    Outputs an under-approximation or the actual result of the pairwise stochastic independence relation.

    :param program_file: the program file of interest
    :param compute_exact: if True, also compute and return the actual stochastic independencies
    :return: under-approximation or the actual relation
    """
    _, prog = checked_program(program_file, syntax)

    start = time.perf_counter()
    indep_rel: Set[frozenset[Var]] = get_independent_vars_of(prog)
    stop = time.perf_counter()
    print(Style.OKBLUE + "Under-approximation: \t" + str(indep_rel) +
          Style.RESET)
    print(f"CPU-time elapsed: {stop - start:04f} seconds")

    if compute_exact:
        underapprox_rel = indep_rel
        start = time.perf_counter()
        indep_rel = _exact_independence(ctx.obj['CONFIG'], prog)
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
@syntax_option
def invariant_synthesis(  # pylint: disable=too-many-arguments  # one parameter per option, as click passes them
        ctx: click.Context, *, program_file: IO, invariant: Optional[str], input_dist: Optional[str],
        max_degree: int, time_limit: int, template_time_limit: int, syntax: Optional[str]) -> None:
    """
    Tries to synthesize an EVT Invariant for the given input file.
    Supports a loop-free program to prefix the actual loop describing an initial distribution.
    """
    _, prog = checked_program(program_file, syntax)
    if invariant:
        invariant = expression_option(invariant, "--invariant", prog)
    with AnalysisSession.detached().activate():
        _invariant_synthesis(ctx.obj["CONFIG"], prog, invariant, input_dist,
                             limits=(max_degree, time_limit, template_time_limit))


def _invariant_synthesis(config: ForwardAnalysisConfig, prog: Program, invariant: Optional[str],
                         input_dist: Optional[str], *, limits: Tuple[int, int, int]) -> None:
    """Checks ``invariant`` for the only loop of ``prog``, or synthesises one if it is not given; ``limits`` are
    the maximal degree of the templates and the time limits of the synthesis and of one template (seconds)."""
    max_degree, time_limit, template_time_limit = limits
    # isolate the initial distribution description from the actual while loop.
    loops = [1 if isinstance(instr, WhileInstr) else 0 for instr in prog.instructions]
    if sum(loops) > 1:
        raise ValueError(f"There are {sum(loops)} loops in the program. Only 1 is supported.")

    # compute the initial distribution with respect to the user specified configuration
    dist, _ = compute_semantics(prog.instructions[:loops.index(1)], ProgramInfo(prog),
                                initial_distribution(config, input_dist, prog), config.factory.from_expr("0"), config)

    # Create the strategy to do invariant synthesis and start synthesis.
    strategy = SynthesisStrategies.from_config(config, prog.variables.keys(), max_degree)
    start = time.perf_counter()
    loop = prog.instructions[loops.index(1)]
    assert isinstance(loop, WhileInstr)
    if invariant:
        invariant_dist = config.factory.from_expr(invariant, *prog.variables.keys())
        evt_invariant_verification(loop, ProgramInfo(prog), dist, invariant_dist, config, compute_semantics)
    else:
        evt_invariant_synthesis(loop, ProgramInfo(prog), dist, config, strategy, compute_semantics,
                                time_limit=time_limit, template_time_limit=template_time_limit)

    stop = time.perf_counter()
    print(f"CPU-time elapsed: {stop - start:04f} seconds")


@cli.command('evt_lower_bound')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.option('--invariant', type=str, required=True,
              help="EVT sub-invariant nu (a generating function in the program variables).")
@click.argument('input_dist', type=str, required=False)
@syntax_option
def evt_lower_bound(ctx: click.Context, *, program_file: IO, invariant: str, input_dist: Optional[str],
                    syntax: Optional[str]) -> None:
    """
    Lower induction: verifies that NU is an EVT sub-invariant of finite mass (NU <= Phi(NU) coefficient-wise and
    NU(1,...,1) < oo) and continues the analysis with the resulting lower bound on the output distribution of the
    loop.  No termination assumption is needed.
    """
    _, prog = checked_program(program_file, syntax)
    prefix, loop, rest = loop_parts(prog)
    invariant = expression_option(invariant, "--invariant", prog)
    config = ctx.obj["CONFIG"]
    dist = initial_distribution(config, input_dist, prog)
    with AnalysisSession.detached().activate():
        dist, err = compute_semantics(prefix, ProgramInfo(prog), dist,
                                      config.factory.from_expr("0", *prog.variables.keys()), config)
        start = time.perf_counter()
        lower = evt_subinvariant_verification(loop, ProgramInfo(prog), dist,
                                              config.factory.from_expr(invariant, *prog.variables.keys()), config,
                                              compute_semantics)
        print(f"{Style.CYAN}Queries below are evaluated on the lower bound (they are lower bounds for the "
              f"unnormalised probabilities and expectations).{Style.RESET}")
        compute_semantics(rest, ProgramInfo(prog), lower, err, config)
    print(f"CPU-time elapsed: {time.perf_counter() - start:04f} seconds")


def _parse_shape(spec: str) -> TemplateShape:
    """'r1,r2;p1,p2;b1,b2' -> TemplateShape"""
    parts = spec.split(";")
    if len(parts) != 3:
        raise click.BadParameter("template shapes are given as 'rates;periods;tail starts', e.g. '0,7/8;1,2;24,48'")
    rates = [Fraction(x) for x in parts[0].split(",")]
    periods = [int(x) for x in parts[1].split(",")]
    tails = [int(x) for x in parts[2].split(",")]
    box = [t + p + 1 if r != 0 else t for r, p, t in zip(rates, periods, tails, strict=True)]
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
@syntax_option
# one parameter per option, as click passes them; they are most of the locals
def evt_sandwich(  # pylint: disable=too-many-arguments,too-many-locals
        *, program_file: IO, size: int, lower_specs: Tuple[str, ...], upper_specs: Tuple[str, ...], no_upper: bool,
        objective: str, per_query: Optional[bool], show_certificates: bool, json_path: Optional[str],
        syntax: Optional[str]) -> None:
    """
    Sandwich bounds for a program with one loop: synthesises an EVT sub-invariant of finite mass (lower bound,
    no termination assumption) and an EVT super-invariant (upper bound) with eventually geometric templates by
    linear programming, verifies both exactly, and prints two-sided bounds for the queries after the loop.
    """
    _, prog = checked_program(program_file, syntax)
    res = run_sandwich(load_problem(prog), lower_shapes=[_parse_shape(s) for s in lower_specs] or None,
                       upper_shapes=[_parse_shape(s) for s in upper_specs] or None, size=size, objective=objective,
                       with_upper=not no_upper, per_query=per_query)
    print_result(res, show_certificates=show_certificates)
    if json_path:
        write_json(res, json_path)
