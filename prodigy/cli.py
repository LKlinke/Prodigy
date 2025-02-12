"""
prodigy also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have prodigy installed globally, you can use `poetry run prodigy INPUT`.

.. click:: prodigy.cli:cli
   :prog: prodigy
   :show-nested:
"""

import logging
import time
from itertools import combinations
from typing import IO, Set

import click
from probably.pgcl import Var, compiler, WhileInstr
from probably.pgcl.check import CheckFail

from prodigy.analysis.analyzer import compute_discrete_distribution, compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.analysis.evtinvariants.invariant_synthesis import evt_invariant_synthesis
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.analysis.independence.independence import independent_vars as ivars
from prodigy.distribution.distribution import State
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup("CLI", logging.DEBUG)


# pylint: disable-msg=too-many-arguments
@click.group()
@click.pass_context
@click.option('--engine', type=str, required=False, default='')
@click.option("--strategy", type=str, required=False, default='default')
@click.option("--solver", type=str, required=False, default='sympy')
@click.option("--template-heuristic", type=str, required=False, default='default')
@click.option("--pos-heuristic", type=str, required=False, default='default')
@click.option('--intermediate-results', is_flag=True, required=False, default=False)
@click.option("--stepwise", is_flag=True, required=False, default=False)
@click.option('--no-simplification', is_flag=True, required=False, default=False)
@click.option('--use-latex', is_flag=True, required=False, default=False)
@click.option("--no-normalize", is_flag=True, required=False, default=False)
@click.option("--show-all-invs", is_flag=True, required=False, default=False)
def cli(ctx,
        engine: str, strategy: str, solver: str, template_heuristic: str, pos_heuristic: str,
        intermediate_results: bool, stepwise: bool, no_simplification: bool, use_latex: bool, no_normalize: bool,
        show_all_invs: bool):
    ctx.ensure_object(dict)
    if solver.upper() not in SolverType.__members__:
        raise ValueError(f"Solver {solver} is not known.")
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
            strategy=SynthesisStrategies.__members__[strategy.upper()],
            templ_heuristic=TemplateHeuristics.__members__[template_heuristic.upper()],
            positivity_heuristic=PositivityHeuristics.__members__[pos_heuristic.upper()],
            show_all_invs=show_all_invs,
            solver_type=SolverType.__members__[solver.upper()]
        )


# pylint: enable-msg=too-many-arguments


@cli.command('main')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('input_dist', type=str, required=False)
@click.option('--show-input-program',
              is_flag=True,
              required=False,
              default=False)
def main(ctx, program_file: IO, input_dist: str,
         show_input_program: bool) -> None:
    """
    Compile the given program and print some information about it.
    """

    logger.info("Prodigy started.")

    # Parse and the input and do typechecking.
    logger.debug("Read file %s", program_file.name)
    logger.debug("Input distribution: %s", input_dist)
    program_source = program_file.read()
    program = compiler.parse_pgcl(program_source)
    # if isinstance(program, CheckFail):
    #    print("Error:", program)
    #    return

    if show_input_program:
        print("Program source:")
        print(program_source)
        print()
    config: ForwardAnalysisConfig = ctx.obj['CONFIG']
    if input_dist is None:
        dist = config.factory.one(*program.variables.keys())
    else:
        dist = config.factory.from_expr(input_dist, *program.variables.keys())

    logger.debug("Start analysis.")
    start = time.perf_counter()
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
def check_equality(ctx, program_file: IO, other_program_file: IO):
    """
    Checks whether a certain loop-free program is an invariant of a specified while loop.
    :param program_file: the file containing the while-loop
    :param invariant_file: the provided invariant
    :return:
    """
    prog_src = program_file.read()
    other_prog_src = other_program_file.read()

    prog = compiler.parse_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        raise ValueError(f"Could not compile the Program. {prog}")

    other_prog = compiler.parse_pgcl(other_prog_src)
    if isinstance(other_prog, CheckFail):
        raise ValueError(f"Could not compile invariant. {other_prog}")

    start = time.perf_counter()
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
def independent_vars(ctx, program_file: IO, compute_exact: bool):
    """
    Outputs an under-approximation or the actual result of the pairwise stochastic independence relation.

    :param program_file: the program file of interest
    :param compute_exact: if True, also compute and return the actual stochastic independencies
    :return: under-approximation or the actual relation
    """
    prog_src = program_file.read()

    prog = compiler.parse_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        raise ValueError(f"Could not compile the Program. {prog}")

    start = time.perf_counter()
    indep_rel: Set[frozenset[Var]] = ivars(prog) #, program_file, compute_exact)
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
@click.argument('input_dist', type=str, required=False)
def invariant_synthesis(ctx, program_file: IO, input_dist: str):
    """
    Tries to synthesize an EVT Invariant for the given input file.
    Supports a loop-free program to prefix the actual loop describing an initial distribution.
    """

    # read the program source file and parse it into a program object.
    prog_src = program_file.read()
    prog = compiler.parse_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        raise ValueError(f"Could not compile the Program. {prog}")

    # isolate the initial distribution description from the actual while loop.
    loops = [1 if isinstance(instr, WhileInstr) else 0 for instr in prog.instructions]
    if sum(loops) > 1:
        raise ValueError(f"There are {sum(loops)} loops in the program. Only 1 is supported.")

    # compute the initial distribution with respect to the user specified configuration
    config = ctx.obj["CONFIG"]
    dist = config.factory.one(*prog.variables.keys())
    if input_dist is not None:
        dist = config.factory.from_expr(input_dist, *prog.variables.keys())
    dist, _ = compute_semantics(prog.instructions[:loops.index(1)],
                                  ProgramInfo(prog),
                                  dist,
                                  config.factory.from_expr("0"),
                                  config
                                  )

    # Create the strategy to do invariant synthesis and start synthesis.
    strategy = SynthesisStrategies.make(config.strategy, prog.variables.keys(), config.factory)
    start = time.perf_counter()
    try:
        evt_invariant_synthesis(prog.instructions[loops.index(1)],
                                             ProgramInfo(prog), dist, config, strategy, compute_semantics)
    except VerificationError as e:
        print(f"{Style.RED} {str(e)} {Style.RESET}")

    stop = time.perf_counter()
    print(f"CPU-time elapsed: {stop - start:04f} seconds")


@cli.command()
def print_strategies():
    print("Currently implemented strategies are: ")
    for strategy in SynthesisStrategies:
        print(strategy.name)


@cli.command()
def print_pos_heuristics():
    print("Currently implemented positivity heuristics are: ")
    for heuristic in PositivityHeuristics:
        print(heuristic.name)


@cli.command()
def print_template_heuristics():
    print("Currently implemented template generation heuristics are: ")
    for heuristic in TemplateHeuristics:
        print(heuristic.name)


if __name__ == "__main__":
    # execute only if run as a script
    cli()  # pylint: disable=no-value-for-parameter
