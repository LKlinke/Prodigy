"""
prodigy also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have prodigy installed globally, you can use `poetry run prodigy INPUT`.

.. click:: prodigy.cli:cli
   :prog: prodigy
   :show-nested:
"""

import logging
import time
from typing import IO

import click
from probably.pgcl import compiler
from probably.pgcl.check import CheckFail

from prodigy import analysis
from prodigy.distribution.distribution import State
from prodigy.util.color import Style


@click.group()
@click.pass_context
@click.option('--engine', type=str, required=False, default='')
@click.option('--intermediate-results',
              is_flag=True,
              required=False,
              default=False)
@click.option('--no-simplification',
              is_flag=True,
              required=False,
              default=False)
@click.option('--use-latex', is_flag=True, required=False, default=False)
def cli(ctx, engine: str, intermediate_results: bool, no_simplification: bool,
        use_latex: bool):
    ctx.ensure_object(dict)
    ctx.obj['CONFIG'] = \
        analysis.ForwardAnalysisConfig(
            engine=analysis.ForwardAnalysisConfig.Engine.GINAC if engine == 'ginac'
                else analysis.ForwardAnalysisConfig.Engine.SYMPY,
            show_intermediate_steps=intermediate_results,
            use_simplification=not no_simplification,
            use_latex=use_latex
        )


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

    # Setup the logging.
    # logging.basicConfig(level=logging.INFO)
    logging.getLogger("prodigy.cli").info("Program started.")

    # Parse and the input and do typechecking.
    program_source = program_file.read()
    program = compiler.parse_pgcl(program_source)
    # if isinstance(program, CheckFail):
    #    print("Error:", program)
    #    return

    if show_input_program:
        print("Program source:")
        print(program_source)
        print()
    config = ctx.obj['CONFIG']
    if input_dist is None:
        dist = config.factory.one(*program.variables.keys())
    else:
        dist = config.factory.from_expr(input_dist,
                                        *program.variables.keys(),
                                        preciseness=1.0)

    start = time.perf_counter()
    dist = analysis.compute_discrete_distribution(program, dist, config)
    stop = time.perf_counter()

    print(Style.OKBLUE + "Result: \t" + Style.OKGREEN + str(dist) +
          Style.RESET)
    print(f"CPU-time elapsed: {stop - start:04f} seconds")


@cli.command('check_equality')
@click.pass_context
@click.argument('program_file', type=click.File('r'))
@click.argument('invariant_file', type=click.File('r'))
def check_equality(ctx, program_file: IO, invariant_file: IO):
    """
    Checks whether a certain loop-free program is an invariant of a specified while loop.
    :param program_file: the file containing the while-loop
    :param invariant_file: the provided invariant
    :return:
    """
    prog_src = program_file.read()
    inv_src = invariant_file.read()

    prog = compiler.parse_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        raise Exception(f"Could not compile the Program. {prog}")

    inv = compiler.parse_pgcl(inv_src)
    if isinstance(inv, CheckFail):
        raise Exception(f"Could not compile invariant. {inv}")

    start = time.perf_counter()
    equiv, result = analysis.equivalence.check_equivalence(
        prog, inv, ctx.obj['CONFIG'])
    stop = time.perf_counter()
    if equiv is True:
        assert isinstance(result, list)
        print(
            f"Program{Style.OKGREEN} is equivalent{Style.RESET} to inavariant",
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
            f"Program{Style.OKRED} is not equivalent{Style.RESET} to invariant. "\
                f"{Style.OKRED}Counterexample:{Style.RESET} {result.valuations}"
        )
    else:
        assert equiv is None
        print(
            f"Program equivalence{Style.OKYELLOW} cannot be determined{Style.RESET}, but depends on {result}."
        )

    print(f"CPU-time elapsed: {stop - start:04f} seconds")
    return equiv, result


if __name__ == "__main__":
    # execute only if run as a script
    cli()  # pylint: disable=no-value-for-parameter
