"""
prodigy also offers a simple command-line interface for quick program inspection.

If you use `poetry` and do not have prodigy installed globally, you can use `poetry run prodigy INPUT`.

.. click:: prodigy.cli:cli
   :prog: prodigy
   :show-nested:
"""

from typing import IO

import logging
import time
import click

from probably.pgcl import compiler
from probably.pgcl.check import CheckFail
from prodigy.util.color import Style
from prodigy import analysis


@click.group()
@click.pass_context
@click.option('--engine', type=str, required=False, default='GF')
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
            engine=analysis.ForwardAnalysisConfig.Engine.GINAC if engine == 'prodigy' else analysis.ForwardAnalysisConfig.Engine.SYMPY,
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
    program = compiler.compile_pgcl(program_source)
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

    prog = compiler.compile_pgcl(prog_src)
    if isinstance(prog, CheckFail):
        raise Exception(f"Could not compile the Program. {prog}")

    inv = compiler.compile_pgcl(inv_src)
    if isinstance(inv, CheckFail):
        raise Exception(f"Could not compile invariant. {inv}")

    start = time.perf_counter()
    equiv = analysis.equivalence.check_equivalence(prog, inv,
                                                   ctx.obj['CONFIG'])
    stop = time.perf_counter()

    print(
        f"Program{f'{Style.OKRED} is not equivalent{Style.RESET}' if not equiv[0] else f'{Style.OKGREEN} is equivalent{Style.RESET}'} to invaraint"
    )
    print(f"CPU-time elapsed: {stop - start:04f} seconds")
    return equiv


if __name__ == "__main__":
    # execute only if run as a script
    cli()  # pylint: disable=no-value-for-parameter
