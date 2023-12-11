from __future__ import annotations

import functools
import logging
from typing import Union, Sequence, get_args

from probably.pgcl import Program, Instr, SkipInstr, AbortInstr, WhileInstr, LoopInstr, QueryInstr, Query, ObserveInstr, \
    ChoiceInstr, AsgnInstr, IfInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import ObserveZeroEventError
from prodigy.analysis.instructionhandler.assignment_handler import AssignmentHandler
from prodigy.analysis.instructionhandler.ite_handler import ITEHandler
from prodigy.analysis.instructionhandler.loop_handler import LoopHandler
from prodigy.analysis.instructionhandler.observe_handler import ObserveHandler
from prodigy.analysis.instructionhandler.probchoice_handler import PChoiceHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.query_handler import QueryHandler
from prodigy.analysis.instructionhandler.queryblock_handler import QueryBlockHandler
from prodigy.analysis.instructionhandler.while_handler import WhileHandler
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG)


def condition_distribution(
        dist: Distribution, error_prob: Distribution,
        config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
    one = config.factory.one(*error_prob.get_variables())
    zero = one * "0"
    if dist == zero and error_prob == one:
        raise ObserveZeroEventError(
            "Undefined semantics: Probability of observing a valid run is 0.")
    result = dist / (one - error_prob)
    return result, error_prob


def compute_discrete_distribution(
        program: Program | ProgramInfo, dist: Distribution,
        config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
    logger.debug("Compute the distribution for\n%s\nand distribution %s", str(program), dist)

    # Inject the context from the parsed program into the initial distribution.
    prog_info = program if isinstance(program,
                                      ProgramInfo) else ProgramInfo(program)
    variables = set(prog_info.variables.keys()).union(dist.get_variables())
    parameters = set(prog_info.parameters.keys()).union(dist.get_parameters())
    initial_dist = dist.set_variables(*variables).set_parameters(*parameters)
    error_prob = config.factory.one(*variables) * 0
    dist, error_prob = compute_semantics(prog_info.instructions,
                                         prog_info, initial_dist,
                                         error_prob, config)
    if config.normalize:
        dist, error_prob = condition_distribution(dist, error_prob, config)
    return dist, error_prob


def compute_semantics(
        instruction: Union[Instr, Sequence[Instr]],
        prog_info: ProgramInfo,
        distribution: Distribution,
        error_prob: Distribution,
        config=ForwardAnalysisConfig()
) -> tuple[Distribution, Distribution]:
    def _show_steps(inp: tuple[Distribution, Distribution],
                    instr: Instr) -> tuple[Distribution, Distribution]:
        dist, error_prob = inp
        res, error_prob = compute_semantics(instr, prog_info, dist,
                                            error_prob, config)
        if isinstance(instr, (WhileInstr, IfInstr, LoopInstr)):
            print("\n")
        output = f"\n{Style.BLUE}Instruction:{Style.RESET} {instr}\t {Style.GREEN}Result:{Style.RESET} {res}"
        print(output, prog_info.so_vars)
        if config.step_wise:
            input("Next step [Enter]")
        return res, error_prob

    def _dont_show_steps(
            inp: tuple[Distribution, Distribution],
            instr: Instr) -> tuple[Distribution, Distribution]:
        dist, error_prob = inp
        return compute_semantics(instr, prog_info, dist, error_prob,
                                 config)

    if isinstance(instruction, list):
        func = _show_steps if config.show_intermediate_steps else _dont_show_steps
        return functools.reduce(func, instruction,
                                (distribution, error_prob))

    elif isinstance(instruction, SkipInstr):
        return distribution, error_prob

    elif isinstance(instruction, AbortInstr):
        return distribution.factory().undefined(
            *distribution.get_variables()), error_prob

    elif isinstance(instruction, WhileInstr):
        logger.info("\n%s gets handled", instruction)
        return WhileHandler.compute(instruction, prog_info, distribution,
                                    error_prob, config, compute_semantics)

    elif isinstance(instruction, IfInstr):
        return ITEHandler.compute(instruction, prog_info, distribution,
                                  error_prob, config, compute_semantics)

    elif isinstance(instruction, AsgnInstr):
        return AssignmentHandler.compute(instruction, prog_info,
                                         distribution, error_prob, config, compute_semantics)

    elif isinstance(instruction, ChoiceInstr):
        return PChoiceHandler.compute(instruction, prog_info, distribution,
                                      error_prob, config, compute_semantics)

    elif isinstance(instruction, ObserveInstr):
        logger.info("%s gets handled", instruction)
        return ObserveHandler.compute(instruction, prog_info, distribution,
                                      error_prob, config, compute_semantics)

    elif isinstance(instruction, get_args(Query)):
        logger.info("%s gets handled", instruction)
        if config.normalize:
            distribution, error_prob = condition_distribution(
                distribution, error_prob,
                config)  # evaluate queries on conditioned distribution
            error_prob *= "0"
        return QueryHandler.compute(instruction, prog_info, distribution,
                                    error_prob, config, compute_semantics)

    elif isinstance(instruction, LoopInstr):
        logger.info("%s gets handled", instruction)
        return LoopHandler.compute(instruction, prog_info, distribution,
                                   error_prob, config, compute_semantics)

    elif isinstance(instruction, QueryInstr):
        logger.info("entering query block")
        return QueryBlockHandler.compute(instruction, prog_info,
                                         distribution, error_prob, config, compute_semantics)

    raise TypeError("illegal instruction")
