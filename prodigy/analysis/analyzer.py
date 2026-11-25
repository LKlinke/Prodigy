from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Iterable, Optional, Union, Sequence, cast, get_args

import sympy

from prodigy.language import Program, Instr, SkipInstr, AbortInstr, WhileInstr, LoopInstr, QueryInstr, Query, \
    ObserveInstr, ChoiceInstr, AsgnInstr, IfInstr, Binop, BinopExpr, NatLitExpr, VarExpr
from prodigy.language.ast.expressions import InferExpr, SampleExpr
from prodigy.language.ast.instructions import CovarianceInstr, ExpectationInstr, ProbabilityQueryInstr, VarianceInstr
from prodigy.language.draws import find_draw

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
from prodigy.analysis import nested_inference
from prodigy.analysis.context import current_context
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.equality import engine_text, sympify_engine_string
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


# one wording on every engine: whether an engine proves the probability to be 0 or only fails to prove it positive
# depends on its simplification (#349)
OBSERVE_ZERO_MESSAGE = ("Undefined semantics: Probability of observing a valid run is 0, or could not be shown to "
                        "be positive.")


def condition_distribution(
        dist: Distribution, error_prob: Distribution,
        config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
    # lets the session rescale the residual mass of approximated loops
    current_context().on_normalize(error_prob)
    one = config.factory.one(*error_prob.get_variables())
    valid = one - error_prob
    if dist.is_provably_zero():
        # Each is compared to the zero measure over its own variables and parameters: SymPy's equality also
        # compares them, and `dist == one * "0"` failed when they differed (#248).
        if valid.is_provably_zero():
            raise ObserveZeroEventError(OBSERVE_ZERO_MESSAGE)
        # Nothing terminates without violating an observation: the result is the zero measure only if the
        # probability of a valid run is positive (the remaining runs diverge). If an engine cannot show that
        # (e.g. an unsimplified sum of roots of unity that is in fact 1 - 1), dividing by it would report 0 as
        # an exact result for a program without semantics (#248).
        if not _provably_positive(valid.get_probability_mass(), valid.get_parameters()):
            raise ObserveZeroEventError(OBSERVE_ZERO_MESSAGE)
    result = dist / valid
    return result, error_prob


def _provably_positive(mass: Union[str, sympy.Basic, Distribution], parameters: Iterable[str]) -> bool:
    """Whether the engine's value ``mass`` is provably positive (SymPy's assumptions; False: not shown)."""
    # parameters stay symbols; the constants pi, E and I keep their meaning
    names = [name for name in parameters if name not in ("pi", "E", "I")]
    try:
        return bool(sympify_engine_string(engine_text(mass), names).is_positive)
    except (SyntaxError, TypeError, ValueError, sympy.SympifyError):
        return False


def compute_program(
        program: Program | ProgramInfo, dist: Distribution,
        config: ForwardAnalysisConfig,
        step_callback: Optional[Callable[[Instr, Distribution, Distribution], None]]
) -> tuple[Distribution, Distribution]:
    """The forward analysis of a whole program in the current context (:mod:`prodigy.analysis.context`): its
    instructions, the upper track, the conditioning on the observations, the values of dist variables. The
    entry point is :func:`~prodigy.analysis.session.compute_discrete_distribution`, which activates a context
    and validates the input."""
    logger.debug("Compute the distribution for\n%s\nand distribution %s", str(program), dist)

    # Inject the context from the parsed program into the initial distribution.
    prog_info = program if isinstance(program,
                                      ProgramInfo) else ProgramInfo(program)
    variables = set(prog_info.variables.keys()).union(dist.get_variables())
    parameters = set(prog_info.parameters.keys()).union(dist.get_parameters())
    initial_dist = dist.set_variables(*variables).set_parameters(*parameters)
    # Variables declared in blocks are 0 outside their block: an input value for them is dropped.
    block_variables = sorted(prog_info.program.block_variables)
    for var in block_variables:
        if var in dist.get_variables():
            initial_dist = initial_dist.update(BinopExpr(Binop.EQ, VarExpr(var), NatLitExpr(0)))
    error_prob = config.factory.one(*variables) * 0
    session = current_context()
    session.start_upper()
    dist = initial_dist
    for instr in prog_info.instructions:
        dist, error_prob = compute_semantics([instr], prog_info, dist, error_prob, config)
        # a proven upper bound of a loop follows the program (prodigy.analysis.upper_track, #451)
        session.advance_upper(instr, prog_info, config, compute_semantics)
        if step_callback is not None:
            step_callback(instr, dist, error_prob)
    if config.normalize:
        tracer = current_context().tracer
        started = time.perf_counter()
        before, error_before = dist, error_prob
        dist, error_prob = condition_distribution(dist, error_prob, config)
        if tracer is not None and not error_before.is_zero_dist():
            tracer.pseudo_step("normalize", "condition on the observations", before, error_before, dist,
                               error_prob, started)
    if block_variables and not dist.get_variables() <= set(block_variables):
        # they are 0 here, so this only removes them from the variables of the result
        dist = dist.marginal(*block_variables, method=MarginalType.EXCLUDE)
        session.restrict_upper(block_variables)
    if nested_inference.dist_variables(prog_info):
        # the values of dist variables are reported apart from the program state
        current_context().dist_values = nested_inference.dist_values(dist, prog_info)
        dist = nested_inference.strip_dist_variables(dist, prog_info)
        error_prob = nested_inference.strip_dist_variables(error_prob, prog_info)
    return dist, error_prob


def compute_semantics(
        instruction: Union[Instr, Sequence[Instr]],
        prog_info: ProgramInfo,
        distribution: Distribution,
        error_prob: Distribution,
        config: ForwardAnalysisConfig = ForwardAnalysisConfig()
) -> tuple[Distribution, Distribution]:
    """The semantics of ``instruction`` (or a list of instructions) applied to ``distribution``.
    Every single instruction passes the context (:meth:`~prodigy.analysis.context.AnalysisContext.on_instruction`)
    and its step trace, if one is recorded (:mod:`prodigy.analysis.trace`)."""
    if isinstance(instruction, list):
        return functools.reduce(
            lambda state, instr: compute_semantics(instr, prog_info, state[0], state[1], config),
            instruction, (distribution, error_prob))

    session = current_context()
    session.on_instruction(instruction)
    tracer = session.tracer
    token = tracer.begin(instruction, distribution, error_prob) if tracer is not None else None
    if token is None:
        return _semantics(instruction, prog_info, distribution, error_prob, config)
    assert tracer is not None
    try:
        result = _semantics(instruction, prog_info, distribution, error_prob, config)
    except BaseException as exc:
        tracer.fail(token, exc)
        raise
    tracer.finish(token, *result)
    return result


def _semantics(
        instruction: Union[Instr, Sequence[Instr]],
        prog_info: ProgramInfo,
        distribution: Distribution,
        error_prob: Distribution,
        config: ForwardAnalysisConfig
) -> tuple[Distribution, Distribution]:
    # pylint: disable=too-many-return-statements
    if not isinstance(instruction, list):
        _check_no_draw_in_condition(cast(Instr, instruction))
        nested = _nested_inference(cast(Instr, instruction), prog_info, distribution, error_prob, config)
        if nested is not None:
            return nested
    if isinstance(instruction, SkipInstr):
        return distribution, error_prob


    elif isinstance(instruction, AbortInstr):
        return distribution.factory().zero(
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
            unconditioned = error_prob.is_zero_dist()
            conditioned, error_prob = condition_distribution(
                distribution, error_prob,
                config)  # evaluate queries on conditioned distribution
            if unconditioned:
                # dividing by one does not change the measure
                current_context().same_measure(distribution, conditioned)
            distribution = conditioned
            error_prob *= "0"
        if nested_inference.dist_variables(prog_info):
            # queries are about the program state; the dist variables stay for later instructions
            _, error_prob = QueryHandler.compute(
                instruction, prog_info, nested_inference.strip_dist_variables(distribution, prog_info),
                error_prob, config, compute_semantics)
            return distribution, error_prob
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


_CONDITIONS = {IfInstr: "the guard of if", WhileInstr: "the guard of while",
               ObserveInstr: "observe", ProbabilityQueryInstr: "a query", ExpectationInstr: "a query",
               VarianceInstr: "a query", CovarianceInstr: "a query"}


def _check_no_draw_in_condition(instruction: Instr) -> None:
    """A draw (e.g. ``geometric(1/2)``) inside a guard, an observation or a query is not
    implemented: unlike an assignment (:mod:`prodigy.language.draws`), there is no variable to draw it
    into. Raises a ``NotImplementedError`` that names the draw."""
    where = _CONDITIONS.get(type(instruction))
    if where is None:
        return
    conditions = [getattr(instruction, name) for name in ("cond", "expr", "lhs", "rhs")
                  if getattr(instruction, name, None) is not None]
    for condition in conditions:
        draw = find_draw(condition)
        if draw is not None:
            session = current_context()
            raise NotImplementedError(
                f"Sampling inside {where} ({session.spell(draw)} in {session.spell(condition)}) is not "
                f"implemented. Draw the value into a variable first and use the variable there.")


def _nested_inference(
        instruction: Instr, prog_info: ProgramInfo, distribution: Distribution,
        error_prob: Distribution, config: ForwardAnalysisConfig
) -> Optional[tuple[Distribution, Distribution]]:
    """The instructions that :mod:`prodigy.analysis.nested_inference` handles: loops with dist
    variables, ``Pr[d …]``/``Ex[d]`` in an instruction, ``infer``, ``sample`` and copies of dist
    values. ``None`` for all others."""
    if isinstance(instruction, (WhileInstr, LoopInstr)):
        result = nested_inference.run_loop(instruction, prog_info, distribution, error_prob, config,
                                           _semantics)
        if result is not None:
            return result
    resolved = nested_inference.resolve_queries(instruction, prog_info, distribution, error_prob, config,
                                                compute_semantics, _semantics)
    if resolved is not None:
        return resolved
    if isinstance(instruction, AsgnInstr):
        if isinstance(instruction.rhs, InferExpr):
            return nested_inference.infer(instruction, prog_info, distribution, error_prob, config,
                                          compute_semantics)
        if isinstance(instruction.rhs, SampleExpr):
            return nested_inference.sample(instruction, prog_info, distribution, error_prob, config,
                                           compute_semantics)
        if instruction.lhs in nested_inference.dist_variables(prog_info):
            return nested_inference.copy_dist(instruction, prog_info, distribution, error_prob)
    return None
