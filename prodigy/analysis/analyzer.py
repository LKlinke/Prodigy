from __future__ import annotations

import contextlib
import functools
import logging
import time
from typing import Callable, Iterator, Optional, Union, Sequence, cast, get_args

from prodigy.pgcl import Program, Instr, SkipInstr, AbortInstr, WhileInstr, LoopInstr, QueryInstr, Query, \
    ObserveInstr, ChoiceInstr, AsgnInstr, IfInstr, Binop, BinopExpr, NatLitExpr, VarExpr
from prodigy.pgcl.ast.expressions import InferExpr, SampleExpr
from prodigy.pgcl.ast.instructions import ExpectationInstr, ProbabilityQueryInstr
from prodigy.pgcl.draws import find_draw

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import ObserveZeroEventError
from prodigy.analysis.input_distribution import validate_input_distribution
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
from prodigy.analysis.session import AnalysisSession, current_session, has_current_session
from prodigy.distribution import Distribution, MarginalType
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def condition_distribution(
        dist: Distribution, error_prob: Distribution,
        config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
    # lets the session rescale the residual mass of approximated loops
    current_session().on_normalize(error_prob)
    one = config.factory.one(*error_prob.get_variables())
    zero = one * "0"
    if dist == zero and error_prob == one:
        raise ObserveZeroEventError(
            "Undefined semantics: Probability of observing a valid run is 0.")
    result = dist / (one - error_prob)
    return result, error_prob


def compute_discrete_distribution(
        program: Program | ProgramInfo, dist: Distribution,
        config: ForwardAnalysisConfig,
        step_callback: Optional[Callable[[Instr, Distribution, Distribution], None]] = None
) -> tuple[Distribution, Distribution]:
    """
    Computes the output distribution of ``program`` for the input ``dist``.

    ``dist`` must be proven to be a power series at 0 with non-negative coefficients and finite
    mass; otherwise :class:`~prodigy.analysis.input_distribution.InputDistributionError` is raised
    (:func:`~prodigy.analysis.input_distribution.validate_input_distribution`).

    :param step_callback: if given, called with (instruction, distribution, error_prob) after each
        top-level instruction.

    With ``config.show_intermediate_steps`` (or ``config.step_wise``) every step of the trace
    (:mod:`prodigy.analysis.trace`) is printed when it is done, with its line (and the analysis
    waits for Enter after each step); a session that records steps itself is left alone.
    """
    if config.engine == ForwardAnalysisConfig.Engine.SYMENGINE:
        # pylint: disable=import-outside-toplevel
        from prodigy.distribution.symengine_distribution import check_names
        check_names([*program.variables, *program.parameters])
    # the input must be proven to be a distribution, else the guarantees of the result mean nothing (#296)
    validate_input_distribution(dist, program)
    with config.activate(), _active_session(), nested_inference.analysis_registry(), _printed_steps(config):
        return _compute_discrete_distribution(program, dist, config, step_callback)


@contextlib.contextmanager
def _active_session() -> Iterator[None]:
    """Without a running session, the analysis gets a detached one of its own, so that all its
    parts (e.g. the exactness of a function's analysis for ``infer``) are booked in one place."""
    if has_current_session():
        yield
        return
    with AnalysisSession.detached().activate():
        yield


@contextlib.contextmanager
def _printed_steps(config: ForwardAnalysisConfig) -> Iterator[None]:
    """Prints the steps of the trace while the analysis runs (``--intermediate-results``, ``--stepwise``)."""
    if not (config.show_intermediate_steps or config.step_wise):
        yield
        return
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.trace import TraceLimits, Tracer, print_step
    session = current_session()
    if session.tracer is not None:
        yield
        return

    def on_step(step):
        print_step(step)
        if config.step_wise:
            session.provider.next_step()

    session.tracer = Tracer(session, TraceLimits(coefficients=0), on_step=on_step)
    try:
        with session.activate():
            yield
    finally:
        session.tracer = None


def _compute_discrete_distribution(
        program: Program | ProgramInfo, dist: Distribution,
        config: ForwardAnalysisConfig,
        step_callback: Optional[Callable[[Instr, Distribution, Distribution], None]]
) -> tuple[Distribution, Distribution]:
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
    if step_callback is None:
        dist, error_prob = compute_semantics(prog_info.instructions,
                                             prog_info, initial_dist,
                                             error_prob, config)
    else:
        dist = initial_dist
        for instr in prog_info.instructions:
            dist, error_prob = compute_semantics([instr], prog_info, dist, error_prob, config)
            step_callback(instr, dist, error_prob)
    if config.normalize:
        tracer = current_session().tracer
        started = time.perf_counter()
        before, error_before = dist, error_prob
        dist, error_prob = condition_distribution(dist, error_prob, config)
        if tracer is not None and not error_before.is_zero_dist():
            tracer.pseudo_step("normalize", "condition on the observations", before, error_before, dist,
                               error_prob, started)
    if block_variables and not dist.get_variables() <= set(block_variables):
        # they are 0 here, so this only removes them from the variables of the result
        dist = dist.marginal(*block_variables, method=MarginalType.EXCLUDE)
    if nested_inference.dist_variables(prog_info):
        # the values of dist variables are reported apart from the program state
        current_session().dist_values = nested_inference.dist_values(dist, prog_info)
        dist = nested_inference.strip_dist_variables(dist, prog_info)
        error_prob = nested_inference.strip_dist_variables(error_prob, prog_info)
    return dist, error_prob


def compute_semantics(
        instruction: Union[Instr, Sequence[Instr]],
        prog_info: ProgramInfo,
        distribution: Distribution,
        error_prob: Distribution,
        config=ForwardAnalysisConfig()
) -> tuple[Distribution, Distribution]:
    """The semantics of ``instruction`` (or a list of instructions) applied to ``distribution``.
    Every single instruction passes the session (:meth:`AnalysisSession.on_instruction`) and its
    step trace, if one is recorded (:mod:`prodigy.analysis.trace`)."""
    if isinstance(instruction, list):
        return functools.reduce(
            lambda state, instr: compute_semantics(instr, prog_info, state[0], state[1], config),
            instruction, (distribution, error_prob))

    session = current_session()
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
                current_session().same_measure(distribution, conditioned)
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
               ObserveInstr: "observe", ProbabilityQueryInstr: "a query", ExpectationInstr: "a query"}


def _check_no_draw_in_condition(instruction: Instr) -> None:
    """A draw (e.g. ``geometric(1/2)``) inside a guard, an observation or a query is not
    implemented: unlike an assignment (:mod:`prodigy.pgcl.draws`), there is no variable to draw it
    into. Raises a ``NotImplementedError`` that names the draw."""
    where = _CONDITIONS.get(type(instruction))
    if where is None:
        return
    condition = getattr(instruction, "cond", None) or getattr(instruction, "expr")
    draw = find_draw(condition)
    if draw is not None:
        session = current_session()
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
