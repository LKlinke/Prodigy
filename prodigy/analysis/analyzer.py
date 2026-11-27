from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional, Union, Sequence, get_args

import sympy

from prodigy.language import Program, Instr, SkipInstr, AbortInstr, WhileInstr, LoopInstr, QueryInstr, Query, \
    ObserveInstr, ChoiceInstr, AsgnInstr, IfInstr, Binop, BinopExpr, NatLitExpr, VarExpr
from prodigy.language.ast.expressions import InferExpr, SampleExpr
from prodigy.language.ast.instructions import CovarianceInstr, ExpectationInstr, ProbabilityQueryInstr, VarianceInstr
from prodigy.language.draws import find_draw

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.context import AnalysisContext, Analyzer, Frame, ProgramState, boundary_context
from prodigy.analysis.exceptions import ObserveZeroEventError
from prodigy.analysis.instructionhandler import (assignment_handler, ite_handler, loop_handler, observe_handler,
                                                 probchoice_handler, query_handler, queryblock_handler,
                                                 while_handler)
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis import nested_inference
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.equality import engine_text, sympify_engine_string
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


# one wording on every engine: whether an engine proves the probability to be 0 or only fails to prove it positive
# depends on its simplification (#349)
OBSERVE_ZERO_MESSAGE = ("Undefined semantics: Probability of observing a valid run is 0, or could not be shown to "
                        "be positive.")


def condition_distribution(context: AnalysisContext, dist: Distribution, error_prob: Distribution,
                           config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
    """``dist`` conditioned on the observations, i.e. divided by the probability of a valid run."""
    # lets the session rescale the residual mass of approximated loops
    context.on_normalize(error_prob)
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
        context: AnalysisContext, program: Program | ProgramInfo, dist: Distribution,
        config: ForwardAnalysisConfig,
        step_callback: Optional[Callable[[Instr, Distribution, Distribution], None]]
) -> tuple[Distribution, Distribution]:
    """The forward analysis of a whole program in ``context`` (:mod:`prodigy.analysis.context`): its
    instructions, the upper track, the conditioning on the observations, the values of dist variables. The
    entry point is :func:`~prodigy.analysis.session.compute_discrete_distribution`, which chooses the context
    and validates the input."""
    logger.debug("Compute the distribution for\n%s\nand distribution %s", program, dist)

    # Inject the context from the parsed program into the initial distribution.
    prog_info = program if isinstance(program, ProgramInfo) else ProgramInfo(program)
    variables = set(prog_info.variables.keys()).union(dist.get_variables())
    parameters = set(prog_info.parameters.keys()).union(dist.get_parameters())
    initial_dist = dist.set_variables(*variables).set_parameters(*parameters)
    # Variables declared in blocks are 0 outside their block: an input value for them is dropped.
    block_variables = sorted(prog_info.program.block_variables)
    for var in block_variables:
        if var in dist.get_variables():
            initial_dist = initial_dist.update(BinopExpr(Binop.EQ, VarExpr(var), NatLitExpr(0)))
    frame = Frame(context, prog_info, config, compute_semantics_in)
    state = ProgramState(initial_dist, config.factory.one(*variables) * 0)
    context.start_upper()
    for instr in prog_info.instructions:
        state = frame.run([instr], state)
        # a proven upper bound of a loop follows the program (prodigy.analysis.upper_track, #451)
        context.advance_upper(instr, frame)
        if step_callback is not None:
            step_callback(instr, state.dist, state.error)
    dist, error_prob = state
    if config.normalize:
        tracer = context.tracer
        started = time.perf_counter()
        before, error_before = dist, error_prob
        dist, error_prob = condition_distribution(context, dist, error_prob, config)
        if tracer is not None and not error_before.is_zero_dist():
            tracer.pseudo_step("normalize", "condition on the observations", before, error_before, dist,
                               error_prob, started)
    if block_variables and not dist.get_variables() <= set(block_variables):
        # they are 0 here, so this only removes them from the variables of the result
        dist = dist.marginal(*block_variables, method=MarginalType.EXCLUDE)
        context.restrict_upper(block_variables)
    if nested_inference.dist_variables(prog_info):
        # the values of dist variables are reported apart from the program state
        context.dist_values = nested_inference.dist_values(dist, prog_info, context.dist_registry)
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
    """The semantics of ``instruction`` (or a list of instructions) applied to ``distribution``, for a direct call
    (the CLI's loop commands, tests): in the active session, or in a detached one that nothing outlives
    (:func:`~prodigy.analysis.context.boundary_context`). The analysis itself runs in a
    :class:`~prodigy.analysis.context.Frame` (:func:`compute_semantics_in`)."""
    return analyzer_in(boundary_context())(instruction, prog_info, distribution, error_prob, config)


def analyzer_in(context: AnalysisContext) -> Analyzer:
    """The forward semantics in ``context`` with positional arguments, for the code that takes an
    :data:`~prodigy.analysis.context.Analyzer` (the comparison, the witness search, the loop checks)."""
    def analyze(instruction: Union[Instr, Sequence[Instr]], prog_info: ProgramInfo, distribution: Distribution,
                error_prob: Distribution, config: ForwardAnalysisConfig) -> tuple[Distribution, Distribution]:
        frame = Frame(context, prog_info, config, compute_semantics_in)
        result = frame.run(instruction, ProgramState(distribution, error_prob))
        return result.dist, result.error
    return analyze


def compute_semantics_in(frame: Frame, instruction: Union[Instr, Sequence[Instr]],
                         state: ProgramState) -> ProgramState:
    """The semantics of ``instruction`` (or a list of instructions) on ``state`` in ``frame``.
    Every single instruction passes the context (:meth:`~prodigy.analysis.context.AnalysisContext.on_instruction`)
    and its step trace, if one is recorded (:mod:`prodigy.analysis.trace`)."""
    if isinstance(instruction, list):
        for instr in instruction:
            state = compute_semantics_in(frame, instr, state)
        return state
    session = frame.context
    session.on_instruction(instruction)
    tracer = session.tracer
    token = tracer.begin(instruction, state.dist, state.error) if tracer is not None else None
    if token is None:
        return untraced_semantics(frame, instruction, state)
    assert tracer is not None
    try:
        result = untraced_semantics(frame, instruction, state)
    except BaseException as exc:
        tracer.fail(token, exc)
        raise
    tracer.finish(token, *result)
    return result


def untraced_semantics(frame: Frame, instruction: Union[Instr, Sequence[Instr]],
                       state: ProgramState) -> ProgramState:
    """The rule of ``instruction`` without the step trace: the checks and the nested inference that apply to
    every instruction, then the handler of its construct (:data:`HANDLERS`). Nested instructions run with
    ``frame``'s semantics."""
    if isinstance(instruction, Sequence):
        for instr in instruction:
            state = untraced_semantics(frame, instr, state)
        return state
    _check_no_draw_in_condition(frame.context, instruction)
    nested = _nested_inference(frame, instruction, state)
    if nested is not None:
        return nested
    for kind in type(instruction).__mro__:
        handler = HANDLERS.get(kind)
        if handler is not None:
            return handler(frame, instruction, state)
    raise TypeError("illegal instruction")


def _skip(frame: Frame, instruction: SkipInstr, state: ProgramState) -> ProgramState:  # pylint: disable=unused-argument
    """``skip``: nothing changes."""
    return state


def _abort(frame: Frame, instruction: AbortInstr, state: ProgramState) -> ProgramState:  # pylint: disable=unused-argument
    """``abort``: the run diverges, the distribution is the zero measure."""
    return ProgramState(state.dist.factory().zero(*state.dist.get_variables()), state.error)


def _query(frame: Frame, instruction: Query, state: ProgramState) -> ProgramState:
    """A query is answered on the distribution conditioned on the observations (if the configuration normalises),
    and without the encoding of the dist variables; the state that follows is the conditioned one."""
    distribution, error_prob = state
    if frame.config.normalize:
        unconditioned = error_prob.is_zero_dist()
        # evaluate queries on the conditioned distribution
        conditioned, error_prob = condition_distribution(frame.context, distribution, error_prob, frame.config)
        if unconditioned:
            # dividing by one does not change the measure
            frame.context.same_measure(distribution, conditioned)
        distribution = conditioned
        error_prob *= "0"
    if nested_inference.dist_variables(frame.program):
        # queries are about the program state; the dist variables stay for later instructions
        stripped = nested_inference.strip_dist_variables(distribution, frame.program)
        answered = query_handler.handle(frame, instruction, ProgramState(stripped, error_prob))
        return ProgramState(distribution, answered.error)
    return query_handler.handle(frame, instruction, ProgramState(distribution, error_prob))


Handler = Callable[[Frame, Any, ProgramState], ProgramState]
"""An instruction handler: one rule of the forward semantics (``docs/source/semantics.md``)."""

HANDLERS: Dict[type, Handler] = {
    SkipInstr: _skip,
    AbortInstr: _abort,
    WhileInstr: while_handler.handle,
    IfInstr: ite_handler.handle,
    AsgnInstr: assignment_handler.handle,
    ChoiceInstr: probchoice_handler.handle,
    ObserveInstr: observe_handler.handle,
    LoopInstr: loop_handler.handle,
    QueryInstr: queryblock_handler.handle,
    **{kind: _query for kind in get_args(Query)},
}
"""The handler of each instruction class (looked up along the class's MRO)."""


_CONDITIONS = {IfInstr: "the guard of if", WhileInstr: "the guard of while",
               ObserveInstr: "observe", ProbabilityQueryInstr: "a query", ExpectationInstr: "a query",
               VarianceInstr: "a query", CovarianceInstr: "a query"}


def _check_no_draw_in_condition(context: AnalysisContext, instruction: Instr) -> None:
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
            session = context
            raise NotImplementedError(
                f"Sampling inside {where} ({session.spell(draw)} in {session.spell(condition)}) is not "
                f"implemented. Draw the value into a variable first and use the variable there.")


def _nested_inference(frame: Frame, instruction: Instr, state: ProgramState) -> Optional[ProgramState]:
    """The instructions that :mod:`prodigy.analysis.nested_inference` handles: loops with dist
    variables, ``Pr[d …]``/``Ex[d]`` in an instruction, ``infer``, ``sample`` and copies of dist
    values. ``None`` for all others."""
    if isinstance(instruction, (WhileInstr, LoopInstr)):
        result = nested_inference.run_loop(frame, instruction, state, untraced_semantics)
        if result is not None:
            return result
    resolved = nested_inference.resolve_queries(frame, instruction, state, untraced_semantics)
    if resolved is not None:
        return resolved
    if isinstance(instruction, AsgnInstr):
        if isinstance(instruction.rhs, InferExpr):
            return nested_inference.infer(frame, instruction, state)
        if isinstance(instruction.rhs, SampleExpr):
            return nested_inference.sample(frame, instruction, state)
        if instruction.lhs in nested_inference.dist_variables(frame.program):
            return nested_inference.copy_dist(frame, instruction, state)
    return None
