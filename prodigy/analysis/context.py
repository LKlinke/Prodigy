"""
--------------------
The analysis context
--------------------

What the parts of a running analysis (instruction handlers, loop strategies, nested inference, the step trace,
the upper track, the witness search) need from the run they belong to: the configuration, the decisions for
loops, where they book guarantees, warnings and queries, and how they start sub-analyses. :class:`AnalysisContext`
names these services; the :class:`~prodigy.analysis.session.AnalysisSession` provides them.

The parts get the context explicitly (T11 of ``docs/source/architecture.md``). An instruction handler is a function
``handle(frame, instruction, state) -> state``: the :class:`ProgramState` is the pair (distribution, violation mass),
and the :class:`Frame` names the run (:attr:`Frame.context`), the program or function body being executed, the
configuration and how nested instructions run. A context lives as long as one run (one
:meth:`~prodigy.analysis.session.AnalysisSession.run` or one call of an entry point); sub-analyses and nested
inference are scopes of the same context. A frame is immutable and lives as long as the program or body it executes.

Only the entry points look a context up: :func:`boundary_context` gives the one made active by
:meth:`~prodigy.analysis.session.AnalysisSession.activate` (the CLI still activates its session, until it passes it
explicitly), or a new detached one. Nothing in the semantics calls it.

This module is part of the vocabulary layer of the analysis (``docs/source/architecture.md``): it imports nothing
of the analysis at run time, so the handlers reach the session without importing it (T1). The types of the
session's own records appear in annotations only.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Mapping, NamedTuple, Optional, Protocol,
                    Sequence, Tuple, Union)

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.language import Expr, Instr, Program, WhileInstr
    from prodigy.language.outline import LoopInfo

    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.nested_inference import DistRegistry
    from prodigy.analysis.loops.api import LoopDecision, LoopOutcome, LoopTask, ParamSpec
    from prodigy.analysis.session import DecisionProvider, LoopRecord, LoopRegistry
    from prodigy.analysis.trace import Tracer
    from prodigy.analysis.upper_track import UpperTrack
    from prodigy.analysis.values import QueryResult
    from prodigy.distribution import Distribution

__all__ = ["Analyzer", "AnalysisContext", "Frame", "ProgramState", "Scope", "Semantics", "boundary_context",
           "has_active_context", "register_detached_context", "reset_context", "use_context"]

_EXACT = "exact"  # Bound.EXACT; the vocabulary of loops.api is not imported here at run time


@dataclass
class Scope:
    """A part of the run that books its guarantee apart (the whole run, a sub-analysis, a nested inference)."""

    silent: bool
    reason: str = ""
    kind: str = _EXACT
    residual: Any = 0
    upper: Optional["UpperTrack"] = None
    """The proven upper bound on the exact state (:mod:`prodigy.analysis.upper_track`), if one is carried."""


class AnalysisContext(Protocol):  # pylint: disable=too-many-public-methods
    """The services of a running analysis (see the module documentation)."""

    # -- what is analysed, and how --------------------------------------------------------
    program: Optional["Program"]
    source: Optional[str]
    config: "ForwardAnalysisConfig"
    coefficient_limit: int
    parameter_synthesis: str
    provider: "DecisionProvider"
    loops: "LoopRegistry"

    # -- what the run records -------------------------------------------------------------
    tracer: Optional["Tracer"]
    loop_records: Dict[Any, "LoopRecord"]
    queries: List[Any]
    warnings: List[str]
    unproven_conditions: List[str]
    dist_values: List[Any]
    dist_registry: "DistRegistry"
    """The ``dist`` sites of the run (:mod:`prodigy.analysis.nested_inference`)."""

    @staticmethod
    def spell(node: Union["Expr", "Instr"]) -> str:
        """``node`` in the syntax of the program."""

    def activate(self) -> ContextManager[Any]:
        """Makes this the current context while the block runs."""

    @property
    def scope(self) -> Scope:
        """The innermost scope."""

    @property
    def silent(self) -> bool:
        """Whether the current scope records nothing (a sub-analysis)."""

    @property
    def collect_details(self) -> bool:
        """Whether certificates and messages are worth collecting."""

    @property
    def bound_kind(self) -> str:
        """The guarantee of the current scope so far (a :class:`~prodigy.analysis.loops.api.Bound`)."""

    @property
    def residual(self) -> Any:
        """The residual mass of the current scope (for a lower bound)."""

    @property
    def upper(self) -> Optional["UpperTrack"]:
        """The proven upper bound carried through the run, if any."""

    @property
    def upper_distribution(self) -> Optional["Distribution"]:
        """The distribution of :attr:`upper`, if any."""

    @property
    def steps(self) -> List[Dict[str, Any]]:
        """The steps of the trace so far."""

    # -- scopes -----------------------------------------------------------------------------
    def sub_analysis(self, reason: str) -> ContextManager[Scope]:
        """A sub-analysis whose guarantee does not count for the run (checks of candidates, witnesses)."""

    def nested_inference(self, reason: str) -> ContextManager[Scope]:
        """The analysis of a function for ``infer`` or a call, whose guarantee counts for the run."""

    def loop_alias(self, copy: "WhileInstr", original: "WhileInstr") -> ContextManager[None]:
        """Looks up the loop ``copy`` as ``original`` while the block runs."""

    def other_program(self, program: "Program", decisions: Optional[Mapping[Any, Any]] = None) -> ContextManager[None]:
        """Decides the loops of ``program`` (not the run's own) by their own ids while the block runs."""

    def upper_replay(self) -> ContextManager[Scope]:
        """The scope in which the upper track replays an instruction."""

    # -- loops ------------------------------------------------------------------------------
    def decide_loop(self, loop: Optional["LoopInfo"], instruction: "WhileInstr") -> "LoopDecision":
        """The decision for ``loop`` (raises :class:`~prodigy.analysis.loops.api.DecisionRequired`)."""

    def ask_param(self, task: "LoopTask", spec: "ParamSpec") -> Any:
        """A parameter of a strategy that the decision does not give."""

    def resolve_alias(self, task: "LoopTask", alias: "ParamSpec", raw: Any) -> Any:
        """The value of a parameter given in its alternative spelling."""

    def run_loop(self, task: "LoopTask") -> "LoopOutcome":
        """Runs the strategy of ``task`` and books its outcome."""

    def loop_iterate(self, task: "LoopTask", iteration: int, terminated: Any, inside: Any = None,
                     residual: Any = None, kind: str = "lower") -> None:
        """Reports an iterate of a loop for the step trace."""

    def loop_query_bounds(self, kind: str, expression: "Expr") -> Optional[Tuple[Any, Any]]:
        """Bounds of a query directly after a loop that its strategy computed."""

    def loop_query_value(self, kind: str, expression: "Expr", distribution: Any) -> Optional[Any]:
        """The exact value of a query directly after a loop, if its strategy knows it."""

    def note(self, task: "LoopTask", message: str) -> None:
        """A message of a strategy for the loop's record."""

    def progress(self, task: "LoopTask", done: int, total: Optional[int], message: str = "") -> None:
        """Progress of a strategy."""

    # -- what the run books -----------------------------------------------------------------
    def on_instruction(self, instruction: Any) -> None:
        """Called before every instruction."""

    def on_normalize(self, error_prob: "Distribution") -> None:
        """Called when the result is conditioned on the observations."""

    def same_measure(self, before: Any, after: Any) -> None:
        """Books that an instruction kept the measure (for the upper track)."""

    def record_query(self, query: "QueryResult") -> None:
        """Books the value of a query."""

    def next_query_position(self) -> Tuple[Optional[int], Optional[int]]:
        """The source position of the next query."""

    def warn(self, message: str) -> None:
        """A warning for the result."""

    def unproven(self, message: str) -> None:
        """A side condition that could not be proven: the result has no guarantee."""

    def start_upper(self) -> None:
        """Starts the upper track of the run, if it carries one."""

    def advance_upper(self, instruction: "Instr", frame: "Frame") -> None:
        """Moves the upper track over ``instruction``."""

    def restrict_upper(self, variables: Sequence[str]) -> None:
        """Removes ``variables`` from the upper track."""

    def own_dist_registry(self) -> ContextManager["DistRegistry"]:
        """A fresh registry of ``dist`` sites for an analysis, unless it runs inside one (e.g. a function)."""


class ProgramState(NamedTuple):
    """The state of the forward analysis: the distribution of the terminated, valid runs and the mass of the runs
    that violated an observation (a distribution over the second-order variables, or a constant)."""

    dist: "Distribution"
    error: "Distribution"


class Semantics(Protocol):  # pylint: disable=too-few-public-methods
    """How a frame runs instructions: :func:`~prodigy.analysis.analyzer.compute_semantics_in` (traced, with nested
    inference) or the variant of nested inference for the shells of loops."""

    def __call__(self, frame: "Frame", instruction: Union["Instr", Sequence["Instr"]],
                 state: ProgramState) -> ProgramState:
        ...


Analyzer = Callable[
    [Union["Instr", Sequence["Instr"]], "ProgramInfo", "Distribution", "Distribution", "ForwardAnalysisConfig"],
    Tuple["Distribution", "Distribution"]
]
"""The forward semantics with positional arguments (instructions, program, distribution, violation mass,
configuration), for the code that is not converted to :class:`Frame` yet (:meth:`Frame.as_analyzer`)."""


@dataclass(frozen=True)
class Frame:
    """What one handler call needs besides the state (see the module documentation)."""

    context: AnalysisContext
    """The run."""
    program: "ProgramInfo"
    """The program or function body being executed."""
    config: "ForwardAnalysisConfig"
    """The configuration of this analysis (the session's, or the one given to a direct call)."""
    semantics: Semantics
    """How nested instructions run."""

    def run(self, instruction: Union["Instr", Sequence["Instr"]], state: ProgramState) -> ProgramState:
        """The semantics of ``instruction`` (or a list of instructions) on ``state`` in this frame."""
        return self.semantics(self, instruction, state)

    def for_program(self, program: "ProgramInfo") -> "Frame":
        """The frame for executing ``program`` (e.g. a function body) in the same run."""
        return Frame(self.context, program, self.config, self.semantics)

    def as_analyzer(self) -> Analyzer:
        """This frame's semantics with positional arguments; the program and configuration passed to it select
        the frame (:attr:`program`, :attr:`config`) of the call."""
        def analyze(instruction: Union["Instr", Sequence["Instr"]], program: "ProgramInfo",
                    distribution: "Distribution", error_prob: "Distribution",
                    config: "ForwardAnalysisConfig") -> Tuple["Distribution", "Distribution"]:
            frame = Frame(self.context, program, config, self.semantics)
            result = frame.run(instruction, ProgramState(distribution, error_prob))
            return result.dist, result.error
        return analyze


_CURRENT: contextvars.ContextVar[Optional[AnalysisContext]] = \
    contextvars.ContextVar("prodigy_analysis_session", default=None)

_DETACHED: List[Callable[[], AnalysisContext]] = []


def register_detached_context(factory: Callable[[], AnalysisContext]) -> None:
    """Registers how :func:`boundary_context` makes a context when none is active (the session module does)."""
    _DETACHED[:] = [factory]


def boundary_context() -> AnalysisContext:
    """
    For the entry points only (:func:`~prodigy.analysis.session.compute_discrete_distribution`,
    :func:`~prodigy.analysis.analyzer.compute_semantics`): the context made active by
    :meth:`~prodigy.analysis.session.AnalysisSession.activate`, or a new detached interactive session (#195), so
    that nothing a direct call records outlives it. The semantics gets its context from the :class:`Frame`.
    """
    context = _CURRENT.get()
    if context is not None:
        return context
    if not _DETACHED:
        raise RuntimeError("No analysis is running; start one with prodigy.analysis.session (AnalysisSession or "
                           "compute_discrete_distribution).")
    return _DETACHED[0]()


def has_active_context() -> bool:
    """Whether a context is active (see :func:`boundary_context`)."""
    return _CURRENT.get() is not None


def use_context(context: AnalysisContext) -> "contextvars.Token[Optional[AnalysisContext]]":
    """Makes ``context`` the active one; undo with :func:`reset_context` (see the session's ``activate``)."""
    return _CURRENT.set(context)


def reset_context(token: "contextvars.Token[Optional[AnalysisContext]]") -> None:
    """Undoes :func:`use_context`."""
    _CURRENT.reset(token)
