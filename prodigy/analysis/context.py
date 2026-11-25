"""
--------------------
The analysis context
--------------------

What the parts of a running analysis (instruction handlers, loop strategies, nested inference, the step trace,
the upper track, the witness search) need from the run they belong to: the configuration, the decisions for
loops, where they book guarantees, warnings and queries, and how they start sub-analyses. :class:`AnalysisContext`
names these services; the :class:`~prodigy.analysis.session.AnalysisSession` provides them and makes itself the
current context while it runs (:meth:`~prodigy.analysis.session.AnalysisSession.activate`), and the parts reach it
with :func:`current_context`.

This module is part of the vocabulary layer of the analysis (``docs/source/architecture.md``): it imports nothing
of the analysis at run time, so the handlers reach the session without importing it (T1). The types of the
session's own records appear in annotations only. Passing the context explicitly instead of through a context
variable is the next step (P5b, T11).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Mapping, Optional, Protocol, Sequence,
                    Tuple, Union)

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.language import Expr, Instr, Program, WhileInstr
    from prodigy.language.outline import LoopInfo

    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.loops.api import LoopDecision, LoopOutcome, LoopTask, ParamSpec
    from prodigy.analysis.session import DecisionProvider, LoopRecord, LoopRegistry
    from prodigy.analysis.trace import Tracer
    from prodigy.analysis.upper_track import Analyzer, UpperTrack
    from prodigy.analysis.values import QueryResult
    from prodigy.distribution import Distribution

__all__ = ["AnalysisContext", "Scope", "current_context", "has_current_context", "register_detached_context",
           "reset_context", "use_context"]

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

    def loop_query_bounds(self, kind: str, expression: str) -> Optional[Tuple[Any, Any]]:
        """Bounds of a query directly after a loop that its strategy computed."""

    def loop_query_value(self, kind: str, expression: str, distribution: Any) -> Optional[Any]:
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

    def advance_upper(self, instruction: "Instr", prog_info: "ProgramInfo", config: "ForwardAnalysisConfig",
                      analyzer: "Analyzer") -> None:
        """Moves the upper track over ``instruction``."""

    def restrict_upper(self, variables: Sequence[str]) -> None:
        """Removes ``variables`` from the upper track."""


_CURRENT: contextvars.ContextVar[Optional[AnalysisContext]] = \
    contextvars.ContextVar("prodigy_analysis_session", default=None)

_DETACHED: List[Callable[[], AnalysisContext]] = []


def register_detached_context(factory: Callable[[], AnalysisContext]) -> None:
    """Registers how :func:`current_context` makes a context when none is active (the session module does)."""
    _DETACHED[:] = [factory]


def has_current_context() -> bool:
    """Whether a context is active (otherwise :func:`current_context` makes a detached one)."""
    return _CURRENT.get() is not None


def current_context() -> AnalysisContext:
    """
    The context of the running analysis (the innermost :func:`use_context`).

    Without an active context, every call returns a new detached interactive session (#195): nothing it
    records (decisions, loop records, steps, queries) outlives the call. This is only for code that runs
    handlers directly, e.g. tests. Every entry point of an analysis activates one session for the whole run,
    so that all its parts share the decisions and the records:
    :func:`~prodigy.analysis.session.compute_discrete_distribution` (a detached one if none is active),
    :meth:`~prodigy.analysis.session.AnalysisSession.run`, the CLI commands and the GUI's jobs.
    """
    context = _CURRENT.get()
    if context is not None:
        return context
    if not _DETACHED:
        raise RuntimeError("No analysis is running; start one with prodigy.analysis.session (AnalysisSession or "
                           "compute_discrete_distribution).")
    return _DETACHED[0]()


def use_context(context: AnalysisContext) -> "contextvars.Token[Optional[AnalysisContext]]":
    """Makes ``context`` the current one; undo with ``_CURRENT.reset(token)`` (see the session's ``activate``)."""
    return _CURRENT.set(context)


def reset_context(token: "contextvars.Token[Optional[AnalysisContext]]") -> None:
    """Undoes :func:`use_context`."""
    _CURRENT.reset(token)
