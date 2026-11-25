"""
-----------------
Analysis sessions
-----------------

An :class:`AnalysisSession` bundles everything a single analysis run needs to know up
front: the program, the input distribution, the configuration and one *decision* per
``while`` loop (which :mod:`loop strategy <prodigy.analysis.loops.api>` to use, with
which parameters). Instruction handlers no longer call :func:`input` themselves; they ask
the session's :class:`DecisionProvider`:

* :class:`InteractiveDecisions` asks on the terminal with the historic prompts. The CLI
  uses it, so ``prodigy main`` behaves as before.
* :class:`FixedDecisions` answers from a dictionary (e.g. parsed from JSON). The web GUI,
  a language server or ``prodigy analyze`` use it. A missing decision raises
  :class:`DecisionRequired` instead of blocking.

While a session runs, it also records what happened: query results with sound intervals
(see :mod:`prodigy.analysis.result`), what each loop strategy produced and certified,
warnings, progress and optionally the step trace (:mod:`prodigy.analysis.trace`).
:meth:`AnalysisSession.run` returns an :class:`~prodigy.analysis.result.AnalysisResult`
whose :meth:`~prodigy.analysis.result.AnalysisResult.to_json` is the exchange format for
front ends.

Code that calls :func:`~prodigy.analysis.analyzer.compute_discrete_distribution` or the
handlers directly (tests, scripts, older code) does not need a session: if none is active,
:func:`current_session` hands out a detached interactive one, which reproduces the old
behaviour.
"""
from __future__ import annotations

import contextlib
import contextvars
import io
import logging
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple,
                    Union)

import attr

from prodigy.language import (CovarianceInstr, ExpectationInstr, Expr, Instr, OptimizationQuery, PlotInstr, PrintInstr,
                              ProbabilityQueryInstr, Program, QueryInstr, VarianceInstr, WhileInstr)

from prodigy.analysis.loops.api import (Bound, DecisionRequired, LoopDecision, LoopOutcome, LoopStrategy, LoopTask,
                                        ParamSpec, get_loop_strategy, loop_strategies)
from prodigy.distribution import known_convergent_series
from prodigy.language.outline import LoopInfo, is_query, outline
from prodigy.language.positions import span_of
from prodigy.language.ppy_printer import spell
from prodigy.util.logger import log_setup, print_progress_bar

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.result import AnalysisResult
    from prodigy.analysis.values import QueryResult
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.trace import TraceLimits
    from prodigy.analysis.upper_track import Analyzer, UpperTrack
    from prodigy.distribution import Distribution

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


# ---------------------------------------------------------------------------------------
# Loops of the analysed program
# ---------------------------------------------------------------------------------------

_QUERY_INSTRS = (CovarianceInstr, VarianceInstr, ExpectationInstr, OptimizationQuery, PlotInstr, PrintInstr,
                 ProbabilityQueryInstr, QueryInstr)


class LoopRegistry:
    """The loops of the analysed program with their ids (:func:`~prodigy.language.outline.outline`); finds the
    :class:`LoopInfo` of a loop instruction, also of a copy of it."""

    def __init__(self, program: Optional[Program]):
        self.loops: List[LoopInfo] = []
        self._by_identity: Dict[int, LoopInfo] = {}
        self._by_position: Dict[Tuple[int, int], LoopInfo] = {}
        self._by_text: Dict[str, List[LoopInfo]] = {}
        self._instructions: List[WhileInstr] = []
        if program is None:
            return
        found = outline(program)
        for info, instr in zip(found.loops, found.loop_instructions, strict=True):
            self.loops.append(info)
            self._instructions.append(instr)
            self._by_identity[id(instr)] = info
            if info.line is not None and info.column is not None:
                self._by_position[(info.line, info.column)] = info
            self._by_text.setdefault(str(instr), []).append(info)

    def lookup(self, instr: WhileInstr) -> Optional[LoopInfo]:
        """The loop ``instr``, or the loop it is a copy of: found by identity, else by its source position (a
        deep copy keeps it, :mod:`prodigy.language.positions`), else by its text if that is unique (a copy
        without position, e.g. of a program built in Python)."""
        info = self._by_identity.get(id(instr))
        if info is not None:
            return info
        span = span_of(instr)
        if span is not None and (span.line, span.column) in self._by_position:
            return self._by_position[(span.line, span.column)]
        candidates = self._by_text.get(str(instr), [])
        return candidates[0] if len(candidates) == 1 else None

    def instruction(self, loop_id: int) -> Optional[WhileInstr]:
        """The loop with id ``loop_id`` (``None`` if there is none)."""
        return self._instructions[loop_id] if 0 <= loop_id < len(self._instructions) else None

    def alias(self, copy: WhileInstr, original: WhileInstr) -> None:
        """Looks up ``copy`` as ``original`` (see :meth:`AnalysisSession.loop_alias`)."""
        info = self.lookup(original)
        if info is not None:
            self._by_identity[id(copy)] = info

    def unalias(self, copy: WhileInstr) -> None:
        self._by_identity.pop(id(copy), None)

    def __iter__(self) -> Iterator[LoopInfo]:
        return iter(self.loops)

    def __len__(self) -> int:
        return len(self.loops)


@dataclass
class IsolatedLoop:
    """A loop of a program as a program of its own (:func:`isolate_loop`)."""

    program: str
    """The program text (``.ppy``)."""
    loop: int
    """The id of the loop in :attr:`program`."""
    declarations: str
    """The declarations alone (a loop-free program to start from)."""
    whole: bool
    """Whether :attr:`program` is the original source (the loop is all the program does)."""


def isolate_loop(program: Program, instruction: WhileInstr,
                 source: Optional[str] = None) -> Optional[IsolatedLoop]:
    """The loop ``instruction`` of ``program`` as a program to compare with a loop-free program (#154,
    #399): the program itself (``source``) if the loop is all it does apart from queries, else its
    declarations and the loop. The code before the loop is left out on purpose: a comparison holds for
    every input, so it covers whatever that code produces. ``None`` if the program cannot be printed in
    ``.ppy``."""
    # pylint: disable=import-outside-toplevel
    from prodigy.language.ppy_printer import PpyPrintError, program_to_ppy

    def text(instructions: List[Instr]) -> str:
        part = attr.evolve(program, instructions=instructions)
        return program_to_ppy(part)

    rest = [instr for instr in program.instructions
            if instr is not instruction and not is_query(instr)]
    try:
        declarations = text([])
        if not rest and source is not None and any(instr is instruction for instr in program.instructions):
            info = LoopRegistry(program).lookup(instruction)
            return IsolatedLoop(source, info.id if info is not None else 0, declarations, True)
        loop_program = attr.evolve(program, instructions=[instruction])
        info = LoopRegistry(loop_program).lookup(instruction)
        return IsolatedLoop(text([instruction]), info.id if info is not None else 0, declarations, False)
    except PpyPrintError:
        return None


# ---------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------


DEFAULT_KEY = "*"
"""Key for the decision that applies to every loop without an own decision."""


def parse_decisions(data: Optional[Mapping[Any, Any]]) -> Dict[Union[int, str], LoopDecision]:
    """Parses ``{"0": {...}, "1": "unroll:iterations=10", "*": {...}}``."""
    result: Dict[Union[int, str], LoopDecision] = {}
    for key, value in (data or {}).items():
        if value is None:
            continue
        if str(key).strip() == DEFAULT_KEY:
            result[DEFAULT_KEY] = LoopDecision.from_json(value)
            continue
        try:
            loop_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"Loop decisions are keyed by loop id or '*', got {key!r}") from None
        result[loop_id] = LoopDecision.from_json(value)
    return result


def validate_decisions(decisions: Mapping[Union[int, str], LoopDecision],
                       loops: Optional[LoopRegistry] = None) -> List[str]:
    """Problems with ``decisions`` (unknown strategies, unknown loops, bad parameters, generating
    functions that are not ``.ppy`` expressions)."""
    problems: List[str] = []
    known_ids = {loop.id for loop in loops} if loops is not None else None
    for key, decision in decisions.items():
        where = "default decision" if key == DEFAULT_KEY else f"loop {key}"
        if known_ids is not None and key != DEFAULT_KEY and key not in known_ids:
            problems.append(f"{where}: the program has no loop with this id")
        try:
            strategy = get_loop_strategy(decision.strategy)
        except KeyError as exc:
            problems.append(f"{where}: {exc.args[0]}")
            continue
        names = {p.name for p in strategy.params}
        for name, raw in decision.params.items():
            if name not in names:
                problems.append(f"{where}: strategy {strategy.name} has no parameter {name!r}")
                continue
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                continue
            try:
                strategy.param(name).parse(raw)
            except (ValueError, TypeError) as exc:
                problems.append(f"{where}: {strategy.param(name).label}: {exc}")
    return problems


class DecisionProvider:
    """Answers the questions an analysis may have while it runs."""

    interactive: bool = False

    def __init__(self, decisions: Optional[Mapping[Any, Any]] = None):
        self.decisions: Dict[Union[int, str], LoopDecision] = \
            {k: LoopDecision.from_json(v) for k, v in (decisions or {}).items()}

    def preset(self, loop: Optional[LoopInfo]) -> Optional[LoopDecision]:
        if loop is not None and loop.id in self.decisions:
            return self.decisions[loop.id]
        return self.decisions.get(DEFAULT_KEY)

    def loop_decision(self, session: "AnalysisSession", loop: Optional[LoopInfo],
                      instruction: WhileInstr) -> LoopDecision:
        raise NotImplementedError

    def ask_param(self, session: "AnalysisSession", task: LoopTask, spec: ParamSpec) -> Any:
        raise NotImplementedError

    def choose_candidate(self, task: LoopTask, candidates: List[str]) -> int:
        raw = task.decision.params.get("candidate")
        index = int(raw) if raw not in (None, "") else 0
        if not 0 <= index < len(candidates):
            raise DecisionRequired(
                f"Candidate {index} does not exist; {len(candidates)} candidate(s) were found",
                loop=task.loop, strategy=task.strategy.name, param="candidate")
        return index

    def next_step(self) -> None:
        """Called after each instruction in step-wise mode."""

    def continue_plot(self, probability: float) -> bool:  # pylint: disable=unused-argument
        return False


class InteractiveDecisions(DecisionProvider):
    """Asks on the terminal, with the prompts the CLI has always used.

    Decisions given up front (e.g. ``prodigy main --loop 0=unroll:iterations=10``) are used
    without asking.
    """

    interactive = True

    def __init__(self, decisions: Optional[Mapping[Any, Any]] = None, *, remember: bool = False):
        super().__init__(decisions)
        self.remember = remember
        """Answer every question about the same loop instruction with the first answer, e.g. when
        ``prodigy check_equality`` analyses a program again on one input after the check."""
        # id of the loop instruction -> (the instruction, kept alive so that the id stays its own; the answer)
        self._choices: Dict[int, Tuple[WhileInstr, LoopDecision]] = {}
        self._params: Dict[Tuple[int, str], Tuple[WhileInstr, Any]] = {}

    def loop_decision(self, session: AnalysisSession, loop: Optional[LoopInfo],
                      instruction: WhileInstr) -> LoopDecision:
        preset = self.preset(loop)
        if preset is not None:
            return preset
        if not self.remember:
            return self._ask_strategy()
        if id(instruction) not in self._choices:
            self._choices[id(instruction)] = (instruction, self._ask_strategy())
        return self._choices[id(instruction)][1]

    def _ask_strategy(self) -> LoopDecision:
        options: Dict[str, LoopStrategy] = {}
        lines = ["While Instruction has only limited support. Choose an option:\n"]
        numbered = [int(s.legacy_key) for s in loop_strategies() if s.legacy_key and s.legacy_key.isdigit()]
        extra_key = max(numbered, default=0) + 1
        for strategy in loop_strategies():
            key = strategy.legacy_key
            if key is None:
                key = str(extra_key)
                extra_key += 1
            options[key] = strategy
            lines.append(f"[{key}]: {strategy.menu_line}\n")
        lines.append("[q]: Quit.\n")
        menu = "".join(lines)
        while True:
            user_choice = input(menu)
            logger.info("User chose %s", user_choice)
            if user_choice in options:
                return LoopDecision(options[user_choice].name)
            if user_choice == "q":
                sys.exit()
            print(f"Invalid input: {user_choice}")

    def ask_param(self, session: AnalysisSession, task: LoopTask, spec: ParamSpec) -> Any:
        if not self.remember:
            return self._ask_param(session, task, spec)
        key = (id(task.instruction), spec.name)
        if key not in self._params:
            self._params[key] = (task.instruction, self._ask_param(session, task, spec))
        return self._params[key][1]

    def _ask_param(self, session: AnalysisSession, task: LoopTask, spec: ParamSpec) -> Any:
        alias = next((p for p in task.strategy.params if p.alias_of == spec.name), None)
        while True:
            if alias is not None and alias.kind == "file":
                path = input(alias.prompt)
                return session.resolve_alias(task, alias, path)
            raw = input(spec.prompt or f"{spec.label}: ")
            try:
                return spec.parse(raw)
            except (ValueError, TypeError) as exc:
                print(f"Invalid input: {exc}")

    def next_step(self) -> None:
        input("Next step [Enter]")

    def continue_plot(self, probability: float) -> bool:
        cont = input(f"Continue with p={probability}? [Y/n]")
        return cont.lower() != 'n'


class FixedDecisions(DecisionProvider):
    """Answers only from decisions given up front; never blocks."""

    interactive = False

    def loop_decision(self, session: AnalysisSession, loop: Optional[LoopInfo],
                      instruction: WhileInstr) -> LoopDecision:
        preset = self.preset(loop)
        if preset is not None:
            return preset
        what = loop.describe() if loop is not None else f"while ({session.spell(instruction.cond)})"
        raise DecisionRequired(f"No strategy chosen for {what}", loop=loop)

    def ask_param(self, session: AnalysisSession, task: LoopTask, spec: ParamSpec) -> Any:
        if spec.default is not None:
            return spec.parse(spec.default)
        where = task.loop.describe() if task.loop is not None else "a loop"
        raise DecisionRequired(
            f"Strategy '{task.strategy.title}' for {where} needs a value for '{spec.label}'",
            loop=task.loop, strategy=task.strategy.name, param=spec.name)


# ---------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------

@dataclass
class LoopRecord:
    """Everything one loop produced during a run (a loop may be entered several times)."""

    loop: Optional[LoopInfo]
    cond: str
    strategy: str
    params: Dict[str, Any]
    invocations: int = 0
    kind: str = Bound.EXACT
    residual: Any = 0
    time: float = 0.0
    certificates: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    progress: Optional[Tuple[int, Optional[int]]] = None
    failed: Optional[str] = None
    """Error message if the strategy failed (e.g. a refuted invariant)."""
    inside_checks: int = 0
    """How many of the invocations happened inside the check of another loop's invariant."""
    suppressed_messages: int = 0
    execution_kinds: Dict[str, int] = field(default_factory=dict)
    """How many executions that returned an outcome gave each kind of guarantee."""
    execution_messages: Dict[str, int] = field(default_factory=dict)
    """The messages of the outcomes, each with the number of executions that gave it (in order)."""

    MAX_MESSAGES = 4

    def add_message(self, message: str) -> None:
        """A message that does not belong to one outcome (a note, the error of a failed execution)."""
        if message in self.messages:
            return
        if len(self.messages) >= self.MAX_MESSAGES:
            self.suppressed_messages += 1
            return
        self.messages.append(message)

    def add_outcome(self, kind: str, messages: Sequence[str]) -> None:
        """Books the guarantee and the messages of one execution of the loop."""
        self.execution_kinds[kind] = self.execution_kinds.get(kind, 0) + 1
        for message in dict.fromkeys(messages):
            self.execution_messages[message] = self.execution_messages.get(message, 0) + 1

    def outcome_messages(self) -> List[str]:
        """The messages of the outcomes. A loop that ran several times (e.g. inside another loop) gets
        the messages that all its executions gave and, if the others differ between executions (e.g.
        the residual mass of each), one summary instead of them (#229)."""
        executions = sum(self.execution_kinds.values())
        common = [message for message, count in self.execution_messages.items() if count == executions]
        if len(common) == len(self.execution_messages):
            return common
        return common + [self._summary(executions)]

    def _summary(self, executions: int) -> str:
        order = (Bound.UNKNOWN, Bound.UPPER, Bound.LOWER, Bound.EXACT)
        labels = {Bound.UNKNOWN: "no guarantee", Bound.UPPER: "upper bound", Bound.LOWER: "lower bound",
                  Bound.EXACT: "exact"}
        counts = [f"{labels[kind]} in {self.execution_kinds[kind]}" for kind in order if kind in self.execution_kinds]
        summary = f"{executions} executions: {', '.join(counts)}"
        if self.execution_kinds.get(Bound.LOWER) and self.residual != 0:
            summary += f"; in total, mass {self.residual} is still inside the loop"
        return summary + "."

    def to_json(self) -> Dict[str, Any]:
        from prodigy.analysis.values import exact_and_float  # pylint: disable=import-outside-toplevel
        residual, residual_float = exact_and_float(self.residual)
        return {
            "id": self.loop.id if self.loop else None,
            "line": self.loop.line if self.loop else None,
            "depth": self.loop.depth if self.loop else None,
            "cond": self.cond,
            "strategy": self.strategy,
            "params": dict(self.params),
            "invocations": self.invocations,
            "kind": self.kind,
            "residual_mass": residual,
            "residual_mass_float": residual_float,
            "time": round(self.time, 6),
            "certificates": self.certificates,
            "messages": self.outcome_messages() + self.messages
            + ([f"… {self.suppressed_messages} similar message(s) omitted."] if self.suppressed_messages else []),
            "failed": self.failed,
            "inside_checks": self.inside_checks,
        }


@dataclass
class _Scope:
    silent: bool
    reason: str = ""
    kind: str = Bound.EXACT
    residual: Any = 0
    upper: Optional["UpperTrack"] = None
    """The proven upper bound on the exact state (:mod:`prodigy.analysis.upper_track`), if one is carried."""


ProgressCallback = Callable[[Dict[str, Any]], None]


_CURRENT: contextvars.ContextVar[Optional["AnalysisSession"]] = \
    contextvars.ContextVar("prodigy_analysis_session", default=None)


def has_current_session() -> bool:
    """Whether a session is active (otherwise :func:`current_session` makes a detached one)."""
    return _CURRENT.get() is not None


def current_session() -> "AnalysisSession":
    """
    The session of the running analysis (the innermost :meth:`AnalysisSession.activate`).

    Without an active session, every call returns a new detached interactive session (#195): nothing it
    records (decisions, loop records, steps, queries) outlives the call. This is only for code that runs
    handlers directly, e.g. tests. Every entry point of an analysis activates one session for the whole run,
    so that all its parts share the decisions and the records: :func:`~prodigy.analysis.analyzer.
    compute_discrete_distribution` (a detached one if none is active), :meth:`AnalysisSession.run`, the CLI
    commands and the GUI's jobs. Use :func:`has_current_session` to tell the two cases apart.
    """
    session = _CURRENT.get()
    if session is None:
        return AnalysisSession.detached()
    return session


# ---------------------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------------------

class AnalysisSession:
    """One analysis run with explicit decisions; see the module documentation."""

    def __init__(self,
                 program: Optional[Program],
                 *,
                 source: Optional[str] = None,
                 syntax: Optional[str] = None,
                 config: Optional[ForwardAnalysisConfig] = None,
                 provider: Optional[DecisionProvider] = None,
                 decisions: Optional[Mapping[Any, Any]] = None,
                 input_distribution: Any = None,
                 allow_files: bool = True,
                 capture_output: bool = False,
                 output_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[ProgressCallback] = None,
                 record_steps: bool = False,
                 coefficient_limit: int = 30,
                 trace_limits: Optional["TraceLimits"] = None,
                 watches: Sequence[str] = (),
                 on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
                 parameter_synthesis: str = "hint",
                 stream_coefficients: bool = False):
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.config import ForwardAnalysisConfig
        from prodigy.analysis.trace import Tracer, parse_watch
        from prodigy.language.parser import check_syntax
        if syntax is not None:
            warnings.warn("AnalysisSession(syntax=...) is deprecated: programs are always .ppy (#489); the "
                          "argument is removed in v0.8.0", DeprecationWarning, stacklevel=2)
            check_syntax(syntax)
        self.program = program
        self.source = source
        self.config = config if config is not None else ForwardAnalysisConfig()
        if provider is None:
            provider = FixedDecisions(parse_decisions(decisions))
        elif decisions:
            provider.decisions.update(parse_decisions(decisions))
        self.provider = provider
        self.input_distribution = input_distribution
        self.allow_files = allow_files
        self.capture_output = capture_output
        self.output_callback = output_callback
        self.progress_callback = progress_callback
        self.record_steps = record_steps
        self.coefficient_limit = coefficient_limit
        self.stream_coefficients = stream_coefficients
        """Whether the coefficients of the output distribution also go to ``progress_callback`` while they
        are computed (the GUI's live plots, :mod:`prodigy.analysis.streaming`)."""
        self.parameter_synthesis = parameter_synthesis
        """What the strategy ``invariant`` does with an invariant that synthesises parameters (template
        parameters, or a check that fixes parameters to values, #154): ``allow`` (comparing two programs,
        ``prodigy check_equality``), ``refuse`` (the GUI's Analyze mode: it is a comparison) or ``hint``
        (the command line: go on as before, but point to ``check_equality``)."""

        self.loops = LoopRegistry(program)
        self.query_positions = outline(program).query_positions if program is not None else []
        self.loop_records: Dict[Any, LoopRecord] = {}
        # the values of the dist variables at the end (prodigy.analysis.nested_inference.DistValue)
        self.dist_values: List[Any] = []
        self.queries: List[Any] = []
        self.tracer: Optional[Tracer] = None
        """Records the step trace (:mod:`prodigy.analysis.trace`) if ``record_steps`` is set."""
        names = None if program is None else list(program.variables) + list(program.parameters)
        # raises InvalidWatchError (a ValueError): the CLI and the GUI report it
        parsed_watches = [parse_watch(text, names) for text in watches]
        if record_steps:
            self.tracer = Tracer(self, trace_limits, parsed_watches, on_step=on_step)
        self.warnings: List[str] = []
        self.unproven_conditions: List[str] = []
        # bounds of a loop strategy for the queries right after its loop (LoopOutcome.query_bounds)
        self._loop_query_bounds: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
        # exact query values on the distribution the loop returned (LoopOutcome.query_values)
        self._loop_query_values: Tuple[Any, Dict[Tuple[str, str], Any]] = (None, {})
        # the upper track that the loop booked last starts (its instruction and the track), see advance_upper
        self._loop_upper: Tuple[Optional[WhileInstr], Optional["UpperTrack"]] = (None, None)
        self._replaying_upper = False
        self.log = io.StringIO()
        self._scopes: List[_Scope] = [_Scope(silent=False)]
        self._query_index = 0
        self.detached_session = False

    @staticmethod
    def spell(node: Union[Expr, Instr]) -> str:
        """``node`` as ``.ppy`` text (see :func:`prodigy.language.ppy_printer.spell`)."""
        return spell(node)

    # -- construction helpers ------------------------------------------------------------

    @classmethod
    def detached(cls) -> "AnalysisSession":
        """Session for code that runs handlers without setting up a session."""
        session = cls(None, provider=InteractiveDecisions())
        session.detached_session = True
        return session

    @classmethod
    def interactive(cls, program: Program, *, source: Optional[str] = None,
                    config: Optional[ForwardAnalysisConfig] = None,
                    decisions: Optional[Mapping[Any, Any]] = None, **kwargs: Any) -> "AnalysisSession":
        return cls(program, source=source, config=config,
                   provider=InteractiveDecisions(parse_decisions(decisions)), **kwargs)

    # -- activation --------------------------------------------------------------------

    @contextlib.contextmanager
    def activate(self) -> Iterator["AnalysisSession"]:
        """Makes this the current session (see :func:`current_session`)."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    # -- questions -----------------------------------------------------------------------

    def decide_loop(self, loop: Optional[LoopInfo], instruction: WhileInstr) -> LoopDecision:
        return self.provider.loop_decision(self, loop, instruction)

    def ask_param(self, task: LoopTask, spec: ParamSpec) -> Any:
        return self.provider.ask_param(self, task, spec)

    def resolve_alias(self, task: LoopTask, alias: ParamSpec, raw: Any) -> Any:
        """Turns an alternative parameter (e.g. a file name) into the main parameter."""
        if alias.kind == "file":
            if not self.allow_files:
                raise DecisionRequired(
                    f"'{alias.label}' is not available here; provide '{alias.alias_of}' instead",
                    loop=task.loop, strategy=task.strategy.name, param=alias.alias_of)
            with open(str(raw), "r", encoding="utf-8") as handle:
                return handle.read()
        assert alias.alias_of is not None
        return task.strategy.param(alias.alias_of).parse(raw)

    # -- scopes and bounds ---------------------------------------------------------------

    @property
    def scope(self) -> _Scope:
        return self._scopes[-1]

    @property
    def silent(self) -> bool:
        return any(scope.silent for scope in self._scopes)

    @contextlib.contextmanager
    def sub_analysis(self, reason: str) -> Iterator[_Scope]:
        """Runs nested analyses whose distributions are not the program's (second-order
        test distributions, invariant candidates, ...). Their loops are still decided as
        usual, but neither their queries nor their residual mass count for the program.
        The yielded scope tells afterwards whether everything in it was exact."""
        scope = _Scope(silent=True, reason=reason)
        self._scopes.append(scope)
        try:
            # their measures need not converge at 1: marginals are proven to be the sums of the series (#393)
            with known_convergent_series(False):
                yield scope
        finally:
            # The strategy that opened the sub-analysis decides what an inexact sub-analysis
            # means for its own outcome (see e.g. the invariant strategy).
            self._scopes.pop()

    @contextlib.contextmanager
    def nested_inference(self, reason: str) -> Iterator[_Scope]:
        """Runs the analysis of a function for ``infer``. Its loops and residual mass are booked
        in a scope of its own, which tells afterwards whether the analysis was exact: only an
        exact output distribution can be normalised."""
        scope = _Scope(silent=self.silent, reason=reason)
        self._scopes.append(scope)
        try:
            yield scope
        finally:
            self._scopes.pop()

    @contextlib.contextmanager
    def loop_alias(self, copy: WhileInstr, original: WhileInstr) -> Iterator[None]:
        """Lets ``copy`` (the loop with its ``Pr[…]``/``Ex[…]`` replaced by their values) have the
        id and decision of ``original`` while it runs."""
        self.loops.alias(copy, original)
        try:
            yield
        finally:
            self.loops.unalias(copy)

    @contextlib.contextmanager
    def other_program(self, program: Program, decisions: Optional[Mapping[Any, Any]] = None) -> Iterator[None]:
        """Decides the loops of ``program``, another program than the session's (the second program of a
        comparison), by their own ids and ``decisions`` while it runs, instead of looking them up among
        the loops of the session's program. Their records are kept apart (keyed ``("other", id)``)."""
        loops, presets, records = self.loops, self.provider.decisions, self.loop_records
        self.loops = LoopRegistry(program)
        self.provider.decisions = parse_decisions(decisions)
        self.loop_records = {}
        try:
            yield
        finally:
            other = self.loop_records
            self.loops, self.provider.decisions, self.loop_records = loops, presets, records
            for key, record in other.items():
                self.loop_records[("other", key)] = record

    def on_normalize(self, error_prob: Distribution) -> None:
        """Called right before the distribution is conditioned on the observations."""
        if self.silent:
            return
        scope = self.scope
        if scope.kind != Bound.LOWER:
            return
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.values import sym
        from prodigy.analysis.upper_track import normalized
        try:
            error_mass = sym(error_prob.get_probability_mass())
        except Exception:  # pylint: disable=broad-except
            scope.upper = None
            return
        if scope.upper is not None:
            # also when the lower track has no violation mass: the upper one may have some
            scope.upper = normalized(scope.upper, error_mass, scope.residual)
        if error_mass == 0:
            return
        scope.residual = (sym(scope.residual) / (1 - error_mass)).simplify()

    @property
    def collect_details(self) -> bool:
        """Whether to compute extra data for front ends (coefficient tables etc.)."""
        return not self.provider.interactive and not self.detached_session

    @property
    def bound_kind(self) -> str:
        return self._scopes[0].kind

    @property
    def residual(self) -> Any:
        return self._scopes[0].residual

    @property
    def upper(self) -> Optional["UpperTrack"]:
        """The proven upper bound on the program's state (:mod:`prodigy.analysis.upper_track`), if one is
        carried; only for a lower bound (for an exact result it is not needed, otherwise it is not sound)."""
        scope = self._scopes[0]
        return scope.upper if scope.kind == Bound.LOWER else None

    @property
    def upper_distribution(self) -> Optional["Distribution"]:
        """The distribution of :attr:`upper`, or ``None``."""
        track = self.upper
        return track.distribution if track is not None else None

    def start_upper(self) -> None:
        """Starts an analysis of the program: no upper track is carried yet."""
        if not self.silent:
            self.scope.upper = None
            self._loop_upper = (None, None)

    def advance_upper(self, instruction: Instr, prog_info: "ProgramInfo", config: "ForwardAnalysisConfig",
                      analyzer: "Analyzer") -> None:
        """Moves the upper track past the top-level ``instruction``, which has just run
        (:mod:`prodigy.analysis.upper_track`): a loop whose strategy proved an upper bound starts it, a query
        leaves it as it is (normalisation is :meth:`on_normalize`), loop-free code is replayed on it, and
        everything else ends it."""
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.nested_inference import dist_variables
        from prodigy.analysis.upper_track import replay
        if self.silent:
            return
        scope = self.scope
        started_by, track = self._loop_upper
        self._loop_upper = (None, None)
        if dist_variables(prog_info):
            scope.upper = None  # nested inference is not followed
        elif started_by is not None and started_by is instruction:
            scope.upper = track
        elif scope.upper is None:
            return
        elif scope.kind != Bound.LOWER:
            scope.upper = None
        elif not isinstance(instruction, _QUERY_INSTRS) or isinstance(instruction, QueryInstr):
            scope.upper = replay(scope.upper, instruction, prog_info, config, analyzer, self)

    def restrict_upper(self, variables: Sequence[str]) -> None:
        """Sums ``variables`` out of the upper track, as out of the program's distribution (a positive linear map,
        so the bound stays one); the track ends if that fails."""
        scope = self.scope
        if self.silent or scope.upper is None:
            return
        # pylint: disable=import-outside-toplevel
        from dataclasses import replace
        from prodigy.distribution import MarginalType
        try:
            with known_convergent_series(True):  # a finite measure (prodigy.analysis.upper_track.start)
                marginal = scope.upper.distribution.marginal(*variables, method=MarginalType.EXCLUDE)
        except Exception:  # pylint: disable=broad-except  # the track is only an addition
            scope.upper = None
            return
        scope.upper = replace(scope.upper, distribution=marginal)

    @contextlib.contextmanager
    def upper_replay(self) -> Iterator[_Scope]:
        """The silent sub-analysis in which an instruction is replayed on the upper track; a loop met there
        ends the track (:class:`~prodigy.analysis.upper_track.UpperTrackEnded`)."""
        replaying = self._replaying_upper
        # the replay passes on_instruction again; the bounds of the loop right before the next query stay
        kept = (self._loop_query_bounds, self._loop_query_values, self._loop_upper)
        self._replaying_upper = True
        try:
            with self.sub_analysis("upper bound") as scope:
                yield scope
        finally:
            self._replaying_upper = replaying
            self._loop_query_bounds, self._loop_query_values, self._loop_upper = kept

    # -- loops -------------------------------------------------------------------------

    def run_loop(self, task: LoopTask) -> LoopOutcome:
        """Runs ``task.strategy`` and books its outcome (used by the while handler)."""
        tracer = self.tracer
        if tracer is None:
            return self._run_loop(task)
        frame = tracer.loop_started(task)
        try:
            outcome = self._run_loop(task)
        except BaseException as exc:
            tracer.loop_finished(task, frame, None, exc)
            raise
        tracer.loop_finished(task, frame, outcome)
        return outcome

    def loop_iterate(self, task: LoopTask, iteration: int, terminated: Any, inside: Any = None,
                     residual: Any = None, kind: str = Bound.LOWER) -> None:
        """An iterate of an unrolling strategy, for the step trace (see :meth:`LoopTask.iterate`)."""
        if self.tracer is not None:
            self.tracer.loop_iterate(task, iteration, terminated, inside, residual, kind)

    def _run_loop(self, task: LoopTask) -> LoopOutcome:
        if self._replaying_upper:
            # pylint: disable=import-outside-toplevel
            from prodigy.analysis.upper_track import UpperTrackEnded
            raise UpperTrackEnded("a loop: its strategy does not bound the exact state from above")
        key = task.loop.id if task.loop is not None else f"while ({task.instruction.cond})"
        record = self.loop_records.get(key)
        if record is None:
            record = LoopRecord(loop=task.loop, cond=self.spell(task.instruction.cond),
                                strategy=task.strategy.name,
                                params=dict(task.decision.params))
            self.loop_records[key] = record
        if self.silent:
            # e.g. an inner loop met while checking the invariant of an outer loop
            record.inside_checks += 1
        task.record = record
        start = time.perf_counter()
        try:
            outcome = task.strategy.run(task)
        except Exception as exc:
            record.time += time.perf_counter() - start
            record.invocations += 1
            record.kind = Bound.UNKNOWN
            record.failed = str(exc) or type(exc).__name__
            details = getattr(exc, "details", None)
            if details:
                record.certificates.append(details)
            # e.g. the undecided candidates of a synthesis that found no certified invariant
            for message in getattr(exc, "messages", ()):
                record.add_message(message)
            raise
        record.time += time.perf_counter() - start
        record.invocations += 1
        record.kind = Bound.combine(record.kind, outcome.kind)
        if outcome.certificate:
            record.certificates.append(outcome.certificate)
        record.add_outcome(outcome.kind, outcome.messages)

        scope = self.scope
        if not self.silent:
            self._loop_query_bounds = self._sound_query_bounds(scope.kind, outcome.query_bounds)
            self._loop_query_values = (outcome.distribution, dict(outcome.query_values))
            self._loop_upper = (task.instruction, self._upper_track(task, outcome, scope.kind))
        scope.kind = Bound.combine(scope.kind, outcome.kind)
        if outcome.kind == Bound.LOWER and not self.silent:
            from prodigy.analysis.values import sym  # pylint: disable=import-outside-toplevel
            residual = sym(outcome.residual)
            record.residual = sym(record.residual) + residual
            scope.residual = sym(scope.residual) + residual
        return outcome

    @staticmethod
    def _upper_track(task: LoopTask, outcome: LoopOutcome, kind_before: str) -> Optional["UpperTrack"]:
        """The upper track that the outcome of ``task`` starts, if any (:func:`~prodigy.analysis.upper_track.start`)."""
        if outcome.upper is None:
            return None
        from prodigy.analysis.upper_track import start  # pylint: disable=import-outside-toplevel
        where = f"loop {task.loop.id}" if task.loop is not None else "a loop"
        return start(outcome.upper, outcome.error_prob, kind_before, f"{where} ({task.strategy.name})")

    @staticmethod
    def _sound_query_bounds(kind_before: str, bounds: Dict[Tuple[str, str], Tuple[Any, Any]]
                            ) -> Dict[Tuple[str, str], Tuple[Any, Any]]:
        """The part of a strategy's query bounds that stays sound for the program: they were
        computed from the loop's incoming distribution, so they hold as they are only if that
        distribution is exact; if it is a lower bound, only their lower sides hold."""
        if kind_before == Bound.EXACT:
            return dict(bounds)
        if kind_before == Bound.LOWER:
            return {key: (lower, None) for key, (lower, _) in bounds.items()}
        return {}

    def on_instruction(self, instruction: Any) -> None:
        """Called for every instruction before it is executed (see ``compute_semantics``)."""
        if not isinstance(instruction, _QUERY_INSTRS):
            self._loop_query_bounds = {}
            self._loop_query_values = (None, {})

    def loop_query_bounds(self, kind: str, expression: str) -> Optional[Tuple[Any, Any]]:
        """Bounds that the strategy of the loop right before this query computed for it."""
        return self._loop_query_bounds.get((kind, expression))

    def same_measure(self, before: Any, after: Any) -> None:
        """Notes that ``after`` is the same measure as ``before`` (e.g. ``before`` divided by one), so
        that the query values of the loop that returned ``before`` hold for ``after`` as well."""
        returned, values = self._loop_query_values
        if returned is not None and before is returned:
            self._loop_query_values = (after, values)

    def loop_query_value(self, kind: str, expression: str, distribution: Any) -> Optional[Any]:
        """The exact value of a query that the strategy of the loop right before it computed, if
        ``distribution`` is the very distribution that loop returned (``None`` otherwise)."""
        returned, values = self._loop_query_values
        if returned is None or distribution is not returned:
            return None
        return values.get((kind, expression))

    def note(self, task: LoopTask, message: str) -> None:
        record = task.record
        if record is not None:
            record.add_message(message)

    def warn(self, message: str) -> None:
        if not self.silent and message not in self.warnings:
            self.warnings.append(message)

    def unproven(self, message: str) -> None:
        """A side condition of the analysis step that is running could not be proven (e.g. that the argument of
        ``sample_pgf`` is a probability generating function, #424): its result, and with it the result of the
        current scope, has no guarantee. ``message`` says which condition and why; it is a warning of the program's
        analysis, and kept in :attr:`unproven_conditions` also inside sub-analyses (e.g. for the reason of a
        comparison's verdict)."""
        self.scope.kind = Bound.combine(self.scope.kind, Bound.UNKNOWN)
        if message not in self.unproven_conditions:
            self.unproven_conditions.append(message)
        self.warn(message)

    def progress(self, task: LoopTask, done: int, total: Optional[int], message: str = "") -> None:
        record = task.record
        if record is not None:
            record.progress = (done, total)
        if self.provider.interactive:
            if total:
                print_progress_bar(done, total, length=50)
            return
        if self.progress_callback is not None and not self.silent:
            self.progress_callback({
                "loop": task.loop.id if task.loop else None,
                "strategy": task.strategy.name,
                "done": done,
                "total": total,
                "message": message,
            })

    # -- queries -----------------------------------------------------------------------

    def next_query_position(self) -> Tuple[Optional[int], Optional[int]]:
        if self._query_index < len(self.query_positions):
            return self.query_positions[self._query_index]
        return None, None

    def record_query(self, query: QueryResult) -> None:
        """Books a :class:`~prodigy.analysis.result.QueryResult` (ignored in sub-analyses)."""
        if self.silent:
            return
        query.index = self._query_index
        query.line, query.column = self.next_query_position()
        self._query_index += 1
        self.queries.append(query)

    # -- steps -------------------------------------------------------------------------

    @property
    def steps(self) -> List[Dict[str, Any]]:
        """The recorded steps (empty unless ``record_steps``); see :mod:`prodigy.analysis.trace`."""
        return self.tracer.steps if self.tracer is not None else []

    # -- running -----------------------------------------------------------------------

    @contextlib.contextmanager
    def _output(self) -> Iterator[None]:
        if not self.capture_output:
            yield
            return
        writer = _TeeWriter(self.log, self.output_callback)
        # Native code (the GiNaC bindings) writes to file descriptor 1 directly; keep that out
        # of the real stdout as well (e.g. for `prodigy analyze --json`).
        with _native_stdout_captured(writer.write), contextlib.redirect_stdout(writer):
            yield

    def run(self) -> AnalysisResult:
        """Runs the analysis; never raises for analysis errors (they end up in the result)."""
        from prodigy.analysis.result import build_result  # pylint: disable=import-outside-toplevel
        return build_result(self)


def _flush_c_stdio() -> None:
    try:
        import ctypes  # pylint: disable=import-outside-toplevel
        ctypes.CDLL(None).fflush(None)
    except Exception:  # pylint: disable=broad-except
        pass


@contextlib.contextmanager
def _native_stdout_captured(append: Callable[[str], Any]) -> Iterator[None]:
    """Redirects file descriptor 1 into a temporary file and hands its content to ``append``."""
    try:
        target = sys.__stdout__
        descriptor = target.fileno() if target is not None else 1
    except (AttributeError, OSError, ValueError):
        yield
        return
    try:
        if target is not None:
            target.flush()
        _flush_c_stdio()
        saved = os.dup(descriptor)
    except OSError:
        yield
        return
    with tempfile.TemporaryFile() as spool:
        os.dup2(spool.fileno(), descriptor)
        try:
            yield
        finally:
            _flush_c_stdio()
            os.dup2(saved, descriptor)
            os.close(saved)
            spool.seek(0)
            data = spool.read().decode("utf-8", "replace")
            if data:
                append(data)


class _TeeWriter(io.TextIOBase):
    """stdout replacement that keeps a copy and forwards chunks to a callback."""

    def __init__(self, buffer: io.StringIO, callback: Optional[Callable[[str], None]]):
        super().__init__()
        self._buffer = buffer
        self._callback = callback

    def write(self, text: str) -> int:
        self._buffer.write(text)
        if self._callback is not None and text:
            try:
                self._callback(text)
            except Exception:  # pylint: disable=broad-except
                pass
        return len(text)

    def flush(self) -> None:
        pass


def _safe(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except Exception:  # pylint: disable=broad-except
        return None
