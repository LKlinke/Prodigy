"""
-----------------
Analysis sessions
-----------------

An :class:`AnalysisSession` bundles everything a single analysis run needs to know up
front: the program, the input distribution, the configuration and one *decision* per
``while`` loop (which :mod:`loop strategy <prodigy.analysis.loop_strategies>` to use, with
which parameters). Instruction handlers no longer call :func:`input` themselves; they ask
the session's :class:`DecisionProvider`:

* :class:`InteractiveDecisions` asks on the terminal with the historic prompts. The CLI
  uses it, so ``prodigy main`` behaves as before.
* :class:`FixedDecisions` answers from a dictionary (e.g. parsed from JSON). The web GUI,
  a language server or ``prodigy analyze`` use it. A missing decision raises
  :class:`DecisionRequired` instead of blocking.

While a session runs, it also records what happened: query results with sound intervals
(see :mod:`prodigy.analysis.result`), what each loop strategy produced and certified,
warnings, progress and optionally the distribution after each top-level instruction.
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
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from probably.pgcl import (ChoiceInstr, IfInstr, Instr, LoopInstr, Program, QueryInstr,
                           WhileInstr)

from prodigy.analysis.loop_strategies import (Bound, LoopOutcome, LoopStrategy, LoopTask, ParamSpec,
                                              get_loop_strategy, loop_strategies)
from prodigy.util.logger import log_setup, print_progress_bar

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


# ---------------------------------------------------------------------------------------
# Program outline: loops and queries with source positions
# ---------------------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"(#|//).*$", re.M)
_WHILE_RE = re.compile(r"\bwhile\b")
_QUERY_RE = re.compile(r"\?Pr\b|\?Ex\b|\?Opt\b|!Print\b|!Plot\b")


def _blank_comments(source: str) -> str:
    """Replaces comments by spaces so that offsets stay valid."""
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), source)


def token_positions(source: str, pattern: "re.Pattern[str]") -> List[Tuple[int, int]]:
    """1-based (line, column) of every match of ``pattern`` outside of comments."""
    text = _blank_comments(source)
    positions = []
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        column = match.start() - (text.rfind("\n", 0, match.start()) + 1) + 1
        positions.append((line, column))
    return positions


@dataclass(frozen=True)
class LoopInfo:
    """A while loop of the analysed program, identified by its position in the text."""

    id: int
    """Index of the loop in textual order (0-based); the key for decisions."""
    cond: str
    """The loop guard as pGCL text."""
    depth: int
    """Number of enclosing while loops."""
    function: Optional[str] = None
    """Name of the enclosing function declaration, if any."""
    line: Optional[int] = None
    column: Optional[int] = None

    def describe(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"loop {self.id}{where}: while ({self.cond})"

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cond": self.cond,
            "depth": self.depth,
            "function": self.function,
            "line": self.line,
            "column": self.column,
        }


def _children(instr: Instr) -> List[List[Instr]]:
    if isinstance(instr, WhileInstr):
        return [instr.body]
    if isinstance(instr, IfInstr):
        return [instr.true or [], instr.false or []]
    if isinstance(instr, ChoiceInstr):
        return [instr.lhs, instr.rhs]
    if isinstance(instr, LoopInstr):
        return [instr.body]
    if isinstance(instr, QueryInstr):
        return [instr.instrs]
    return []


def _collect_loops(instrs: List[Instr], depth: int, function: Optional[str],
                   out: List[Tuple[WhileInstr, int, Optional[str]]]) -> None:
    for instr in instrs:
        if isinstance(instr, WhileInstr):
            out.append((instr, depth, function))
            _collect_loops(instr.body, depth + 1, function, out)
        else:
            for block in _children(instr):
                _collect_loops(block, depth, function, out)


class LoopRegistry:
    """Assigns stable ids to the while loops of a program (textual pre-order)."""

    def __init__(self, program: Optional[Program], source: Optional[str] = None):
        self.loops: List[LoopInfo] = []
        self._by_identity: Dict[int, LoopInfo] = {}
        self._by_text: Dict[str, List[LoopInfo]] = {}
        if program is None:
            return
        found: List[Tuple[WhileInstr, int, Optional[str]]] = []
        for name, function in program.functions.items():
            _collect_loops(function.instructions, 0, name, found)
        _collect_loops(program.instructions, 0, None, found)
        positions = token_positions(source, _WHILE_RE) if source is not None else []
        if len(positions) != len(found):
            positions = []
        for index, (instr, depth, function) in enumerate(found):
            line, column = positions[index] if positions else (None, None)
            info = LoopInfo(id=index, cond=str(instr.cond), depth=depth, function=function,
                            line=line, column=column)
            self.loops.append(info)
            self._by_identity[id(instr)] = info
            self._by_text.setdefault(str(instr), []).append(info)

    def lookup(self, instr: WhileInstr) -> Optional[LoopInfo]:
        info = self._by_identity.get(id(instr))
        if info is not None:
            return info
        candidates = self._by_text.get(str(instr), [])
        return candidates[0] if len(candidates) == 1 else None

    def __iter__(self) -> Iterator[LoopInfo]:
        return iter(self.loops)

    def __len__(self) -> int:
        return len(self.loops)


QUERY_INSTRUCTIONS = ("ExpectationInstr", "ProbabilityQueryInstr", "PrintInstr", "PlotInstr",
                      "OptimizationQuery")


def query_instructions(program: Program) -> List[Instr]:
    return [instr for instr in program.instructions if type(instr).__name__ in QUERY_INSTRUCTIONS]


# ---------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------

@dataclass
class LoopDecision:
    """Which strategy to use for a loop, and its parameters (raw, not yet parsed)."""

    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse_spec(spec: str) -> "LoopDecision":
        """Parses the command-line form ``STRATEGY[:name=value,name=value]``.

        Example: ``unroll:iterations=100`` or ``invariant:invariant_file=inv.pgcl``.
        """
        name, _, rest = spec.partition(":")
        params: Dict[str, Any] = {}
        if rest.strip():
            for item in rest.split(","):
                key, sep, value = item.partition("=")
                if not sep:
                    raise ValueError(f"Expected name=value in {item!r} (decision {spec!r})")
                params[key.strip()] = value.strip()
        return LoopDecision(name.strip(), params)

    @staticmethod
    def from_json(data: Union[str, Mapping[str, Any], "LoopDecision"]) -> "LoopDecision":
        if isinstance(data, LoopDecision):
            return data
        if isinstance(data, str):
            return LoopDecision.parse_spec(data)
        if "strategy" not in data:
            raise ValueError(f"A loop decision needs a 'strategy': {data!r}")
        params = dict(data.get("params") or {})
        return LoopDecision(str(data["strategy"]), params)

    def to_json(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "params": dict(self.params)}


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
    """Problems with ``decisions`` (unknown strategies, unknown loops, bad parameters)."""
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


class DecisionRequired(Exception):
    """A non-interactive analysis reached a choice that was not decided up front."""

    def __init__(self, message: str, *, loop: Optional[LoopInfo] = None,
                 strategy: Optional[str] = None, param: Optional[str] = None):
        super().__init__(message)
        self.loop = loop
        self.strategy = strategy
        self.param = param

    def to_json(self) -> Dict[str, Any]:
        return {
            "message": str(self),
            "loop": self.loop.to_json() if self.loop else None,
            "strategy": self.strategy,
            "param": self.param,
        }


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

    def continue_plot(self, probability: float) -> bool:
        return False


class InteractiveDecisions(DecisionProvider):
    """Asks on the terminal, with the prompts the CLI has always used.

    Decisions given up front (e.g. ``prodigy main --loop 0=unroll:iterations=10``) are used
    without asking.
    """

    interactive = True

    def loop_decision(self, session, loop, instruction):
        preset = self.preset(loop)
        if preset is not None:
            return preset
        options: Dict[str, LoopStrategy] = {}
        lines = ["While Instruction has only limited support. Choose an option:\n"]
        extra_key = 7
        for strategy in loop_strategies():
            key = strategy.legacy_key
            if key is None:
                key = str(extra_key)
                extra_key += 1
            options[key] = strategy
            lines.append(f"[{key}]: {strategy.menu_text}\n")
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

    def ask_param(self, session, task, spec):
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

    def loop_decision(self, session, loop, instruction):
        preset = self.preset(loop)
        if preset is not None:
            return preset
        what = loop.describe() if loop is not None else f"while ({instruction.cond})"
        raise DecisionRequired(f"No strategy chosen for {what}", loop=loop)

    def ask_param(self, session, task, spec):
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

    MAX_MESSAGES = 4

    def add_message(self, message: str) -> None:
        if message in self.messages:
            return
        if len(self.messages) >= self.MAX_MESSAGES:
            self.suppressed_messages += 1
            return
        self.messages.append(message)

    def to_json(self) -> Dict[str, Any]:
        from prodigy.analysis.result import exact_and_float  # pylint: disable=import-outside-toplevel
        residual, residual_float = exact_and_float(self.residual)
        return {
            "id": self.loop.id if self.loop else None,
            "line": self.loop.line if self.loop else None,
            "depth": self.loop.depth if self.loop else None,
            "cond": self.cond,
            "strategy": self.strategy,
            "params": {k: v for k, v in self.params.items()},
            "invocations": self.invocations,
            "kind": self.kind,
            "residual_mass": residual,
            "residual_mass_float": residual_float,
            "time": round(self.time, 6),
            "certificates": self.certificates,
            "messages": self.messages + ([f"… {self.suppressed_messages} similar message(s) omitted."]
                                         if self.suppressed_messages else []),
            "failed": self.failed,
            "inside_checks": self.inside_checks,
        }


@dataclass
class _Scope:
    silent: bool
    reason: str = ""
    kind: str = Bound.EXACT
    residual: Any = 0


ProgressCallback = Callable[[Dict[str, Any]], None]


_CURRENT: contextvars.ContextVar[Optional["AnalysisSession"]] = \
    contextvars.ContextVar("prodigy_analysis_session", default=None)


def current_session() -> "AnalysisSession":
    """The session of the running analysis, or a detached interactive one."""
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
                 config=None,
                 provider: Optional[DecisionProvider] = None,
                 decisions: Optional[Mapping[Any, Any]] = None,
                 input_distribution: Any = None,
                 allow_files: bool = True,
                 capture_output: bool = False,
                 output_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[ProgressCallback] = None,
                 record_steps: bool = False,
                 coefficient_limit: int = 30):
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.config import ForwardAnalysisConfig
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

        self.loops = LoopRegistry(program, source)
        self.query_positions = token_positions(source, _QUERY_RE) if source is not None else []
        self.loop_records: Dict[Any, LoopRecord] = {}
        self.queries: List[Any] = []
        self.steps: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.log = io.StringIO()
        self._scopes: List[_Scope] = [_Scope(silent=False)]
        self._query_index = 0
        self.detached_session = False

    # -- construction helpers ------------------------------------------------------------

    @classmethod
    def detached(cls) -> "AnalysisSession":
        """Session for code that runs handlers without setting up a session."""
        session = cls(None, provider=InteractiveDecisions())
        session.detached_session = True
        return session

    @classmethod
    def interactive(cls, program: Program, *, source: Optional[str] = None, config=None,
                    decisions: Optional[Mapping[Any, Any]] = None, **kwargs) -> "AnalysisSession":
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
            yield scope
        finally:
            # The strategy that opened the sub-analysis decides what an inexact sub-analysis
            # means for its own outcome (see e.g. the invariant strategy).
            self._scopes.pop()

    def on_normalize(self, error_prob) -> None:
        """Called right before the distribution is conditioned on the observations."""
        if self.silent:
            return
        scope = self.scope
        if scope.kind != Bound.LOWER:
            return
        from prodigy.analysis.result import sym  # pylint: disable=import-outside-toplevel
        try:
            error_mass = sym(error_prob.get_probability_mass())
        except Exception:  # pylint: disable=broad-except
            return
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

    # -- loops -------------------------------------------------------------------------

    def run_loop(self, task: LoopTask) -> LoopOutcome:
        """Runs ``task.strategy`` and books its outcome (used by the while handler)."""
        key = task.loop.id if task.loop is not None else f"while ({task.instruction.cond})"
        record = self.loop_records.get(key)
        if record is None:
            record = LoopRecord(loop=task.loop, cond=str(task.instruction.cond),
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
            raise
        record.time += time.perf_counter() - start
        record.invocations += 1
        record.kind = Bound.combine(record.kind, outcome.kind)
        if outcome.certificate:
            record.certificates.append(outcome.certificate)
        for message in outcome.messages:
            record.add_message(message)

        scope = self.scope
        scope.kind = Bound.combine(scope.kind, outcome.kind)
        if outcome.kind == Bound.LOWER and not self.silent:
            from prodigy.analysis.result import sym  # pylint: disable=import-outside-toplevel
            residual = sym(outcome.residual)
            record.residual = sym(record.residual) + residual
            scope.residual = sym(scope.residual) + residual
        return outcome

    def note(self, task: LoopTask, message: str) -> None:
        record = task.record
        if record is not None:
            record.add_message(message)

    def warn(self, message: str) -> None:
        if not self.silent and message not in self.warnings:
            self.warnings.append(message)

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

    def record_query(self, query) -> None:
        """Books a :class:`~prodigy.analysis.result.QueryResult` (ignored in sub-analyses)."""
        if self.silent:
            return
        query.index = self._query_index
        query.line, query.column = self.next_query_position()
        self._query_index += 1
        self.queries.append(query)

    # -- steps -------------------------------------------------------------------------

    def record_step(self, instruction: Instr, distribution, error_prob) -> None:
        if self.silent or not self.record_steps:
            return
        from prodigy.analysis.result import exact_and_float  # pylint: disable=import-outside-toplevel
        mass, mass_float = exact_and_float(_safe(distribution.get_probability_mass))
        self.steps.append({
            "index": len(self.steps),
            "instruction": str(instruction),
            "distribution": str(distribution),
            "mass": mass,
            "mass_float": mass_float,
            "error_probability": str(error_prob),
            "bound": self.bound_kind,
        })

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

    def run(self):
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
        fd = target.fileno() if target is not None else 1
    except (AttributeError, OSError, ValueError):
        yield
        return
    try:
        if target is not None:
            target.flush()
        _flush_c_stdio()
        saved = os.dup(fd)
    except OSError:
        yield
        return
    with tempfile.TemporaryFile() as spool:
        os.dup2(spool.fileno(), fd)
        try:
            yield
        finally:
            _flush_c_stdio()
            os.dup2(saved, fd)
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

    def write(self, text: str) -> int:  # type: ignore[override]
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
