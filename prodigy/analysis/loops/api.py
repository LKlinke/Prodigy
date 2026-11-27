"""
--------------------------
Loop strategies (registry)
--------------------------

A *loop strategy* tells the forward analysis how to deal with one ``while`` loop:
unroll it a fixed number of times, check a user-provided invariant, synthesise an
EVT invariant, and so on. Up to now the strategy and its parameters were asked for
interactively inside the handler of while loops (:mod:`prodigy.analysis.instructionhandler.while_handler`).

This module makes strategies explicit data:

* :class:`ParamSpec` describes one parameter (type, label, legacy CLI prompt, default).
* :class:`LoopStrategy` describes one strategy (stable name, menu text, parameters,
  nominal soundness) and points to the function that runs it.
* :class:`LoopTask` is what a strategy receives: the loop, the incoming distribution and
  a way to read its parameters (from the session's decisions, or interactively).
* :class:`LoopOutcome` is what a strategy returns: the outgoing distribution plus what
  is known about it (exact, lower bound with residual mass, upper bound, unknown) and a
  certificate that a front end can display or re-check.

Strategies are registered by name, so additional strategies (for example from the
lower-bounds work) appear in the CLI menu, the JSON API and the web GUI without touching
those front ends: call :func:`register_loop_strategy` at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import attr

from prodigy.language import Expr, Instr, Program, WhileInstr
from prodigy.language.outline import LoopInfo, is_query, outline

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.context import AnalysisContext, Analyzer, Frame, ProgramState
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.distribution import Distribution


class Bound:
    """What is known about a distribution computed by (part of) an analysis.

    The values are compared to the *exact* semantics of the program:

    * ``EXACT``: equal to the exact semantics (certified).
    * ``LOWER``: coefficient-wise below the exact semantics; the mass that is not
      accounted for is at most the recorded *residual mass*. No termination assumption.
    * ``UPPER``: coefficient-wise above the exact semantics (Park's lemma).
    * ``UNKNOWN``: an approximation without a guaranteed direction, or a result whose
      side conditions could not be checked.
    """
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"
    UNKNOWN = "unknown"

    ALL = (EXACT, LOWER, UPPER, UNKNOWN)

    @staticmethod
    def combine(first: str, second: str) -> str:
        """Kind of a composition of two analysis steps (sequential or parallel)."""
        if first == Bound.EXACT:
            return second
        if second == Bound.EXACT:
            return first
        if first == second:
            return first
        return Bound.UNKNOWN


def _missing(raw: Any) -> bool:
    return raw is None or (isinstance(raw, str) and raw.strip() == "")


def _parse_int(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("must be a whole number")
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ValueError(f"must be a whole number, not {str(raw).strip()!r}") from None


def _parse_probability(raw: Any) -> str:
    """Accepts 0.99, "0.99", "99/100"; returns a normalised string."""
    text = str(raw).strip()
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"must be a probability such as 0.99 or 99/100, not {text!r}") from None
    if not 0 <= value <= 1:
        raise ValueError("must be a probability between 0 and 1")
    return text


def _parse_text(raw: Any) -> str:
    text = str(raw)
    if text.strip() == "":
        raise ValueError("expected a non-empty value")
    return text


def _parse_expression(raw: Any) -> str:
    """A generating function given from outside (CLI, decisions, GUI), checked before any engine
    sees it: SymPy's parser evaluates Python code. Returns it in the engines' spelling (powers as
    ``^``, which GiNaC needs); raises ``ValueError`` (a
    :class:`~prodigy.analysis.ai_candidates.CandidateError`) for anything else
    (:func:`~prodigy.language.expression_allowlist.admissible_expression`)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.language.expression_allowlist import MATH_FUNCTIONS, admissible_expression
    return admissible_expression(_parse_text(raw).strip(), functions=MATH_FUNCTIONS)


_PARSERS: Dict[str, Callable[[Any], Any]] = {
    "int": _parse_int,
    "probability": _parse_probability,
    "program": _parse_text,
    "expression": _parse_expression,
    "file": _parse_text,
    "choice": _parse_text,
}


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a loop strategy."""

    name: str
    """Key in :attr:`LoopDecision.params`."""

    kind: str
    """One of ``int``, ``probability``, ``program`` (program source), ``expression``
    (generating-function expression), ``file`` (path, CLI only), ``choice``."""

    label: str
    """Short label for forms."""

    prompt: str = ""
    """Prompt used by the interactive CLI (kept identical to the historic prompts)."""

    default: Any = None
    """Default used by non-interactive front ends when the parameter is missing."""

    help: str = ""
    """One-sentence explanation for forms and ``--help`` texts."""

    minimum: Optional[int] = None
    """Lower bound for ``int`` parameters."""

    choices: Tuple[str, ...] = ()
    """Allowed values for ``choice`` parameters."""

    alias_of: Optional[str] = None
    """If set, this parameter is an alternative way to provide ``alias_of``
    (e.g. an invariant *file* instead of the invariant *source*)."""

    optional: bool = False
    """Optional parameters are never asked for; if missing, :attr:`default` (or ``None``)
    is used."""

    def parse(self, raw: Any) -> Any:
        """The value of ``raw``; raises ``ValueError`` if it is not admissible. ``expression``
        parameters must be generating functions written as ``.ppy`` expressions."""
        value: Any = _parse_expression(raw) if self.kind == "expression" else _PARSERS[self.kind](raw)
        if self.kind == "int" and self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum}")
        if self.kind == "choice" and self.choices and value not in self.choices:
            raise ValueError(f"{self.label} must be one of {', '.join(self.choices)}")
        return value

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "default": self.default,
            "help": self.help,
            "minimum": self.minimum,
            "choices": list(self.choices),
            "alias_of": self.alias_of,
            "optional": self.optional,
        }


QueryKey = Tuple[str, str]
"""The key of a query directly after a loop in :attr:`LoopOutcome.query_bounds` and ``query_values``."""


def query_key(kind: str, expression: "Expr") -> QueryKey:
    """
    The key of the query ``kind`` (``"?Pr"`` or ``"?Ex"``) of ``expression``: its kind and the structure of the
    expression tree (the ``repr`` that ``attrs`` derives from the fields), so that the strategy and the query
    handler find the same query whatever ``str()`` or a printer writes.
    """
    return kind, repr(expression)


@dataclass
class LoopOutcome:
    """Result of running a loop strategy once."""

    distribution: "Distribution"
    error_prob: "Distribution"
    kind: str = Bound.UNKNOWN
    residual: Any = 0
    """For ``LOWER``: upper bound on the mass that is not accounted for, in the units of
    the incoming distribution (i.e. before any later normalisation)."""
    certificate: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    query_bounds: Dict["QueryKey", Tuple[Any, Any]] = field(default_factory=dict)
    """Bounds that the strategy computed itself for the queries directly after the loop, as
    ``query_key(kind, expression) -> (lower, upper)`` with kind ``"?Pr"`` or ``"?Ex"`` and ``None`` for an
    unknown side. They refer to the incoming distribution of the loop and tighten the intervals of
    exactly these queries (e.g. the upper bounds of an EVT super-invariant)."""
    query_values: Dict["QueryKey", Any] = field(default_factory=dict)
    """Exact values of the queries directly after the loop on ``distribution`` itself, keyed like
    ``query_bounds``, if the strategy knows them anyway (e.g. from its certificate). They must be
    exactly what evaluating the query on ``distribution`` gives; the query handler then skips that
    evaluation, which can be expensive for large closed forms."""
    upper: Optional["Distribution"] = None
    """A proven coefficient-wise upper bound on the loop's exact output for the incoming ``distribution``
    (e.g. ``[not guard]·I`` for an EVT super-invariant I, Park induction), or ``None``. A strategy sets it only
    for a power series with non-negative coefficients and finite mass, and only if the loop adds no violation
    mass (``error_prob`` is the incoming one). The session carries it through the rest of the program as a
    second track (:mod:`prodigy.analysis.upper_track`) if the incoming distribution is exact; it tightens the
    upper ends of the coefficients and queries that ``residual`` alone gives."""


@dataclass
class LoopTask:
    """Everything a strategy needs to analyse one occurrence of a loop: the
    :class:`~prodigy.analysis.context.Frame` it runs in and the
    :class:`~prodigy.analysis.context.ProgramState` before the loop. ``prog_info``, ``distribution``,
    ``error_prob``, ``config``, ``analyzer`` and ``session`` read them, so that registered strategies
    keep their interface."""

    instruction: "WhileInstr"
    frame: "Frame"
    state: "ProgramState"
    loop: Optional["LoopInfo"]
    decision: "LoopDecision"
    strategy: "LoopStrategy"
    record: Any = None
    """The session's :class:`~prodigy.analysis.session.LoopRecord` for this loop."""

    @property
    def prog_info(self) -> "ProgramInfo":
        return self.frame.program

    @property
    def distribution(self) -> "Distribution":
        return self.state.dist

    @property
    def error_prob(self) -> "Distribution":
        return self.state.error

    @property
    def config(self) -> "ForwardAnalysisConfig":
        return self.frame.config

    @property
    def analyzer(self) -> "Analyzer":
        """Runs instructions of the loop's program (the body, an invariant program) on a given state, in
        the loop's frame."""
        return self.frame.as_analyzer()

    @property
    def session(self) -> "AnalysisContext":
        return self.frame.context

    def param(self, name: str) -> Any:
        """Value of parameter ``name``: from the decision if present, otherwise from the
        session's decision provider (asks on the terminal in the interactive CLI, uses the
        default or reports a missing decision elsewhere)."""
        spec = self.strategy.param(name)
        raw = self.decision.params.get(name)
        if not _missing(raw):
            try:
                return spec.parse(raw)
            except ValueError as exc:
                where = self.loop.describe() if self.loop is not None else "a loop"
                raise ValueError(f"{where}: parameter {name!r} of strategy {self.strategy.name} "
                                 f"({spec.label}): {exc}") from exc
        # alternative spelling, e.g. invariant_file instead of invariant
        for alias in self.strategy.params:
            if alias.alias_of == name and not _missing(self.decision.params.get(alias.name)):
                return self.session.resolve_alias(self, alias, self.decision.params[alias.name])
        if spec.optional:
            return spec.parse(spec.default) if spec.default is not None else None
        return self.session.ask_param(self, spec)

    def progress(self, done: int, total: Optional[int], message: str = "") -> None:
        self.session.progress(self, done, total, message)

    def note(self, message: str) -> None:
        self.session.note(self, message)

    def iterate(self, iteration: int, terminated: Any, inside: Any = None, residual: Any = None,
                kind: str = Bound.LOWER) -> None:
        """Reports an iterate for the step trace (:mod:`prodigy.analysis.trace`): ``terminated`` (a
        distribution, or a function computing it, called only if the iterate is kept) has left the
        loop after ``iteration`` iterations and is a lower bound on the loop's output of kind
        ``kind``; the mass not accounted for is the mass of ``inside`` (still in the loop) or
        ``residual``. Costs nothing unless steps are recorded."""
        self.session.loop_iterate(self, iteration, terminated, inside, residual, kind)

    @property
    def interactive(self) -> bool:
        return self.session.provider.interactive


@dataclass(frozen=True)
class LoopStrategy:
    """A registered way of analysing a while loop."""

    name: str
    """Stable identifier used in JSON decisions and on the command line."""

    title: str
    """Short title for menus and forms."""

    soundness: str
    """Nominal guarantee (see :class:`Bound`); the actual outcome may be weaker, e.g. when
    a side condition cannot be checked."""

    description: str
    """A few sentences for help texts and the GUI."""

    run: Callable[[LoopTask], LoopOutcome]

    params: Tuple[ParamSpec, ...] = ()

    group: str = "other"
    """What the strategy is for, one of the keys of :data:`STRATEGY_GROUPS` (``automatic``, ``exact``,
    ``bounds``, ``explore``, ``other``). Menus and forms list the strategies by group."""

    summary: str = ""
    """One plain sentence: what you get, what you must provide, roughly what it costs. The CLI menu,
    ``prodigy loop_strategies`` and the GUI show the same sentence."""

    outcome: str = ""
    """What the result is, in a few words (e.g. ``lower bound``, ``exact or upper bound``,
    ``exploration``), for a chip next to the strategy."""

    menu_text: str = ""
    """Text of the entry in the interactive CLI menu; empty for ``title: summary``."""

    legacy_key: Optional[str] = None
    """Number of the entry in the historic CLI menu (``"1"`` … ``"6"``)."""

    requires_unconditioned: bool = False
    """True if the strategy cannot handle observe statements before or inside the loop."""

    def param(self, name: str) -> ParamSpec:
        for spec in self.params:
            if spec.name == name:
                return spec
        raise KeyError(f"Strategy {self.name} has no parameter {name}")

    @property
    def menu_line(self) -> str:
        """The entry of the interactive CLI menu."""
        if self.menu_text:
            return self.menu_text
        return f"{self.title}: {self.summary}" if self.summary else self.title

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "group": self.group,
            "summary": self.summary,
            "outcome": self.outcome,
            "menu_text": self.menu_line,
            "soundness": self.soundness,
            "description": self.description,
            "legacy_key": self.legacy_key,
            "requires_unconditioned": self.requires_unconditioned,
            "params": [p.to_json() for p in self.params if p.kind != "file"],
        }


STRATEGY_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ("automatic", "Automatic", "Prodigy picks: exact if it can prove it quickly, otherwise a lower bound."),
    ("exact", "Exact result", "Prove the loop's output with an invariant: exact, or an upper bound if a side "
                              "condition fails; without a proof, no result."),
    ("bounds", "Bounds", "Sound bounds without proving termination: every query becomes an interval."),
    ("explore", "Explore", "Insight into the loop for guessing an invariant; the result is only a lower bound."),
    ("other", "More", "Strategies registered by extensions."),
)
"""The groups of loop strategies, in menu order: (key, title, one sentence)."""


def strategy_groups() -> List[Tuple[str, str, str, List["LoopStrategy"]]]:
    """The non-empty groups with their strategies, in menu order (within a group in the order of
    :func:`loop_strategies`); strategies with an unknown group go to ``other``."""
    known = {key for key, _, _ in STRATEGY_GROUPS}
    strategies = loop_strategies()
    groups = []
    for key, title, text in STRATEGY_GROUPS:
        members = [s for s in strategies if (s.group if s.group in known else "other") == key]
        if members:
            groups.append((key, title, text, members))
    return groups


_REGISTRY: Dict[str, LoopStrategy] = {}


def register_loop_strategy(strategy: LoopStrategy, *, replace: bool = False) -> LoopStrategy:
    """Makes ``strategy`` available to the CLI menu, the JSON API and the web GUI."""
    if strategy.name in _REGISTRY and not replace:
        raise ValueError(f"A loop strategy named {strategy.name!r} is already registered")
    _REGISTRY[strategy.name] = strategy
    return strategy


def loop_strategies() -> List[LoopStrategy]:
    """All registered strategies; the six historic ones first, in menu order. The built-in ones are registered
    by :mod:`prodigy.analysis.loops.builtin` (imported by the session and the front-end module
    :mod:`prodigy.analysis.loop_strategies`)."""

    def key(strategy: LoopStrategy) -> Tuple[int, int, str]:
        if strategy.legacy_key is not None:
            return (0, int(strategy.legacy_key), "")
        return (1, 0, strategy.name)

    return sorted(_REGISTRY.values(), key=key)


def get_loop_strategy(name: str) -> LoopStrategy:
    """The registered strategy ``name`` (raises :class:`KeyError` naming the known ones)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(s.name for s in loop_strategies())
        raise KeyError(f"Unknown loop strategy {name!r}. Known strategies: {known}") from None


@dataclass
class LoopDecision:
    """Which strategy to use for a loop, and its parameters (raw, not yet parsed)."""

    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse_spec(spec: str) -> "LoopDecision":
        """Parses the command-line form ``STRATEGY[:name=value,name=value]``.

        Example: ``unroll:iterations=100`` or ``invariant:invariant_file=inv.ppy``.
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


def _loop_id(program: Program, instruction: WhileInstr) -> int:
    """The id of the loop ``instruction`` of ``program`` (:func:`~prodigy.language.outline.outline`), 0 if it is
    none."""
    found = outline(program)
    return next((info.id for info, instr in zip(found.loops, found.loop_instructions, strict=True)
                 if instr is instruction), 0)


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
            return IsolatedLoop(source, _loop_id(program, instruction), declarations, True)
        loop_program = attr.evolve(program, instructions=[instruction])
        return IsolatedLoop(text([instruction]), _loop_id(loop_program, instruction), declarations, False)
    except PpyPrintError:
        return None
