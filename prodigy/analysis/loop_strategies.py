"""
--------------------------
Loop strategies (registry)
--------------------------

A *loop strategy* tells the forward analysis how to deal with one ``while`` loop:
unroll it a fixed number of times, check a user-provided invariant, synthesise an
EVT invariant, and so on. Up to now the strategy and its parameters were asked for
interactively inside :class:`~prodigy.analysis.instructionhandler.while_handler.WhileHandler`.

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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.pgcl import Instr, WhileInstr

    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.session import AnalysisSession, LoopDecision, LoopInfo
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


_PARSERS: Dict[str, Callable[[Any], Any]] = {
    "int": _parse_int,
    "probability": _parse_probability,
    "pgcl": _parse_text,
    "expression": _parse_text,
    "file": _parse_text,
    "choice": _parse_text,
}


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a loop strategy."""

    name: str
    """Key in :attr:`LoopDecision.params`."""

    kind: str
    """One of ``int``, ``probability``, ``pgcl`` (program source), ``expression``
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
        value = _PARSERS[self.kind](raw)
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


@dataclass
class LoopTask:
    """Everything a strategy needs to analyse one occurrence of a loop."""

    instruction: "WhileInstr"
    prog_info: "ProgramInfo"
    distribution: "Distribution"
    error_prob: "Distribution"
    config: "ForwardAnalysisConfig"
    analyzer: Callable[
        [Union["Instr", Sequence["Instr"]], "ProgramInfo", "Distribution", "Distribution",
         "ForwardAnalysisConfig"],
        Tuple["Distribution", "Distribution"]
    ]
    session: "AnalysisSession"
    loop: Optional["LoopInfo"]
    decision: "LoopDecision"
    strategy: "LoopStrategy"
    record: Any = None
    """The session's :class:`~prodigy.analysis.session.LoopRecord` for this loop."""

    def param(self, name: str) -> Any:
        """Value of parameter ``name``: from the decision if present, otherwise from the
        session's decision provider (asks on the terminal in the interactive CLI, uses the
        default or reports a missing decision elsewhere)."""
        spec = self.strategy.param(name)
        raw = self.decision.params.get(name)
        if not _missing(raw):
            return spec.parse(raw)
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

    menu_text: str
    """Text of the entry in the interactive CLI menu."""

    soundness: str
    """Nominal guarantee (see :class:`Bound`); the actual outcome may be weaker, e.g. when
    a side condition cannot be checked."""

    description: str
    """A few sentences for help texts and the GUI."""

    run: Callable[[LoopTask], LoopOutcome]

    params: Tuple[ParamSpec, ...] = ()

    legacy_key: Optional[str] = None
    """Number of the entry in the historic CLI menu (``"1"`` … ``"6"``)."""

    requires_unconditioned: bool = False
    """True if the strategy cannot handle observe statements before or inside the loop."""

    def param(self, name: str) -> ParamSpec:
        for spec in self.params:
            if spec.name == name:
                return spec
        raise KeyError(f"Strategy {self.name} has no parameter {name}")

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "menu_text": self.menu_text,
            "soundness": self.soundness,
            "description": self.description,
            "legacy_key": self.legacy_key,
            "requires_unconditioned": self.requires_unconditioned,
            "params": [p.to_json() for p in self.params if p.kind != "file"],
        }


_REGISTRY: Dict[str, LoopStrategy] = {}


def register_loop_strategy(strategy: LoopStrategy, *, replace: bool = False) -> LoopStrategy:
    """Makes ``strategy`` available to the CLI menu, the JSON API and the web GUI."""
    if strategy.name in _REGISTRY and not replace:
        raise ValueError(f"A loop strategy named {strategy.name!r} is already registered")
    _REGISTRY[strategy.name] = strategy
    return strategy


def _ensure_builtins() -> None:
    # The built-in strategies are registered when the while handler is imported.
    # pylint: disable=import-outside-toplevel,unused-import
    import prodigy.analysis.instructionhandler.while_handler  # noqa: F401


def loop_strategies() -> List[LoopStrategy]:
    """All registered strategies; the six historic ones first, in menu order."""
    _ensure_builtins()

    def key(strategy: LoopStrategy):
        if strategy.legacy_key is not None:
            return (0, int(strategy.legacy_key), "")
        return (1, 0, strategy.name)

    return sorted(_REGISTRY.values(), key=key)


def get_loop_strategy(name: str) -> LoopStrategy:
    _ensure_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(s.name for s in loop_strategies())
        raise KeyError(f"Unknown loop strategy {name!r}. Known strategies: {known}") from None
