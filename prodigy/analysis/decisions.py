"""
Decisions: which strategy analyses a loop, and the answers to the questions a strategy asks.

A decision (:class:`~prodigy.analysis.loops.api.LoopDecision`) is given per loop id, or for every loop
(:data:`DEFAULT_KEY`). The session asks its :class:`DecisionProvider`:

* :class:`InteractiveDecisions` asks on the terminal with the historic prompts (the CLI).
* :class:`FixedDecisions` answers from decisions given up front (the GUI, ``prodigy analyze``); a missing
  decision raises :class:`~prodigy.analysis.loops.api.DecisionRequired` instead of blocking.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from prodigy.language import WhileInstr
from prodigy.language.outline import LoopInfo

from prodigy.analysis.loops.api import (DecisionRequired, LoopDecision, LoopStrategy, LoopTask, ParamSpec,
                                        get_loop_strategy, loop_strategies)
from prodigy.util.logger import log_setup

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.context import AnalysisContext
    from prodigy.analysis.loop_records import LoopRegistry

__all__ = ["DEFAULT_KEY", "DecisionProvider", "FixedDecisions", "InteractiveDecisions", "parse_decisions",
           "validate_decisions"]

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


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

    def loop_decision(self, session: AnalysisContext, loop: Optional[LoopInfo],
                      instruction: WhileInstr) -> LoopDecision:
        raise NotImplementedError

    def ask_param(self, session: AnalysisContext, task: LoopTask, spec: ParamSpec) -> Any:
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

    def loop_decision(self, session: AnalysisContext, loop: Optional[LoopInfo],
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

    def ask_param(self, session: AnalysisContext, task: LoopTask, spec: ParamSpec) -> Any:
        if not self.remember:
            return self._ask_param(session, task, spec)
        key = (id(task.instruction), spec.name)
        if key not in self._params:
            self._params[key] = (task.instruction, self._ask_param(session, task, spec))
        return self._params[key][1]

    def _ask_param(self, session: AnalysisContext, task: LoopTask, spec: ParamSpec) -> Any:
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


class FixedDecisions(DecisionProvider):
    """Answers only from decisions given up front; never blocks."""

    interactive = False

    def loop_decision(self, session: AnalysisContext, loop: Optional[LoopInfo],
                      instruction: WhileInstr) -> LoopDecision:
        preset = self.preset(loop)
        if preset is not None:
            return preset
        what = loop.describe() if loop is not None else f"while ({session.spell(instruction.cond)})"
        raise DecisionRequired(f"No strategy chosen for {what}", loop=loop)

    def ask_param(self, session: AnalysisContext, task: LoopTask, spec: ParamSpec) -> Any:
        if spec.default is not None:
            return spec.parse(spec.default)
        where = task.loop.describe() if task.loop is not None else "a loop"
        raise DecisionRequired(
            f"Strategy '{task.strategy.title}' for {where} needs a value for '{spec.label}'",
            loop=task.loop, strategy=task.strategy.name, param=spec.name)
