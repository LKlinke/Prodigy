"""
The registry of loop strategies.

Strategies register by name at import time (:func:`register_loop_strategy`); the built-in ones are registered by
:mod:`prodigy.analysis.loops.builtin`. The CLI menu, the JSON API and the GUI list them (:func:`loop_strategies`,
:func:`strategy_groups`); :func:`strategies_for` finds the strategies that can give a guarantee.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from prodigy.analysis.loops.api import LoopStrategy

__all__ = ["STRATEGY_GROUPS", "get_loop_strategy", "loop_strategies", "register_loop_strategy", "strategies_for",
           "strategy_groups"]


STRATEGY_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ("automatic", "Automatic", "Prodigy picks: exact if it can prove it quickly, otherwise a lower bound."),
    ("exact", "Exact result", "Prove the loop's output with an invariant: exact, or an upper bound if a side "
                              "condition fails; without a proof, no result."),
    ("bounds", "Bounds", "Sound bounds without proving termination: every query becomes an interval."),
    ("explore", "Explore", "Insight into the loop for guessing an invariant; the result is only a lower bound."),
    ("other", "More", "Strategies registered by extensions."),
)
"""The groups of loop strategies, in menu order: (key, title, one sentence)."""


def strategy_groups() -> List[Tuple[str, str, str, List[LoopStrategy]]]:
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


def strategies_for(goal: str) -> List[LoopStrategy]:
    """The strategies that can give the guarantee ``goal`` (one of :class:`~prodigy.analysis.loops.api.Bound`),
    unconditionally or under their declared condition (:attr:`~prodigy.analysis.loops.api.LoopStrategy.guarantees`),
    in the order of :func:`loop_strategies` (the cheapest first among the built-in ones)."""
    return [strategy for strategy in loop_strategies() if any(given.bound == goal for given in strategy.guarantees)]
