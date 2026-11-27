"""
-----------------------------------
Loop strategies (front-end API)
-----------------------------------

The loop strategies for the front ends (CLI, web GUI, scripts): the registry and its vocabulary from
:mod:`prodigy.analysis.loops.api` and :mod:`prodigy.analysis.loops.registry`, with the built-in strategies
registered. The analysis itself imports
:mod:`prodigy.analysis.loops.api`, never this module (``docs/source/architecture.md``, T1).
"""
# registers the built-in strategies
import prodigy.analysis.loops.builtin  # pylint: disable=unused-import
from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import (STRATEGY_GROUPS, get_loop_strategy, loop_strategies,
                                             register_loop_strategy, strategies_for, strategy_groups)

__all__ = ["STRATEGY_GROUPS", "Bound", "Delivers", "LoopOutcome", "LoopStrategy", "LoopTask", "ParamSpec",
           "get_loop_strategy", "loop_strategies", "register_loop_strategy", "strategies_for", "strategy_groups"]
