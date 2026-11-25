"""
-----------------------------------
Loop strategies (front-end API)
-----------------------------------

The loop strategies for the front ends (CLI, web GUI, scripts): the registry and its vocabulary from
:mod:`prodigy.analysis.loops.api`, with the built-in strategies registered. The analysis itself imports
:mod:`prodigy.analysis.loops.api`, never this module (``docs/source/architecture.md``, T1).
"""
from prodigy.analysis.loops.api import (STRATEGY_GROUPS, Bound, LoopOutcome, LoopStrategy, LoopTask, ParamSpec,
                                        get_loop_strategy, loop_strategies, register_loop_strategy, strategy_groups)

__all__ = ["STRATEGY_GROUPS", "Bound", "LoopOutcome", "LoopStrategy", "LoopTask", "ParamSpec", "get_loop_strategy",
           "loop_strategies", "register_loop_strategy", "strategy_groups"]
