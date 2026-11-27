"""
Errors and helpers shared by the loop strategies.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import sympy

from prodigy.language import Expr

from prodigy.analysis.exceptions import ForwardAnalysisError, VerificationError
from prodigy.analysis.loops.api import LoopTask
from prodigy.distribution import Distribution


class UnsupportedLoopError(ForwardAnalysisError):
    """The chosen strategy cannot handle this loop (e.g. EVT reasoning with conditioning)."""


class InvariantRefutedError(VerificationError):
    """
    A proposed invariant was not certified. ``details`` holds the evidence; its ``verdict`` is ``refuted``
    if the invariant is provably wrong and ``unknown`` if it could be neither proven nor refuted.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def _mass(dist: Distribution) -> sympy.Expr:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import sym
    return sym(dist.get_probability_mass())


# What may stand for an invariant or generating function: its text, a SymPy expression or a distribution.
InvariantLike = Union[str, sympy.Expr, Distribution]


def _exit_part(dist: Distribution, cond: Expr) -> Distribution:
    """The part of ``dist`` that leaves a loop with guard ``cond``."""
    return dist - dist.filter(cond)


def _require_unconditioned(task: LoopTask) -> None:
    if not task.error_prob.is_zero_dist():
        raise UnsupportedLoopError(
            "Currently EVT reasoning does not support conditioning: the strategy "
            f"'{task.strategy.title}' cannot be used after an observe statement.")
