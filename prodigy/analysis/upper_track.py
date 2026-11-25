"""
------------------------------------
Proven upper bounds per state (#451)
------------------------------------

A loop strategy may prove a coefficient-wise upper bound U on the output of its loop
(:attr:`~prodigy.analysis.loops.api.LoopOutcome.upper`; the EVT sandwich gives ``[not guard]·I`` for a
super-invariant I, Park induction). A bound that has been proven is never discarded: this module carries U
through the rest of the program as a second track next to the lower bound L and its residual mass r, so that
every coefficient, marginal and query gets the upper end ``min(U, L + r, 1)`` instead of only ``min(L + r, 1)``
(:func:`~prodigy.analysis.result.upper_with_source`). Where the track ends, the upper ends fall back to
``L + r``, which is what they were before.

Soundness
---------
Let μ be the exact measure of the terminated runs at some point of the program and ε the exact mass of the
runs that violated an observation. The track holds a pair (U, ε_U) with μ ≤ U coefficient-wise and ε ≤ ε_U
(the mass of ``error_prob``).

* **Start** (:func:`start`): right after the loop, μ = Post ≤ U by the strategy's certificate, and ε = ε_U
  because the loop adds no violation mass. The certificate bounds the loop's output *for the distribution it
  was computed from*, so the track starts only if the loop's incoming distribution is exact (otherwise it is a
  lower bound, and the certificate says nothing about the exact output). U must be a power series with
  non-negative coefficients and finite mass: then the series converges absolutely on the closed unit
  polydisc, its sums there are the values of the closed form (Abel), and the engines may evaluate U at 1
  exactly as they evaluate the program's own distributions (``known_convergent_series(True)``, #393). An
  upper bound of infinite mass starts no track.
* **Loop-free code** (:func:`replay`): assignments, ``if``, probabilistic choice, ``abort``, marginalisation and
  ``observe`` act on pairs (μ, ε) as a positive linear map T (``observe`` only moves mass from μ to ε).
  Positivity and linearity give monotonicity: (μ, ε) ≤ (U, ε_U) implies T(μ, ε) ≤ T(U, ε_U). So the analysis
  applies the same instruction to the track. The replay runs as a silent sub-analysis; if it is not exact
  (e.g. an unproven side condition) or fails, the track ends.
* **Normalisation** (:func:`normalized`; conditioning on the observations, at the end and before each query):
  the exact result is μ / (1 − ε). The lower track divides L by 1 − ε_L ≥ 1 − ε (unchanged, see
  :mod:`prodigy.analysis.result`). The violation mass is at most ε_hi = min(ε_U, ε_L + r): ε_U by the
  track, ε_L + r because the violation mass that L misses is part of the mass r that L does not account for.
  So μ / (1 − ε) ≤ U / (1 − ε_hi) coefficient-wise if ε_hi < 1, and the track continues with
  (U / (1 − ε_hi), 0), like the program's distribution, whose violation mass is reset as well. If ε_hi < 1 is
  not proven, the track ends.
* **Everything else ends the track**: a later loop (its strategy works on the lower bound, and an upper
  certificate computed from L does not bound the exact output), ``dist`` variables (nested inference),
  query blocks, and any instruction whose replay raises.

Each cell's upper end is the least of three proven upper bounds on the exact value: U's coefficient (this
track), L's coefficient plus r (the residual argument), and 1 (a probability). Where U's value is not a
finite number, or its comparison with ``L + r`` is undecided, the cell keeps ``L + r``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import sympy

from prodigy.analysis.loops.api import Bound
from prodigy.util.logger import log_setup

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.analysis.session import AnalysisSession
    from prodigy.distribution import Distribution
    from prodigy.language import Instr

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[
    [Union["Instr", Sequence["Instr"]], "ProgramInfo", "Distribution", "Distribution", "ForwardAnalysisConfig"],
    Tuple["Distribution", "Distribution"]
]
"""The forward semantics (:func:`~prodigy.analysis.analyzer.compute_semantics`)."""


class UpperTrackEnded(Exception):
    """Raised inside a replay for an instruction that the track cannot follow (a loop): the track ends."""


@dataclass(frozen=True)
class UpperTrack:
    """A proven coefficient-wise upper bound on the program's exact state, see the module documentation."""

    distribution: "Distribution"
    """U: lies coefficient-wise above the exact measure of the terminated runs."""

    error_prob: "Distribution"
    """ε_U: its mass is at least the exact mass of the runs that violated an observation."""

    origin: str
    """Where U comes from (e.g. ``loop 0 (evt_sandwich)``), for the result."""


def start(upper: "Distribution", error_prob: "Distribution", kind_before: str, origin: str
          ) -> Optional[UpperTrack]:
    """The track that starts with a loop's proven upper bound ``upper`` (a finite measure with non-negative
    coefficients, see :attr:`~prodigy.analysis.loops.api.LoopOutcome.upper`), or ``None`` if the
    certificate does not bound the program's exact state: the loop's incoming distribution was not exact
    (``kind_before``), or the loop added violation mass."""
    if kind_before != Bound.EXACT:
        return None
    if not error_prob.is_zero_dist():
        # the strategies that prove upper bounds run without observe; with violation mass before the loop,
        # ε_U = ε holds only if the loop adds none, which a strategy does not report
        return None
    return UpperTrack(upper, error_prob, origin)


def replay(track: UpperTrack, instruction: "Instr", prog_info: "ProgramInfo", config: "ForwardAnalysisConfig",
           analyzer: Analyzer, session: "AnalysisSession") -> Optional[UpperTrack]:
    """``track`` after the loop-free top-level ``instruction`` (a positive linear map, see the module
    documentation), or ``None`` if the track ends there: a loop, a query block, or a replay that fails or is
    not exact. The replay is a silent sub-analysis: it records no queries, steps or residual mass."""
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution import known_convergent_series
    from prodigy.language import QueryInstr
    if isinstance(instruction, QueryInstr):
        return None  # a query block normalises inside, which is not linear
    try:
        with session.upper_replay() as scope, known_convergent_series(True):
            # U is a finite measure with non-negative coefficients (see start), as the program's distributions
            distribution, error_prob = analyzer(instruction, prog_info, track.distribution, track.error_prob,
                                                config)
    except UpperTrackEnded:
        return None
    except Exception:  # pylint: disable=broad-except  # the track is only an addition: it ends, the bounds fall back
        logger.debug("The upper track ends at %s", instruction, exc_info=True)
        return None
    if scope.kind != Bound.EXACT:
        return None
    return replace(track, distribution=distribution, error_prob=error_prob)


def normalized(track: UpperTrack, lower_error: object, residual: object) -> Optional[UpperTrack]:
    """``track`` conditioned on the observations, where the lower track's violation mass is ``lower_error``
    (ε_L) and its residual mass before rescaling is ``residual`` (r): divided by 1 − min(ε_U, ε_L + r), with its
    violation mass reset, or ``None`` if min(ε_U, ε_L + r) < 1 is not proven (see the module documentation)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import sym
    try:
        error = sym(track.error_prob.get_probability_mass())
        alternative = sym(lower_error) + sym(residual)
        if bool(alternative < error):
            error = alternative
        if not error.is_number or not bool(error < 1):
            return None
    except (TypeError, ValueError, ArithmeticError, sympy.SympifyError):
        return None  # e.g. parameters: the comparison is undecided
    zero = track.error_prob * "0"
    if error == 0:
        return replace(track, error_prob=zero)
    return replace(track, distribution=track.distribution / str(1 - error), error_prob=zero)
