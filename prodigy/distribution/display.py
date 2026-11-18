"""
The shown form of a result: a bounded search for a shorter form of the same generating function (#404).

The analysis keeps closed forms factored and leaves polynomials as the operations produce them, whatever
the settings (``GeneratingFunction._arithmetic``); that is
fast to compute with, but often not the shortest form to read. Only the form that is shown is simplified,
and only with bounded effort:

- the candidates are exact rewrites of the same rational function (``together``, ``cancel``, ``factor``),
  tried in the order of their cost; ``simplify`` is never called;
- the candidates that need a polynomial GCD are only tried below a size and a degree threshold (sparse
  polynomials of high degree such as ``x**(2**k)`` make GCDs and factorisations explode, #224), and
  ``factor`` only on smaller expressions still;
- every candidate is checked to be equal to the computed form (``cancel(candidate - computed) == 0``)
  before it is shown;
- the whole search has a time limit (it interrupts a running GCD where :func:`~prodigy.util.time_limit.time_limit`
  can); on the time limit the best form found so far is kept;
- the shortest form by ``count_ops`` wins; on a tie the computed form stays.

The decision when and how much to simplify is shared by all engines. The rewrites run in SymPy on the
engine's SymPy form (``_sympy_expression``); the chosen form goes back to the engine
(``_from_sympy_expression``), so that it is printed in the engine's notation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import sympy

from prodigy.util.logger import log_setup
from prodigy.util.time_limit import TimeLimitReached, time_limit

if TYPE_CHECKING:
    from prodigy.distribution.distribution import Distribution

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

DISPLAY_TIME_LIMIT = 2.0
"""The time limit of the whole search, in seconds."""

SMALL_OPS = 2
"""A computed form with at most this many operations is shown as it is."""

GCD_MAX_OPS = 600
"""Candidates that need a polynomial GCD (``cancel``, ``factor``, and the equality check of every candidate)
are only tried on computed forms with at most this many operations ..."""

GCD_MAX_DEGREE = 64
"""... and no exponent above this one."""

FACTOR_MAX_OPS = 200
"""``factor`` is only tried on computed forms with at most this many operations."""

COMPUTED = "computed"
"""The name of the computed form among the candidates."""

Rewrite = Callable[[sympy.Expr], sympy.Expr]
JsonValue = Union[str, bool, int, float, None]

# ordered by their cost: together needs no GCD, cancel one, factor a factorisation
_REWRITES: Tuple[Tuple[str, Rewrite, int], ...] = (
    ("together", sympy.together, GCD_MAX_OPS),
    ("cancel", sympy.cancel, GCD_MAX_OPS),
    ("factor", sympy.factor, FACTOR_MAX_OPS),
)


@dataclass(frozen=True)
class DisplayForm:
    """The form of a distribution that is shown, and why."""

    distribution: "Distribution"
    """The shown distribution; equal to the computed one as a generating function."""

    form: str
    """``"computed"`` or the name of the rewrite that gave the shown form (``together``, ``cancel``, ``factor``)."""

    note: str
    """Why this form is shown, in a few words."""

    computed_ops: Optional[int] = None
    """The size of the computed form (``count_ops``), if known."""

    shown_ops: Optional[int] = None
    """The size of the shown form (``count_ops``), if known."""

    seconds: float = 0.0
    """The time the search took."""

    @property
    def simplified(self) -> bool:
        return self.form != COMPUTED

    def to_json(self) -> Dict[str, JsonValue]:
        return {"form": self.form, "simplified": self.simplified, "note": self.note,
                "computed_ops": self.computed_ops, "shown_ops": self.shown_ops, "seconds": round(self.seconds, 6)}


def _max_exponent(expr: sympy.Expr) -> int:
    """The largest absolute integer exponent in ``expr`` (0 without powers)."""
    exponents = [abs(int(power.exp)) for power in expr.atoms(sympy.Pow) if power.exp.is_Integer]
    return max(exponents, default=0)


def _equal(candidate: sympy.Expr, computed: sympy.Expr) -> bool:
    """Whether ``candidate`` and ``computed`` are the same rational function (in their generators)."""
    return bool(sympy.cancel(candidate - computed) == 0)


def display_form(dist: "Distribution", *, seconds: float = DISPLAY_TIME_LIMIT) -> DisplayForm:
    """The shortest form of ``dist`` the bounded search finds (see the module); ``dist`` itself if none is
    shorter, if it is too large for the search, or if the time is up before a shorter form is found."""
    start = time.perf_counter()

    def kept(note: str, ops: Optional[int] = None) -> DisplayForm:
        return DisplayForm(dist, COMPUTED, note, ops, ops, time.perf_counter() - start)

    try:
        computed = dist._sympy_expression()  # pylint: disable=protected-access
    except NotImplementedError:
        return kept("Kept as computed: this engine has no SymPy form.")
    if not isinstance(computed, sympy.Expr):
        return kept("Kept as computed.")
    computed_ops = int(sympy.count_ops(computed))
    if computed_ops <= SMALL_OPS:
        return kept("Kept as computed: already short.", computed_ops)
    if computed_ops > GCD_MAX_OPS:
        return kept("Kept as computed: too large to simplify within the bounds.", computed_ops)
    if _max_exponent(computed) > GCD_MAX_DEGREE:
        return kept("Kept as computed: the degree is too high to simplify within the bounds.", computed_ops)

    best_name, best, best_ops = COMPUTED, computed, computed_ops
    reason = "no shorter form found"
    try:
        with time_limit(seconds) as expired:
            for name, rewrite, max_ops in _REWRITES:
                if expired():
                    raise TimeLimitReached()
                if computed_ops > max_ops:
                    continue
                try:
                    candidate = rewrite(computed)
                    ops = int(sympy.count_ops(candidate))
                    if ops < best_ops and _equal(candidate, computed):
                        best_name, best, best_ops = name, candidate, ops
                except Exception as exc:  # pylint: disable=broad-except
                    # a failing rewrite (e.g. a polynomial error of SymPy) only gives no candidate
                    logger.info("The rewrite %s failed: %s", name, exc)
    except TimeLimitReached:
        reason = f"time limit of {seconds:g} s reached"
    if best_name == COMPUTED:
        return kept(f"Kept as computed: {reason}.", computed_ops)
    try:
        shown = dist._from_sympy_expression(best)  # pylint: disable=protected-access
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("The simplified form does not convert back to the engine: %s", exc)
        return kept("Kept as computed: the simplified form does not convert back to this engine.", computed_ops)
    return DisplayForm(shown, best_name, f"Simplified ({best_name}).", computed_ops, best_ops,
                       time.perf_counter() - start)
