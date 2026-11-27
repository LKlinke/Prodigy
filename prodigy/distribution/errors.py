"""
The errors of the :class:`~prodigy.distribution.distribution.Distribution` interface (#511), and the declaration
of convergent series (:func:`known_convergent_series`) that decides when :class:`NonFiniteMarginalError` is raised.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional


class NonFiniteMarginalError(ArithmeticError):
    """
    A marginal that a computation needs as a distribution over finitely or infinitely many states is
    not a finite measure, e.g. `oo`, `zoo` (complex infinity), `nan` or a division by zero. This happens
    on measures with infinite mass (candidate measures, not program distributions): summing out the
    other variables diverges, so the marginal does not describe which values the variables take.
    """


# Whether every generating function in the current context is a measure whose marginals are the sums of
# their series (see `known_convergent_series`); None outside analyses (the Python API, #398)
_KNOWN_CONVERGENT: ContextVar[Optional[bool]] = ContextVar("prodigy_known_convergent_series", default=None)


@contextlib.contextmanager
def known_convergent_series(known: bool) -> Iterator[None]:
    """
    Declares whether the generating functions computed in this context are known to be measures whose
    marginals are the sums of their series, so that :meth:`Distribution.checked_marginal` may skip the
    proof (#393). This holds for the analysis of a program from a validated input (#300): its
    distributions are non-negative measures with finite mass for every admissible value of the
    parameters, also those that a proof would have to bound (e.g. `geometric(p)` with a parameter p),
    and the outcomes of loop strategies keep this or are not guaranteed at all. It does not hold for
    candidate measures (invariants, second-order generating functions, signed differences in checks):
    the sub-analyses that compute with them set it to False.

    Outside such a declaration (the Python API, #398), every evaluation at 1 of a public entry point
    (:meth:`Distribution.marginal`, :meth:`Distribution.get_probability_mass`, and through them `Pr` and
    `Ex`) is proven to be the sum of the series, or raises `NonFiniteMarginalError`. With False, only
    :meth:`Distribution.checked_marginal` proves it; the candidate checks guard their other evaluations
    themselves (#276, finite mass of sub-invariants).
    """
    token = _KNOWN_CONVERGENT.set(known)
    try:
        yield
    finally:
        _KNOWN_CONVERGENT.reset(token)


class NotAnIntegerError(ValueError):
    """A division whose quotient is not a natural number in a state of positive probability."""


class ZeroDivisorError(ValueError):
    """
    A modulus or divisor (``%``, ``/``) is 0 in a reachable state: the update is undefined (#273). All
    engines raise this error, for a literal 0 and for a variable that takes the value 0; the update of
    an expression adds the expression to the message (``expression``).
    """

    def __init__(self, message: str, expression: Optional[str] = None):
        super().__init__(message if expression is None else f"{message} (in {expression})")
        self.reason = message
        self.expression = expression



def declared_convergence() -> Optional[bool]:
    """What the innermost :func:`known_convergent_series` declares, or None outside of one (the Python API)."""
    return _KNOWN_CONVERGENT.get()


class UndecidedComparisonError(RuntimeError):
    """
    Neither ``f <= g`` nor its negation could be proven, e.g. for two infinite generating functions whose
    difference is not non-negative by construction and has no negative coefficient among the first ones.
    """
