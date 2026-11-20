"""
The coefficient-wise order of generating functions, shared by all engines (#441).

``f <= g`` holds iff every coefficient of ``g - f`` is non-negative. The order is decided three-valued:

* ``True`` only if this is proven: the difference is finite and all its coefficients are proven non-negative,
  or the difference is non-negative by construction
  (:func:`~prodigy.distribution.coefficient_signs.nonneg_by_construction`),
  or ``f`` is finite, ``g`` is proven non-negative and ``g`` dominates ``f`` on the finite support of ``f``;
* ``False`` only with a witness: a state whose coefficient in ``g - f`` is proven negative, found among all
  coefficients of a finite difference or among the first :data:`SEARCH_COEFFICIENTS` coefficients of an
  infinite one;
* ``None`` otherwise.

Coefficients that contain parameters count only if their sign holds for every real value of the parameters
(:mod:`~prodigy.distribution.coefficient_signs`); a SymPy relational is never used as a truth value.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, Iterable, Optional, Tuple, Union

import sympy

from prodigy.distribution.coefficient_signs import nonneg_by_construction, series_form, sign
from prodigy.pgcl.pgcl_operations import state_to_equality_expression

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.distribution import Distribution, State

SEARCH_COEFFICIENTS = 20
"""How many coefficients of an infinite difference are searched for a negative one."""

Coefficient = Union[str, sympy.Expr]
"""A coefficient as an engine gives it: a string in the engine's syntax or a SymPy expression."""


class UndecidedComparisonError(RuntimeError):
    """
    Neither ``f <= g`` nor its negation could be proven, e.g. for two infinite generating functions whose
    difference is not non-negative by construction and has no negative coefficient among the first ones.
    """


@dataclass(frozen=True)
class OrderVerdict:
    """The verdict on ``f <= g``: ``holds`` is three-valued; ``witness`` is a state whose coefficient in
    ``g - f`` is proven negative (only if ``holds`` is False)."""
    holds: Optional[bool]
    witness: Optional[State] = None


def coefficient_verdict(terms: Iterable[Tuple[Coefficient, State]], variables: Collection[str],
                        complete: bool) -> OrderVerdict:
    """
    Whether the coefficients `terms` are all non-negative. `complete` says that `terms` are all coefficients
    of the series (a finite series); otherwise only a negative coefficient decides (``False``), as the
    remaining coefficients are unknown. `terms` yields each state once (the iteration of a distribution,
    :meth:`Distribution.__iter__`, #444).
    """
    # lazy: the search stops at the first negative coefficient (expanding is costly)
    coefficients = ((series_form(coefficient, variables)[0], state) for coefficient, state in terms)
    verdict: Optional[bool] = True
    for coefficient, state in coefficients:
        result = sign(sympy.simplify(coefficient))
        if result is False:
            return OrderVerdict(False, state)
        if result is None:
            verdict = None
    return OrderVerdict(verdict if complete else None)


def nonneg_verdict(dist: Distribution, expression: sympy.Expr) -> OrderVerdict:
    """Whether every coefficient of `dist` (whose generating function is `expression`) is non-negative."""
    variables = dist.get_variables()
    if dist.is_finite():
        return coefficient_verdict(dist, variables, complete=True)
    if nonneg_by_construction(expression, variables):
        return OrderVerdict(True)
    try:
        return coefficient_verdict(itertools.islice(dist, SEARCH_COEFFICIENTS), variables, complete=False)
    except (ArithmeticError, NotImplementedError, ValueError, TypeError, sympy.PolynomialError) as error:
        # the series of the difference cannot be expanded on this engine: no witness
        del error
        return OrderVerdict(None)


def dominates_finite(lower: Distribution, upper: Distribution) -> Optional[bool]:
    """
    For a finite `lower`: True if ``lower <= upper`` follows from the coefficients of `upper` at the finitely
    many states of `lower`, given that `upper` has non-negative coefficients. None if a coefficient of `upper`
    cannot be computed or a sign is not proven; never False (the caller refutes on the difference).
    """
    variables = lower.get_variables()
    for probability, state in lower:
        coefficient, _ = series_form(probability, variables)
        try:
            upper_coefficient = upper.get_probability_of(state_to_equality_expression(state.valuations))
        except (ArithmeticError, NotImplementedError, ValueError) as error:
            del error
            return None
        upper_expr, _ = series_form(upper_coefficient, variables)
        if sign(sympy.simplify(upper_expr - coefficient)) is not True:
            return None
    return True
