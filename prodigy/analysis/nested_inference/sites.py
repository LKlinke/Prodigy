"""
Sites of ``dist`` values and their encoding in the joint distribution.

A :class:`DistSite` is a function with its arguments; the :class:`DistRegistry` of the run numbers the sites (ids from
1) and keeps their closed forms. A ``dist`` variable is the id of its site, its state-dependent arguments are the slot
variables (:func:`slot_variable`). This module also splits a distribution by the values of ``dist`` variables.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from prodigy.analysis.exceptions import UndefinedDistributionError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.context import ProgramState
from prodigy.distribution import Distribution, MarginalType
from prodigy.language import Binop, BinopExpr, Expr, Function, NatLitExpr, VarExpr
from prodigy.language.ast.types import DistributionType
from prodigy.language.ppy_printer import spell
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

VALUE_VARIABLE = "dist_value__"
"""The variable of the generating function of a closed site."""


@dataclass(frozen=True)
class Slot:
    """An argument of a site whose value is kept in the slot variable ``index`` of the ``dist``
    variable (see :func:`slot_variable`)."""
    index: int


@dataclass(frozen=True)
class DistSite:
    """The value ``infer(function, args)``: ``args`` gives every variable of the function its
    value (a natural number, a closed site for a ``dist`` parameter, or a :class:`Slot`)."""
    function: str
    args: Tuple[Tuple[str, Union[int, "DistSite", Slot]], ...]
    given: Tuple[str, ...] = field(default=(), compare=False)
    """The parameters the call names (for :meth:`describe`; the other variables are 0)."""

    @property
    def closed(self) -> bool:
        return not any(isinstance(value, Slot) for _, value in self.args)

    @property
    def slots(self) -> List[int]:
        return sorted(value.index for _, value in self.args if isinstance(value, Slot))

    def with_slots(self, values: Sequence[int]) -> "DistSite":
        """The closed site with the slots ``0, 1, …`` replaced by ``values``."""
        args = tuple((var, values[value.index] if isinstance(value, Slot) else value)
                     for var, value in self.args)
        return DistSite(self.function, args, self.given)

    def describe(self) -> str:
        """``infer(f, a = 2, d = infer(g))`` (``.ppy``); a slot is shown as ``…``."""
        values = dict(self.args)
        arguments = []
        for var in self.given:
            value = values.get(var, 0)
            if isinstance(value, DistSite):
                text = value.describe()
            elif isinstance(value, Slot):
                text = "…"
            else:
                text = str(value)
            arguments.append(f"{var} = {text}")
        return f"infer({', '.join([self.function] + arguments)})"


@dataclass
class ClosedDist:
    """The normalised distribution of a closed site."""
    site: DistSite
    distribution: Distribution
    """The generating function, in :data:`VALUE_VARIABLE`."""
    mass: Any
    """The normalising constant Z (a sympy number)."""
    values: Dict[str, Any] = field(default_factory=dict)
    """Cache of ``Pr``/``Ex`` values (sympy), by the text of the expression."""


class DistRegistry:
    """The sites one analysis has met, with their ids and closed forms."""

    def __init__(self) -> None:
        self._sites: List[DistSite] = []
        self._ids: Dict[DistSite, int] = {}
        self._closed: Dict[DistSite, ClosedDist] = {}
        self.functions: Dict[str, Function] = {}

    def site_id(self, site: DistSite) -> int:
        found = self._ids.get(site)
        if found is None:
            self._sites.append(site)
            found = self._ids[site] = len(self._sites)
        return found

    def site(self, site_id: int) -> Optional[DistSite]:
        """The site with ``site_id``, ``None`` for 0 (no value)."""
        if site_id == 0:
            return None
        if not 0 < site_id <= len(self._sites):
            raise UndefinedDistributionError(f"Unknown value of a dist variable (site {site_id}).")
        return self._sites[site_id - 1]

    def closed(self, site: DistSite) -> ClosedDist:
        return self._closed[site]

    def has_closed(self, site: DistSite) -> bool:
        return site in self._closed

    def store(self, closed: ClosedDist) -> None:
        self._closed[closed.site] = closed


# ---------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------

def slot_variable(var: str, index: int) -> str:
    """The name of slot ``index`` of the ``dist`` variable ``var``."""
    return f"{var}_arg{index}__"


def dist_variables(prog_info: ProgramInfo) -> Set[str]:
    """The ``dist`` variables of the program (or function) that is analysed."""
    return {var for var, typ in prog_info.program.variables.items()
            if isinstance(typ, DistributionType)}


def _slots_present(distribution: Distribution, var: str) -> List[str]:
    present = distribution.get_variables()
    result = []
    index = 0
    while slot_variable(var, index) in present:
        result.append(slot_variable(var, index))
        index += 1
    return result


def encoding_variables(distribution: Distribution, var: str) -> List[str]:
    """The variables of ``distribution`` that encode the ``dist`` variable ``var``."""
    found = [var] if var in distribution.get_variables() else []
    return found + _slots_present(distribution, var)


def _add_variables(distribution: Distribution, *variables: str) -> Distribution:
    missing = [var for var in variables if var not in distribution.get_variables()]
    if not missing:
        return distribution
    return distribution.set_variables(*distribution.get_variables(), *missing)


def _assign(distribution: Distribution, var: str, value: Union[int, str]) -> Distribution:
    distribution = _add_variables(distribution, var)
    rhs: Expr = NatLitExpr(value) if isinstance(value, int) else VarExpr(value)
    return distribution.update(BinopExpr(Binop.EQ, VarExpr(var), rhs))


def without(distribution: Distribution, variables: Sequence[str]) -> Distribution:
    """``distribution`` without ``variables`` (those it has)."""
    present = [var for var in variables if var in distribution.get_variables()]
    if not present:
        return distribution
    if not distribution.get_variables() - set(present):
        # nothing would be left; keep the mass in a distribution without variables
        return distribution.factory().from_expr(distribution.get_probability_mass())
    return distribution.marginal(*present, method=MarginalType.EXCLUDE)


def strip_dist_variables(distribution: Distribution, prog_info: ProgramInfo) -> Distribution:
    """The distribution of the program state without the encoding of the ``dist`` variables."""
    variables: List[str] = []
    for var in sorted(dist_variables(prog_info)):
        variables.extend(encoding_variables(distribution, var))
    return without(distribution, variables)


# ---------------------------------------------------------------------------------------
# Splitting the distribution by the values of dist variables
# ---------------------------------------------------------------------------------------

@dataclass
class DistPart:
    """The part of a distribution where a ``dist`` variable has one value."""
    distribution: Distribution
    site: Optional[DistSite]
    """``None``: the variable has no value; a site with slots: their values have infinite
    support."""


def _values(distribution: Distribution, variables: Sequence[str]) -> Optional[List[Tuple[int, ...]]]:
    """
    The values of ``variables`` in the support of ``distribution``, ``None`` for infinite support
    (``NonFiniteMarginalError`` if, in addition, their marginal is not a finite measure). The values
    come from the support (:meth:`Distribution.finite_support`), not the marginal, whose mass of a
    value can cancel to 0 on a signed measure (#256).
    """
    support = distribution.finite_support(*variables)
    if support is None:
        distribution.checked_marginal(*variables)
        return None
    values: List[Tuple[int, ...]] = []
    for coefficient, state in support:
        if _sym(coefficient) == 0:
            continue
        valuation = dict(state.items())
        value = tuple(int(valuation.get(var, 0)) for var in variables)
        if value not in values:
            values.append(value)
    return values


def _split(distribution: Distribution, variables: Sequence[str],
           values: List[Tuple[int, ...]]) -> Iterator[Tuple[Tuple[int, ...], Distribution]]:
    if len(values) == 1:
        yield values[0], distribution
        return
    for value in values:
        conditions = [BinopExpr(Binop.EQ, VarExpr(var), NatLitExpr(val))
                      for var, val in zip(variables, value, strict=True)]
        yield value, distribution.filter(BinopExpr.reduce(Binop.AND, conditions))


def split_by_value(distribution: Distribution, var: str, prog_info: ProgramInfo,
                   reg: DistRegistry) -> List[DistPart]:
    """Splits ``distribution`` by the value of the ``dist`` variable ``var``; slots with finite
    support are split into their values, too."""
    bound = prog_info.dist_bindings.get(var)
    if bound is not None:
        return [DistPart(distribution, bound)]
    if var not in distribution.get_variables():
        return [DistPart(distribution, None)]
    found = _values(distribution, [var])
    assert found is not None, "a dist variable has finitely many values"
    parts: List[DistPart] = []
    for (site_id,), part in _split(distribution, [var], found):
        site = reg.site(site_id)
        if site is None or site.closed:
            parts.append(DistPart(part, site))
            continue
        slots = [slot_variable(var, index) for index in site.slots]
        slot_values = _values(part, slots)
        if slot_values is None:
            parts.append(DistPart(part, site))
            continue
        for values, sub in _split(part, slots, slot_values):
            parts.append(DistPart(sub, site.with_slots(values)))
    if not parts:
        parts.append(DistPart(distribution, None))
    return parts


def split_by_values(distribution: Distribution, variables: Sequence[str], prog_info: ProgramInfo,
                    reg: DistRegistry) -> List[Tuple[Distribution, Dict[str, Optional[DistSite]]]]:
    """Splits by the values of several ``dist`` variables."""
    parts: List[Tuple[Distribution, Dict[str, Optional[DistSite]]]] = [(distribution, {})]
    for var in variables:
        refined = []
        for part, sites in parts:
            for sub in split_by_value(part, var, prog_info, reg):
                refined.append((sub.distribution, {**sites, var: sub.site}))
        parts = refined
    return parts


def _sum(parts: Sequence[Tuple[Distribution, Distribution]], distribution: Distribution,
         error_prob: Distribution) -> ProgramState:
    """The sum of the results of the parts of ``distribution``; their error probabilities are
    added to ``error_prob``."""
    if not parts:
        return ProgramState(distribution * "0", error_prob)
    result, error = parts[0]
    error = error_prob + error
    for other, other_error in parts[1:]:
        variables = result.get_variables() | other.get_variables()
        result = result.set_variables(*variables) + other.set_variables(*variables)
        error = error + other_error
    return ProgramState(result, error)


def _spell(node: Any) -> str:
    """``node`` as ``.ppy`` text."""
    return spell(node)


def _unassigned(var: str, construct: str) -> UndefinedDistributionError:
    return UndefinedDistributionError(
        f"{construct}: the dist variable {var} has no value here; assign it first, "
        f"with {var} = infer(f, …).")


def _sym(value: Any) -> Any:
    from prodigy.analysis.values import sym  # pylint: disable=import-outside-toplevel
    return sym(value)


def _set_site(distribution: Distribution, var: str, site_id: int) -> Distribution:
    return _assign(distribution, var, site_id)


def _rename_value(closed: ClosedDist, var: str) -> Distribution:
    """The generating function of ``closed`` in the variable ``var``."""
    source = closed.distribution
    return source.set_variables(VALUE_VARIABLE, var).update(
        BinopExpr(Binop.EQ, VarExpr(var), VarExpr(VALUE_VARIABLE))).marginal(var)
