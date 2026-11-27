"""
The values of the :class:`~prodigy.distribution.distribution.Distribution` interface (#511): states, the kind of
a marginal, the parameters of the common distributions and the verdict of a comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, ItemsView, Iterable, Iterator, List, Optional, Tuple, Union

import sympy

from prodigy.language import Expr


def canonical_order(items: Iterable[Tuple[str, State]]) -> List[Tuple[str, State]]:
    """
    The (probability, state) pairs of a finite distribution in the order in which every engine iterates
    them (#361): by the sum of the values, then by the values of the variables in the order of their names
    (the graded lexicographic order). The order depends only on the states, not on how an engine stores
    the terms.
    """
    def key(item: Tuple[str, State]) -> Tuple[int, Tuple[int, ...]]:
        values = tuple(value for _, value in sorted(item[1].items()))
        return sum(values), values
    return sorted(items, key=key)


class MarginalType(Enum):
    """ Specifies the type of marginalization. """
    INCLUDE = auto()
    EXCLUDE = auto()


DistributionParam = Union[str, Expr]


Value = Union[int, sympy.Expr]
"""The value of an expression in a state (:meth:`Distribution.evaluate`): an engine's number (GiNaC's) or a
SymPy number or expression."""


@dataclass
class State:
    """Describes a state of a distribution, i.e., an assignment of variables to values"""

    valuations: Dict[str, int] = field(default_factory=dict)
    """The variable assignment of this state"""

    def __iter__(self) -> Iterator[str]:
        return self.valuations.__iter__()

    def __getitem__(self, item: str) -> int:
        return self.valuations.__getitem__(item)

    def __setitem__(self, key: str, value: int) -> None:
        self.valuations.__setitem__(key, value)

    def __str__(self) -> str:
        return self.valuations.__str__()

    def __hash__(self) -> int:
        # Order-independent, consistent with __eq__. A state must not change while it is used as a key.
        return hash(frozenset(self.valuations.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.valuations == other
        if isinstance(other, State):
            return self.valuations == other.valuations
        return False

    def items(self) -> ItemsView[str, int]:
        """Provides a view on all (variable, value) pairs"""
        return self.valuations.items()

    def to_monomial(self) -> str:
        """
        Provides a string representation of this state in the form of a polynomial. If the state is empty, returns an
        empty string.

        For example, the variable assignment `{x: 3, y: 6}` gives rise to the string `x^3*y^6`.
        """
        if self.valuations:
            result = "*"
            addends = []
            for variable, value in self.valuations.items():
                addends.append(f"{variable}^{value}")
            return result.join(addends)
        return ""

    def copy(self) -> State:
        return State(valuations=self.valuations.copy())

    def extend(self, other: State) -> State:
        res = self.copy()
        for var, val in other.items():
            if var not in res.valuations:
                res[var] = val
            else:
                raise ValueError(f"Variable {var} already has a value")
        return res



@dataclass(frozen=True)
class OrderVerdict:
    """The verdict on ``f <= g``: ``holds`` is three-valued; ``witness`` is a state whose coefficient in
    ``g - f`` is proven negative (only if ``holds`` is False)."""
    holds: Optional[bool]
    witness: Optional[State] = None
