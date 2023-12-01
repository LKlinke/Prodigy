from abc import ABC, abstractmethod
from typing import Optional


class PositivityHeuristic(ABC):

    def __init__(self, sub_heuristic: 'PositivityHeuristic' = None):
        self.sub_heuristic = sub_heuristic

    def is_positive(self, f: str) -> Optional[bool]:
        """
        Checks heuristically whether a function _f_ has a non-negative power series
        expansion.

        ..returns:
            - None: We do not know
            - True: It has a non-negative power series expansion
            - False: There exists at least one negative coefficient in the power series expansion
        """
        if self.sub_heuristic:
            result = self.sub_heuristic.is_positive(f)
            if result is False or result is True:
                return result
        return self._is_positive(f)

    @abstractmethod
    def _is_positive(self, f: str) -> Optional[bool]:
        ...

    def __str__(self) -> str:
        return self.__class__.__name__
