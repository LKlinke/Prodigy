from typing import List, Optional, Collection

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic


class OrHeuristic(PositivityHeuristic):

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *heuristics: PositivityHeuristic):
        super().__init__(sub_heuristic)
        self._heuristics: List[PositivityHeuristic] = list(heuristics)

    def _is_positive(self, f: str) -> Optional[bool]:
        result = None
        for heuristic in self._heuristics:
            result = heuristic.is_positive(f)
            if result is True or result is False:
                return result
        return result


class AndHeuristic(PositivityHeuristic):

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *heuristics: Collection[PositivityHeuristic]):
        super().__init__(sub_heuristic)
        self._heuristics: List[PositivityHeuristic] = list(*heuristics)

    def _is_positive(self, f: str) -> Optional[bool]:
        result = None
        for heuristic in self._heuristics:
            result = heuristic.is_positive(f)
            if result is False or result is None:
                return False
        return result


class NotHeuristic(PositivityHeuristic):

    def __init__(self, heuristic: PositivityHeuristic, sub_heuristic: Optional[PositivityHeuristic] = None):
        super().__init__(sub_heuristic)
        self._heuristic = heuristic

    def _is_positive(self, f: str) -> Optional[bool]:
        result = self._heuristic.is_positive(f)
        if result is None:
            return None
        return not result
