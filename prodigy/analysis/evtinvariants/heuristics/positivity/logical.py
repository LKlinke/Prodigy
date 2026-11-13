from typing import List, Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.distribution.equality import sympify_symbolic_string


# pylint: disable-msg=keyword-arg-before-vararg

class OrHeuristic(PositivityHeuristic):
    """The verdict of the first heuristic that decides (all of them are sound, so they cannot disagree)."""

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *heuristics: PositivityHeuristic):
        super().__init__(sub_heuristic)
        self._heuristics: List[PositivityHeuristic] = list(heuristics)

    def _is_positive(self, f) -> Optional[bool]:
        for heuristic in self._heuristics:
            result = heuristic.is_positive(f)
            if result is True or result is False:
                return result
        return None


class AndHeuristic(PositivityHeuristic):
    """Non-negative if every heuristic shows it, negative if one shows a negative coefficient, else undecided."""

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *heuristics: PositivityHeuristic):
        super().__init__(sub_heuristic)
        self._heuristics: List[PositivityHeuristic] = list(heuristics)

    def _is_positive(self, f) -> Optional[bool]:
        verdict: Optional[bool] = True
        for heuristic in self._heuristics:
            result = heuristic.is_positive(f)
            if result is False:
                return False
            if result is not True:
                verdict = None
        return verdict


class NotHeuristic(PositivityHeuristic):
    """Refutes with the negation: if the heuristic shows that ``-f`` has non-negative coefficients and ``f`` is not
    zero, then ``f`` has a negative coefficient. Inverting a verdict on ``f`` itself would not be sound: a
    function without negative coefficients may still have positive ones and vice versa (#295)."""

    def __init__(self, heuristic: PositivityHeuristic, sub_heuristic: Optional[PositivityHeuristic] = None):
        super().__init__(sub_heuristic)
        self._heuristic = heuristic

    def _is_positive(self, f) -> Optional[bool]:
        s_f = sympify_symbolic_string(f) if isinstance(f, str) else sympy.S(f)
        if sympy.cancel(sympy.together(s_f)) == 0:
            return True
        if self._heuristic.is_positive(-s_f) is True:
            return False
        return None

# pylint: enable-msg=keyword-arg-before-vararg
