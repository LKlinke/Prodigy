import logging
from typing import Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.util.logger import log_setup


class SumPositivity(PositivityHeuristic):
    logger = log_setup("SumPositivity", logging.DEBUG)

    def __init__(self, term_heuristic: PositivityHeuristic, sub_heuristic: Optional[PositivityHeuristic] = None):
        if term_heuristic is None:
            raise HeuristicsError("The SumPositivity heuristic needs a term heuristic.")
        super().__init__(sub_heuristic)
        self._term_heuristic = term_heuristic

    def _is_positive(self, f: str) -> Optional[bool]:
        s_f = sympy.S(f)
        if isinstance(s_f, sympy.Add):
            for term in s_f.args:
                self.logger.debug("checking positivity for %s", term)
                result = self._term_heuristic.is_positive(str(term))
                if result is False or result is None:
                    self.logger.debug("Could not determinine positivity")
                    return None
            self.logger.debug("Positive %s", f)
            return True
        return self._term_heuristic.is_positive(f)
