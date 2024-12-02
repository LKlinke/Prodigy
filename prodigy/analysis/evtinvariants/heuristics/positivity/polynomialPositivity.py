import logging
from typing import Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.util.logger import log_setup


class FinitePolynomialPositivity(PositivityHeuristic):
    logger = log_setup("FinitePolynomialPositivity", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None):
        super().__init__(sub_heuristic)

    def _is_positive(self, f: str) -> Optional[bool]:
        s_f: sympy.Expr = sympy.S(f)
        if s_f.is_polynomial():
            self.logger.debug("Determine positivity of polynomial %s", s_f)
            if s_f.is_constant():
                return s_f.is_positive
            return all([coefficient >= 0 for coefficient in s_f.as_poly().coeffs()])
        raise HeuristicsError(f"Given function {f} is not a polynomial.")
