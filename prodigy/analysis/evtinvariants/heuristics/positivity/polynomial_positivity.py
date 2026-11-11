import logging
from typing import Collection, Optional

from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import polynomial_verdict, series_form
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.util.logger import log_setup


class FinitePolynomialPositivity(PositivityHeuristic):
    """Decides polynomials in ``variables``: their coefficients are the coefficients of the power series. A
    coefficient that contains a parameter counts only if its sign holds for every real value of the parameter."""
    logger = log_setup("FinitePolynomialPositivity", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *, variables: Collection[str]):
        super().__init__(sub_heuristic)
        self.variables = frozenset(map(str, variables))

    def _is_positive(self, f) -> Optional[bool]:
        s_f, gens = series_form(f, self.variables)
        if not s_f.is_polynomial(*gens):
            raise HeuristicsError(f"Given function {f} is not a polynomial.")
        self.logger.debug("Determine positivity of polynomial %s", s_f)
        return polynomial_verdict(s_f, gens)
