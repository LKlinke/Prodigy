import logging
from typing import Collection, Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import series_form
from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import VerificationError
from prodigy.util.logger import log_setup


class DominantRootHeuristic(PositivityHeuristic):
    """
    This heuristic tries to falsify positivity for univariate rational functions by using results about dominant
    roots in the characteristic polynomial of the linear recurrence sequence related to it.
    More details can be found in https://arxiv.org/pdf/1309.1550.pdf in particular Proposition 2.
    It only refutes (by Pringsheim's theorem, a series with non-negative coefficients has a positive real
    singularity of least modulus); it never proves non-negativity.
    """

    logger = log_setup("DominantRootHeuristic", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *, variables: Collection[str]):
        super().__init__(sub_heuristic)
        self.variables = frozenset(map(str, variables))

    def _is_positive(self, f) -> Optional[bool]:
        s_f, gens = series_form(f, self.variables)
        if len(s_f.free_symbols) > 1 or set(s_f.free_symbols) - set(gens):
            raise VerificationError(f"Currently only univariate rational functions without parameters are "
                                    f"supported. Given {f}")
        if not s_f.is_rational_function():
            raise VerificationError(f"The heuristic does not work on non-rational funtions. Given {f}")

        self.logger.debug("Put rational function into canoncial form.")
        s_f = s_f.cancel()
        if s_f.is_polynomial():
            # It was not a proper rational function but a polynomial instead.
            return FinitePolynomialPositivity(variables=self.variables).is_positive(s_f)
        self.logger.debug("Cancelled fraction: %s", s_f)
        _, s_denom = s_f.as_numer_denom()
        if s_denom.subs(gens[0], 0) == 0:
            # a pole at the origin: not a power series
            return None
        s_denom_as_poly = sympy.Poly(s_denom.as_poly().all_coeffs()[::-1], *s_f.free_symbols)
        self.logger.debug("Characteristic Polynomial of LRS: %s", s_denom_as_poly)

        real_roots = s_denom_as_poly.real_roots()
        if len(real_roots) == 0:
            # No dominant real root -> thus there definetly is a negative sign somewhere.
            self.logger.debug("No real roots found. Not positive!")
            return False
        else:
            # We have at least one real root.
            self.logger.debug("Real roots of polynomial: %s", real_roots)
            all_roots = s_denom_as_poly.all_roots()
            self.logger.debug("All roots of polynomial: %s", all_roots)
            max_all_roots = max(sympy.Abs(root) for root in all_roots)
            max_real_roots = max(sympy.Abs(root) for root in real_roots)
            if max_real_roots >= max_all_roots:
                # real roots are dominant.
                self.logger.debug("We have at least one real dominant root. -> Result UNKNOWN")
                return None
            else:
                # No real dominant root. The LRS has infinitely many negative coefficients.
                self.logger.debug("No real dominant root. -> Not positive!")
                return False
