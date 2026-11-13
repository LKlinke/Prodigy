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

    It refutes only when this is proven: a univariate rational function without parameters and without
    floating-point numbers, whose reduced denominator has no zero at 0 (a power series) and whose roots of
    least modulus are all non-real. Every other case, in particular any precondition that cannot be established
    (parameters, several variables, floats, too high a degree, moduli that agree to 40 digits, a failing computer
    algebra call), is ``None`` ("cannot decide"); it never refutes on a numerical guess: the moduli are compared
    at 50 digits (algebraic numbers are evaluated to any precision), and a difference below ``1e-40`` counts as a tie.
    """

    logger = log_setup("DominantRootHeuristic", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *, variables: Collection[str]):
        super().__init__(sub_heuristic)
        self.variables = frozenset(map(str, variables))

    MAX_DEGREE = 40
    PRECISION = 50
    TIE = sympy.Float("1e-40")

    def _is_positive(self, f) -> Optional[bool]:
        try:
            return self._refute(f)
        except (VerificationError, sympy.PolynomialError, NotImplementedError, TypeError, IndexError,
                ValueError, ZeroDivisionError, sympy.polys.polyerrors.BasePolynomialError):
            return None

    def _refute(self, f) -> Optional[bool]:
        s_f, gens = series_form(f, self.variables)
        if len(s_f.free_symbols) > 1 or set(s_f.free_symbols) - set(gens):
            raise VerificationError(f"Currently only univariate rational functions without parameters are "
                                    f"supported. Given {f}")
        if not s_f.is_rational_function():
            raise VerificationError(f"The heuristic does not work on non-rational funtions. Given {f}")

        self.logger.debug("Put rational function into canoncial form.")
        if s_f.has(sympy.Float) or len(gens) != 1:
            return None
        s_f = s_f.cancel()
        if s_f.is_polynomial():
            # It was not a proper rational function but a polynomial instead.
            return FinitePolynomialPositivity(variables=self.variables).is_positive(s_f)
        self.logger.debug("Cancelled fraction: %s", s_f)
        _, s_denom = s_f.as_numer_denom()
        if s_denom.subs(gens[0], 0) == 0:
            # a pole at the origin: not a power series
            return None
        denominator_poly = s_denom.as_poly(gens[0])
        if denominator_poly.degree() > self.MAX_DEGREE or not all(c.is_rational for c in denominator_poly.all_coeffs()):
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
            max_all_roots = max(sympy.Abs(root).evalf(self.PRECISION) for root in all_roots)
            max_real_roots = max(sympy.Abs(root).evalf(self.PRECISION) for root in real_roots)
            if max_real_roots >= max_all_roots - self.TIE:
                # a real root is dominant (or ties): Pringsheim's singularity may be there -> undecided
                self.logger.debug("A real root is dominant or ties. -> Result UNKNOWN")
                return None
            # no real root of least modulus: the radius of convergence is not a singularity on the real
            # axis, let alone a positive one, so the coefficients cannot all be non-negative
            self.logger.debug("No real dominant root. -> Not positive!")
            return False
