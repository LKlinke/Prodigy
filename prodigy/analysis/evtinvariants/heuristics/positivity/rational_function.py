import logging
from typing import Collection, Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import nonneg_by_construction, \
    nonneg_quotient, series_form
from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.distribution import CommonDistributionsFactory
from prodigy.util.logger import log_setup


class SingleRationalFunction(PositivityHeuristic):
    """Decides a single quotient ``N / D`` of polynomials in ``variables``: non-negative if ``D = c - R`` with
    ``c > 0`` and ``N``, ``R`` with non-negative coefficients (up to a common sign). Other symbols are
    parameters (:mod:`.coefficients`)."""
    logger = log_setup("SingleRationalFunction", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None, *, variables: Collection[str]):
        super().__init__(sub_heuristic)
        self.variables = frozenset(map(str, variables))
        self._poly_heuristic = FinitePolynomialPositivity(sub_heuristic, variables=self.variables)

    def _is_positive(self, f) -> Optional[bool]:
        s_f, gens = series_form(f, self.variables)
        if not s_f.is_rational_function(*gens):
            raise HeuristicsError(f"The given function is not rational {f=}")
        if s_f.is_polynomial(*gens):
            self.logger.debug("Forward to poly heuristic")
            return self._poly_heuristic.is_positive(s_f)
        if isinstance(s_f, sympy.Add):
            raise HeuristicsError(f"The given function is a sum of rational functions {f=}.")

        self.logger.debug("try do decide positivity for %s", f)
        numerator, denominator = s_f.as_numer_denom()
        if nonneg_quotient(numerator, denominator, gens) or nonneg_quotient(-numerator, -denominator, gens):
            return True
        return None


def _apart(f: sympy.Expr) -> sympy.Expr:
    """The partial fraction decomposition of ``f``, or ``f`` itself where SymPy has none (multivariate)."""
    try:
        return f.apart()
    except (NotImplementedError, sympy.PolynomialError):
        return f


class RationalFunctionDenomSign(PositivityHeuristic):
    logger = log_setup("RationalFunctionDenomSign", logging.DEBUG)

    def __init__(self, dist_fact: CommonDistributionsFactory, sub_heuristic: Optional[PositivityHeuristic] = None,
                 *, variables: Collection[str]):
        """``variables`` are the variables of the power series; other symbols are parameters, real numbers of
        unknown sign (#295)."""
        super().__init__(sub_heuristic)
        self.dist_fact = dist_fact
        self.variables = frozenset(map(str, variables))

    def _is_positive(self, f) -> Optional[bool]:
        """
        Checks heuristically whether a rational function _f_ = N / D (in any number of variables) has a
        non-negative power series expansion: it does if D has a positive constant term, all other
        coefficients of D are non-positive and all coefficients of N are non-negative (up to a common sign).
        Sums are checked term by term first, and N may also be Q * D + r with non-negative Q and r
        (:func:`.coefficients.nonneg_by_construction`).

        ..returns:
            - None: We do not know
            - True: It has a non-negative power series expansion
            - False: never; this heuristic only proves non-negativity
        """
        self.logger.debug("check non-negativity for %s", f)
        f, gens = series_form(f, self.variables)
        if nonneg_by_construction(f, self.variables):
            return True

        if isinstance(f, sympy.Add):
            self.logger.debug("Try to solve positivity by looking at sum terms individually.")
            if all(self.is_positive(_apart(sub_f)) is True for sub_f in f.args):
                return True
            self.logger.debug("Sum decomposition did not work. Try together()")
            f = f.together()
            if not isinstance(f, sympy.Add):
                return self.is_positive(f)

        # Getting f as numerator/denominator and check whether it is indeed a rational function.
        numerator, denominator = f.as_numer_denom()
        if not (numerator.is_polynomial(*gens) and denominator.is_polynomial(*gens)):
            self.logger.info("%s is not a rational function.", f)
            return None
        if nonneg_quotient(numerator, denominator, gens) or nonneg_quotient(-numerator, -denominator, gens):
            self.logger.info("Invariant validates as non-negative FPS.")
            return True

        # We don't know otherwise
        self.logger.info("Heuristic failed, we dont know!")
        return None
