import logging
from typing import Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.exceptions import HeuristicsError
from prodigy.distribution import CommonDistributionsFactory
from prodigy.util.logger import log_setup


class SingleRationalFunction(PositivityHeuristic):
    logger = log_setup("SingleRationalFunction", logging.DEBUG)

    def __init__(self, sub_heuristic: Optional[PositivityHeuristic] = None):
        super().__init__(sub_heuristic)
        self._poly_heuristic = FinitePolynomialPositivity(sub_heuristic)

    def _num_denom_check(self, numerator: sympy.Expr, denominator: sympy.Expr) -> Optional[bool]:
        s_denom_cterm, s_denom_rest = denominator.as_coeff_Add()
        numerator_positivity = self._poly_heuristic.is_positive(str(numerator))
        if numerator_positivity is False:
            return None
        if s_denom_cterm <= 0:
            return None
        if any(coefficient > 0 for coefficient in s_denom_rest.as_poly().coeffs()):
            return None
        return True

    def _is_positive(self, f: str) -> Optional[bool]:
        s_f: sympy.Expr = sympy.S(f)
        if not s_f.is_rational_function():
            raise HeuristicsError(f"The given function is not rational {f=}")
        if s_f.is_polynomial():
            self.logger.debug("Forward to poly heuristic")
            return self._poly_heuristic.is_positive(str(s_f))
        if isinstance(s_f, sympy.Add):
            raise HeuristicsError(f"The given function is a sum of rational functions {f=}.")

        self.logger.debug("try do decide positivity for %s", f)
        s_numerator, s_denominator = s_f.as_numer_denom()
        res1 = self._num_denom_check(s_numerator, s_denominator)
        res2 = self._num_denom_check(-s_numerator, -s_denominator)
        self.logger.debug("result: %s", res1 or res2)
        return res1 or res2


class RationalFunctionDenomSign(PositivityHeuristic):
    logger = log_setup("RationalFunctionDenomSign", logging.DEBUG)

    def __init__(self, dist_fact: CommonDistributionsFactory, sub_heuristic: Optional[PositivityHeuristic] = None):
        super().__init__(sub_heuristic)
        self.dist_fact = dist_fact

    def _is_positive(self, f: sympy.Expr) -> Optional[bool]:
        """
        Checks heuristically whether a univariate rational function _f_ has a non-negative power series
        expansion.

        ..returns:
            - None: We do not know
            - True: It has a non-negative power series expansion
            - False: There exists at least one negative coefficient in the power series expansion
        """
        self.logger.debug("check non-negativity for %s", f)

        # TODO  f might be a sum of rational functions (partial fraction decomposition).
        if isinstance(f, sympy.Add):
            self.logger.debug("Try to solve positivity by looking at sum terms individually.")
            for sub_f in f.args:
                res = self.is_positive(sub_f.apart())
                if res is not True:
                    self.logger.debug("Sum decomposition did not work. Try together()")
                    return self.is_positive(f.together())
            return True

        # Getting f as numerator/denominator and check whether it is indeed a rational function.
        numerator, denominator = f.as_numer_denom()

        # Check whether f is univariate (others not supported currently)
        if len(denominator.free_symbols) > 1:
            self.logger.info("Multivariate functions like %s are currently not supported", f)
            return None

        self.logger.debug("\nnumerator (non-poly) %s\ndenominator (non-poly) %s", numerator, denominator)
        if not (numerator.is_polynomial() and denominator.is_polynomial()):
            self.logger.info("%s is not a rational function.", f)
            return None

        # convert numerator and denominator into polynomial objects
        # as_poly() fails converting constant polynomials without specified variables
        # so we give it a _DUMMY_ variable.
        maybe_numerator = numerator.as_poly(*numerator.free_symbols)
        numerator = maybe_numerator if maybe_numerator else numerator.as_poly(sympy.S("DUMMY"))
        maybe_denominator = denominator.as_poly(*denominator.free_symbols)
        denominator = maybe_denominator if maybe_denominator else denominator.as_poly(sympy.S("DUMMY"))
        self.logger.debug("\nnumerator %s\ndenominator %s", numerator, denominator)

        # Check the coefficients of the denominator
        d_coeffs = denominator.all_coeffs()
        n_coeffs = numerator.all_coeffs()

        # Check the constant coefficient and factor a minus sign if necessary
        if d_coeffs[-1] < 0:
            self.logger.debug("Try factoring a minus sign in %s", f)
            d_coeffs = [-x for x in d_coeffs]
            n_coeffs = [-x for x in n_coeffs]
        if all(map(lambda x: x <= 0, d_coeffs[:-1])):  # are all other coefficients non-positive?
            if all(map(lambda x: x >= 0, n_coeffs)):  # are all nominator coefficients non-negative?
                self.logger.info("Invariant validates as non-negative FPS.")
                return True

        # We don't know otherwise
        self.logger.info("Heuristic failed, we dont know!")
        return None


class RationalFunctionDenomGeometricLike(PositivityHeuristic):
    logger = log_setup("RationalFunctionDenomGeometricLike", logging.DEBUG)

    def __init__(self, numerator_heuristic: PositivityHeuristic, sub_heuristic: Optional[PositivityHeuristic] = None):
        if numerator_heuristic is None:
            raise HeuristicsError("The RationalFunctionDenomGeometricLike heuristic needs a numerator heuristic.")
        super().__init__(sub_heuristic)
        self._numerator_heuristic = numerator_heuristic

    def _is_geometric_like(self, f: sympy.Expr) -> Optional[bool]:
        """
        Checks if _f_ is 'geometric-like' if in a denominator, i.e. f is a polynomial of the form 
        - positive constant term, and
        - all other monome coefficients are negative.
        """
        f_const_term, f_rest = f.as_coeff_Add()
        if f_const_term <= 0:
            return None
        if any(coefficient > 0 for coefficient in f_rest.as_poly().coeffs()):
            return None
        return True

    def _denominator_is_prod_of_gemoetric_like(self, denom: sympy.Expr) -> Optional[bool]:
        if isinstance(denom, sympy.Mul):
            for term in denom.args:
                self.logger.debug("checking denominator geometric-like for %s", term)
                result = self._is_geometric_like(term)
                if result is False or result is None:
                    self.logger.debug("Could not determinine positivity")
                    return None
            self.logger.debug("Geometric-like denominator %s", denom)
            return True
        return self._is_geometric_like(denom)        

    def _num_denom_check(self, numerator: sympy.Expr, denominator: sympy.Expr) -> Optional[bool]:
        # if numerator is not positive, we don't know about positivity of fraction.
        # The denominator could cancel negativity, but we don't check this.
        if self._numerator_heuristic.is_positive(str(numerator)):
            return self._denominator_is_prod_of_gemoetric_like(denominator)
        else:
            return None
        
    def _is_positive(self, f: str) -> Optional[bool]:
        """
        Checks heuristically whether _f_ is a rational function g/h with
        - g is positive,
        - h is a product of 'geometric-like' terms, see _is_geometric_like
        """ 
        s_f: sympy.Expr = sympy.S(f)
        if not s_f.is_rational_function():
            raise HeuristicsError(f"The given function is not rational {f=}")
        if s_f.is_polynomial():
            self.logger.debug("Forward to numerator heuristic")
            return self._numerator_heuristic.is_positive(str(s_f))
        if isinstance(s_f, sympy.Add):
            raise HeuristicsError(f"The given function is a sum of rational functions {f=}.")

        self.logger.debug("try do decide positivity for %s", f)
        s_numerator, s_denominator = s_f.as_numer_denom()
        res = self._num_denom_check(s_numerator, s_denominator)
    
        self.logger.debug("result: %s", res)
        return res
