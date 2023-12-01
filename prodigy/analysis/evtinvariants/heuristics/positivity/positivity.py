import logging
from abc import ABC, abstractmethod
from typing import Optional, Collection, List

import sympy

from prodigy.analysis.exceptions import HeuristicsError
from prodigy.distribution import CommonDistributionsFactory
from prodigy.util.logger import log_setup


class FPSPositivityHeuristic(ABC):
    def __init__(self, sub_heuristic: 'FPSPositivityHeuristic' = None):
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


class FinitePolynomialPositivity(FPSPositivityHeuristic):
    logger = log_setup("FinitePolynomialPositivity", logging.DEBUG)

    def __init__(self, sub_heuristic: 'FPSPositivityHeuristic' = None):
        super().__init__(sub_heuristic)

    def _is_positive(self, f: str) -> Optional[bool]:
        s_f: sympy.Expr = sympy.S(f)
        if s_f.is_polynomial():
            self.logger.debug("Determine positivity of polynomial %s", s_f)
            if s_f.is_constant():
                return s_f.is_positive
            return all([coefficient >= 0 for coefficient in s_f.as_poly().coeffs()])
        raise HeuristicsError(f"Given function {f} is not a polynomial.")


class SumPositivity(FPSPositivityHeuristic):
    logger = log_setup("SumPositivity", logging.DEBUG)

    def __init__(self, term_heuristic: FPSPositivityHeuristic, sub_heuristic: FPSPositivityHeuristic = None):
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
            self.logger.debug(f"Positive %s", f)
            return True
        return self._term_heuristic.is_positive(f)


class OrHeuristic(FPSPositivityHeuristic):

    def __init__(self, sub_heuristic: 'FPSPositivityHeuristic' = None, *heuristics: FPSPositivityHeuristic):
        super().__init__(sub_heuristic)
        self._heuristics: List[FPSPositivityHeuristic] = list(heuristics)

    def _is_positive(self, f: str) -> Optional[bool]:
        result = None
        for heuristic in self._heuristics:
            result = heuristic.is_positive(f)
            if result is True or result is False:
                return result
        return result


class AndHeuristic(FPSPositivityHeuristic):

    def __init__(self, sub_heuristic: 'FPSPositivityHeuristic' = None, *heuristics: Collection[FPSPositivityHeuristic]):
        super().__init__(sub_heuristic)
        self._heuristics: List[FPSPositivityHeuristic] = list(*heuristics)

    def _is_positive(self, f: str) -> Optional[bool]:
        result = None
        for heuristic in self._heuristics:
            result &= heuristic.is_positive(f)
        return result


class SingleRationalFunction(FPSPositivityHeuristic):
    logger = log_setup("SingleRationalFunction", logging.DEBUG)

    def __init__(self, sub_heuristic: 'FPSPositivityHeuristic' = None):
        super().__init__(sub_heuristic)
        self._poly_heuristic = FinitePolynomialPositivity(sub_heuristic)

    def _num_denom_check(self, numerator: sympy.Expr, denominator: sympy.Expr) -> Optional[bool]:
        s_denom_cterm, s_denom_rest = denominator.as_coeff_Add()
        numerator_positivity = self._poly_heuristic.is_positive(str(numerator))
        if numerator_positivity is False:
            return None
        if s_denom_cterm <= 0:
            return None
        if any([coefficient > 0 for coefficient in s_denom_rest.coeffs()]):
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


class RationalFunctionDenomSign(FPSPositivityHeuristic):
    logger = log_setup("RationalFunctionDenomSign", logging.DEBUG)

    def __init__(self, dist_fact: CommonDistributionsFactory, sub_heuristic: 'FPSPositivityHeuristic' = None):
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

        # Search for bad coefficients
        for i, (coef, state) in enumerate(self.dist_fact.from_expr(str(f).replace("**", "^"))):
            if i > 10:
                break
            if sympy.S(coef) < 0:
                self.logger.info("Invariant is spurious. Coefficient of state %s is %s", state.valuations, coef)
                return False

        # We don't know otherwise
        self.logger.info("Heuristic failed, we dont know!")
        return None


class FPSPositivityHeuristicFactory:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"Static class {str(cls)} cannot be instantiated.")

    @classmethod
    def rational_function_denominator_sign(cls, dist_fact: CommonDistributionsFactory) -> RationalFunctionDenomSign:
        return RationalFunctionDenomSign(dist_fact)

    @classmethod
    def create_sum_then_rational(cls) -> SumPositivity:
        return SumPositivity(SingleRationalFunction())

    @classmethod
    def create_or(cls, *heuristics: FPSPositivityHeuristic) -> OrHeuristic:
        return OrHeuristic(*heuristics)
