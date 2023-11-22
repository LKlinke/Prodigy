import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import sympy

from prodigy.distribution import CommonDistributionsFactory
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class FPSPositivityHeuristic(ABC):

    @abstractmethod
    def is_positive(self, f) -> Optional[bool]:
        ...

    def __str__(self) -> str:
        return self.__class__.__name__


@dataclass
class RationalFunctionDenomSign(FPSPositivityHeuristic):
    dist_fact: CommonDistributionsFactory

    def is_positive(self, f: sympy.Expr) -> Optional[bool]:
        """
        Checks heuristically whether a univariate rational function _f_ has a non-negative power series
        expansion.

        ..returns:
            - None: We do not know
            - True: It has a non-negative power series expansion
            - False: There exists at least one negative coefficient in the power series expansion
        """
        logger.debug("check non-negativity for %s", f)

        # TODO  f might be a sum of rational functions (partial fraction decomposition).
        if isinstance(f, sympy.Add):
            logger.debug("Try to solve positivity by looking at sum terms individually.")
            for sub_f in f.args:
                res = self.is_positive(sub_f.apart())
                if res is not True:
                    logger.debug("Sum decomposition did not work. Try together()")
                    return self.is_positive(f.together())
            return True

        # Getting f as numerator/denominator and check whether it is indeed a rational function.
        numerator, denominator = f.as_numer_denom()

        # Check whether f is univariate (others not supported currently)
        if len(denominator.free_symbols) > 1:
            logger.info("Multivariate functions like %s are currently not supported", f)
            return None

        logger.debug("\nnumerator (non-poly) %s\ndenominator (non-poly) %s", numerator, denominator)
        if not (numerator.is_polynomial() and denominator.is_polynomial()):
            logger.info("%s is not a rational function.", f)
            return None

        # convert numerator and denominator into polynomial objects
        # as_poly() fails converting constant polynomials without specified variables
        # so we give it a _DUMMY_ variable.
        maybe_numerator = numerator.as_poly(*numerator.free_symbols)
        numerator = maybe_numerator if maybe_numerator else numerator.as_poly(sympy.S("DUMMY"))
        maybe_denominator = denominator.as_poly(*denominator.free_symbols)
        denominator = maybe_denominator if maybe_denominator else denominator.as_poly(sympy.S("DUMMY"))
        logger.debug("\nnumerator %s\ndenominator %s", numerator, denominator)

        # Check the coefficients of the denominator
        d_coeffs = denominator.all_coeffs()
        n_coeffs = numerator.all_coeffs()

        # Check the constant coefficient and factor a minus sign if necessary
        if d_coeffs[-1] < 0:
            logger.debug("Try factoring a minus sign in %s", f)
            d_coeffs = [-x for x in d_coeffs]
            n_coeffs = [-x for x in n_coeffs]
        if all(map(lambda x: x <= 0, d_coeffs[:-1])):  # are all other coefficients non-positive?
            if all(map(lambda x: x >= 0, n_coeffs)):  # are all nominator coefficients non-negative?
                logger.info("Invariant validates as non-negative FPS.")
                return True

        # Search for bad coefficients
        for i, (coef, state) in enumerate(self.dist_fact.from_expr(str(f).replace("**", "^"))):
            if i > 10:
                break
            if sympy.S(coef) < 0:
                logger.info("Invariant is spurious. Coefficient of state %s is %s", state.valuations, coef)
                return False

        # We don't know otherwise
        logger.info("Heuristic failed, we dont know!")
        return None


class FPSPositivityHeuristicFactory:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"Static class {str(cls)} cannot be instantiated.")

    @classmethod
    def rational_function_denominator_sign(cls, dist_fact: CommonDistributionsFactory) -> RationalFunctionDenomSign:
        return RationalFunctionDenomSign(dist_fact)
