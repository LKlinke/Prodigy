import logging
from typing import Collection, Optional

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


def _nonneg_quotient(numerator: sympy.Expr, denominator: sympy.Expr,
                     variables: Optional[Collection[str]] = None) -> bool:
    """Whether numerator / denominator is shown to have a non-negative power series expansion in
    ``variables`` (default: all free symbols): denominator = c - R with a constant c > 0 and a polynomial R
    with non-negative coefficients and no constant term, so 1 / denominator = sum_k R^k / c^(k+1); and the
    numerator has non-negative coefficients. Works for any number of variables; a coefficient whose sign is
    not known (e.g. one with a parameter) is not decided."""
    symbols = numerator.free_symbols | denominator.free_symbols
    if variables is not None:
        symbols = {symbol for symbol in symbols if str(symbol) in variables}
    gens = sorted(symbols, key=str) or [sympy.Dummy("DUMMY")]
    num = sympy.Poly(numerator, *gens)
    den = sympy.Poly(denominator, *gens)
    constant = den.as_dict().get((0,) * len(gens), sympy.S.Zero)
    if constant.is_positive is not True:
        return False
    rest = [coeff for monom, coeff in den.terms() if any(monom)]
    return all(coeff.is_nonpositive is True for coeff in rest) \
        and all(coeff.is_nonnegative is True for coeff in num.coeffs())


def _apart(f: sympy.Expr) -> sympy.Expr:
    """The partial fraction decomposition of ``f``, or ``f`` itself where SymPy has none (multivariate)."""
    try:
        return f.apart()
    except (NotImplementedError, sympy.PolynomialError):
        return f


class RationalFunctionDenomSign(PositivityHeuristic):
    logger = log_setup("RationalFunctionDenomSign", logging.DEBUG)

    def __init__(self, dist_fact: CommonDistributionsFactory, sub_heuristic: Optional[PositivityHeuristic] = None,
                 variables: Optional[Collection[str]] = None):
        """``variables`` are the variables of the power series; other symbols are parameters of unknown
        sign. Without them every free symbol counts as a variable."""
        super().__init__(sub_heuristic)
        self.dist_fact = dist_fact
        self.variables = None if variables is None else frozenset(map(str, variables))

    def _is_positive(self, f: sympy.Expr) -> Optional[bool]:
        """
        Checks heuristically whether a rational function _f_ = N / D (in any number of variables) has a
        non-negative power series expansion: it does if D has a positive constant term, all other
        coefficients of D are non-positive and all coefficients of N are non-negative (up to a common sign).

        ..returns:
            - None: We do not know
            - True: It has a non-negative power series expansion
            - False: There exists at least one negative coefficient in the power series expansion
        """
        self.logger.debug("check non-negativity for %s", f)
        f = sympy.S(f)

        # TODO  f might be a sum of rational functions (partial fraction decomposition).
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
        self.logger.debug("\nnumerator (non-poly) %s\ndenominator (non-poly) %s", numerator, denominator)
        if not (numerator.is_polynomial() and denominator.is_polynomial()):
            self.logger.info("%s is not a rational function.", f)
            return None
        if _nonneg_quotient(numerator, denominator, self.variables) \
                or _nonneg_quotient(-numerator, -denominator, self.variables):
            self.logger.info("Invariant validates as non-negative FPS.")
            return True

        # We don't know otherwise
        self.logger.info("Heuristic failed, we dont know!")
        return None
