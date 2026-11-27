"""
The engine hooks of the generating-function logic (#511, ``docs/source/architecture.md``): CAS operations on an
already planned case. A hook does not branch on the program, does not choose between cases and does not raise
"not implemented" by itself; :class:`~prodigy.distribution.pgf.PGF` decides.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast, Optional, Self, Sequence, Set, Union

import sympy

from prodigy.language import BinopExpr, Expr, VarExpr
from prodigy.distribution.distribution import CommonDistributionsFactory, Distribution, Operand
if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.values import MarginalType, State


class PGFHooks(Distribution, CommonDistributionsFactory):
    """The operations that every engine of generating functions implements."""

    def _pgf(self) -> PGF:
        """This object with the type of the whole logic: the layers of :class:`~prodigy.distribution.pgf.pgf.PGF`
        are only instantiated through it."""
        return cast("PGF", self)

    @abstractmethod
    def _series_expression(self) -> sympy.Expr:
        """
        The generating function as a SymPy expression, for the shared coefficient-sign checks (a default from
        the engine text is :meth:`prodigy.distribution.pgf.base.PGFBase._series_expression`).
        """

    @abstractmethod
    def _probability_mass(self) -> str:
        """Engine hook of :meth:`get_probability_mass`: the closed form at 1, without a convergence proof."""

    @abstractmethod
    def _expected_value_of(self, expression: Union[Expr, str]) -> str:
        """The expected value of `expression`, computed by the engine (see :meth:`get_expected_value_of`)."""

    @abstractmethod
    def _normalize(self) -> PGF:
        """Engine hook of :meth:`normalize`: divides by the closed form at 1, without a convergence proof."""

    @abstractmethod
    def _sympy_expression(self) -> sympy.Expr:
        """This generating function as a SymPy expression (for exact computations in SymPy)."""

    @abstractmethod
    def _from_sympy_expression(self, expression: sympy.Expr) -> PGF:
        """A distribution of this engine with the variables and parameters of `self` from a SymPy
        expression."""

    @abstractmethod
    def _exhaustive_search(self, condition: Expr) -> PGF:
        """
        Given that `self` is finite, iterates over `self` and returns the sum of all terms where
        the condition holds.
        """

    @abstractmethod
    def _filter_constant_condition(self,
                                   condition: BinopExpr) -> PGF:
        """
        Filters out the terms that satisfy a constant condition, e.g., (var <= 5) (var > 5) (var = 5).

        :param condition: The condition to filter.

        :return: The filtered distribution.
        """

    @abstractmethod
    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> Sequence[PGF]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """

    @abstractmethod
    def _find_symbols(self, expr: str) -> Set[str]:
        "Returns a set of all free symbols in the given expression."

    @abstractmethod
    def _update_var(self, updated_var: str,
                    assign_var: str | int) -> PGF:
        """
        Applies the update `updated_var = assign_var` to this distribution.
        `assign_var` may be a variable or integer literal.
        """

    @abstractmethod
    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> PGF:
        """
        Applies the expression `temp_var = fist_summand + second_summand` to this distribution.
        """

    @abstractmethod
    def _update_product(self, temp_var: str, first_factor: str,
                        second_factor: str,
                        approximate: str | float | None) -> PGF:
        """
        Applies the update `temp_var = first_factor * second_factor` to this distribution.

        If the distribution is infinite and both factors are variables, multiplication is
        only supported if at least one of the factors has finite range (i.e., a finite marginal)
        if approximation is disabled.
        """

    @abstractmethod
    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> PGF:
        """
        Applies the expression `temp_var = sub_from - sub` to this distribution, where the
        difference is never negative on the support (the caller ensures this, see
        :meth:`_update_monus`).
        """

    @abstractmethod
    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> PGF:
        """
        Applies the expression `temp_var = left % right` to this distribution. If `self` is
        infinite, `right` must be a literal or a variable with finite range if approximation is disabled.
        """

    @abstractmethod
    def _update_division(self, temp_var: str, numerator: str | int,
                         denominator: str | int,
                         approximate: str | float | None) -> PGF:
        """
        Applies the expression `temp_var = numerator / denominator` to this distribution.
        If in some state of the GF, the numerator is not divisible by the denominator, this function
        raises an error.

        Infinite distributions are only supported if both sides of the division have finite range
        (i.e., they are either literals or have a finite marginal) if approximation is disabled. The
        exception is a numerator with infinite marginal and a literal denominator: the division is
        exact if the generating function is rational in the numerator and all its mass provably lies
        on multiples of the denominator (see :mod:`prodigy.distribution.pgf.rules.exact_division`).
        """

    @abstractmethod
    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> PGF:
        """
        Applies the expression `temp_var := base^exp` to this distribution.

        All variables occuring in the expression must have a finite marginal if approximation is disabled.
        """

    @abstractmethod
    def _compose_iid(self, sample: PGF, count: Union[str, int], variable: str) -> PGF:
        """
        Engine hook of :meth:`update_iid`: ``variable`` becomes the sum of ``count`` independent copies of
        ``sample`` (a generating function in ``variable`` of mass 1). With ``count`` a natural number ``n``:
        ``variable`` summed out, times ``sample^n``; with ``count == variable``: ``variable`` substituted by
        ``sample(variable)``; with another variable ``count``: ``variable`` summed out, then ``count`` substituted
        by ``count * sample(variable)``. The variables are those of this distribution (``sample`` may contain other
        variables of the state); its parameters gain those of ``sample``.
        """

    @abstractmethod
    def _marginal(self, *variables: Union[str, VarExpr],
                  method: MarginalType = MarginalType.INCLUDE) -> PGF:
        """Engine hook of :meth:`marginal`: the closed form at 1 in the summed-out variables, without a
        convergence proof."""

    @abstractmethod
    def _with_names(self, variables: Set[str], parameters: Set[str]) -> Self:
        """Engine hook of :meth:`set_variables` and :meth:`set_parameters`: the same generating function with
        exactly these (disjoint) variables and parameters."""

    # The operations of the interface that return a distribution, with the type of the engines: the shared logic
    # applies the hooks above to their results.

    @abstractmethod
    def __add__(self, other: Operand) -> PGF:
        """ The addition of two distributions. """

    @abstractmethod
    def __sub__(self, other: Operand) -> PGF:
        """ The subtraction of two distributions. """

    @abstractmethod
    def __mul__(self, other: Operand) -> PGF:
        """ The multiplication of two distributions. """

    @abstractmethod
    def __truediv__(self, other: Operand) -> PGF:
        """ The division of two distributions. """

    @abstractmethod
    def copy(self, deep: bool = True) -> PGF:
        """ Returns a full copy of itself."""

    @abstractmethod
    def filter(self, condition: Expr) -> PGF:
        """ Filters the distribution such that only the parts which satisfy the `condition` are left."""

    @abstractmethod
    def filter_state(self, state: State) -> PGF:
        """The distribution restricted to the extensions of `state` (see :meth:`Distribution.filter_state`)."""

    @abstractmethod
    def update(self, expression: Expr, approximate: str | float | None = None) -> PGF:
        """The distribution after the assignment `expression` (see :meth:`Distribution.update`)."""

    @abstractmethod
    def marginal(self, *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> PGF:
        """The marginal distribution (see :meth:`Distribution.marginal`)."""

    @abstractmethod
    def checked_marginal(self, *variables: Union[str, VarExpr], proof: bool = True) -> PGF:
        """The marginal for a case split (see :meth:`Distribution.checked_marginal`)."""

    @abstractmethod
    def finite_support(self, *variables: Union[str, VarExpr]) -> Optional[PGF]:
        """The values of `variables` in the support (see :meth:`Distribution.finite_support`)."""

    @abstractmethod
    def approximate_unilaterally(self, variable: str, probability_mass: str | float) -> PGF:
        """The series in `variable` up to `probability_mass` (see :meth:`Distribution.approximate_unilaterally`)."""

    @staticmethod
    @abstractmethod
    def from_expr(expression: Union[str, Expr], *variables: Union[str, VarExpr],
                  **kwargs: Optional[bool]) -> PGF:
        """ A distribution of this engine represented by the expression."""


def as_pgf(dist: Distribution) -> PGFHooks:
    """`dist` as a generating function, for the rules that compute on its closed form; every engine is one."""
    if not isinstance(dist, PGFHooks):
        raise TypeError(f"{type(dist).__name__} is not a generating function")
    return dist
