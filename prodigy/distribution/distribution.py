"""
The :class:`Distribution` interface and the :class:`CommonDistributionsFactory` (#511). The values, errors and
generating-function helpers that this module defined before are re-exported until the end of v0.7.0.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import FrozenSet, Generator, Iterator, List, Optional, Self, Set, Tuple, Type, Union

from prodigy.language import Expr, NatLitExpr, VarExpr
from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.errors import (NonFiniteMarginalError, NotAnIntegerError, UndecidedComparisonError,
                                         ZeroDivisorError, known_convergent_series)
from prodigy.distribution.values import (DistributionParam, MarginalType, OrderVerdict, State, Value,
                                         canonical_order)
# re-exported from their new modules until the end of v0.7.0 (#511)
from prodigy.distribution.terms import constant_value

# Spellings of non-finite values (nan, infinity, complex infinity, unevaluated limits) in the engines'
# string forms; a program variable or parameter of the same name is not one of them
_NON_FINITE_NAMES = frozenset({"nan", "NaN", "zoo", "oo", "inf", "Inf", "infinity", "Infinity",
                               "ComplexInfinity", "UnsignedInfinity", "Limit"})


Operand = Union["Distribution", str, int, float]
"""The other operand of the arithmetic of distributions: a distribution, or a closed form (text or number) in the
variables of the first one. Each engine accepts its own distributions and some of the others."""


class Distribution(ABC):  # pylint: disable=too-many-public-methods  # the semantic interface of all engines
    """
    The semantic interface of a distribution (#511, ``docs/source/architecture.md``): the operations of the
    semantics (filter, update, marginal, probabilities, expectations, normalisation, approximation and the
    comparisons). It does not mention generating functions; their shared logic is
    :class:`prodigy.distribution.pgf.PGF`, and the engines implement it.
    """

    # declared to be a non-negative measure with finite mass (:meth:`declare_finite_measure`, #398)
    _declared_finite_measure: bool = False

    def declare_finite_measure(self) -> Distribution:
        """
        Declares that this object is a power series with non-negative coefficients and finite mass, e.g. an
        output of :func:`~prodigy.analysis.session.compute_discrete_distribution` (from a validated input), and
        returns it. Its evaluations at 1 (:meth:`marginal`, :meth:`get_probability_mass`,
        :meth:`get_probability_of`, :meth:`get_expected_value_of`) then skip the convergence proof, as inside an
        analysis (:func:`known_convergent_series`). The declaration belongs to this object only: distributions
        computed from it are not declared.
        """
        self._declared_finite_measure = True
        return self

    @staticmethod
    @abstractmethod
    def factory() -> Type[CommonDistributionsFactory]:
        """Returns a factory for this subclass of distribution"""

    @abstractmethod
    def __add__(self, other: Operand) -> Distribution:
        """ The addition of two distributions. """

    @abstractmethod
    def __sub__(self, other: Operand) -> Distribution:
        """ The subtraction of two distributions. """

    @abstractmethod
    def __mul__(self, other: Operand) -> Distribution:
        """ The multiplication of two distributions. """

    @abstractmethod
    def __truediv__(self, other: Operand) -> Distribution:
        """ The division of two distributions. """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Checks whether two distributions are equal. True must be backed by a proof; False only means
        "not shown equal" (see :meth:`decide_equal`).
        """

    @abstractmethod
    def decide_equal(self, other: object) -> Optional[bool]:
        """
        Three-valued equality: True if ``self == other`` is proven, False if ``self != other`` is proven
        (e.g. the difference is a nonzero rational function with rational coefficients), None if neither.

        The default takes True from ``==`` and otherwise decides on the string forms with SymPy's exact
        cancellation (:func:`prodigy.distribution.pgf.rules.equality.decide_equal_strings`).
        """

    def _engine_text(self) -> str:
        """
        The engine's expression as text that its ``from_expr`` reads back, independent of the printing flags
        (``str`` of a SymPy generating function is LaTeX while ``latex_output`` is active, #445).
        """
        return engine_text_of(self)

    @abstractmethod
    def decide_le_verdict(self, other: Distribution) -> OrderVerdict:
        """
        Three-valued ``self <= other`` (every coefficient of ``other - self`` is non-negative) with a witness
        state for a refutation; see :mod:`prodigy.distribution.pgf.rules.ordering`.

        :raises TypeError: if `other` is of another engine or has other variables or parameters.
        """

    def decide_le(self, other: Distribution) -> Optional[bool]:
        """
        Three-valued ``self <= other``: True if proven, False if a coefficient of ``other - self`` is proven
        negative, None if neither (see :meth:`decide_le_verdict`).
        """
        return self.decide_le_verdict(other).holds

    def decide_lt(self, other: Distribution) -> Optional[bool]:
        """Three-valued ``self < other``: ``self <= other`` and ``self`` not equal to `other`."""
        less_equal = self.decide_le(other)
        if less_equal is not True:
            return less_equal
        equal = self.decide_equal(other)
        return None if equal is None else not equal

    def __le__(self, other: Distribution) -> bool:
        """
        Whether every coefficient of ``other - self`` is non-negative. The answer is proven either way.

        :raises TypeError: if `other` is of another engine or has other variables or parameters.
        :raises UndecidedComparisonError: if neither ``self <= other`` nor its negation is proven.
        """
        return self._decided(self.decide_le(other), "<=", other)

    def __lt__(self, other: Distribution) -> bool:
        """``self <= other`` and not equal; proven either way (see :meth:`__le__`)."""
        return self._decided(self.decide_lt(other), "<", other)

    def _decided(self, verdict: Optional[bool], relation: str, other: Distribution) -> bool:
        if verdict is None:
            raise UndecidedComparisonError(
                f"Cannot decide whether {self} {relation} {other}: the coefficients of their difference are "
                "neither proven non-negative nor is one proven negative.")
        return verdict

    @abstractmethod
    def __str__(self) -> str:
        """ The string representation of a distribution. """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, State]]:
        """
        Returns an iterator that iterates over the (probability, state) pairs of the distribution.

        Contract (all engines, #444): each state occurs at most once, with its full coefficient (the sum of
        all terms of the generating function with this monomial, also when the coefficient contains
        parameters, e.g. `1-p` for x = 0 in `p*x + (1-p)`). A finite distribution yields no state whose
        coefficient is zero; an infinite one may (it enumerates the monomials). Callers (case splits,
        filters, the result tables) take the yielded probability as Pr[state] and may rely on this.

        A finite distribution yields its states in the order of :func:`canonical_order` on all engines
        (#361); GiNaC's order of terms changed between runs (#418).
        """

    @abstractmethod
    def copy(self, deep: bool = True) -> Distribution:
        """ Returns a full copy of itself."""

    def native_coefficients(self, variable: str) -> Optional[Iterator[str]]:
        """
        Engine hook of :func:`prodigy.distribution.pgf.coefficient_stream.marginal_stream` (#405): the
        coefficients Pr[variable = 0], Pr[variable = 1], ... of this univariate distribution with infinite
        support, computed by the engine itself one at a time, or None (the default), in which case the shared
        logic expands the generating function as a SymPy expression.
        """
        # pylint: disable=unused-argument
        return None

    @abstractmethod
    def get_probability_of(self, condition: Union[Expr, str]) -> str:
        """
        Returns the probability of a given `condition` or variable.

        :param condition: The condition, as an AST or in the ``.ppy`` syntax.

        :return: The probability that the condition is satisfied.
        """

    @abstractmethod
    def query_marginal(self, expression: Expr, error: NotImplementedError) -> Distribution:
        """
        The marginal of the variables of `expression` for a query (`Pr`, `Ex`) that could not be computed on
        this distribution with `error`: the query only depends on these variables, and on the marginal
        e.g. `x < y` may be computable when it is not on the joint distribution (x and y independent,
        but dependent given another variable, as after `z := x - y`). Raises `error` if the marginal has
        all variables of this distribution or is not a finite measure.
        """

    @abstractmethod
    def get_probability_mass(self) -> str:
        """
        Returns the probability mass of the distribution, the closed form at 1. Outside an analysis
        (:func:`known_convergent_series`), raises `NonFiniteMarginalError` if it is not proven that this
        value is the sum of the series (#398).
        """

    @abstractmethod
    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        """
        The expected value of `expression`. Floor divisions and subtractions of natural numbers
        (truncated at 0) are computed into fresh variables first, exactly as an assignment of them
        (:meth:`without_computed_terms`), so every engine gives the same value and no engine
        simplifies `n - 7 + 7` to `n` before the subtraction is truncated.

        The moments are derivatives of the closed form at 1. Outside an analysis
        (:func:`known_convergent_series`), raises `NonFiniteMarginalError` if it is not proven that the
        series converges there (#398); then the derivatives of the closed form are those of the series.
        """

    @abstractmethod
    def normalize(self) -> Distribution:
        """
        Normalizes the probability mass of the distribution: divides by the closed form at 1. Outside an
        analysis (:func:`known_convergent_series`), raises `NonFiniteMarginalError` if it is not proven that
        this value is the sum of the series (#427).
        """

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Returns the free variables of the distribution. """

    @abstractmethod
    def get_parameters(self) -> Set[str]:
        """ Returns the parameters of the distribution. """

    @abstractmethod
    def filter_state(self, state: State) -> Distribution:
        """
        Filters the distribution such that only the specified state is left. If the state does not contain
        assignments for all variables of the distribution, the resulting distribution contains all extensions of the
        state.
        """

    @abstractmethod
    def filter(self, condition: Expr) -> Distribution:
        """ Filters the distribution such that only the parts which satisfy the `condition` are left."""

    @abstractmethod
    def without_computed_terms(self, expression: Expr) -> Tuple[Distribution, Expr, List[str]]:
        """
        Replaces every floor division `e // q` and every subtraction `a - b` of natural-number
        expressions (monus) in `expression` by its value: a literal if the term is constant, else a
        fresh variable `t` with the update `t := term` applied to this distribution. So guards and
        queries (`Pr[...]`, `Ex[...]`) evaluate these terms exactly as an assignment does, on every
        engine, and no engine simplifies e.g. `(n - 7) + 7` to `n` before the subtraction is
        truncated. Returns the extended distribution, the expression without these terms and the
        fresh variables (each a function of the state, so they can be marginalised out afterwards).
        """

    @abstractmethod
    def hadamard_product(self, other: Distribution) -> Distribution:
        """
        Computes the Hadamard product of the distributions, i.e., the point-wise product of the probabilities.
        """

    @abstractmethod
    def get_symbols(self) -> Set[str]:
        """
        Returns all symbols that occur in this distribution (a subset of `self.get_variables() |
        self.get_parameters()`).
        """

    @staticmethod
    @abstractmethod
    def evaluate(expression: str, state: State) -> Value:
        """ Evaluates the expression in a given state. """

    @classmethod
    @abstractmethod
    def evaluate_condition(cls, condition: Expr,
                           state: State) -> bool:
        """Evaluates whether the condition holds in the specified state."""

    @abstractmethod
    def is_zero_dist(self) -> bool:
        """ Returns whether the distribution encodes the 0 distribution."""

    def is_provably_zero(self) -> bool:
        """
        Whether this is provably the zero measure: :meth:`is_zero_dist`, or equal to it by the engine's
        equality (which never claims an equality without proof). False means "not shown zero".

        A case split over :meth:`finite_support` may meet a value whose coefficient is zero but not
        simplified to 0; an update that fails on such a part (e.g. a divisor 0 in a state that does
        not occur) is skipped instead of raising (#256).
        """
        try:
            return self.is_zero_dist() or self == self * "0"
        except ValueError:
            return False

    @abstractmethod
    def is_finite(self) -> bool:
        """ Returns whether the distribution has finite support."""

    @abstractmethod
    def update(self,
               expression: Expr,
               approximate: str | float | None = None) -> Distribution:
        """ Updates the current distribution by applying the expression to itself.

            Subtraction is truncated at 0 (monus, `a - b = max(a - b, 0)`) in every intermediate result, e.g.
            `(n - 7) + 7` is 7 for `n = 5` (see :meth:`_update_monus`).

            Some operations are illegal and will cause this function to raise an error. These operations include
            division that may cause a variable to have a value that is not an integer, and certain operations on
            infinite generating functions if the variables involved have an infinite marginal (such as multiplication
            of two variables) and approximation is disabled.

            Every intermediate result that depends on a variable must be a natural number. A constant that is not a
            natural number may therefore only be a factor or a divisor: `p/q * e` is evaluated as `(p * e) / q` and
            is subject to the same divisibility check as a division, which is exact for infinite marginals of rational
            generating functions as well. Any other use, such as `x + 1/2`, raises an error.

            The `approximate` parameter is used to determine up to which precision unilateral approximation should be
            performed if an update is not possible on infinite distributions.

            Parameters are not allowed in an update expression.
        """

    @abstractmethod
    def finite_support(self, *variables: Union[str, VarExpr]) -> Optional[Distribution]:
        """
        The values of `variables` in the support of this measure as a finite distribution over
        `variables` (each of its states is such a value), or None if the variables take infinitely
        many values (or finiteness is not decided).

        Unlike :meth:`marginal`, this sums nothing out: the other variables stay symbols (parameters)
        in the coefficients, so it is defined for measures with infinite mass as well (#249). The
        coefficient of a value of `variables` is a power series in the other variables, which is zero
        if and only if no state of the measure has this value (for signed measures as well). The
        coefficients are therefore not probabilities; only the states are meaningful. A coefficient
        that the engine does not simplify to zero adds a state without mass, which is harmless for a
        case split over the states.
        """

    @abstractmethod
    def get_fresh_variable(
            self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        """
        Returns a str that is the name of neither an existing variable nor an existing
        parameter of this distribution nor contained in the `exclude` parameter.
        """

    @abstractmethod
    def update_iid(self, sampling_dist: Expr, count: Union[VarExpr, NatLitExpr],
                   variable: Union[str, VarExpr]) -> Distribution:
        """
        Updates the distribution by the iid-sampling rules: `variable := iid(sampling_dist, count)`.

        The old value of `variable` is summed out (set to 1), or, if `count` is `variable`, composed with the
        generating function of `sampling_dist`, which converges where the series in `variable` converges
        absolutely at 1. Outside an analysis (:func:`known_convergent_series`), raises `NonFiniteMarginalError`
        if this is not proven (#427).
        """

    @abstractmethod
    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> Distribution:
        """
        Computes the marginal distribution for the given variables (MarginalType.Include),
        or for all but the given variables (MarginalType.Exclude).

        Outside an analysis (:func:`known_convergent_series`), raises `NonFiniteMarginalError` if it is
        not proven that the closed form at 1 in the summed-out variables is the sum of the series (#398).

        :param variables: A list of variables for which the marginal distribution should be computed. If this list is
            empty or contains symbols that are not known variables of this distribution, this function will raise an
            exception.

        :param method: The method of marginalization.

        :return: The marginal distribution.
        """

    @abstractmethod
    def checked_marginal(self, *variables: Union[str, VarExpr], proof: bool = True) -> Distribution:
        """
        The marginal of `variables` (MarginalType.INCLUDE) for code that unfolds it into its states,
        e.g. a filter or update that does a case split over the values of these variables.

        Raises `NonFiniteMarginalError` if the marginal is not a finite measure (infinite mass, see
        :class:`NonFiniteMarginalError`): then its terms do not tell which values the variables take,
        and unfolding it would give a wrong result (#242). With `proof` (the default), also if it is
        not proven that the closed form is the sum of the series (#393); without, only a non-finite
        value is detected, for callers that use the marginal only to choose their error.
        """

    def has_non_finite_value(self) -> bool:
        """
        Whether this function contains a non-finite value (`nan`, ∞, complex infinity) or an
        unevaluated limit. Engines override this with a check on their expression; this default scans
        the string form (program variables and parameters are never taken for such a value).
        """
        tokens = set(re.findall(r"[A-Za-z_]\w*", engine_text_of(self)))
        names = self.get_variables() | self.get_parameters()
        return "∞" in engine_text_of(self) or bool((tokens & _NON_FINITE_NAMES) - names)

    @abstractmethod
    def set_variables(self, *variables: str) -> Self:
        """
        The distribution with the variables `variables`; the receiver is not changed.

        The same on all engines (#361): every name that occurs in the generating function stays a
        variable or a parameter. A parameter named in `variables` becomes a variable. A variable that
        is not named becomes a parameter if it occurs, and is dropped otherwise. No variables at all is
        allowed (the marginal that sums out every variable has none).
        """

    @abstractmethod
    def set_parameters(self, *parameters: str) -> Self:
        """
        The distribution with the parameters `parameters`; the receiver is not changed.

        The same on all engines (#361): raises `ValueError` if a name is a variable. A parameter
        that is not named becomes a variable if it occurs, and is dropped otherwise.
        """

    @abstractmethod
    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        """
        The partial sums of the series until their probability mass reaches `threshold`, a number in engine text
        (might not terminate for a threshold equal to the mass of an infinite distribution). A count of states is
        refused (`TypeError`), see :meth:`~prodigy.distribution.pgf.base.PGFBase.approximate`.

        :param threshold: The probability mass to reach.

        :return: The approximated (truncated) probability distributions, one more state each.
        """

    @abstractmethod
    def approximate_unilaterally(
            self, variable: str,
            probability_mass: str | float) -> Distribution:
        """
        Approximates the distribution in one variable via its series expansion, up until the specified probability
        mass is reached.
        """

    def get_state(self) -> Tuple[State, str]:
        """Returns a state of this distribution that has probability > 0, and its probability"""

        for prob, state in self:
            if prob != '0':
                return state, prob

        raise ValueError("There is no state with probability > 0")


class CommonDistributionsFactory(ABC):
    """ Abstract Factory Class implementing a Factory for common distributions."""

    @staticmethod
    @abstractmethod
    def geometric(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        """ A geometric distribution with success probability `p`, i.e., `p + p*(1-p)*x + p*(1-p)^2*x^2 + ...`."""

    @staticmethod
    @abstractmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> Distribution:
        """ A uniform distribution with bounds [`lower`,`upper`]."""

    @staticmethod
    @abstractmethod
    def bernoulli(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        """ A bernoulli distribution with parameter `p`."""

    @staticmethod
    @abstractmethod
    def poisson(var: Union[str, VarExpr],
                lam: DistributionParam) -> Distribution:
        """ A poisson distribution with parameter `lam`."""

    @staticmethod
    @abstractmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        """ A logarithmic distribution with parameter `p`."""

    @staticmethod
    @abstractmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> Distribution:
        """ A binomial distribution with parameters `n` and `p`."""

    @staticmethod
    @abstractmethod
    def negative_binomial(var: Union[str, VarExpr], r: DistributionParam,
                          p: DistributionParam) -> Distribution:
        """ A negative binomial distribution: the number of failures before the `r`-th success with success
        probability `p`, i.e., `(p / (1 - (1-p)*x))^r`. It is the sum of `r` independent samples of
        :meth:`geometric` with parameter `p`."""

    @staticmethod
    @abstractmethod
    def zero(*variables: Union[str, VarExpr]) -> Distribution:
        """ The zero distribution over `variables`: no mass on any state.

        This is the result of `abort`, where all mass goes to non-termination."""

    @staticmethod
    @abstractmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where all variables are initialized with 0."""

    @staticmethod
    @abstractmethod
    def from_expr(expression: Union[str, Expr], *variables: Union[str, VarExpr],
                  **kwargs: Optional[bool]) -> Distribution:
        """ A distribution represented by the expression."""


__all__ = ["CommonDistributionsFactory", "Distribution", "DistributionParam", "MarginalType", "NonFiniteMarginalError",
           "NotAnIntegerError", "Operand", "OrderVerdict", "State", "UndecidedComparisonError", "Value",
           "ZeroDivisorError", "canonical_order", "constant_value", "known_convergent_series"]
