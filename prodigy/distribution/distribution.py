from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (Dict, FrozenSet, Generator, Iterator, List, Sequence, Set,
                    Tuple, Type, Union)

from probably.pgcl import (Binop, BinopExpr, BoolLitExpr, Expr, NatLitExpr,
                           RealLitExpr, Unop, UnopExpr, VarExpr)
from probably.pgcl.parser import parse_expr

from prodigy.pgcl.pgcl_checks import (check_is_constant_constraint,
                                      check_is_modulus_condition, has_variable)
from prodigy.pgcl.pgcl_operations import state_to_equality_expression


class MarginalType(Enum):
    """ Specifies the type of marginalization. """
    INCLUDE = auto()
    EXCLUDE = auto()


DistributionParam = Union[str, Expr]


@dataclass
class State:
    """Describes a state of a distribution, i.e., an assignment of variables to values"""

    valuations: Dict[str, int] = field(default_factory=dict)
    """The variable assignment of this state"""
    def __iter__(self):
        return self.valuations.__iter__()

    def __getitem__(self, item):
        return self.valuations.__getitem__(item)

    def __setitem__(self, key, value):
        return self.valuations.__setitem__(key, value)

    def __str__(self):
        return self.valuations.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return self.valuations == other
        if isinstance(other, State):
            return self.valuations == other.valuations
        return False

    def items(self):
        """Provides a view on all (variable, value) pairs"""
        return self.valuations.items()

    def to_monomial(self) -> str:
        """
        Provides a string representation of this state in the form of a polynomial. If the state is empty, returns an
        empty string.

        For example, the variable assignment `{x: 3, y: 6}` gives rise to the string `x^3*y^6`.
        """
        if self.valuations:
            result = "*"
            addends = []
            for variable, value in self.valuations.items():
                addends.append(f"{variable}^{value}")
            return result.join(addends)
        return ""

    def copy(self):
        return State(valuations=self.valuations.copy())


class Distribution(ABC):
    """ Abstract class that models different representations of probability distributions. """
    @staticmethod
    @abstractmethod
    def factory() -> Type[CommonDistributionsFactory]:
        """Returns a factory for this subclass of distribution"""

    @abstractmethod
    def __add__(self, other) -> Distribution:
        """ The addition of two distributions. """

    @abstractmethod
    def __sub__(self, other) -> Distribution:
        """ The subtraction of two distributions. """

    @abstractmethod
    def __mul__(self, other) -> Distribution:
        """ The multiplication of two distributions. """

    @abstractmethod
    def __truediv__(self, other) -> Distribution:
        """ The division of two distributions. """

    @abstractmethod
    def __eq__(self, other) -> bool:
        """ Checks whether two distributions are equal. """

    @abstractmethod
    def __le__(self, other) -> bool:
        """ Checks whether `self` is less or equal than `other`."""

    def __lt__(self, other) -> bool:
        return self <= other and not self == other

    @abstractmethod
    def __str__(self) -> str:
        """ The string representation of a distribution. """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, State]]:
        """ Returns an iterator that iterates over the (probability, state) pairs of the distribution."""

    @abstractmethod
    def copy(self, deep: bool = True) -> Distribution:
        """ Returns a full copy of itself."""

    def get_probability_of(self, condition: Union[Expr, str]):
        """
        Returns the probability of a given `condition` or variable.

        :param condition: The condition.

        :return: The probability that the condition is satisfied.
        """

        expr = condition
        if isinstance(expr, str):
            expr = parse_expr(expr)
        return self.filter(expr).get_probability_mass()

    @abstractmethod
    def get_probability_mass(self) -> Union[Expr, str]:
        """ Returns the probability mass of the distribution. """

    @abstractmethod
    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        """ Returns the expected value of the expression `expression` evaluated in the distribution. """

    @abstractmethod
    def normalize(self) -> Distribution:
        """ Normalizes the probability mass of the distribution."""

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Returns the free variables of the distribution. """

    @abstractmethod
    def get_parameters(self) -> Set[str]:
        """ Returns the parameters of the distribution. """

    def filter_state(self, state: State) -> Distribution:
        """
        Filters the distribution such that only the specified state is left. If the state does not contain
        assignments for all variables of the distribution, the resulting distribution contains all extensions of the
        state.
        """

        if not state.valuations.keys() <= self.get_variables():
            raise ValueError("Unknown variable in state")

        res = self
        for var, val in state.items():
            #pylint: disable=protected-access
            # this accesses the same type which is fine
            res = res._filter_constant_condition(
                BinopExpr(Binop.EQ, lhs=VarExpr(var), rhs=NatLitExpr(val)))
            #pylint: enable=protected-access

        return res

    def filter(self, condition: Expr) -> Distribution:
        """ Filters the distribution such that only the parts which satisfy the `condition` are left."""
        # Boolean literals
        if isinstance(condition, BoolLitExpr):
            return self if condition.value else self.factory().from_expr(
                "0", *self.get_variables())

        if isinstance(condition, BinopExpr):
            if condition.operator == Binop.AND:
                return self.filter(condition.lhs).filter(condition.rhs)
            if condition.operator == Binop.OR:
                filtered_left = self.filter(condition.lhs)
                return filtered_left + self.filter(
                    condition.rhs) - filtered_left.filter(condition.rhs)

        if isinstance(condition, UnopExpr):
            # unary relation
            if condition.operator == Unop.NEG:
                return self - self.filter(condition.expr)
            raise SyntaxError(
                f"We do not support filtering for {type(Unop.IVERSON)} expressions."
            )

        if isinstance(condition,
                      BinopExpr) and not has_variable(condition, None):
            return self.filter(
                BoolLitExpr(self.evaluate_condition(condition, State())))

        if isinstance(condition, BinopExpr) and not self._find_symbols(
                str(condition.lhs)) | self._find_symbols(str(
                    condition.rhs)) <= (self.get_variables()
                                        | self.get_parameters()):
            raise ValueError(
                f"Cannot filter based on the expression {str(condition)} because it contains unknown variables"
            )

        # Modulo extractions
        if check_is_modulus_condition(condition):
            return self._arithmetic_progression(
                str(condition.lhs.lhs),
                str(condition.lhs.rhs))[condition.rhs.value]

        # Constant expressions
        if check_is_constant_constraint(condition, self.get_parameters()):
            return self._filter_constant_condition(condition)

        # all other conditions given that the Generating Function is finite (exhaustive search)
        if self.is_finite():
            return self._exhaustive_search(condition)

        # Worst case: infinite Generating function and  non-standard condition.
        # Here we try marginalization and hope that the marginal is finite so we can do
        # exhaustive search again. If this is not possible, we raise an NotComputableException
        expression = self._explicit_state_unfolding(condition)
        return self.filter(expression)

    @abstractmethod
    def _exhaustive_search(self, condition: Expr) -> Distribution:
        """
        Given that `self` is finite, iterates over `self` and returns the sum of all terms where
        the condition holds.
        """

    @abstractmethod
    def _filter_constant_condition(self, condition: Expr) -> Distribution:
        """
        Filters out the terms that satisfy a constant condition, e.g., (var <= 5) (var > 5) (var = 5).

        :param condition: The condition to filter.

        :return: The filtered distribution.
        """

    def _explicit_state_unfolding(self, condition: Expr) -> BinopExpr:
        """
        Checks whether one side of the condition has only finitely many valuations and explicitly creates a new
        condition which is the disjunction of each individual evaluations.

        :param condition: The condition to unfold.

        :return: The disjunction condition of explicitly encoded state conditions.
        """
        expr: str = str(condition.rhs)
        syms = self._find_symbols(expr)
        if not len(syms) == 0:
            marginal = self.marginal(*syms)

        # Marker to express which side of the equation has only finitely many interpretations.
        left_side_original = True

        # Check whether left hand side has only finitely many interpretations.
        if len(syms) == 0 or not marginal.is_finite():
            # Failed, so we have to check the right hand side
            left_side_original = False
            expr = str(condition.lhs)
            marginal = self.marginal(*self._find_symbols(expr))

            if not marginal.is_finite():
                # We are not able to marginalize into a finite amount of states! -> FAIL filtering.
                raise NotImplementedError(
                    f"Instruction {condition} is not computable on infinite generating function"
                    f" {self}")
        assert isinstance(marginal, Distribution)

        # Now we know that `expr` can be instantiated with finitely many states.
        # We generate these explicit state.
        state_expressions: List[BinopExpr] = []

        # Compute for all states the explicit condition checking that specific valuation.
        for _, state in marginal:

            # Evaluate the current expression
            evaluated_expr = self.evaluate(expr, state)

            # create the equalities for each variable, value pair in a given state
            # i.e., {x:1, y:0, c:3} -> [x=1, y=0, c=3]
            encoded_state = state_to_equality_expression(state.valuations)

            # Create the equality which assigns the original side the anticipated value.
            other_side_expr = BinopExpr(condition.operator, condition.lhs,
                                        NatLitExpr(int(evaluated_expr)))
            if not left_side_original:
                other_side_expr = BinopExpr(condition.operator,
                                            NatLitExpr(int(evaluated_expr)),
                                            condition.rhs)

            state_expressions.append(
                BinopExpr(Binop.AND, encoded_state, other_side_expr))

        # Get all individual conditions and make one big disjunction.
        return functools.reduce(
            lambda left, right: BinopExpr(
                operator=Binop.OR, lhs=left, rhs=right), state_expressions)

    @abstractmethod
    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> Sequence[Distribution]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """

    @staticmethod
    @abstractmethod
    def _find_symbols(expr: str) -> Set[str]:
        "Returns a set of all free symbols in the given expression."

    def get_symbols(self) -> Set[str]:
        """
        Returns all symbols that occur in this distribution (a subset of `self.get_variables() |
        self.get_parameters()`).
        """
        return self._find_symbols(str(self))

    @staticmethod
    @abstractmethod
    def evaluate(expression: str, state: State):
        """ Evaluates the expression in a given state. """

    @classmethod
    def evaluate_condition(cls, condition: BinopExpr, state: State) -> bool:
        """Evaluates whether the condition holds in the specified state."""

        if not isinstance(condition, BinopExpr):
            raise AssertionError(
                f"Expression must be an (in-)equation, was {condition}")

        lhs = str(condition.lhs)
        rhs = str(condition.rhs)
        op = condition.operator

        if op == Binop.EQ:
            return cls.evaluate(lhs, state) == cls.evaluate(rhs, state)
        elif op == Binop.LEQ:
            return cls.evaluate(lhs, state) <= cls.evaluate(rhs, state)
        elif op == Binop.LT:
            return cls.evaluate(lhs, state) < cls.evaluate(rhs, state)
        elif op == Binop.GT:
            return cls.evaluate(lhs, state) > cls.evaluate(rhs, state)
        elif op == Binop.GEQ:
            return cls.evaluate(lhs, state) >= cls.evaluate(rhs, state)
        raise AssertionError(f"Unexpected condition type. {condition}")

    def evaluate_expression(self,
                            expression: Expr,
                            new_var: str | None = None) -> Distribution:
        """
        Creates a new distribution with a single variable that represents all
        possible results of the given expression.
        """

        if new_var is None:
            new_var = self.get_fresh_variable()
        return self.set_variables(*self.get_variables(), new_var).update(
            BinopExpr(Binop.EQ, VarExpr(new_var),
                      expression)).marginal(new_var)

    @abstractmethod
    def is_zero_dist(self) -> bool:
        """ Returns whether the distribution encodes the 0 distribution."""

    @abstractmethod
    def is_finite(self) -> bool:
        """ Returns whether the distribution has finite support."""

    def update(self,
               expression: Expr,
               approximate: str | float | None = None) -> Distribution:
        """ Updates the current distribution by applying the expression to itself.

            Some operations are illegal and will cause this function to raise an error. These operations include
            subtraction that may cause a variable to have a negative value, division that may cause a variable to have a
            value that is not an integer, and certain operations on infinite generating functions if the variables
            involved have an infinite marginal (such as multiplication of two variables) and approximation is disabled.

            The `approximate` parameter is used to determine up to which precision unilateral approximation should be
            performed if an update is not possible on infinite distributions.

            Parameters are not allowed in an update expression.
        """

        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."

        variable = expression.lhs.var
        if variable not in self.get_variables():
            raise ValueError(
                f"Cannot assign to variable {variable} because it does not exist"
            )

        # pylint: disable=protected-access
        def evaluate(function: Distribution, expression: Expr,
                     temp_var: str) -> Tuple[Distribution, str]:
            # TODO handle reals in every case
            if isinstance(expression, BinopExpr):
                xl = function.get_fresh_variable()
                xr = function.get_fresh_variable({xl})
                f = function.set_variables(*(function.get_variables()
                                             | {xl, xr}))
                f, t_1 = evaluate(f, expression.lhs, xl)
                f, t_2 = evaluate(f, expression.rhs, xr)
                if expression.operator == Binop.PLUS:
                    f = f._update_sum(temp_var, t_1, t_2)
                elif expression.operator == Binop.TIMES:
                    f = f._update_product(temp_var, t_1, t_2, approximate)
                elif expression.operator == Binop.MINUS:
                    f = f._update_subtraction(temp_var, t_1, t_2)
                elif expression.operator == Binop.MODULO:
                    f = f._update_modulo(temp_var, t_1, t_2, approximate)
                elif expression.operator == Binop.DIVIDE:
                    f = f._update_division(temp_var, t_1, t_2, approximate)
                elif expression.operator == Binop.POWER:
                    f = f._update_power(temp_var, t_1, t_2, approximate)
                else:
                    raise ValueError(
                        f"Unsupported binary operator: {expression.operator}")

                f = f.marginal(xl, xr, method=MarginalType.EXCLUDE)
                return f, temp_var

            if isinstance(expression, VarExpr):
                f = function._update_var(temp_var, expression.var)
                return f, temp_var

            if isinstance(expression, (NatLitExpr, RealLitExpr)):
                return function, str(expression.value)

            else:
                raise ValueError(
                    f"Unsupported type of subexpression: {expression}")

        # pylint: enable=protected-access

        value: int | None = None
        if isinstance(expression.rhs, RealLitExpr):
            if expression.rhs.to_fraction().denominator == 1:
                value = expression.rhs.to_fraction().numerator
            else:
                raise ValueError(
                    f'Cannot assign the real value {str(expression.rhs)} to {variable}'
                )
        if isinstance(expression.rhs, NatLitExpr):
            value = expression.rhs.value
        if value is not None:
            result = self._update_var(variable, value)
        else:
            result, _ = evaluate(self, expression.rhs, variable)
        return result

    @abstractmethod
    def get_fresh_variable(
        self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        """
        Returns a str that is the name of neither an existing variable nor an existing
        parameter of this distribution nor contained in the `exclude` parameter.
        """

    @abstractmethod
    def _update_var(self, updated_var: str,
                    assign_var: str | int) -> Distribution:
        """
        Applies the update `updated_var = assign_var` to this distribution.
        `assign_var` may be a variable or integer literal.
        """

    @abstractmethod
    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> Distribution:
        """
        Applies the expression `temp_var = fist_summand + second_summand` to this distribution.
        """

    @abstractmethod
    def _update_product(self, temp_var: str, first_factor: str,
                        second_factor: str,
                        approximate: str | float | None) -> Distribution:
        """
        Applies the update `temp_var = first_factor * second_factor` to this distribution.

        If the distribution is infinite and both factors are variables, multiplication is
        only supported if at least one of the factors has finite range (i.e., a finite marginal)
        if approximation is disabled.
        """

    @abstractmethod
    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> Distribution:
        """
        Applies the expression `temp_var = sub_from - sub` to this distribution. If this
        difference might be negative, this function will raise an error.
        """

    @abstractmethod
    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> Distribution:
        """
        Applies the expression `temp_var = left % right` to this distribution. If `self` is
        infinite, `right` must be a literal or a variable with finite range if approximation is disabled.
        """

    @abstractmethod
    def _update_division(self, temp_var: str, numerator: str | int,
                         denominator: str | int,
                         approximate: str | float | None) -> Distribution:
        """
        Applies the expression `temp_var = numerator / denominator` to this distribution.
        If in some state of the GF, the numerator is not divisible by the denominator, this function
        raises an error.

        Infinite distributions are only supported if both sides of the division have finite range
        (i.e., they are either literals or have a finite marginal) if approximation is disabled.
        """

    @abstractmethod
    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> Distribution:
        """
        Applies the expression `temp_var := base^exp` to this distribution.

        All variables occuring in the expression must have a finite marginal if approximation is disabled.
        """

    @abstractmethod
    def update_iid(self, sampling_dist: Expr, count: VarExpr,
                   variable: Union[str, VarExpr]) -> 'Distribution':
        """ Updates the distribution by the the iid-sampling rules. """

    @abstractmethod
    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> Distribution:
        """
        Computes the marginal distribution for the given variables (MarginalType.Include),
        or for all but the given variables (MarginalType.Exclude).

        :param variables: A list of variables for which the marginal distribution should be computed. If this list is
            empty or contains symbols that are not known variables of this distribution, this function will raise an
            exception.

        :param method: The method of marginalization.

        :return: The marginal distribution.
        """

    @abstractmethod
    def set_variables(self, *variables: str) -> Distribution:
        """
        Sets the free variables in a distribution.

        :param variables: The variables.

        :return: The distribution with free variables `variables`.
        """

    @abstractmethod
    def set_parameters(self, *parameters: str) -> Distribution:
        """
        Sets the parameters in a distribution.

        :param parameters: The parameters.

        :return: The distribution with parameters `parameters`.
        """

    @abstractmethod
    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        """
        Computes the approximation until the given threshold is reached (might not terminate).

        :param threshold: The threshold either as a maximum number of states (str), or a certain probability mass (int).

        :return: The approximated (truncated) probability distribution.
        """

    @abstractmethod
    def approximate_unilaterally(
            self, variable: str,
            probability_mass: str | float) -> Distribution:
        """
        Approximates the distribution in one variable via its series expansion, up until the specified probability
        mass is reached.
        """

    def approximate_until_finite(
            self, probability_mass: str | float) -> Distribution:
        """Unilaterally approximates this distribution in only the variables that have an infinite marginal."""

        res = self
        for var in self.get_variables():
            if not res.marginal(var).is_finite():
                res = res.approximate_unilaterally(var, probability_mass)
        return res

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
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""

    @staticmethod
    @abstractmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where all variables are initialized with 0."""

    @staticmethod
    @abstractmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> Distribution:
        """ A distribution represented by the expression."""
