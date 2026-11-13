from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import (Any, Callable, Dict, FrozenSet, Generator, Iterable, Iterator, List, Optional,
                    Sequence, Set, Tuple, Type, Union)

from lark.exceptions import LarkError

from prodigy.pgcl import (Binop, BinopExpr, BoolLitExpr, Expr, NatLitExpr,
                           RealLitExpr, Unop, UnopExpr, VarExpr)
from prodigy.pgcl.ast.walk import Walk, walk_expr
from prodigy.pgcl.parser import parse_expr
from prodigy.pgcl.util.ref import Mut
from prodigy.distribution.equality import decide_equal_strings
from prodigy.distribution.algebraic_comparison import filter_difference_algebraic
from prodigy.distribution.independent_comparison import (DEPENDENT, NOT_A_DIFFERENCE, ComparisonNotImplemented,
                                                          filter_difference)
from prodigy.distribution.linear_guards import difference_comparison, normalize_guard

from prodigy.pgcl.pgcl_checks import (check_is_constant_constraint,
                                      has_variable, modulus_condition)
from prodigy.pgcl.pgcl_operations import state_to_equality_expression


def constant_value(expression: Expr) -> Fraction | None:
    """
    The value of an arithmetic expression without variables, or None if the expression contains
    variables, an operator other than + - * / % ^, or an intermediate result that natural-number
    arithmetic does not allow (division by zero, non-integer modulo or exponent).

    A difference of two natural numbers is truncated at 0 (monus, `3 - 5 = 0`), as `-` on `nat`
    is everywhere (see :meth:`Distribution.update`). A difference with an operand that is not a
    natural number (e.g. `1 - 1/3` in a probability) is the rational difference if it is not
    negative, and None otherwise.
    """
    if isinstance(expression, NatLitExpr):
        return Fraction(expression.value)
    if isinstance(expression, RealLitExpr):
        return expression.to_fraction()
    if not isinstance(expression, BinopExpr):
        return None
    left, right = constant_value(expression.lhs), constant_value(expression.rhs)
    if left is None or right is None:
        return None
    if expression.operator == Binop.PLUS:
        return left + right
    if expression.operator == Binop.TIMES:
        return left * right
    if expression.operator == Binop.MINUS:
        if _is_natural(left) and _is_natural(right):
            return max(left - right, Fraction(0))
        return left - right if left >= right else None
    if expression.operator == Binop.DIVIDE:
        return left / right if right != 0 else None
    if expression.operator == Binop.MODULO:
        if left.denominator == 1 and right.denominator == 1 and right != 0:
            return Fraction(left.numerator % right.numerator)
        return None
    if expression.operator == Binop.FLOOR_DIVIDE:
        if left.denominator == 1 and right.denominator == 1 and right != 0:
            return Fraction(left.numerator // right.numerator)
        return None
    if expression.operator == Binop.POWER:
        return left**right.numerator if right.denominator == 1 else None
    return None


def _is_natural(value: Fraction) -> bool:
    return value.denominator == 1 and value >= 0


def is_natural_expression(expression: Expr, variables: Set[str]) -> bool:
    """
    Whether `expression` has a natural number as value in every state: it is built from the
    variables in `variables`, natural number literals and `+ - * % // ^` (no division `/`, no
    parameters, no constants that are not natural numbers). A `-` between two such expressions is
    truncated subtraction (monus).
    """
    if isinstance(expression, NatLitExpr):
        return True
    if isinstance(expression, RealLitExpr):
        return _is_natural(expression.to_fraction())
    if isinstance(expression, VarExpr):
        return expression.var in variables
    if isinstance(expression, BinopExpr) and expression.operator in _NATURAL_OPERATORS:
        return is_natural_expression(expression.lhs, variables) and \
            is_natural_expression(expression.rhs, variables)
    return False


_NATURAL_OPERATORS = {Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.MODULO, Binop.FLOOR_DIVIDE, Binop.POWER}


def _without_fractional_constants(expression: BinopExpr) -> BinopExpr | None:
    """
    Rewrites a binary expression with an operand that is a constant but not a natural number (e.g.
    1/2 in `1/2*x`) such that every intermediate result is a natural number again, or returns None
    if neither operand is such a constant.

    A product `p/q * e` becomes `(p * e) / q` and a quotient `e / (p/q)` becomes `(q * e) / p`, so
    the division update checks that the result is a natural number in every state, exactly as for
    `e / q`. Any other operator with such an operand cannot yield a natural number in general and
    raises an error.
    """
    left, right = constant_value(expression.lhs), constant_value(expression.rhs)
    fraction, other = (left, expression.rhs) if left is not None and left.denominator != 1 \
        else (right, expression.lhs)
    if fraction is None or fraction.denominator == 1:
        return None
    numerator, denominator = NatLitExpr(fraction.numerator), NatLitExpr(fraction.denominator)
    if expression.operator == Binop.TIMES:
        return BinopExpr(Binop.DIVIDE, BinopExpr(Binop.TIMES, numerator, other), denominator)
    if expression.operator == Binop.DIVIDE and other is expression.lhs:
        return BinopExpr(Binop.DIVIDE, BinopExpr(Binop.TIMES, denominator, other), numerator)
    if expression.operator == Binop.DIVIDE:
        return BinopExpr(Binop.DIVIDE, numerator, BinopExpr(Binop.TIMES, denominator, other))
    if expression.operator == Binop.FLOOR_DIVIDE:
        raise ValueError(f"Cannot compute {expression}: floor division by or of the non-integer constant "
                         f"{fraction} is not implemented")
    raise ValueError(
        f"Cannot evaluate {expression} because the constant {fraction} is not a natural number")


def _computed_term(expression: Expr, variables: Set[str]) -> BinopExpr | None:
    """
    An innermost term of `expression` that the engines cannot evaluate themselves, or None: a
    floor division `e // q`, or a subtraction `a - b` of natural-number expressions (monus, see
    :func:`is_natural_expression`).
    """
    if isinstance(expression, UnopExpr):
        return _computed_term(expression.expr, variables)
    if not isinstance(expression, BinopExpr):
        return None
    inner = _computed_term(expression.lhs, variables) or _computed_term(expression.rhs, variables)
    if inner is not None:
        return inner
    if expression.operator == Binop.FLOOR_DIVIDE:
        return expression
    if expression.operator == Binop.MINUS and is_natural_expression(expression, variables):
        return expression
    return None


def _state_value(expression: Expr, state: State) -> Fraction | None:
    """The value of `expression` in `state` (see :func:`constant_value`), or None."""
    def assigned(expr: Expr) -> Expr:
        if isinstance(expr, VarExpr) and expr.var in state.valuations:
            return NatLitExpr(state[expr.var])
        if isinstance(expr, BinopExpr):
            return BinopExpr(expr.operator, assigned(expr.lhs), assigned(expr.rhs))
        return expr
    return constant_value(assigned(expression))


def _replace_term(expression: Expr, term: Expr, replacement: Expr) -> Expr:
    """`expression` with the subexpression `term` (by identity) replaced."""
    if expression is term:
        return replacement
    if isinstance(expression, UnopExpr):
        return UnopExpr(expression.operator, _replace_term(expression.expr, term, replacement))
    if isinstance(expression, BinopExpr):
        return BinopExpr(expression.operator, _replace_term(expression.lhs, term, replacement),
                         _replace_term(expression.rhs, term, replacement))
    return expression


def _variables_of(expression: Expr) -> Set[str]:
    """The names of the variables in `expression`."""
    return {node.val.var for node in walk_expr(Walk.DOWN, Mut.alloc(expression)) if isinstance(node.val, VarExpr)}


class NonFiniteMarginalError(ArithmeticError):
    """
    A marginal that a computation needs as a distribution over finitely or infinitely many states is
    not a finite measure, e.g. `oo`, `zoo` (complex infinity), `nan` or a division by zero. This happens
    on measures with infinite mass (candidate measures, not program distributions): summing out the
    other variables diverges, so the marginal does not describe which values the variables take.
    """


class ZeroDivisorError(ValueError):
    """
    A modulus or divisor (``%``, ``/``) is 0 in a reachable state: the update is undefined (#273). All
    engines raise this error, for a literal 0 and for a variable that takes the value 0; the update of
    an expression adds the expression to the message (``expression``).
    """

    def __init__(self, message: str, expression: Optional[str] = None):
        super().__init__(message if expression is None else f"{message} (in {expression})")
        self.reason = message
        self.expression = expression


# Spellings of non-finite values (nan, infinity, complex infinity, unevaluated limits) in the engines'
# string forms; a program variable or parameter of the same name is not one of them
_NON_FINITE_NAMES = frozenset({"nan", "NaN", "zoo", "oo", "inf", "Inf", "infinity", "Infinity",
                               "ComplexInfinity", "UnsignedInfinity", "Limit"})


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

    def __hash__(self):
        # Order-independent, consistent with __eq__. A state must not change while it is used as a key.
        return hash(frozenset(self.valuations.items()))

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

    def extend(self, other: State):
        res = self.copy()
        for var, val in other.items():
            if var not in res.valuations:
                res[var] = val
            else:
                raise ValueError(f"Variable {var} already has a value")
        return res



def _is_true(variable: VarExpr) -> Expr:
    """The condition that the ``bool`` variable ``variable`` holds: ``not (variable = 0)``."""
    return UnopExpr(Unop.NEG, BinopExpr(Binop.EQ, variable, NatLitExpr(0)))

def _spelled(expression: Expr) -> str:
    """The expression in the program's syntax if an analysis session is running (else pGCL)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import current_session, has_current_session
    return current_session().spell(expression) if has_current_session() else str(expression)


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
        """
        Checks whether two distributions are equal. True must be backed by a proof; False only means
        "not shown equal" (see :meth:`decide_equal`).
        """

    def decide_equal(self, other) -> Optional[bool]:
        """
        Three-valued equality: True if ``self == other`` is proven, False if ``self != other`` is proven
        (e.g. the difference is a nonzero rational function with rational coefficients), None if neither.

        The default takes True from ``==`` and otherwise decides on the string forms with SymPy's exact
        cancellation (:func:`prodigy.distribution.equality.decide_equal_strings`).
        """
        if self == other:
            return True
        if not isinstance(other, Distribution):
            return None
        names = self.get_variables() | self.get_parameters() | other.get_variables() | other.get_parameters()
        return decide_equal_strings(str(self), str(other), names)

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

    def get_probability_of(self, condition: Union[Expr, str]) -> str:
        """
        Returns the probability of a given `condition` or variable.

        :param condition: The condition.

        :return: The probability that the condition is satisfied.
        """

        expr = condition
        if isinstance(expr, str):
            expr = parse_expr(expr)
        try:
            return self.filter(expr).get_probability_mass()
        except NotImplementedError as error:
            marginal = self.query_marginal(expr, error)
            return marginal.filter(expr).get_probability_mass()

    def query_marginal(self, expression: Expr, error: NotImplementedError) -> Distribution:
        """
        The marginal of the variables of `expression` for a query (`Pr`, `Ex`) that could not be computed on
        this distribution with `error`: the query only depends on these variables, and on the marginal
        e.g. `x < y` may be computable when it is not on the joint distribution (x and y independent,
        but dependent given another variable, as after `z := x - y`). Raises `error` if the marginal has
        all variables of this distribution or is not a finite measure.
        """
        names = _variables_of(expression) & self.get_variables()
        if not names or names == self.get_variables():
            raise error
        try:
            return self.checked_marginal(*sorted(names))
        except NonFiniteMarginalError:
            raise error from None

    @abstractmethod
    def get_probability_mass(self) -> str:
        """ Returns the probability mass of the distribution. """

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        """
        The expected value of `expression`. Floor divisions and subtractions of natural numbers
        (truncated at 0) are computed into fresh variables first, exactly as an assignment of them
        (:meth:`without_computed_terms`), so every engine gives the same value and no engine
        simplifies `n - 7 + 7` to `n` before the subtraction is truncated.
        """
        parsed: Union[Expr, str] = expression
        if isinstance(expression, str):
            try:
                parsed = parse_expr(expression)
            except LarkError:
                # not in pGCL syntax (e.g. `x**2`): the engine reads it itself
                return self._expected_value_of(expression)
        assert not isinstance(parsed, str)
        try:
            extended, replaced, _ = self.without_computed_terms(parsed)
        except NotImplementedError as error:
            marginal = self.query_marginal(parsed, error)
            extended, replaced, _ = marginal.without_computed_terms(parsed)
        return extended._expected_value_of(replaced)  # pylint: disable=protected-access

    @abstractmethod
    def _expected_value_of(self, expression: Union[Expr, str]) -> str:
        """The expected value of `expression`, computed by the engine (see :meth:`get_expected_value_of`)."""


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
            # pylint: disable=protected-access
            # this accesses the same type which is fine
            res = res._filter_constant_condition(
                BinopExpr(Binop.EQ, lhs=VarExpr(var), rhs=NatLitExpr(val)))
            # pylint: enable=protected-access

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

        if isinstance(condition, VarExpr):
            # a bool variable as condition: it holds where the variable is 1, i.e. not 0
            return self.filter(_is_true(condition))

        if isinstance(condition, UnopExpr):
            # unary relation
            if condition.operator == Unop.NEG:
                return self - self.filter(condition.expr)
            raise SyntaxError(
                f"We do not support filtering for {condition.operator} expressions."
            )

        # Floor divisions and subtractions (monus) are computed into fresh variables first, e.g.
        # x // 2 < 3 becomes t < 3 for t := x // 2 (an exact update); t is a function of the state,
        # so it can be dropped again
        if _computed_term(condition, self.get_variables()) is not None:
            extended, replaced, fresh = self.without_computed_terms(condition)
            filtered = extended.filter(replaced)
            return filtered.marginal(*fresh, method=MarginalType.EXCLUDE) if fresh else filtered

        # Linear comparisons are evaluated exactly, e.g. 1/2*x < 2 becomes x < 4
        normalized = normalize_guard(condition, self.get_variables())
        if normalized is not None and normalized != condition:
            return self.filter(normalized)

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
        modulus = modulus_condition(condition)
        if modulus is not None:
            return self._arithmetic_progression(
                str(modulus.variable),
                str(modulus.modulus))[modulus.remainder.value]

        # Constant expressions
        if check_is_constant_constraint(condition, self.get_parameters()):
            return self._filter_constant_condition(condition)

        # all other conditions given that the Generating Function is finite (exhaustive search)
        if self.is_finite():
            return self._exhaustive_search(condition)

        # Worst case: infinite Generating function and  non-standard condition.
        # Here we try marginalization and hope that the marginal is finite so we can do
        # exhaustive search again. If this is not possible, we raise an NotComputableException
        try:
            expression = self._explicit_state_unfolding(condition)
        except NotImplementedError as error:
            # two variables with infinite support: exact if they are independent (#243)
            return self._filter_difference(condition, error)
        return self.filter(expression)

    def _filter_difference(self, condition: Expr, error: NotImplementedError) -> Distribution:
        """
        Filters a comparison of two variables such as `x < y + c` exactly if the generating
        function is a product of a rational function in x and a rational function in y (given the
        other variables, see :mod:`prodigy.distribution.independent_comparison`). Raises a
        :class:`ComparisonNotImplemented` naming the reason for every other comparison of several
        variables, and `error` (from the unfolding of finite supports) for all other conditions.
        """
        comparison = difference_comparison(condition, self.get_variables())
        if isinstance(comparison, BoolLitExpr):
            return self.filter(comparison)
        if comparison is None:
            if isinstance(condition, BinopExpr) and len(_variables_of(condition) & self.get_variables()) >= 2:
                raise ComparisonNotImplemented(str(condition), NOT_A_DIFFERENCE) from error
            raise error
        function = self._sympy_expression()
        arguments = (comparison.x, comparison.y, comparison.relation, comparison.shift)
        try:
            filtered = self._from_sympy_expression(filter_difference(function, *arguments, str(condition)))
        except ComparisonNotImplemented as not_independent:
            if not_independent.reason != DEPENDENT:
                raise
            # dependent variables: the residue method, whose result may be algebraic (not rational)
            filtered = self._from_sympy_expression(filter_difference_algebraic(
                function, *arguments, self.get_variables(), str(condition)))
        return self - filtered if comparison.complement else filtered

    @abstractmethod
    def _sympy_expression(self) -> Any:
        """This generating function as a SymPy expression (for exact computations in SymPy)."""

    @abstractmethod
    def _from_sympy_expression(self, expression: Any) -> Distribution:
        """A distribution of this engine with the variables and parameters of `self` from a SymPy
        expression."""

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
        dist: Distribution = self
        fresh: List[str] = []
        term = _computed_term(expression, self.get_variables())
        while term is not None:
            constant = constant_value(term)
            if constant is not None:
                replacement: Expr = NatLitExpr(int(constant))
            else:
                name = dist.get_fresh_variable()
                dist = dist.set_variables(*dist.get_variables(), name).update(
                    BinopExpr(Binop.EQ, VarExpr(name), term))
                fresh.append(name)
                replacement = VarExpr(name)
            expression = _replace_term(expression, term, replacement)
            term = _computed_term(expression, dist.get_variables())
        return dist, expression, fresh

    @abstractmethod
    def _exhaustive_search(self, condition: Expr) -> Distribution:
        """
        Given that `self` is finite, iterates over `self` and returns the sum of all terms where
        the condition holds.
        """

    @abstractmethod
    def _filter_constant_condition(self,
                                   condition: BinopExpr) -> Distribution:
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
        if not isinstance(condition, BinopExpr):
            raise NotImplementedError(
                f"Instruction {condition} is not computable on infinite generating function"
                f" {self}")
        # The side whose variables take finitely many values is unfolded: the right-hand side if
        # possible (`left_side_original`: the left-hand side stays in the condition), else the left.
        # Only the values of the side matter, not their mass, so its finite support is unfolded
        # (#249), not its marginal: a marginal that is not a finite measure (infinite mass) says
        # nothing about these values (e.g. `zoo`, a constant, would claim the single value 0, #242),
        # and in the marginal of a signed measure the mass of a value can cancel to 0.
        marginal: Optional[Distribution] = None
        for left_side_original, side in ((True, condition.rhs), (False, condition.lhs)):
            expr: str = str(side)
            syms = self._find_symbols(expr)
            if not syms:
                continue
            marginal = self.finite_support(*syms)
            if marginal is not None:
                break
        if marginal is None:
            for side in (condition.rhs, condition.lhs):
                syms = self._find_symbols(str(side))
                if syms:
                    # raises NonFiniteMarginalError on a measure with infinite mass
                    self.checked_marginal(*syms)
            # We are not able to marginalize into a finite amount of states! -> FAIL filtering.
            raise NotImplementedError(
                f"Instruction {condition} is not computable on infinite generating function"
                f" {self}")

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

    @abstractmethod
    def hadamard_product(self, other: Distribution) -> Distribution:
        """
        Computes the Hadamard product of the distributions, i.e., the point-wise product of the probabilities.
        """

    @abstractmethod
    def _find_symbols(self, expr: str) -> Set[str]:
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
    def evaluate_condition(cls, condition: Expr,
                           state: State) -> bool:
        """Evaluates whether the condition holds in the specified state."""

        if isinstance(condition, UnopExpr) and condition.operator == Unop.NEG:
            return not cls.evaluate_condition(condition.expr, state)
        if isinstance(condition, VarExpr):
            return cls.evaluate_condition(_is_true(condition), state)
        if not isinstance(condition, BinopExpr):
            raise AssertionError(
                f"Expression must be an (in-)equation or negation, was {condition}"
            )

        op = condition.operator
        if op in (Binop.EQ, Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ):
            # exactly, with truncated subtraction, if both sides are numbers in this state
            left, right = _state_value(condition.lhs, state), _state_value(condition.rhs, state)
            lhs: Any
            rhs: Any
            if left is not None and right is not None:
                lhs, rhs = left, right
            else:
                lhs, rhs = cls.evaluate(str(condition.lhs), state), cls.evaluate(str(condition.rhs), state)
            if op == Binop.EQ:
                return lhs == rhs
            if op == Binop.LEQ:
                return lhs <= rhs
            if op == Binop.LT:
                return lhs < rhs
            if op == Binop.GT:
                return lhs > rhs
            return lhs >= rhs
        if op == Binop.OR:
            return cls.evaluate_condition(condition.lhs,
                                          state) or cls.evaluate_condition(
                condition.rhs, state)
        if op == Binop.AND:
            return cls.evaluate_condition(condition.lhs,
                                          state) and cls.evaluate_condition(
                condition.rhs, state)
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
        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."
        variable = expression.lhs.var
        if variable not in self.get_variables():
            raise ValueError(
                f"Cannot assign {expression.rhs} to variable {variable} because {variable} does not exist"
            )

        # pylint: disable=protected-access
        def evaluate(function: Distribution, expression: Expr,
                     temp_var: str) -> Tuple[Distribution, str]:
            # TODO handle reals in every case

            # Subexpressions without variables are computed directly, e.g. 9/2*2 becomes 9
            constant = constant_value(expression) if isinstance(expression, BinopExpr) else None
            if constant is not None:
                return function, str(constant)

            # Constants that are not natural numbers are rewritten, e.g. 1/2*x becomes (1*x)/2
            if isinstance(expression, BinopExpr):
                rewritten = _without_fractional_constants(expression)
                if rewritten is not None:
                    return evaluate(function, rewritten, temp_var)

            # For binary operator, we need to introduce fresh variables to prevent clashes
            # and recursively evaluate the left- and right-hand side of the operator
            if isinstance(expression, BinopExpr):
                if expression.operator == Binop.FLOOR_DIVIDE:
                    for operand in (expression.lhs, expression.rhs):
                        if isinstance(operand, VarExpr) and operand.var in function.get_parameters():
                            raise NotImplementedError(
                                f"Cannot compute {_spelled(expression)}: floor division with the parameter "
                                f"{operand.var} is not implemented")
                # Get two fresh variables
                xl = function.get_fresh_variable()
                xr = function.get_fresh_variable({xl})
                #print(xl, xr)
                # Add variables to functions' variables
                f = function.set_variables(*(function.get_variables()
                                             | {xl, xr}))
                # Recursively evaluate the lhs and rhs of the binary operator
                f, t_1 = evaluate(f, expression.lhs, xl)
                f, t_2 = evaluate(f, expression.rhs, xr)

                # With the two resulting temporary variables, execute the effect of the binary operator
                # by calling the corresponding update method within the concrete distribution class
                if expression.operator == Binop.PLUS:
                    f = f._update_sum(temp_var, t_1, t_2)
                elif expression.operator == Binop.TIMES:
                    f = f._split_on_finite_support(
                        lambda part, left, right: part._update_product(temp_var, left, right, approximate),
                        t_1, t_2)
                elif expression.operator == Binop.MINUS:
                    try:
                        f = f._update_monus(temp_var, t_1, t_2)
                    except NotImplementedError as error:
                        raise NotImplementedError(
                            f"Cannot compute {expression} exactly: subtraction on natural numbers is truncated "
                            f"at 0, which needs the states with {expression.lhs} < {expression.rhs}, and these "
                            f"cannot be filtered here (not implemented). Details: {error}") from error
                elif expression.operator in (Binop.MODULO, Binop.DIVIDE):
                    f = f._divisor_update(expression, temp_var, t_1, t_2, approximate)
                elif expression.operator == Binop.POWER:
                    f = f._split_on_finite_support(
                        lambda part, left, right: part._update_power(temp_var, left, right, approximate),
                        t_1, t_2)
                elif expression.operator == Binop.FLOOR_DIVIDE:
                    f = f._update_floor_division(temp_var, t_1, t_2, approximate, _spelled(expression))
                else:
                    raise ValueError(
                        f"Unsupported binary operator: {expression.operator}")

                # Marginalize the introduced variables as they are only present to prevent clashes
                f = f.marginal(xl, xr, method=MarginalType.EXCLUDE)
                #print(f"marginal: {f}")
                #print(f.get_variables())
                return f, temp_var

            if isinstance(expression, VarExpr):
                f = function._update_var(temp_var, expression.var)
                return f, temp_var

            if isinstance(expression, (NatLitExpr, RealLitExpr)):
                return function, str(expression.value)

            if isinstance(expression, BoolLitExpr):
                # booleans are the values 1 (true) and 0 (false) of the generating function (#283)
                return function, str(int(expression.value))

            else:
                raise ValueError(
                    f"Unsupported type of subexpression: {expression}")

        # pylint: enable=protected-access

        value: int | None = None

        # Check if the value is a natural number
        if isinstance(expression.rhs, RealLitExpr):
            if expression.rhs.to_fraction().denominator == 1:
                value = expression.rhs.to_fraction().numerator
            else:
                raise ValueError(
                    f'Cannot assign the real value {str(expression.rhs)} to {variable}'
                )

        if isinstance(expression.rhs, NatLitExpr):
            value = expression.rhs.value

        # A bool literal is the value 1 (true) or 0 (false), e.g. "b = true" (#283)
        if isinstance(expression.rhs, BoolLitExpr):
            value = int(expression.rhs.value)

        # An expression without variables, e.g. "x = 9/2*2" or "x = 5 % 3", is a literal assignment
        if value is None and isinstance(expression.rhs, BinopExpr):
            constant = constant_value(expression.rhs)
            if constant is not None:
                if constant.denominator != 1:
                    raise ValueError(f'Cannot assign the real value {constant} to {variable}')
                value = constant.numerator

        # If expression is of form "x = n" where n is a natual number
        # simply assign the value
        if value is not None:
            result = self._update_var(variable, value)
        else:
            # Not just a literal assignment, need to evaluate the expression first
            result, _ = evaluate(self, expression.rhs, variable)
        return result

    def _divisor_update(self, expression: BinopExpr, temp_var: str, left: str, right: str,
                        approximate: str | float | None) -> Distribution:
        """
        `temp_var = left % right` or `left / right` for the operands `left`, `right` of `expression` (a
        modulo or division); a divisor 0 names `expression` in the error (#273).
        """
        def update(part: Distribution, lhs: str, rhs: str) -> Distribution:
            # pylint: disable=protected-access
            if expression.operator == Binop.MODULO:
                return part._update_modulo(temp_var, lhs, rhs, approximate)
            return part._update_division(temp_var, lhs, rhs, approximate)

        try:
            return self._split_on_finite_support(update, left, right)
        except ZeroDivisorError as error:
            if error.expression is not None:
                raise
            raise ZeroDivisorError(error.reason, str(expression)) from error

    def _split_on_finite_support(self, update: Callable[[Distribution, str, str], Distribution],
                                 left: str, right: str) -> Distribution:
        """
        `update(self, left, right)`, a binary update whose operands are variables or literals. If it
        raises `NonFiniteMarginalError` (a case split over the values of an operand on a measure with
        infinite mass, #242), the operands with a finite support (:meth:`finite_support`) are fixed
        instead (:meth:`_split_over_finite_supports`, #249). If no operand has a finite support, or the
        update of a part still needs a marginal that is not finite, the error is raised.
        """
        try:
            return update(self, left, right)
        except NonFiniteMarginalError:
            result = self._split_over_finite_supports(update, left, right, (left, right))
            if result is None:
                raise
            return result

    def _split_over_finite_supports(self, update: Callable[[Distribution, str, str], Distribution],
                                    left: str | int, right: str | int,
                                    names: Iterable[str | int]) -> Optional[Distribution]:
        """
        `update(self, left, right)` for a binary update whose operands are variables or literals, as
        the sum of the updates of the parts in which the variables among `names` that have a finite
        support (:meth:`finite_support`) have one value each, with these values passed as literals.
        The update is linear, so this is exact; the parts follow the support, not the marginal, so no
        value is lost whose mass cancels in the marginal of a signed measure (#256). Returns None if
        none of `names` is a variable with a finite support.
        """
        variables = self.get_variables()
        supports = {name: self.finite_support(name)
                    for name in dict.fromkeys(str(name) for name in names) if name in variables}
        fixed = [name for name, support in supports.items() if support is not None]
        if not fixed:
            return None
        support = supports[fixed[0]] if len(fixed) == 1 else self.finite_support(*fixed)
        assert support is not None
        result: Optional[Distribution] = None
        for state in self._unique_states(support):
            values = state.valuations
            section = self.filter_state(State({name: values.get(name, 0) for name in fixed}))
            try:
                part = update(section,
                              str(values.get(str(left), 0)) if str(left) in fixed else str(left),
                              str(values.get(str(right), 0)) if str(right) in fixed else str(right))
            except ValueError:
                # a value without mass (coefficient not simplified to 0), see is_provably_zero
                if section.is_provably_zero():
                    continue
                raise
            result = part if result is None else result + part
        if result is None:
            # no state at all: the zero measure, whose update is the zero measure
            return self.filter(BoolLitExpr(False))
        return result

    @staticmethod
    def _unique_states(values: Distribution) -> List[State]:
        """
        The states of `values` (a :meth:`finite_support` or a marginal) for a case split, each once.
        An engine may list a state once per term: GiNaC lists the support `1/2 + 1/2*x` of `y` (with
        the variable `x` as a parameter) as the state y = 0 twice, and a case split over both would
        count its part twice.
        """
        states: Dict[Tuple[Tuple[str, int], ...], State] = {}
        for _, state in values:
            states.setdefault(tuple(sorted(state.valuations.items())), state)
        return list(states.values())

    def _case_split_support(self, variable: str) -> Optional[Distribution]:
        """
        The values of `variable` for an update that does a case split over them: its
        :meth:`finite_support` (each state is one value), or None if `variable` takes infinitely many
        values. The support, not the marginal, is unfolded: in the marginal of a signed measure the
        mass of a value can cancel to 0, and the case split would lose the part with this value (#256).
        The coefficients of the support are not probabilities; an update that approximates needs the
        real marginal (:meth:`checked_marginal`).

        Raises `NonFiniteMarginalError` if `variable` takes infinitely many values and its marginal is
        not a finite measure (#242).
        """
        support = self.finite_support(variable)
        if support is None:
            self.checked_marginal(variable)
        return support

    def _approximated_support(self, variable: str, approximate: str | float | None,
                              message: str) -> Distribution:
        """
        For an update that does a case split over the values of `variable`, which takes infinitely
        many values: the approximation of its marginal with mass `approximate`. Raises
        `NonFiniteMarginalError` if the marginal is not a finite measure (#242), and `ValueError` with
        `message` if `approximate` is None.
        """
        marginal = self.checked_marginal(variable)
        if approximate is None:
            raise ValueError(message)
        return marginal.approximate_unilaterally(variable, approximate)

    def _product_case_split(self, first_factor: str, second_factor: str,
                            approximate: str | float | None) -> Tuple[Distribution, str, str]:
        """
        For `first_factor * second_factor` of two variables on a distribution that is not finite:
        `(values, fixed, other)`, a split over the values of the factor `fixed` (the first one with a
        finite support, else the approximated marginal of the first factor) and the other factor.
        """
        for fixed, other in ((first_factor, second_factor), (second_factor, first_factor)):
            support = self.finite_support(fixed)
            if support is not None:
                return support, fixed, other
        self.checked_marginal(second_factor)
        # TODO can we choose which side to approximate in a smarter way?
        return self._approximated_support(
            first_factor, approximate,
            f'Cannot perform the multiplication {first_factor} * {second_factor} because both variables have '
            f'infinite range'), first_factor, second_factor

    def _power_case_split(self, base: str, exp: str,
                          approximate: str | float | None) -> Tuple[Distribution, Distribution]:
        """
        For `base ^ exp` of two variables: the values of `base` and of `exp` for the case split, each
        its finite support or else its approximated marginal.
        """
        message = "Can only perform exponentiation of variables if both have a finite marginal"
        values = []
        for variable in (base, exp):
            support = self.finite_support(variable)
            values.append(support if support is not None
                          else self._approximated_support(variable, approximate, message))
        return values[0], values[1]

    def _require_nonzero_divisor(self, right: str | int, operator: str) -> None:
        """
        Raises `ZeroDivisorError` if `right`, the right operand of `%` or `/` (`operator`), is 0: the
        literal 0, or a variable that takes the value 0 on this finite distribution. A modulus or divisor
        0 is undefined. All engines raise this error before any case split or call of the bindings
        (GiNaC reported a non-finite marginal on measures with infinite mass, #257, and its bindings a
        bare division by zero, #273). On an infinite distribution, the case split over the values of a
        variable `right` passes each value as a literal, so the value 0 is caught there.
        """
        text = str(right).strip()
        if text == "0":
            zero = True
        elif text in self.get_variables() and self.is_finite():
            support = self.finite_support(text)
            zero = support is not None and any(state[text] == 0 for state in self._unique_states(support))
        else:
            zero = False
        if not zero:
            return
        if operator == "%":
            raise ZeroDivisorError("The modulus has to be positive, got 0.")
        raise ZeroDivisorError("Division by zero: the divisor is 0.")

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
        names = [str(var) for var in variables]
        support = self.factory().from_expr(str(self), *names)
        return support if support.is_finite() else None

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

    def _update_monus(self, temp_var: str, minuend: str | int, subtrahend: str | int) -> Distribution:
        """
        Applies `temp_var = minuend - subtrahend` with truncated subtraction (monus): states with
        `minuend < subtrahend` get the value 0, the others the difference. `minuend` and
        `subtrahend` are variables of this distribution or natural numbers.

        The split is exact: the states below are filtered (`minuend < subtrahend`), the rest gets
        the ordinary subtraction, which is never negative there. If the subtraction is provably
        never negative (:meth:`_subtraction_never_negative`), there is nothing to split. Raises
        NotImplementedError if the filter cannot be computed exactly (e.g. two variables with
        infinite marginals) or an operand is a parameter.
        """
        variables = self.get_variables()
        operands: List[Expr] = []
        for operand in (minuend, subtrahend):
            text = str(operand).strip()
            if text in variables:
                operands.append(VarExpr(text))
            elif text.isdigit():
                operands.append(NatLitExpr(int(text)))
            else:
                raise NotImplementedError(
                    f"Cannot compute '{minuend} - {subtrahend}' (truncated at 0) because '{text}' is not a program "
                    "variable or a natural number")
        left, right = operands
        if isinstance(left, NatLitExpr) and isinstance(right, NatLitExpr):
            return self._update_var(temp_var, max(left.value - right.value, 0))
        if isinstance(left, VarExpr) and isinstance(right, VarExpr) and left.var == right.var:
            return self._update_var(temp_var, 0)
        if isinstance(right, NatLitExpr) and right.value == 0:
            return self._update_var(temp_var, str(minuend))
        if isinstance(left, VarExpr) and isinstance(right, VarExpr) \
                and self._subtraction_never_negative(str(minuend), str(subtrahend)):
            return self._update_subtraction(temp_var, minuend, subtrahend)
        below = self.filter(BinopExpr(Binop.LT, left, right))
        if below.is_zero_dist() is True:
            return self._update_subtraction(temp_var, minuend, subtrahend)
        above = self - below
        # pylint: disable=protected-access
        truncated = below._update_var(temp_var, 0)
        if above.is_zero_dist() is True:
            return truncated
        return truncated + above._update_subtraction(temp_var, minuend, subtrahend)

    def _subtraction_never_negative(self, minuend: str, subtrahend: str) -> bool:
        """
        Whether `minuend - subtrahend` (two variables) is provably never negative on the support
        of this distribution; False if it can be negative or the question is not decided. Engines
        override this with an exact check that needs no filter (see
        :mod:`prodigy.distribution.negative_values`).
        """
        # pylint: disable=unused-argument
        return False

    @abstractmethod
    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> Distribution:
        """
        Applies the expression `temp_var = sub_from - sub` to this distribution, where the
        difference is never negative on the support (the caller ensures this, see
        :meth:`_update_monus`).
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
        (i.e., they are either literals or have a finite marginal) if approximation is disabled. The
        exception is a numerator with infinite marginal and a literal denominator: the division is
        exact if the generating function is rational in the numerator and all its mass provably lies
        on multiples of the denominator (see :mod:`prodigy.distribution.exact_division`).
        """

    def _update_floor_division(self, temp_var: str, numerator: str | int, divisor: str | int,
                               approximate: str | float | None, text: str | None = None) -> Distribution:
        """
        Applies the expression `temp_var = numerator // divisor` (rounding down) to this distribution. The
        divisor must be a positive natural literal.

        With a variable `numerator`, the distribution is split into the residue classes of `numerator`
        modulo `divisor` (exact multisection, see :meth:`_arithmetic_progression`). On the class of r,
        `numerator // divisor` is `(numerator - r) / divisor`, an exact division. If the division
        cannot be proven exact (an infinite marginal of `numerator` that is not rational, without
        approximation), this raises an error.
        """
        text = text or f"{numerator} // {divisor}"  # the program's expression, not the temporary names
        if not str(divisor).isdigit() or int(str(divisor)) < 1:
            raise ValueError(f"Cannot compute {text}: the divisor must be a positive natural "
                             f"constant (not implemented otherwise).")
        q = int(str(divisor))
        if str(numerator) not in self.get_variables():
            value = Fraction(str(numerator))
            if value.denominator != 1:
                raise ValueError(f"Cannot compute {text}: not a natural number")
            return self._update_var(temp_var, value.numerator // q)
        if q == 1:
            return self._update_var(temp_var, str(numerator))
        shifted = self.get_fresh_variable({temp_var})
        result: Distribution | None = None
        # pylint: disable=protected-access
        for remainder, section in enumerate(self._arithmetic_progression(str(numerator), str(q))):
            if section.is_zero_dist():
                continue
            part = section.set_variables(*section.get_variables(), shifted)
            part = part._update_subtraction(shifted, str(numerator), str(remainder))
            try:
                part = part._update_division(temp_var, shifted, str(q), approximate)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot compute the floor division by {q} exactly: it is implemented for a dividend with "
                    f"finite support or with a generating function that is rational in it ({exc}).") from exc
            part = part.marginal(shifted, method=MarginalType.EXCLUDE)
            result = part if result is None else result + part
        # pylint: enable=protected-access
        if result is None:
            return self._update_var(temp_var, 0)  # the zero distribution
        return result.set_variables(*self.get_variables()).set_parameters(*self.get_parameters())

    @abstractmethod
    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> Distribution:
        """
        Applies the expression `temp_var := base^exp` to this distribution.

        All variables occuring in the expression must have a finite marginal if approximation is disabled.
        """

    @abstractmethod
    def update_iid(self, sampling_dist: Expr, count: Union[VarExpr, NatLitExpr],
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

    def checked_marginal(self, *variables: Union[str, VarExpr]) -> Distribution:
        """
        The marginal of `variables` (MarginalType.INCLUDE) for code that unfolds it into its states,
        e.g. a filter or update that does a case split over the values of these variables.

        Raises `NonFiniteMarginalError` if the marginal is not a finite measure (infinite mass, see
        :class:`NonFiniteMarginalError`): then its terms do not tell which values the variables take,
        and unfolding it would give a wrong result (#242).
        """
        names = ", ".join(str(var) for var in variables)
        try:
            marginal = self.marginal(*variables)
        except ZeroDivisionError as exc:
            raise NonFiniteMarginalError(
                f"The marginal of {names} is undefined (division by zero) in {self}") from exc
        except ValueError as exc:
            # e.g. GiNaC: power::eval(): division by zero
            if "division by zero" not in str(exc):
                raise
            raise NonFiniteMarginalError(
                f"The marginal of {names} is undefined (division by zero) in {self}") from exc
        self._require_finite_values(marginal, *variables)
        return marginal

    def _require_finite_values(self, marginal: Distribution, *variables: Union[str, VarExpr]) -> None:
        """Raises `NonFiniteMarginalError` if `marginal`, the marginal of `variables`, is not finite."""
        if marginal.has_non_finite_value():
            names = ", ".join(str(var) for var in variables)
            raise NonFiniteMarginalError(
                f"The marginal of {names} is {marginal}, not a finite measure: {self} has infinite "
                f"mass, so a case split over the values of {names} is not computable.")

    def has_non_finite_value(self) -> bool:
        """
        Whether this function contains a non-finite value (`nan`, ∞, complex infinity) or an
        unevaluated limit. Engines override this with a check on their expression; this default scans
        the string form (program variables and parameters are never taken for such a value).
        """
        tokens = set(re.findall(r"[A-Za-z_]\w*", str(self)))
        names = self.get_variables() | self.get_parameters()
        return "∞" in str(self) or bool((tokens & _NON_FINITE_NAMES) - names)

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

    # @abstractmethod
    def set_variables_and_parameters(self, variables, parameters):
        """
        todo
        """

    @abstractmethod
    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        """
        Computes the approximation until the given threshold is reached (might not terminate).

        :param threshold: The threshold either as a maximum number of states (int), or a certain probability mass (str).

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
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> Distribution:
        """ A distribution represented by the expression."""
