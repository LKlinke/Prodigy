# pylint: disable=protected-access
"""
The SymPy engine :class:`SympyPGF` (#511): the CAS hooks of :class:`~prodigy.distribution.pgf.hooks.PGFHooks` on
SymPy expressions, and the factory of the common distributions (``SympyPGF.geometric(…)``). The case analysis
of the operations is the shared logic of :class:`~prodigy.distribution.pgf.pgf.PGF`. ``GeneratingFunction`` is
a deprecated alias for one release.
"""
from __future__ import annotations

import logging
import contextlib
import operator
from typing import Callable, FrozenSet, Generator, Iterator, List, Optional, Set, Type, Union, cast

import sympy

from prodigy.distribution.distribution import CommonDistributionsFactory, Distribution, Operand
from prodigy.distribution.values import MarginalType, State
from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.pgf.sympy.terms import SympyTerms
from prodigy.distribution.pgf.sympy.settings import _SETTINGS, GeneratingFunctionSettings
from prodigy.distribution.pgf.sympy.hadamard import rational_hadamard_product
from prodigy.distribution.pgf.updating import binary_value
from prodigy.distribution.pgf.rules.value_at_one import _substitute_one
from prodigy.distribution.pgf.sympy.expressions import (_has_radical, _is_finite_function, _limit_at_one,
                                                        _parse_to_sympy, _polynomial_coefficients,
                                                        _sympy_symbol, _taylor_coefficient_by_limit,
                                                        provably_equal)
from prodigy.distribution.engine_text import sympy_text
from prodigy.distribution.pgf.rules.equality import decide_equal
from prodigy.distribution.pgf.rules.rational_series import rational_taylor_coefficients
from prodigy.distribution.pgf.rules.negative_values import can_be_negative
from prodigy.distribution.pgf.rules.multisection import rational_sections
from prodigy.distribution.pgf.rules.roots_of_unity import require_single_valued, root_of_unity
from prodigy.language import (Binop, BinopExpr, Expr,
                          NatLitExpr, Unop, UnopExpr, VarExpr)
from prodigy.language.engine_text import engine_text_of
from prodigy.util.logger import log_setup

logger = log_setup("generating_function", logging.DEBUG)


MAX_FACTORED_PARAMETERS = 3
"""A closed form is factored after an arithmetic operation only if it has at most this many symbols besides the
variables (parameters, the unknown coefficients of a template, see :meth:`SympyPGF._arithmetic`)."""


class SympyPGF(SympyTerms, PGF):  # pylint: disable=too-many-ancestors  # the layers of PGF and the two interfaces
    """
    This class represents a generating function. It wraps the sympy library.
    A SympyPGF object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """

    @staticmethod
    def settings() -> GeneratingFunctionSettings:
        """The settings that hold for the current analysis (the defaults outside one)."""
        return _SETTINGS.get()

    @staticmethod
    @contextlib.contextmanager
    def configured(*, latex_output: bool) -> Iterator[GeneratingFunctionSettings]:
        """
        Applies the settings within the ``with`` block of the current thread or task only and restores
        the previous ones afterwards, so analyses with different settings do not influence each other.
        """
        settings = GeneratingFunctionSettings(latex_output=latex_output)
        token = _SETTINGS.set(settings)
        try:
            yield settings
        finally:
            _SETTINGS.reset(token)

    # ==================================== CONSTRUCTORS ====================================

    def __init__(self,
                 function: Union[str, sympy.Expr, Expr],
                 *variables: Union[str, sympy.Symbol],
                 closed: bool | None = None,
                 finite: bool | None = None):

        # Set the basic information.
        self._function: sympy.Expr = _parse_to_sympy(function)
        # The constants of the roots of unity read back from a string are symbols named I and pi; unless they are
        # declared variables, they are the constants again and never become parameters (#342).
        declared = {str(v) for v in variables}
        constants = {sym: value for sym in self._function.free_symbols
                     for name, value in (("I", sympy.I), ("pi", sympy.pi))
                     if getattr(sym, "name", None) == name and name not in declared}
        if constants:
            self._function = self._function.xreplace(constants)

        # Set variables and parameters
        self._variables: Set[
            sympy.Symbol] = self._function.free_symbols
        self._parameters: Set[sympy.Symbol] = set()
        if variables:
            self._variables = self._variables.union(
                map(lambda v: _sympy_symbol(engine_text_of(v)),
                    filter(lambda v: v != "", variables)))
            self._parameters = self._variables.difference(
                map(lambda v: _sympy_symbol(engine_text_of(v)),
                    filter(lambda v: v != "", variables)))
            self._variables -= self._parameters

        # Do closed form and finiteness heuristics
        self._is_closed_form = closed if closed else not self._function.is_polynomial(
            *self._variables)
        self._is_finite = finite if finite else _is_finite_function(
            self._function, frozenset(self._variables))

    @staticmethod
    def factory() -> Type[CommonDistributionsFactory]:
        return SympyPGF

    def _filter_constant_condition(self,
                                   condition: BinopExpr) -> SympyPGF:
        # TODO refactor this in superclass?
        # Normalize the condition into the format _var_ (< | <= | =) const. I.e., having the variable on the lhs.

        # Mapping of operator from expression
        #   <const> <op> <var>
        # to
        #   <var> <op> <const>
        flipped_operators = {
            Binop.LT: Binop.GT,
            Binop.LEQ: Binop.GEQ,
            Binop.GT: Binop.LT,
            Binop.GEQ: Binop.LEQ,
            Binop.EQ: Binop.EQ
        }

        # Mapping of operator from
        #   <var> <op> <const>
        # to
        #   ¬(<var> <op> <const>)
        neg_operator = {
            Binop.GT: Binop.LEQ,
            Binop.GEQ: Binop.LT
        }

        # If condition of form
        #   <const> <op> <var>
        # flip operator and move to
        #   <var> <op> <const>
        if isinstance(condition.rhs, VarExpr):
            return self.filter(
                BinopExpr(
                    operator=flipped_operators[condition.operator],
                    rhs=condition.lhs,
                    lhs=condition.rhs
                )
            )

        # Now we have the form <var> <op> <const>
        # As we only allow <op> \in {<, <=, =}, we need to remove the other operator
        # by using negation
        if condition.operator in neg_operator:
            return self.filter(
                UnopExpr(operator=Unop.NEG, expr=BinopExpr(
                    operator=neg_operator[condition.operator],
                    rhs=condition.rhs,
                    lhs=condition.lhs
                ))
            )

        # Now we have an expression of the form _var_ (< | <=, =) _const_.
        variable = _sympy_symbol(engine_text_of(condition.lhs))
        if not isinstance(condition.rhs, NatLitExpr):
            raise ValueError(
                f"Cannot filter for {condition}: the bound is not a natural number")
        constant = condition.rhs.value
        result = 0
        ranges = {
            Binop.LT: range(constant),
            Binop.LEQ: range(constant + 1),
            Binop.EQ: [constant]
        }

        # Compute the probabilities of the states _var_ = i where i ranges depending on the operator (< , <=, =).
        indices = ranges[condition.operator]
        if not indices:
            coefficients: List[sympy.Expr] | None = []
        elif not self._is_closed_form and self._function.is_polynomial(variable):
            coefficients = _polynomial_coefficients(self._function, variable, indices[-1] + 1)
        else:
            coefficients = rational_taylor_coefficients(self._function, variable, indices[-1] + 1)
        for i in indices:
            if coefficients is None:
                coefficient = _taylor_coefficient_by_limit(self._function, variable, i)
            else:
                coefficient = coefficients[i] if i < len(coefficients) else sympy.Integer(0)
            result += coefficient * variable ** i

        return SympyPGF(result,
                                  *self._variables,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite).set_parameters(*self.get_parameters())

    def _sympy_expression(self) -> sympy.Expr:
        return self._function

    def _from_sympy_expression(self, expression: sympy.Expr) -> SympyPGF:
        # a reduced fraction: finite iff the denominator has no variable; an algebraic function (with
        # square roots of the variables) is not rational and never finite
        finite = not sympy.denom(expression).free_symbols & self._variables \
            and expression.is_rational_function(*self._variables)
        # (the flags are set afterwards: passing False makes the constructor run its heuristics)
        gf = SympyPGF(expression, *self._variables, closed=True, finite=True)
        gf._is_closed_form, gf._is_finite = not finite, finite
        gf._parameters = set(self._parameters)
        return gf

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[SympyPGF]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        a = _parse_to_sympy(modulus)
        var = _sympy_symbol(variable)

        # Rational in var: exact multisection without roots of unity.
        sections = rational_sections(self._function, var, int(a))
        if sections is not None:
            result = []
            for section in sections:
                # the sections are reduced fractions: finite iff the denominator has no variable
                finite = self._is_finite or not sympy.denom(section).free_symbols & self._variables
                # (the flags are set afterwards: passing False makes the constructor run its heuristics)
                gf = SympyPGF(section, *self._variables, closed=True, finite=True)
                gf._is_closed_form, gf._is_finite = not finite, finite
                gf._parameters = set(self._parameters)
                result.append(gf)
            return result

        require_single_valued(self._function, variable, f"{variable} % {modulus}")
        # This is the general algorithm
        q = int(a)
        roots = [root_of_unity(k, q) for k in range(q)]
        images = [self._function.subs(var, roots[m] * var) for m in range(q)]
        result = []
        for remainder in range(q):
            psum = sympy.Add(*(roots[(-m * remainder) % q] * images[m] for m in range(q)))
            gf = SympyPGF(psum / q, *self._variables, closed=self._is_closed_form, finite=self._is_finite)
            gf._parameters = set(self._parameters)
            result.append(gf)
        return result

    def hadamard_product(self, other: Distribution) -> SympyPGF:
        """
            Computes the Hadamard product of two univariate rational functions, i.e. the point-wise product of the
            coefficients in their formal power series representation.
        """
        logger.debug("Compute Hadamard product of %s and %s", self, other)
        if len(self._variables) > 1:
            raise NotImplementedError("Currently only univariate distributions are supported for the Hadamard product.")
        var = next(iter(self._variables))
        if isinstance(other, SympyPGF):
            if not other._variables == self._variables:
                raise NotImplementedError("Only univariate distributions are supported for the Hadamard product.")

            return SympyPGF(rational_hadamard_product(self._function, other._function, var), var)

        raise TypeError(f"Cannot compute the Hadamard Product, where {other=} is of type {type(other)}.")

    def get_fresh_variable(
            self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        i = 0
        while _sympy_symbol(f'_{i}') in (
                self._variables
                | self._parameters) or f'_{i}' in exclude:
            i += 1
        return f'_{i}'

    def approximate_unilaterally(
            self, variable: str,
            probability_mass: str | float) -> SympyPGF:
        logger.debug("approximate_unilaterally(%s, %s) call on %s", variable,
                     probability_mass, self)
        mass = _parse_to_sympy(probability_mass)
        if mass == 0:
            return SympyPGF('0',
                                      *self._variables,
                                      closed=True,
                                      finite=True)
        elif mass > self._coefficient_sum():
            raise ValueError("Given probability mass is too large")
        elif mass < 0:
            raise ValueError("Given probability mass must be non-negative")
        var = _sympy_symbol(variable)
        if var not in self._variables:
            raise ValueError(f'Not a variable: {variable}')
        result = 0
        mass_res = 0

        for element in self._function.series(var, n=None):
            result += element
            mass_res += element.subs([(sym, 1)
                                      for sym in element.free_symbols])
            if mass_res >= mass:
                return SympyPGF(result, *self._variables)

        raise NotImplementedError("unreachable")

    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> SympyPGF:
        update_var = _sympy_symbol(temp_var)
        sub_1, sub_2 = _parse_to_sympy(sub_from), _parse_to_sympy(sub)
        result = self._function

        # we subtract a variable from another variable
        if sub_1 in self._variables and sub_2 in self._variables:
            if sub_2 == update_var:
                if sub_1 == update_var:
                    result = _substitute_one(result, update_var)
                else:
                    result = result.subs([(update_var, update_var ** (-1)),
                                          (sub_1, sub_1 * update_var)])
            else:
                if not sub_1 == update_var:
                    result = _substitute_one(result, update_var).subs(sub_1, sub_1 * update_var)
                result = result.subs(sub_2, sub_2 * update_var ** (-1))

        # we subtract a literal / parameter from a variable
        elif sub_1 in self._variables:
            if not update_var == sub_1:
                result = _substitute_one(result, update_var).subs(sub_1, sub_1 * update_var)
            result = result * update_var ** (-sub_2)

        # we subtract a variable from a literal / parameter
        elif sub_2 in self._variables:
            if sub_2 == update_var:
                result = result.subs(update_var, update_var
                                     ** (-1)) * update_var ** sub_1
            else:
                result = _substitute_one(result, update_var) * update_var ** sub_1
                result = result.subs(sub_2, sub_2 * update_var ** (-1))

        # we subtract two literals / parameters from each other
        else:
            diff = sub_1 - sub_2
            if sub_1 not in self._parameters and sub_2 not in self._parameters and diff < 0:
                raise ValueError(
                    f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it is negative"
                )
            result = _substitute_one(result, update_var) * update_var ** diff

        result = sympy.expand(result)
        return SympyPGF(result,
                                  *self._variables).set_parameters(
            *self.get_parameters())

    def _subtraction_never_negative(self, minuend: str, subtrahend: str) -> bool:
        return can_be_negative(self._function, set(self._variables), _parse_to_sympy(minuend),
                               _parse_to_sympy(subtrahend)) is False

    def _update_var(self, updated_var: str,
                    assign_var: str | int) -> SympyPGF:
        if _sympy_symbol(engine_text_of(assign_var)) in self._parameters:
            raise ValueError(("Not implemented: a parameter cannot be the source of an assignment "
                             f"({updated_var} = {assign_var}); parameters are unknown constants, "
                             "not values a program variable can take"))
        if _parse_to_sympy(engine_text_of(assign_var)).is_Symbol and _parse_to_sympy(
                engine_text_of(assign_var)) not in self._variables:
            raise ValueError(f"Unknown symbol: {assign_var}")

        if not updated_var == assign_var:
            if _parse_to_sympy(assign_var) in self._variables:
                result = _substitute_one(self._function, _sympy_symbol(updated_var)).subs(
                    _parse_to_sympy(assign_var), _parse_to_sympy(assign_var) * _sympy_symbol(updated_var))
            else:
                result = _substitute_one(self._function, _sympy_symbol(updated_var)) \
                    * _sympy_symbol(updated_var) ** _parse_to_sympy(assign_var)
            return SympyPGF(
                result,
                *self._variables,
                closed=self._is_closed_form,
                finite=self._is_finite).set_parameters(*self.get_parameters())
        else:
            return self.copy()

    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> SympyPGF:
        update_var = _sympy_symbol(temp_var)
        sum_1, sum_2 = _parse_to_sympy(first_summand), _parse_to_sympy(
            second_summand)
        result = self._function

        # we add two variables
        if sum_1 in self._variables and sum_2 in self._variables:
            if sum_2 == update_var:
                sum_1, sum_2 = sum_2, sum_1
            if sum_1 == update_var:
                if sum_2 == update_var:
                    result = result.subs(update_var, update_var ** 2)
                else:
                    result = result.subs(sum_2, sum_2 * update_var)
            else:
                result = _substitute_one(result, update_var).subs([(sum_1, sum_1 * update_var),
                                                                   (sum_2, sum_2 * update_var)])

        # we add a variable and a literal / parameter
        elif sum_1 in self._variables or sum_2 in self._variables:
            if sum_1 in self._variables:
                var = sum_1
                lit = sum_2
            else:
                var = sum_2
                lit = sum_1
            if not var == update_var:
                result = _substitute_one(result, update_var).subs(var, update_var * var)
            result = result * (update_var ** lit)

        # we add two literals / parameters
        else:
            result = _substitute_one(result, update_var) * (update_var ** (sum_1 + sum_2))

        return SympyPGF(
            result,
            *self._variables,
            closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _update_scaled(self, temp_var: str, variable: str, factor: int) -> SympyPGF:
        # the simplification of powers of powers needs non-negative symbols
        update_var = _sympy_symbol(temp_var, nonnegative=True)
        var = _sympy_symbol(variable)
        if variable == temp_var:
            result = self._function.subs(_sympy_symbol(temp_var), update_var ** factor)
        else:
            result = _substitute_one(self._function, _sympy_symbol(temp_var)).subs(var, var * update_var ** factor)
        return SympyPGF(
            # remove the nonnegativity assumption from all symbols
            result.replace(lambda expr: expr.is_Symbol, lambda expr: _sympy_symbol(expr.name)),
            *self._variables, closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _termwise(self, binop: Binop, temp_var: str, left: str, right: str) -> SympyPGF:
        update_var = _sympy_symbol(temp_var)
        result = sympy.Integer(0)
        for probability, monomial in self._iter_expr_expr():
            state = self._monomial_to_state(monomial)
            values = [state[operand] if operand in state.valuations else int(operand) for operand in (left, right)]
            result += probability * monomial.subs(update_var, 1) * update_var ** binary_value(binop, *values)
        return SympyPGF(result, *self._variables, closed=self._is_closed_form,
                        finite=self._is_finite).set_parameters(*self.get_parameters())

    def _compose_iid(self, sample: PGF, count: Union[str, int], variable: str) -> SympyPGF:
        sample_gf = cast(SympyPGF, sample)._function
        symbol = _sympy_symbol(variable)
        result = self if count == variable else self._marginal(variable, method=MarginalType.EXCLUDE)
        result = result.set_variables(*self.get_variables(), variable)
        if isinstance(count, int):
            # a fixed number of samples: the sum has the generating function sample^count
            result._function = result._function * sample_gf ** count
        elif count == variable:
            result._function = result._function.subs(symbol, sample_gf)
        else:
            counter = _sympy_symbol(count)
            result._function = result._function.subs(counter, counter * sample_gf)
        return result.set_parameters(*(self.get_parameters() | (sample.get_parameters() - self.get_variables())))

    def copy(self, deep: bool = True) -> SympyPGF:
        res = SympyPGF(self._function, *self._variables, closed=True, finite=True)
        # set directly: the constructor reads no variables as "all symbols are variables", and it recomputes
        # the flags that are False
        res._variables = self._variables.copy()
        res._parameters = self._parameters.copy()
        res._is_closed_form, res._is_finite = self._is_closed_form, self._is_finite
        return res

    def _with_names(self, variables: Set[str], parameters: Set[str]) -> SympyPGF:
        result = self.copy()
        result._variables = {_sympy_symbol(name) for name in variables}
        result._parameters = {_sympy_symbol(name) for name in parameters}
        occurring, old = self.get_symbols(), self.get_variables()
        if (variables - old) & occurring:
            # a former parameter may make the function a closed form, or infinite (e.g. 1/(1 - p) for p)
            result._is_closed_form = self._is_closed_form or not self._function.is_polynomial(*result._variables)
            result._is_finite = _is_finite_function(self._function, frozenset(result._variables))
        elif not self._is_finite and (old - variables) & occurring:
            # a variable became a parameter: an infinite function may be a polynomial in the remaining ones
            result._is_finite = _is_finite_function(self._function, frozenset(result._variables))
        return result

    def _arithmetic(self, other: Operand, op: Callable[[sympy.Expr, sympy.Expr], sympy.Expr]) -> SympyPGF:
        """
        Computes the basic arithmetic of two generating functions.
        """
        logger.debug("Computing the %s of %s with %s", op, self, other)

        # other object is already a generating function.
        if isinstance(other, SympyPGF):
            self._require_compatible_names(other)

            # collect the meta information.
            is_closed_form = self._is_closed_form and other._is_closed_form
            is_finite = self._is_finite and other._is_finite

            # do the actual operation
            function = op(self._function, other._function)
            variables = self._variables.union(other._variables)

            # The normal form of the computation, independent of the settings (#404): a closed form gets its
            # numerator and denominator factored, a polynomial is expanded. Both are measured to pay off: the
            # analysis of loops is 2-6 times slower without the factors (limits at 1, substitutions and the
            # equality checks of invariants work on them), and programs with many variables and finite support
            # twice as slow without expanded polynomials. Expanding never factors, so sparse polynomials of high
            # degree such as the x**(2**k) of a doubled variable stay cheap (#224). Closed forms are not
            # factored if they contain radicals or exponentials (factoring an algebraic function after a
            # subtraction does not finish, #243; factored Poisson products make the coefficients slow) or more
            # than MAX_FACTORED_PARAMETERS symbols besides the variables (the many unknown coefficients of the
            # templates of evt_synthesis: +40 % in test_template_solver). The shown result is simplified on its
            # own (prodigy.distribution.display).
            if is_closed_form and len(function.free_symbols - variables) <= MAX_FACTORED_PARAMETERS \
                    and not _has_radical(function):
                numerator, denominator = sympy.fraction(function)
                function = numerator.factor() / denominator.factor()
            elif not is_closed_form and function.free_symbols <= variables:
                function = function.expand()

            parameters = self._parameters.union(other._parameters)
            result = SympyPGF(function,
                                        *variables,
                                        closed=is_closed_form,
                                        finite=is_finite)
            result._parameters = parameters
            return result

        # other object is either an expression, or literal
        elif isinstance(other, (str, float, int)):
            # we try to convert this into a Generatingfunction and compute the arithmetic from there on.
            return self._arithmetic(
                SympyPGF(engine_text_of(other), *self._variables), op)
        # We don't know how to do arithmetic on other types.
        else:
            raise SyntaxError(
                f"You cannot {str(op)} {type(self)} with {type(other)}.")

    def __add__(self, other: Operand) -> SympyPGF:
        return self._arithmetic(other, operator.add)

    def __sub__(self, other: Operand) -> SympyPGF:
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other: Operand) -> SympyPGF:
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other: Operand) -> SympyPGF:
        logger.debug("Isn't it weird to divide Distributions?")
        return self._arithmetic(other, operator.truediv)

    def _engine_text(self) -> str:
        return sympy_text(self._function)

    def __str__(self) -> str:
        if SympyPGF.settings().latex_output:
            return sympy.latex(self._function)
        else:
            output = sympy_text(self._function)
        return output

    def __repr__(self) -> str:
        return repr(self._function)

    # ====== Comparison of Generating Functions ======

    def _series_expression(self) -> sympy.Expr:
        return self._function

    def __eq__(self, other: object) -> bool:
        """
        True only if the two generating functions are provably equal (see :func:`provably_equal`).
        False means "not shown equal", which is not a proof of inequality for non-rational functions.
        """
        if not isinstance(other, SympyPGF):
            return False
        return self._variables == other._variables \
            and self._parameters == other._parameters \
            and provably_equal(self._function, other._function)

    def decide_equal(self, other: object) -> Optional[bool]:
        if isinstance(other, SympyPGF) and self._variables == other._variables \
                and self._parameters == other._parameters:
            return decide_equal(self._function, other._function)
        return super().decide_equal(other)

    def approximate(
            self,
            threshold: Union[str,
            int]) -> Generator[SympyPGF, None, None]:
        """
        Generate an approximation of a generating function, until `threshold` percent or terms of the probability
        mass is captured.

        :param threshold: The probability percentage threshold of the probability mass of the distribution.

        :return: A Generating Function generator.
        """
        logger.debug("expand_until() call")
        approx = sympy.Integer(0)
        precision = sympy.Integer(0)

        if isinstance(threshold, int):
            assert threshold > 0, "Expanding to less than 0 terms is not valid."
            for n, (prob, state) in enumerate(self._iter_expr_expr()):
                if n >= threshold:
                    break
                approx += prob * state
                precision += prob
                yield SympyPGF(approx,
                                         *self._variables,
                                         closed=False,
                                         finite=True)

        elif isinstance(threshold, str):
            s_threshold = _parse_to_sympy(threshold)
            assert s_threshold < self._coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {s_threshold}, CSum {self._coefficient_sum()}"
            for s_prob, state in self._iter_expr_expr():
                if precision >= _parse_to_sympy(threshold):
                    break
                approx += s_prob * state
                precision += s_prob
                yield SympyPGF(engine_text_of(approx.expand()),
                                         *self._variables,
                                         closed=False,
                                         finite=True)
        else:
            raise TypeError(
                f"Parameter threshold can only be of type str or int,"
                f" not {type(threshold)}.")

    def coefficient_sum(self) -> sympy.Expr:
        """
        The closed form at 1. Outside an analysis
        (:func:`~prodigy.distribution.distribution.known_convergent_series`), raises
        `NonFiniteMarginalError` if it is not proven to be the sum of the series (#427).
        """
        self._require_convergent_evaluation(set(), "The coefficient sum")
        return self._coefficient_sum()

    def _coefficient_sum(self) -> sympy.Expr:
        """:meth:`coefficient_sum` without a convergence proof."""
        logger.debug("coefficient_sum() call")
        coefficient_sum = self._function
        for var in self._variables:
            coefficient_sum = _substitute_one(
                coefficient_sum, var) if self._is_closed_form else coefficient_sum.subs(
                var, 1)
        return coefficient_sum

    def _probability_mass(self) -> str:
        return sympy_text(self._coefficient_sum())

    def get_parameters(self) -> Set[str]:
        return set(map(engine_text_of, self._parameters))

    def get_variables(self) -> Set[str]:
        return set(map(engine_text_of, self._variables))

    def is_zero_dist(self) -> bool:
        return self._function == 0

    def _normalize(self) -> SympyPGF:
        logger.debug("normalized() call")
        mass = self._coefficient_sum()
        if mass == 0:
            raise ZeroDivisionError("Cannot normalize: the probability mass is 0")
        result = self.copy()  # keeps the parameters, also those that do not occur (#361)
        result._function = self._function / mass
        return result

    def is_finite(self) -> bool:
        """
        Checks whether the generating function is finite.

        :return: True if the GF is a polynomial, False otherwise.
        """
        return self._is_finite

    @staticmethod
    def evaluate(expression: str, state: State) -> sympy.Expr:
        """ Evaluates the expression in a given state. """

        s_exp = _parse_to_sympy(expression)
        # Iterate over the variable, value pairs in the state
        for var, value in state.items():
            # Convert the variable into a sympy symbol and substitute
            s_var = _sympy_symbol(var)
            s_exp = s_exp.subs(s_var, value)

        # If the state did not interpret all variables in the condition, interpret the remaining variables as 0
        for free_var in s_exp.free_symbols:
            s_exp = s_exp.subs(free_var, 0)
        return s_exp

    @classmethod
    def evaluate_condition(cls, condition: Expr, state: State) -> bool:
        logger.debug("evaluate_condition() call")
        return super().evaluate_condition(condition, state)

    def _marginal(
            self,
            *variables: Union[str, VarExpr],
            method: MarginalType = MarginalType.INCLUDE) -> SympyPGF:
        logger.debug(
            "Creating marginal for variables %s and joint probability distribution %s",
            variables, self)

        marginal_vars = set(
            map(_sympy_symbol, filter(lambda v: v != '', map(engine_text_of, variables))))
        marginal = self.copy()
        s_var: str | VarExpr | sympy.Symbol
        # _limit_at_one, not _substitute_one: it also keeps the result small (later steps rely on that)
        if method == MarginalType.INCLUDE:
            for s_var in marginal._variables.difference(marginal_vars):
                if marginal._is_closed_form:
                    marginal._function = _limit_at_one(marginal._function, s_var)
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = marginal_vars
        else:
            for s_var in marginal_vars:
                if marginal._is_closed_form:
                    marginal._function = _limit_at_one(marginal._function, s_var)
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = marginal._variables.difference(marginal_vars)

        marginal._is_closed_form = not marginal._function.is_polynomial()
        marginal._is_finite = _is_finite_function(
            marginal._function, frozenset(marginal._function.free_symbols))

        return marginal

    def checked_marginal(self, *variables: Union[str, VarExpr], proof: bool = True) -> SympyPGF:
        res = super().checked_marginal(*variables, proof=proof)
        assert isinstance(res, SympyPGF)
        return res

    def has_non_finite_value(self) -> bool:
        return bool(self._function.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo, sympy.Limit,
                                       sympy.AccumBounds))

    def filter(self, condition: Expr) -> SympyPGF:
        logger.debug("filter(%s) call on %s", condition, self)
        res = super().filter(condition)
        assert isinstance(res, SympyPGF)
        return res

    def _exhaustive_search(self, condition: Expr) -> SympyPGF:
        res = sympy.Integer(0)
        for prob, state in self._iter_expr_expr():
            if self.evaluate_condition(condition,
                                       self._monomial_to_state(state)):
                res += prob * state
        return SympyPGF(res,
                                  *self._variables,
                                  closed=False,
                                  finite=True).set_parameters(*self.get_parameters())

    def _find_symbols(self, expr: str) -> Set[str]:
        return {str(s) for s in _parse_to_sympy(expr).free_symbols}

    def get_symbols(self) -> Set[str]:
        return {str(s) for s in self._function.free_symbols}

    # The common distributions (the factory, #511)

    @staticmethod
    def zero(*variables: Union[str, VarExpr]) -> SympyPGF:
        return SympyPGF("0",
                                  *variables,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> SympyPGF:
        return SympyPGF("1",
                                  *map(engine_text_of, variables),
                                  closed=True,
                                  finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables: Union[str, VarExpr],
                  **kwargs: Optional[bool]) -> SympyPGF:
        return SympyPGF(engine_text_of(expression), *map(engine_text_of, variables),
                                  **kwargs)


GeneratingFunction = SympyPGF
"""Deprecated name of :class:`SympyPGF` (until v0.8.0, #511)."""
