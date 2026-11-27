"""
The shared logic of generating functions, third layer (#511): updates, with the case splits over the values of
operands and the operations that the engines do not compute themselves (truncated subtraction, floor division).
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple, Union


from prodigy.language import Binop, BinopExpr, BoolLitExpr, Expr, FunctionCallExpr, NatLitExpr, RealLitExpr, VarExpr
from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.errors import NonFiniteMarginalError, NotAnIntegerError, ZeroDivisorError
if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.values import MarginalType, State
from prodigy.distribution.terms import constant_value, spelled, variables_of, without_fractional_constants
from prodigy.distribution.pgf.messages import in_programs_terms
from prodigy.distribution.pgf.filtering import PGFFiltering


class PGFUpdating(PGFFiltering):  # pylint: disable=abstract-method  # a layer of PGF
    """Updates of generating functions."""

    def update(self,
               expression: Expr,
               approximate: str | float | None = None) -> PGF:
        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."
        variable = expression.lhs.var
        if variable not in self.get_variables():
            raise ValueError(
                f"Cannot assign {expression.rhs} to variable {variable} because {variable} does not exist"
            )
        unknown = variables_of(expression.rhs) - self.get_variables() - self.get_parameters()
        if unknown:
            # GiNaC read an unknown name as a variable with value 0 (#361)
            raise ValueError(f"Cannot assign {expression.rhs} to variable {variable} because "
                             f"{', '.join(sorted(unknown))} is neither a variable nor a parameter")
        if variable not in variables_of(expression.rhs):
            # the old value is summed out, i.e. set to 1 (#427); an update that reads it substitutes instead
            self._require_convergent_evaluation(self.get_variables() - {variable},
                                                f"The update {variable} := {expression.rhs}")

        # pylint: disable=protected-access
        def evaluate(function: PGF, expression: Expr,
                     temp_var: str) -> Tuple[PGF, str]:
            # TODO handle reals in every case

            # Subexpressions without variables are computed directly, e.g. 9/2*2 becomes 9
            constant = constant_value(expression) if isinstance(expression, BinopExpr) else None
            if constant is not None:
                return function, engine_text_of(constant)

            # Constants that are not natural numbers are rewritten, e.g. 1/2*x becomes (1*x)/2
            if isinstance(expression, BinopExpr):
                rewritten = without_fractional_constants(expression)
                if rewritten is not None:
                    return evaluate(function, rewritten, temp_var)

            # For binary operator, we need to introduce fresh variables to prevent clashes
            # and recursively evaluate the left- and right-hand side of the operator
            if isinstance(expression, BinopExpr):
                if expression.operator == Binop.FLOOR_DIVIDE:
                    for operand in (expression.lhs, expression.rhs):
                        if isinstance(operand, VarExpr) and operand.var in function.get_parameters():
                            raise NotImplementedError(
                                f"Cannot compute {spelled(expression)}: floor division with the parameter "
                                f"{operand.var} is not implemented")
                # Get two fresh variables
                xl = function.get_fresh_variable()
                xr = function.get_fresh_variable({xl})
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
                    try:
                        f = f._split_on_finite_support(
                            lambda part, left, right: part._update_product(temp_var, left, right, approximate),
                            t_1, t_2)
                    except (ValueError, NotImplementedError, NonFiniteMarginalError) as error:
                        raise in_programs_terms(
                            error, {str(t_1): spelled(expression.lhs), engine_text_of(t_2): spelled(expression.rhs)}
                        ) from error
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
                    try:
                        f = f._split_on_finite_support(
                            lambda part, left, right: part._update_power(temp_var, left, right, approximate),
                            t_1, t_2)
                    except (ValueError, NotImplementedError, NonFiniteMarginalError) as error:
                        raise in_programs_terms(
                            error, {str(t_1): spelled(expression.lhs), engine_text_of(t_2): spelled(expression.rhs)}
                        ) from error
                elif expression.operator == Binop.FLOOR_DIVIDE:
                    f = f._update_floor_division(temp_var, t_1, t_2, approximate, spelled(expression))
                else:
                    raise ValueError(
                        f"Unsupported binary operator: {expression.operator}")

                # Marginalize the introduced variables as they are only present to prevent clashes
                f = f._marginal(xl, xr, method=MarginalType.EXCLUDE)
                return f, temp_var

            if isinstance(expression, VarExpr):
                f = function._update_var(temp_var, expression.var)
                return f, temp_var

            if isinstance(expression, (NatLitExpr, RealLitExpr)):
                return function, engine_text_of(expression.value)

            if isinstance(expression, BoolLitExpr):
                # booleans are the values 1 (true) and 0 (false) of the generating function (#283)
                return function, engine_text_of(int(expression.value))

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
            result, _ = evaluate(self._pgf(), expression.rhs, variable)
        return result

    def _divisor_update(self, expression: BinopExpr, temp_var: str, left: str, right: str,
                        approximate: str | float | None) -> PGF:
        """
        `temp_var = left % right` or `left / right` for the operands `left`, `right` of `expression` (a
        modulo or division); a divisor 0 names `expression` in the error (#273).
        """
        def update(part: PGF, lhs: str, rhs: str) -> PGF:
            # pylint: disable=protected-access
            if expression.operator == Binop.MODULO:
                return part._update_modulo(temp_var, lhs, rhs, approximate)
            return part._update_division(temp_var, lhs, rhs, approximate)

        try:
            return self._split_on_finite_support(update, left, right)
        except ZeroDivisorError as error:
            if error.expression is not None:
                raise
            raise ZeroDivisorError(error.reason, engine_text_of(expression)) from error
        except NotAnIntegerError as error:
            # the engines named temporary variables (SymPy) or the values of a case split (GiNaC, #361)
            raise NotAnIntegerError(f"Cannot compute {spelled(expression)} because it is not always an "
                                    f"integer") from error

    def _split_on_finite_support(self, update: Callable[[PGF, str, str], PGF],
                                 left: str, right: str) -> PGF:
        """
        `update(self, left, right)`, a binary update whose operands are variables or literals. If it
        raises `NonFiniteMarginalError` (a case split over the values of an operand on a measure with
        infinite mass, #242), the operands with a finite support (:meth:`finite_support`) are fixed
        instead (:meth:`_split_over_finite_supports`, #249). If no operand has a finite support, or the
        update of a part still needs a marginal that is not finite, the error is raised.
        """
        try:
            return update(self._pgf(), left, right)
        except NonFiniteMarginalError:
            result = self._split_over_finite_supports(update, left, right, (left, right))
            if result is None:
                raise
            return result

    def _split_over_finite_supports(self, update: Callable[[PGF, str, str], PGF],
                                    left: str | int, right: str | int,
                                    names: Iterable[str | int]) -> Optional[PGF]:
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
                    for name in dict.fromkeys(engine_text_of(name) for name in names) if name in variables}
        fixed = [name for name, support in supports.items() if support is not None]
        if not fixed:
            return None
        support = supports[fixed[0]] if len(fixed) == 1 else self.finite_support(*fixed)
        assert support is not None
        result: Optional[PGF] = None
        for state in self._unique_states(support):
            values = state.valuations
            section = self.filter_state(State({name: values.get(name, 0) for name in fixed}))
            try:
                left_text, right_text = engine_text_of(left), engine_text_of(right)
                part = update(section,
                              str(values.get(left_text, 0)) if left_text in fixed else left_text,
                              str(values.get(right_text, 0)) if right_text in fixed else right_text)
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

    def _case_split_support(self, variable: str) -> Optional[PGF]:
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
                              message: str) -> PGF:
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
                            approximate: str | float | None) -> Tuple[PGF, str, str]:
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
                          approximate: str | float | None) -> Tuple[PGF, PGF]:
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
        text = engine_text_of(right).strip()
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

    def _update_monus(self, temp_var: str, minuend: str | int, subtrahend: str | int) -> PGF:
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
            text = engine_text_of(operand).strip()
            if text in variables:
                operands.append(VarExpr(text))
            elif text.isdigit():
                operands.append(NatLitExpr(int(text)))
            else:
                raise NotImplementedError(
                    f"Cannot compute '{minuend} - {subtrahend}' (truncated at 0) because '{text}' is not a program "
                    "variable or a natural number")
        left, right = operands[0], operands[1]
        if isinstance(left, NatLitExpr) and isinstance(right, NatLitExpr):
            return self._update_var(temp_var, max(left.value - right.value, 0))
        if isinstance(left, VarExpr) and isinstance(right, VarExpr) and left.var == right.var:
            return self._update_var(temp_var, 0)
        if isinstance(right, NatLitExpr) and right.value == 0:
            return self._update_var(temp_var, engine_text_of(minuend))
        if isinstance(left, VarExpr) and isinstance(right, VarExpr) \
                and self._subtraction_never_negative(engine_text_of(minuend), engine_text_of(subtrahend)):
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
        :mod:`prodigy.distribution.pgf.rules.negative_values`).
        """
        # pylint: disable=unused-argument
        return False

    @staticmethod
    def _not_an_integer_quotient(numerator: str | int, denominator: str | int, temp_var: str) -> NotAnIntegerError:
        """The error of a division `temp_var := numerator / denominator` whose quotient is not a natural number
        in some state of positive probability, in the same words on all engines (#361). The operands are
        the engine's: temporary names or the values of a case split; :meth:`update` names the program's term."""
        return NotAnIntegerError(f"Cannot assign {numerator} / {denominator} to {temp_var} because it is not "
                                 f"always an integer")

    def _update_floor_division(self, temp_var: str, numerator: str | int, divisor: str | int,
                               approximate: str | float | None, text: str | None = None) -> PGF:
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
        if not engine_text_of(divisor).isdigit() or int(engine_text_of(divisor)) < 1:
            raise ValueError(f"Cannot compute {text}: the divisor must be a positive natural "
                             f"constant (not implemented otherwise).")
        q = int(engine_text_of(divisor))
        if engine_text_of(numerator) not in self.get_variables():
            value = Fraction(engine_text_of(numerator))
            if value.denominator != 1:
                raise ValueError(f"Cannot compute {text}: not a natural number")
            return self._update_var(temp_var, value.numerator // q)
        if q == 1:
            return self._update_var(temp_var, engine_text_of(numerator))
        shifted = self.get_fresh_variable({temp_var})
        result: PGF | None = None
        # pylint: disable=protected-access
        for remainder, section in enumerate(self._arithmetic_progression(engine_text_of(numerator), engine_text_of(q))):
            if section.is_zero_dist():
                continue
            part = section.set_variables(*section.get_variables(), shifted)
            part = part._update_subtraction(shifted, engine_text_of(numerator), engine_text_of(remainder))
            try:
                part = part._update_division(temp_var, shifted, engine_text_of(q), approximate)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot compute the floor division by {q} exactly: it is implemented for a dividend with "
                    f"finite support or with a generating function that is rational in it ({exc}).") from exc
            part = part._marginal(shifted, method=MarginalType.EXCLUDE)
            result = part if result is None else result + part
        # pylint: enable=protected-access
        if result is None:
            return self._update_var(temp_var, 0)  # the zero distribution
        return result.set_variables(*self.get_variables()).set_parameters(*self.get_parameters())

    def update_iid(self, sampling_dist: Expr, count: Union[VarExpr, NatLitExpr],
                   variable: Union[str, VarExpr]) -> PGF:
        name = engine_text_of(variable)
        if isinstance(sampling_dist, FunctionCallExpr) and sampling_dist.function == "sample_pgf":
            # iid(sample_pgf(G, y), n): the summands have the generating function G in y; the engines
            # handle G like the generating function given to iid directly
            generating_function, pgf_variable = sampling_dist.params[0]
            if pgf_variable != VarExpr(name):
                raise NotImplementedError(
                    f"The sampling {name} := iid({sampling_dist}, {count}) reads the generating function in "
                    f"{pgf_variable}, but only sampling in the assigned variable {name} is supported.")
            sampling_dist = generating_function
        self._require_convergent_evaluation(self.get_variables() - {name},
                                            f"The sampling {name} := iid({sampling_dist}, {count})")
        return self._update_iid(sampling_dist, count, variable)
