"""
The shared logic of generating functions, third layer (#511): updates, with the case splits over the values of
operands and the operations that the engines do not compute themselves (truncated subtraction, floor division).
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Set, Tuple, Union


from prodigy.language import Binop, BinopExpr, BoolLitExpr, Expr, FunctionCallExpr, NatLitExpr, RealLitExpr, VarExpr
from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.errors import NonFiniteMarginalError, NotAnIntegerError, ZeroDivisorError
if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.values import MarginalType, State
from prodigy.distribution.terms import constant_value, spelled, variables_of, without_fractional_constants
from prodigy.distribution.pgf.messages import in_programs_terms
from prodigy.distribution.pgf.filtering import PGFFiltering
from prodigy.distribution.pgf.sampling import is_sampling_distribution, refutes_unit_mass, sampling_pgf
from prodigy.distribution.pgf.rules.exact_division import divide_exponents

# The operators whose update is a case split over the values of their operands (_binary_by_cases)
_SYMBOLS = {Binop.TIMES: "*", Binop.DIVIDE: "/", Binop.MODULO: "%", Binop.POWER: "**", Binop.FLOOR_DIVIDE: "//"}
_NAMES = {Binop.TIMES: "multiplication", Binop.DIVIDE: "division", Binop.MODULO: "modulo",
          Binop.POWER: "exponentiation", Binop.FLOOR_DIVIDE: "floor division"}


def binary_value(operator: Binop, left: int, right: int) -> int:
    """
    The value of ``left operator right`` on natural numbers (``*``, ``/``, ``%``, ``**``). Raises
    ``NotAnIntegerError`` for a quotient that is not a natural number and ``ZeroDivisorError`` for a divisor 0.

    >>> binary_value(Binop.POWER, 2, 10), binary_value(Binop.MODULO, 7, 3), binary_value(Binop.DIVIDE, 6, 3)
    (1024, 1, 2)
    """
    if operator in (Binop.DIVIDE, Binop.MODULO) and right == 0:
        raise ZeroDivisorError("The modulus has to be positive, got 0." if operator == Binop.MODULO
                               else "Division by zero: the divisor is 0.")
    if operator == Binop.TIMES:
        return left * right
    if operator == Binop.MODULO:
        return left % right
    if operator == Binop.POWER:
        return left ** right
    if operator == Binop.DIVIDE:
        if left % right != 0:
            raise NotAnIntegerError(f"Cannot compute {left} / {right} because it is not always an integer")
        return left // right
    raise ValueError(f"Unsupported binary operator: {operator}")


def _literal_value(expression: Expr, variable: str) -> Optional[int]:
    """The natural number that `expression` is if it contains no variables (a literal, `true`/`false` as 1/0,
    #283, or a constant expression like `9/2*2`), else None. Raises ValueError for a constant that is no
    natural number."""
    if isinstance(expression, NatLitExpr):
        return expression.value
    if isinstance(expression, BoolLitExpr):
        return int(expression.value)
    if isinstance(expression, RealLitExpr):
        constant: Optional[Fraction] = expression.to_fraction()
    elif isinstance(expression, BinopExpr):
        constant = constant_value(expression)
    else:
        return None
    if constant is None:
        return None
    if constant.denominator != 1:
        shown = str(expression) if isinstance(expression, RealLitExpr) else str(constant)
        raise ValueError(f'Cannot assign the real value {shown} to {variable}')
    return constant.numerator


def _approximated_operands(operator: Binop, left: str, right: str, variables: Set[str]) -> Tuple[List[str], str]:
    """The operands of `left operator right` whose marginals the split of step 7 of `_binary_by_cases` approximates,
    and the message if no approximation is given: the right operand if it is a variable, and the left one as well
    if it stays inexact with a number on the right (`/`, `**`)."""
    left_variable, right_variable = left in variables, right in variables
    infinite = [name for name, is_variable in ((left, left_variable), (right, right_variable)) if is_variable]
    message = (f"Cannot compute {left} {_SYMBOLS[operator]} {right} exactly: {' and '.join(dict.fromkeys(infinite))} "
               f"{'have' if len(set(infinite)) > 1 else 'has'} infinite range, and the case split over the values is "
               f"not implemented without an approximation")
    approximated = [right] if right_variable else []
    if left_variable and left not in approximated and (not right_variable or operator in (Binop.DIVIDE, Binop.POWER)):
        approximated.append(left)
    return approximated, message


def _monus_without_split(left: Expr, right: Expr) -> Union[int, str, None]:
    """The value of `left - right` (truncated at 0) if no case split is needed: two numbers, `x - x` (0), `x - 0`
    (the variable's name); None otherwise."""
    if isinstance(left, NatLitExpr) and isinstance(right, NatLitExpr):
        return max(left.value - right.value, 0)
    if isinstance(left, VarExpr) and isinstance(right, VarExpr) and left.var == right.var:
        return 0
    if isinstance(right, NatLitExpr) and right.value == 0:
        return engine_text_of(left)
    return None


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

        value = _literal_value(expression.rhs, variable)
        if value is not None:
            # e.g. "x = 9/2*2": the literal is assigned
            return self._update_var(variable, value)
        # Not just a literal assignment, need to evaluate the expression first
        result, _ = self._evaluate(expression.rhs, variable, approximate)
        return result

    def _evaluate(self, expression: Expr, temp_var: str, approximate: str | float | None) -> Tuple[PGF, str]:
        """
        Evaluates `expression` for an update: returns the distribution with the value of a composite expression
        in the variable `temp_var`, and the name of that variable, or the distribution unchanged and the text of
        a variable or literal. Rational constants are computed exactly (:func:`constant_value`) or rewritten into
        natural-number arithmetic (:func:`without_fractional_constants`).
        """
        if isinstance(expression, VarExpr):
            return self._update_var(temp_var, expression.var), temp_var
        if isinstance(expression, (NatLitExpr, RealLitExpr, BoolLitExpr)):
            # booleans are the values 1 (true) and 0 (false) of the generating function (#283)
            value = int(expression.value) if isinstance(expression, BoolLitExpr) else expression.value
            return self._pgf(), engine_text_of(value)
        if not isinstance(expression, BinopExpr):
            raise ValueError(f"Unsupported type of subexpression: {expression}")
        # Subexpressions without variables are computed directly, e.g. 9/2*2 becomes 9
        constant = constant_value(expression)
        if constant is not None:
            return self._pgf(), engine_text_of(constant)
        # Constants that are not natural numbers are rewritten, e.g. 1/2*x becomes (1*x)/2
        rewritten = without_fractional_constants(expression)
        if rewritten is not None:
            return self._evaluate(rewritten, temp_var, approximate)
        return self._evaluate_binop(expression, temp_var, approximate), temp_var

    def _evaluate_binop(self, expression: BinopExpr, temp_var: str, approximate: str | float | None) -> PGF:
        """`temp_var := expression` for a binary operator: the operands are evaluated into two fresh variables
        (no clash with the variables they read), the operator is applied, and the fresh variables are summed out."""
        if expression.operator in _NAMES:
            # a case split needs the values of its operands; a parameter has none (B1 of #511)
            for operand in (expression.lhs, expression.rhs):
                if isinstance(operand, VarExpr) and operand.var in self.get_parameters():
                    raise NotImplementedError(
                        f"Cannot compute {spelled(expression)}: {_NAMES[expression.operator]} with the "
                        f"parameter {operand.var} is not implemented")
        xl = self.get_fresh_variable()
        xr = self.get_fresh_variable({xl})
        f = self.set_variables(*(self.get_variables() | {xl, xr}))
        f, t_1 = f._evaluate(expression.lhs, xl, approximate)  # pylint: disable=protected-access
        f, t_2 = f._evaluate(expression.rhs, xr, approximate)  # pylint: disable=protected-access
        # pylint: disable=protected-access
        if expression.operator == Binop.PLUS:
            f = f._update_sum(temp_var, t_1, t_2)
        elif expression.operator == Binop.MINUS:
            try:
                f = f._update_monus(temp_var, t_1, t_2)
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"Cannot compute {expression} exactly: subtraction on natural numbers is truncated "
                    f"at 0, which needs the states with {expression.lhs} < {expression.rhs}, and these "
                    f"cannot be filtered here (not implemented). Details: {error}") from error
        elif expression.operator in (Binop.TIMES, Binop.MODULO, Binop.DIVIDE, Binop.POWER):
            f = f._binary_update(expression, temp_var, t_1, t_2, approximate)
        elif expression.operator == Binop.FLOOR_DIVIDE:
            f = f._update_floor_division(temp_var, t_1, t_2, approximate, spelled(expression))
        else:
            raise ValueError(f"Unsupported binary operator: {expression.operator}")
        # the fresh variables are only there to prevent clashes
        return f._marginal(xl, xr, method=MarginalType.EXCLUDE)
        # pylint: enable=protected-access

    def _binary_update(self, expression: BinopExpr, temp_var: str, left: str, right: str,
                       approximate: str | float | None) -> PGF:
        """
        `temp_var = left op right` for the operands `left`, `right` of `expression` (`*`, `/`, `%`, `**`) by
        :meth:`_binary_by_cases`; the errors name the program's terms instead of the temporary variables (#349),
        a divisor 0 names `expression` (#273).
        """
        try:
            return self._binary_by_cases(expression.operator, temp_var, left, right, approximate)
        except ZeroDivisorError as error:
            if error.expression is not None:
                raise
            raise ZeroDivisorError(error.reason, engine_text_of(expression)) from error
        except NotAnIntegerError as error:
            # the operands were temporary variables or the values of a case split (#361)
            raise NotAnIntegerError(f"Cannot compute {spelled(expression)} because it is not always an "
                                    f"integer") from error
        except (ValueError, NotImplementedError, NonFiniteMarginalError) as error:
            raise in_programs_terms(
                error, {engine_text_of(left): spelled(expression.lhs), engine_text_of(right): spelled(expression.rhs)}
            ) from error

    def _binary_by_cases(self, operator: Binop, temp_var: str, left: str | int, right: str | int,
                         approximate: str | float | None) -> PGF:
        """
        The update `temp_var := left operator right` for `*`, `/`, `%`, `**` (#511, P4c), with operands that are
        variables or natural numbers; the one case analysis of all engines:

        1. a divisor 0 (`/`, `%`) raises `ZeroDivisorError` (:meth:`_require_nonzero_divisor`), also for each
           value of a case split below;
        2. two numbers: :func:`binary_value`, on a distribution of positive mass (a quotient that is no
           natural number on a zero section is no error);
        3. `*` with a number: the substitution of the engine hook `_update_scaled` (exact everywhere);
        4. a finite distribution: the engine hook `_termwise`, if the engine has it for the operator;
        5. the operands with a finite support are fixed, value by value (:meth:`_split_over_finite_supports`:
           the support, not the marginal, #256), and the parts are updated with numbers;
        6. exact cases of an operand with infinitely many values: `% q` by its residue classes
           (`_arithmetic_progression`), `/ q` if all mass lies on multiples of `q`
           (:mod:`~prodigy.distribution.pgf.rules.exact_division`);
        7. otherwise a case split over the approximated marginals (:meth:`_approximated_support`) of the right
           operand if it is a variable, and of the left one if the right one is a number or the operator is `/`
           or `**` (which stay inexact with a number on the right), all approximated on this distribution:
           without `approximate`, the one `ValueError` "not implemented without an approximation"; with a
           marginal that is no finite measure, `NonFiniteMarginalError`.

        Parameters are no operands here; :meth:`update` refuses them before (B1 of #511).
        """
        if operator in (Binop.DIVIDE, Binop.MODULO):
            self._require_nonzero_divisor(right, _SYMBOLS[operator])
        left_text, right_text = engine_text_of(left).strip(), engine_text_of(right).strip()
        left_variable, right_variable = (text in self.get_variables() for text in (left_text, right_text))
        if not (left_variable or right_variable):
            return self._literal_update(operator, temp_var, _natural(left_text), _natural(right_text))
        if operator == Binop.TIMES and not (left_variable and right_variable):
            scaled = (left_text, _natural(right_text)) if left_variable else (right_text, _natural(left_text))
            return self._update_scaled(temp_var, *scaled)
        if self.is_finite():
            termwise = self._termwise(operator, temp_var, left_text, right_text)
            if termwise is not None:
                return termwise

        def update(part: PGF, lhs: str, rhs: str) -> PGF:
            return part._binary_by_cases(operator, temp_var, lhs, rhs, approximate)  # pylint: disable=protected-access

        split = self._split_over_finite_supports(update, left_text, right_text, (right_text, left_text))
        if split is not None:
            return split
        # every variable among the operands takes infinitely many values
        exact = None if right_variable else self._by_constant(operator, temp_var, left_text, _natural(right_text))
        if exact is not None:
            return exact
        return self._approximated_split(operator, temp_var, (left_text, right_text), approximate)

    def _approximated_split(self, operator: Binop, temp_var: str, operands: Tuple[str, str],
                            approximate: str | float | None) -> PGF:
        """Step 7 of :meth:`_binary_by_cases`: `temp_var := left operator right` as the sum over the states of the
        approximated marginals of the operands that are variables with infinitely many values."""
        left_text, right_text = operands
        approximated, message = _approximated_operands(operator, left_text, right_text, self.get_variables())
        supports = [self._approximated_support(name, approximate, message) for name in approximated]
        result: Optional[PGF] = None
        for states in itertools.product(*(list(self._unique_states(support)) for support in supports)):
            values = {name: state[name] for name, state in zip(approximated, states)}
            part = self.filter_state(State(values))._binary_by_cases(  # pylint: disable=protected-access
                operator, temp_var, str(values.get(left_text, left_text)), str(values.get(right_text, right_text)),
                approximate)
            result = part if result is None else result + part
        return result if result is not None else self.filter(BoolLitExpr(False))

    def _literal_update(self, operator: Binop, temp_var: str, left: int, right: int) -> PGF:
        """`temp_var := left operator right` for two numbers (:func:`binary_value`); a quotient that is no natural
        number is an error only where the distribution has mass (a value without mass, see is_provably_zero)."""
        try:
            value = binary_value(operator, left, right)
        except NotAnIntegerError:
            if self.is_provably_zero():
                return self._update_var(temp_var, 0)
            raise
        return self._update_var(temp_var, value)

    def _by_constant(self, operator: Binop, temp_var: str, left: str, right: int) -> Optional[PGF]:
        """The exact cases of `left operator right` for a variable `left` with infinitely many values and a
        constant `right`: `%` by the residue classes (:meth:`_residue_update`), `/` if all mass lies on multiples
        of `right` (:func:`~prodigy.distribution.pgf.rules.exact_division.divide_exponents`); None otherwise."""
        if operator == Binop.MODULO:
            return self._residue_update(temp_var, left, str(right))
        if operator == Binop.DIVIDE:
            divided = divide_exponents(self._sympy_expression(), left, temp_var, right)
            if divided is not None:
                return self._from_sympy_expression(divided)
        return None

    def _residue_update(self, temp_var: str, left: str, modulus: str) -> PGF:
        """`temp_var := left % modulus` for a constant `modulus`, from the residue classes of `left` (exact for every
        measure, #249)."""
        result: Optional[PGF] = None
        for remainder, section in enumerate(self._arithmetic_progression(left, modulus)):
            part = section._update_var(temp_var, remainder)  # pylint: disable=protected-access
            result = part if result is None else result + part
        assert result is not None
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
        supports = {name: self.finite_support(name)
                    for name in dict.fromkeys(engine_text_of(name) for name in names) if name in self.get_variables()}
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
        value = _monus_without_split(left, right)
        if value is not None:
            return self._update_var(temp_var, value)
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
                part = part._binary_by_cases(Binop.DIVIDE, temp_var, shifted, engine_text_of(q), approximate)
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
        if isinstance(sampling_dist, FunctionCallExpr):
            if not is_sampling_distribution(sampling_dist.function):
                raise NotImplementedError(f"Unsupported distribution: {sampling_dist}")
            text = sampling_pgf(sampling_dist.function, sampling_dist.params[0], name).text
        else:
            text = engine_text_of(sampling_dist)
            if refutes_unit_mass(text, name):
                raise ValueError(f"The sampling {name} := iid({sampling_dist}, {count}) needs a probability "
                                 f"generating function of mass 1, but {sampling_dist} has another mass.")
        # the names of the state stay variables in the sample (a parameter of its distribution may be one)
        sample = self.from_expr(text, name, *sorted(self.get_variables() - {name}))
        return self._compose_iid(sample, count.value if isinstance(count, NatLitExpr) else count.var, name)


def _natural(text: str) -> int:
    """The natural number `text` (an operand of a case split that is no variable)."""
    value = Fraction(text)
    if value.denominator != 1 or value < 0:
        raise ValueError(f"Cannot compute with {text}: not a natural number")
    return value.numerator
