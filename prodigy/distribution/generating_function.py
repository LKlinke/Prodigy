# pylint: disable=protected-access
from __future__ import annotations

import operator
from fractions import Fraction
from typing import (Any, Callable, FrozenSet, Generator, Iterator, List, Set,
                    Tuple, Type, Union, get_args)

import sympy
from probably.pgcl import (Binop, BinopExpr, Expr, FunctionCallExpr,
                           NatLitExpr, RealLitExpr, Unop, UnopExpr, VarExpr)
from probably.pgcl.parser import parse_expr
from probably.util.ref import Mut
from sympy.assumptions.assume import global_assumptions

# TODO Implement these checks in probably
from prodigy.distribution import (CommonDistributionsFactory, Distribution,
                                  DistributionParam, MarginalType, State)
from prodigy.pgcl.pgcl_checks import has_variable
from prodigy.util.logger import log_setup, logging

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG, file="GF_operations.log")


def _term_generator(function: sympy.Poly):
    assert isinstance(function,
                      sympy.Poly), "Terms can only be generated for finite GF"
    poly = function
    while poly.as_expr() != 0:
        yield poly.EC(), poly.EM().as_expr()
        poly -= poly.EC() * poly.EM().as_expr()


def _parse_to_sympy(expr: Any, real=True, **kwargs) -> sympy.Expr:
    """
    Parses something (`float`s, `bool`s, `Fraction`s, `probably.Expr`s, anything that can be converted to a valid
    string) to a sympy expression. Can convert a probably expression faster than passing it as a string to sympy.
    Also will correctly determine if something is a symbol or a function (e.g., if we have a variable called `exp` or
    `log`). All symbols are given the assumptions that they are real unless specified otherwise.
    """

    # first we cover some cases that we can handle faster than by parsing a string
    if getattr(expr, '__sympy__', None):
        return expr
    if isinstance(expr, (float, bool, Fraction)):
        return sympy.S(expr, rational=True, **kwargs)

    s: sympy.Expr

    def get_function(name: str):
        functions = {'sum': sum}
        if name in functions:
            return functions[name]
        return getattr(sympy, name)

    if isinstance(expr, get_args(Expr)):

        def probably_to_sympy(prob_expr: Expr) -> sympy.Expr:
            if isinstance(prob_expr, VarExpr):
                return _sympy_symbol(prob_expr.var, real, **kwargs)
            if isinstance(prob_expr, NatLitExpr):
                return sympy.Integer(prob_expr.value)
            if isinstance(prob_expr, RealLitExpr):
                frac = prob_expr.to_fraction()
                return sympy.Rational(frac.numerator, frac.denominator)
            if isinstance(prob_expr, FunctionCallExpr):
                if len(prob_expr.params[1]) != 0:
                    raise ValueError(
                        f'Cannot parse function with named parameters: {prob_expr}'
                    )
                return get_function(prob_expr.function)(
                    *map(probably_to_sympy, prob_expr.params[0]))
            if isinstance(prob_expr, BinopExpr):
                op = prob_expr.operator
                if op == Binop.DIVIDE:
                    return probably_to_sympy(
                        prob_expr.lhs) / probably_to_sympy(prob_expr.rhs)
                if op == Binop.MINUS:
                    return probably_to_sympy(
                        prob_expr.lhs) - probably_to_sympy(prob_expr.rhs)
                if op == Binop.TIMES:
                    return probably_to_sympy(
                        prob_expr.lhs) * probably_to_sympy(prob_expr.rhs)
                if op == Binop.POWER:
                    return probably_to_sympy(prob_expr.lhs) ** probably_to_sympy(
                        prob_expr.rhs)
                if op == Binop.MODULO:
                    return probably_to_sympy(
                        prob_expr.lhs) % probably_to_sympy(prob_expr.rhs)
                if op == Binop.PLUS:
                    return probably_to_sympy(
                        prob_expr.lhs) + probably_to_sympy(prob_expr.rhs)
                raise ValueError(f'Unsupported operand: {op}')
            raise ValueError(f'Unsupported expression type: {prob_expr}')

        s = probably_to_sympy(expr)

    else:
        s = sympy.parse_expr(
            str(expr),
            transformations=sympy.parsing.sympy_parser.standard_transformations
                            + (sympy.parsing.sympy_parser.convert_xor,
                               sympy.parsing.sympy_parser.rationalize),
            global_dict={
                'Symbol': lambda x: _sympy_symbol(x, real, **kwargs),
                'Function': get_function,
                'Integer': sympy.Integer,
                'Float': sympy.Float,
                'Rational': sympy.Rational
            })

    return s


def _sympy_symbol(name: Any, real=True, **kwargs) -> sympy.Symbol:
    """
    Creates a real sympy Symbol.
    
    Note that `sympy.Symbol('x') == sympy.Symbol('x', real=True)` evaluates to false, as variables
    with the same name but different domains are not considered equal by sympy.
    """
    return sympy.Symbol(name, real=real, **kwargs)


class GeneratingFunction(Distribution):
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """

    use_simplification = False
    use_latex_output = False

    # ==================================== CONSTRUCTORS ====================================

    def __init__(self,
                 function: Union[str, sympy.Expr, Expr],
                 *variables: Union[str, sympy.Symbol],
                 closed: bool | None = None,
                 finite: bool | None = None):

        # Set the basic information.
        self._function: sympy.Expr = _parse_to_sympy(function)

        # Set variables and parameters
        self._variables: Set[
            sympy.Symbol] = self._function.free_symbols  # type: ignore
        self._parameters: Set[sympy.Symbol] = set()
        if variables:
            self._variables = self._variables.union(
                map(lambda v: _sympy_symbol(str(v)),
                    filter(lambda v: v != "", variables)))
            self._parameters = self._variables.difference(
                map(lambda v: _sympy_symbol(str(v)),
                    filter(lambda v: v != "", variables)))
            self._variables -= self._parameters

        # pylint: disable = too-many-function-args
        for var in self._variables:
            global_assumptions.add(sympy.Q.nonnegative(var))
        for param in self._parameters:
            global_assumptions.add(sympy.Q.positive(param))
        # pylint: enable = too-many-function-args

        # Do closed form and finiteness heuristics
        self._is_closed_form = closed if closed else not self._function.is_polynomial(
            *self._variables)
        self._is_finite = finite if finite else self._function.ratsimp(
        ).is_polynomial(*self._variables)

    @staticmethod
    def factory() -> Type[CommonDistributionsFactory]:
        return SympyPGF

    def _filter_constant_condition(self,
                                   condition: Expr) -> GeneratingFunction:
        # Normalize the condition into the format _var_ (< | <= | =) const. I.e., having the variable on the lhs.

        # FIXME variable comparison does not work as condition.rhs.var is str and self._var of type
        #   set[sympy.Symbol]
        if isinstance(condition.rhs,
                      VarExpr) and condition.rhs.var not in self._variables:
            if condition.operator == Binop.LEQ:
                return self.filter(
                    UnopExpr(operator=Unop.NEG,
                             expr=BinopExpr(operator=Binop.LT,
                                            lhs=condition.rhs,
                                            rhs=condition.lhs)))
            elif condition.operator == Binop.LT:
                return self.filter(
                    UnopExpr(operator=Unop.NEG,
                             expr=BinopExpr(operator=Binop.LEQ,
                                            lhs=condition.rhs,
                                            rhs=condition.lhs)))
            elif condition.operator == Binop.GT:
                return self.filter(
                    BinopExpr(operator=Binop.LT,
                              lhs=condition.rhs,
                              rhs=condition.lhs))
            elif condition.operator == Binop.GEQ:
                return self.filter(
                    BinopExpr(operator=Binop.LEQ,
                              lhs=condition.rhs,
                              rhs=condition.lhs))

        # FIXME variable comparison does not work as condition.rhs.var is str and self._var of type
        #   set[sympy.Symbol]
        if isinstance(condition.lhs,
                      VarExpr) and condition.lhs.var not in self._variables:
            if condition.operator == Binop.GT:
                return self.filter(
                    UnopExpr(operator=Unop.NEG,
                             expr=BinopExpr(operator=Binop.LEQ,
                                            lhs=condition.lhs,
                                            rhs=condition.rhs)))

            elif condition.operator == Binop.GEQ:
                return self.filter(
                    BinopExpr(operator=Binop.LEQ,
                              lhs=condition.rhs,
                              rhs=condition.lhs))

        # Now we have an expression of the form _var_ (< | <=, =) _const_.
        variable = _sympy_symbol(str(condition.lhs))
        constant = condition.rhs.value
        result = 0
        ranges = {
            Binop.LT: range(constant),
            Binop.LEQ: range(constant + 1),
            Binop.EQ: [constant]
        }

        # Compute the probabilities of the states _var_ = i where i ranges depending on the operator (< , <=, =).
        for i in ranges[condition.operator]:
            result += (self._function.diff(variable, i) / sympy.factorial(i)
                       ).limit(variable, 0, '-') * variable ** i

        return GeneratingFunction(result,
                                  *self._variables,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def _mult_term_generator(self):
        """
            Generates terms of multivariate generating function in `grlex` order.
        """
        i = 1
        while True:
            # The easiest method is to create all monomials of until total degree `i` and sort them.
            logger.debug("Generating and sorting of new monomials")
            new_monomials = sorted(
                sympy.polys.monomials.itermonomials(self._variables, i),
                key=sympy.polys.orderings.monomial_key("grlex",
                                                       list(self._variables)))
            # instead of repeating all the monomials for higher total degrees, just cut-off the already created ones.
            if i > 1:
                new_monomials = new_monomials[
                                sympy.polys.monomials.
                                monomial_count(len(self._variables), i - 1):]
            logger.debug("Monomial_generation done")

            # after we have the list of new monomials, create (probability, state) pais and yielding them.
            for monomial in new_monomials:
                state = {}
                if monomial == 1:
                    for var in self._variables:
                        state[var.name] = 0
                else:
                    state = {
                        str(var): val
                        for var, val in
                        monomial.as_expr().as_powers_dict().items()
                    }
                yield self._probability_of_state(state), monomial.as_expr()
            logger.debug("\t>Terms generated until total degree of %d", i)
            i += 1

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[GeneratingFunction]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        a = _parse_to_sympy(modulus)
        var = _sympy_symbol(variable)

        # This should be a faster variant for univariate distributions.
        if self._variables == {var} and self._function.is_rational_function():
            result = []
            for remainder in range(a):
                other = GeneratingFunction(f"({var}^{remainder})/(1-{var}^{modulus})", *self._variables)
                result.append(self.hadamard_product(other))
            return result

        # This is the general algorithm
        primitive_uroot = sympy.exp(2 * sympy.pi * sympy.I / a)
        result = []
        for remainder in range(a):  # type: ignore
            psum = 0
            for m in range(a):  # type: ignore
                psum += primitive_uroot ** (-m *
                                            remainder) * self._function.subs(
                    var, (primitive_uroot ** m) * var)
            result.append(
                GeneratingFunction(f"(1/{a}) * ({psum})",
                                   *self._variables,
                                   closed=self._is_closed_form,
                                   finite=self._is_finite))
        return result

    def hadamard_product(self, other: Distribution) -> GeneratingFunction:
        """
            Computes the Hadamard product of two univariate rational functions, i.e. the point-wise product of the
            coefficients in their formal power series representation.
        """
        logger.debug("Compute Hadamard product of %s and %s", self, other)
        if len(self._variables) > 1:
            raise NotImplementedError("Currently only univariate distributions are supported for the Hadamard product.")
        var = next(iter(self._variables))
        if isinstance(other, GeneratingFunction):
            if not other._variables == self._variables:
                raise NotImplementedError("Only univariate distributions are supported for the Hadamard product.")

            s_f, s_g = self._function, other._function

            if not (s_f.is_rational_function() and s_g.is_rational_function()):
                raise ValueError(f"Distributions need to be rational functions. Got {s_f} and {s_g}")

            if s_f.is_polynomial(var):
                s_f, s_g = s_g, s_f
            if s_g.is_polynomial(var):
                if s_g.is_constant(var):
                    return GeneratingFunction(s_g * s_f.subs(var, 0), var)
                g_deg = s_g.as_poly(var).degree()
                f_series = s_f.series(var, n=g_deg + 1).removeO().as_poly(var)
                result = sympy.polys.Poly(reversed([cf * cg for cf, cg in zip(reversed(f_series.all_coeffs()), reversed(
                    s_g.as_poly(var).all_coeffs()))]), var)
                return GeneratingFunction(result.as_expr(), var)

            _, s_f_denom = s_f.as_numer_denom()
            _, s_g_denom = s_g.as_numer_denom()
            f_degree = s_f_denom.as_poly(var).degree()
            g_degree = s_g_denom.as_poly(var).degree()
            logger.debug("Degree f: %i\nDegree g: %i", f_degree, g_degree)

            s_compm_f = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
                (var ** f_degree * s_f_denom.subs(var, 1 / var)).as_poly(var).monic()))
            s_compm_g = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
                (var ** g_degree * s_g_denom.subs(var, 1 / var)).as_poly(var).monic()))

            s_result_denom = sympy.det(
                sympy.eye(f_degree * g_degree) - (sympy.matrices.kronecker_product(s_compm_f, s_compm_g)) * var)
            logger.debug("Denominator: %s", s_result_denom)
            logger.debug("Comparing %i terms", f_degree * g_degree)

            result = sympy.polys.Poly(reversed([cf * cg for cf, cg in zip(reversed(
                s_f.series(var, n=f_degree * g_degree).removeO().as_poly(var).all_coeffs()), reversed(
                s_g.series(var, n=f_degree * g_degree).removeO().as_poly(var).all_coeffs()))]), var)
            result *= s_result_denom
            s_num = result.as_expr().series(var, n=f_degree * g_degree).removeO()

            logger.debug("Resulting function: %s", s_num / s_result_denom)
            return GeneratingFunction(str(s_num / s_result_denom), *self._variables)
        
        raise TypeError(f"Cannot compute the Hadamard Product, where {other=} is of type {type(other)}.")

        raise ValueError("Hadamard product is only applicable with distributions.")

    @staticmethod
    def _split_addend(addend: sympy.Expr) -> Tuple[sympy.Expr, sympy.Expr]:
        r"""
        This method assumes that the addend is given in terms of :math:`\alpha \times X^{\sigma}`, where
        :math:`\alpha \in [0,1], \sigma \in \mathbb{N}^k`.

        :param addend: the addend to split into its factor and monomial.

        :return: a tuple (factor, monomial)
        """
        if addend.free_symbols == set():
            return addend, sympy.Integer(1)
        else:
            factor_powers = addend.as_powers_dict()
            result = (sympy.Integer(1), sympy.Integer(1))
            for factor in factor_powers:
                if factor in addend.free_symbols:
                    result = (result[0],
                              result[1] * factor ** factor_powers[factor])
                else:
                    result = (result[0] * factor ** factor_powers[factor],
                              result[1])
            return result

    def _monomial_to_state(self, monomial: sympy.Expr) -> State:
        """ Converts a `monomial` into a state representation, i.e., a (variable_name, variable_value) dict.
            This is tied to a specific instance of a generating function as a variable and parameter context
            is needed to resolve the types of indeterminates in the monomial.
        """
        result = State()
        if monomial.free_symbols == set():
            for var in self._variables:
                result[str(var)] = 0
        else:
            variables_and_powers = monomial.as_powers_dict()
            for var in self._variables:
                if var in variables_and_powers.keys():
                    result[str(var)] = int(variables_and_powers[var])
                else:
                    result[str(var)] = 0
        return result

    def _probability_of_state(self, state: State) -> sympy.Expr:
        """
        Determines the probability of a single program state encoded by a monomial (discrete only).

        :param state: The queried program state.

        :return: The probability for that state.
        """
        logger.debug("probability_of(%s) call", state)

        # complete state by assuming every variable which is not occurring is zero.
        complete_state = state
        for s_var in self._variables:
            if str(s_var) not in state:
                complete_state[str(s_var)] = 0

        # When its a closed form, or not finite, do the maths.
        if self._is_closed_form or not self._is_finite:
            result = self._function
            for variable, value in complete_state.items():
                result = result.diff(_sympy_symbol(variable), value)
                result /= sympy.factorial(value)
                result = result.limit(_sympy_symbol(variable), 0, "-")
            return result
        # Otherwise use poly module and simply extract the correct coefficient from it
        else:
            monomial = 1
            for variable, value in complete_state.items():
                monomial *= _sympy_symbol(variable) ** value
            probability = self._function.as_poly(
                *self._variables).coeff_monomial(monomial)
            return probability if probability else sympy.core.numbers.Zero()

    # Distribution interface implementation

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
            probability_mass: str | float) -> GeneratingFunction:
        logger.debug("approximate_unilaterally(%s, %s) call on %s", variable,
                     probability_mass, self)
        mass = _parse_to_sympy(probability_mass)
        if mass == 0:
            return GeneratingFunction('0',
                                      *self._variables,
                                      closed=True,
                                      finite=True)
        elif mass > self.coefficient_sum():
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
                return GeneratingFunction(result, *self._variables)

        raise NotImplementedError("unreachable")

    def _update_division(
            self, temp_var: str, numerator: str | int, denominator: str | int,
            approximate: str | float | None) -> GeneratingFunction:
        update_var = _sympy_symbol(temp_var)
        div_1, div_2 = _parse_to_sympy(numerator), _parse_to_sympy(denominator)

        if div_1 in self._parameters or div_2 in self._parameters:
            raise ValueError('Division containing parameters is not allowed')

        marginal_l = self.marginal(
            numerator) if div_1 in self._variables else div_1
        marginal_r = self.marginal(
            denominator) if div_2 in self._variables else div_2

        if isinstance(marginal_l,
                      GeneratingFunction) and not marginal_l._is_finite:
            if approximate is None:
                raise ValueError(
                    f'Cannot perform the division {numerator} / {denominator} because at least one has infinite range'
                )
            assert isinstance(numerator, str)
            marginal_l = marginal_l.approximate_unilaterally(
                numerator, approximate)
        if isinstance(marginal_r,
                      GeneratingFunction) and not marginal_r._is_finite:
            if approximate is None:
                raise ValueError(
                    f'Cannot perform the division {numerator} / {denominator} because at least one has infinite range'
                )
            assert isinstance(denominator, str)
            marginal_r = marginal_r.approximate_unilaterally(
                denominator, approximate)

        result = 0
        if isinstance(marginal_l, GeneratingFunction):
            for _, state_l in marginal_l:
                if isinstance(marginal_r, GeneratingFunction):
                    for _, state_r in marginal_r:
                        val_l, val_r = state_l[numerator], state_r[denominator]
                        x = self.filter(
                            parse_expr(
                                f'{numerator}={val_l} & {denominator}={val_r}')
                        )._function
                        if val_l % val_r != 0 and x != 0:
                            raise ValueError(
                                f"Cannot assign {numerator} / {denominator} to {temp_var} " \
                                "because it is not always an integer"
                            )
                        result += x.subs(update_var,
                                         1) * update_var ** (val_l / val_r)
                else:
                    val_l, val_r = state_l[numerator], div_2
                    x = self.filter(
                        parse_expr(f'{numerator}={val_l}'))._function
                    if val_l % val_r != 0 and x != 0:
                        raise ValueError(
                            f"Cannot assign {numerator} / {denominator} to {temp_var} " \
                            "because it is not always an integer"
                        )
                    result += x.subs(update_var,
                                     1) * update_var ** (val_l / val_r)
        else:
            if isinstance(marginal_r, GeneratingFunction):
                for _, state_r in marginal_r:
                    val_l, val_r = div_1, state_r[denominator]
                    x = self.filter(
                        parse_expr(f'{denominator}={val_r}'))._function
                    if val_l % val_r != 0 and x != 0:
                        raise ValueError(
                            f"Cannot assign {numerator} / {denominator} to {temp_var} " \
                            "because it is not always an integer"
                        )
                    result += x.subs(update_var,
                                     1) * update_var ** (val_l / val_r)
            else:
                if div_1 % div_2 != 0:
                    raise ValueError(
                        f"Cannot assign {numerator} / {denominator} to {temp_var} because it is not always an integer"
                    )
                result = self._function.subs(update_var,
                                             1) * update_var ** (div_1 / div_2)

        return GeneratingFunction(
            result,
            *self._variables,
            closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> GeneratingFunction:
        # TODO this function is very inefficient and I'm not sure why (maybe because of all the filtering?)

        left_sym, right_sym = _sympy_symbol(str(left)), _sympy_symbol(
            str(right))
        if left_sym in self._parameters or right_sym in self._parameters:
            raise ValueError('Cannot perform modulo operation on parameters')

        update_var = _sympy_symbol(temp_var)
        result = 0

        # On finite GFs, iterate over all states
        if self._is_finite:
            for prob, state_r in self._iter_expr_state():
                if left_sym in self._variables:
                    left_var = state_r[left]
                else:
                    left_var = _parse_to_sympy(left)
                if right_sym in self._variables:
                    right_var = state_r[right]
                else:
                    right_var = _parse_to_sympy(right)
                result += prob * _parse_to_sympy(state_r.to_monomial()).subs(
                    update_var, 1) * update_var ** (left_var % right_var)

        # If the GF is infinite and right is a variable, it needs to have finite range
        elif right_sym in self._variables:
            assert isinstance(right, str)
            marginal_r = self.marginal(right)
            if not marginal_r._is_finite:
                if approximate is None:
                    raise ValueError(
                        f'Cannot perform modulo operation with infinite right hand side {right}'
                    )
                marginal_r = marginal_r.approximate_unilaterally(
                    right, approximate)
            for _, state_r in marginal_r:
                result += self.filter(
                    parse_expr(f'{right}={state_r[right]}'))._update_modulo(
                    temp_var, left, state_r[right], None)._function

        # If left is a variable, it doesn't have to have finite range
        elif left_sym in self._variables:
            assert isinstance(left, str)
            marginal_l = self.marginal(left)
            if marginal_l._is_finite:
                for _, state_l in marginal_l:
                    result += self.filter(
                        parse_expr(f'{left}={state_l[left]}'))._update_modulo(
                        temp_var, state_l[left], right, None)._function
            else:
                for index, gf in enumerate(
                        self._arithmetic_progression(left, str(right))):
                    # TODO this seems to compute the correct result, but it can't always be simplified to 0
                    result = result + gf._function.subs(update_var,
                                                        1) * update_var ** index

        # If both are not variables, simply compute the result
        else:
            return self._update_var(temp_var, int(left) % int(right))

        return GeneratingFunction(
            result,
            *self._variables,
            closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> GeneratingFunction:
        update_var = _sympy_symbol(temp_var)
        sub_1, sub_2 = _parse_to_sympy(sub_from), _parse_to_sympy(sub)
        result = self._function

        # we subtract a variable from another variable
        if sub_1 in self._variables and sub_2 in self._variables:
            if sub_2 == update_var:
                if sub_1 == update_var:
                    result = result.subs(update_var, 1)
                else:
                    result = result.subs([(update_var, update_var ** (-1)),
                                          (sub_1, sub_1 * update_var)])
            else:
                if not sub_1 == update_var:
                    result = result.subs([(update_var, 1),
                                          (sub_1, sub_1 * update_var)])
                result = result.subs(sub_2, sub_2 * update_var ** (-1))

        # we subtract a literal / parameter from a variable
        elif sub_1 in self._variables:
            if not update_var == sub_1:
                result = result.subs([(update_var, 1),
                                      (sub_1, sub_1 * update_var)])
            result = result * update_var ** (-sub_2)

        # we subtract a variable from a literal / parameter
        elif sub_2 in self._variables:
            if sub_2 == update_var:
                result = result.subs(update_var, update_var
                                     ** (-1)) * update_var ** sub_1
            else:
                result = result.subs(update_var, 1) * update_var ** sub_1
                result = result.subs(sub_2, sub_2 * update_var ** (-1))

        # we subtract two literals / parameters from each other
        else:
            diff = sub_1 - sub_2
            if sub_1 not in self._parameters and sub_2 not in self._parameters and diff < 0:
                raise ValueError(
                    f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it is negative"
                )
            result = result.subs(update_var, 1) * update_var ** diff

        result = sympy.expand(result)
        gf = GeneratingFunction(result,
                                *self._variables).set_parameters(
            *self.get_parameters())
        test_fun: sympy.Basic = gf.marginal(temp_var)._function.subs(
            update_var, 0)
        if test_fun.has(sympy.S('zoo')) or test_fun == sympy.nan:
            raise ValueError(
                f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it can be negative"
            )
        return gf

    def _update_product(self, temp_var: str, first_factor: str,
                        second_factor: str,
                        approximate: str | float | None) -> GeneratingFunction:
        # certain simplifications are only possible if we assume the symbols to be nonnegative
        update_var_with_assumptions = _sympy_symbol(temp_var, nonnegative=True)
        update_var = _sympy_symbol(temp_var)
        prod_1, prod_2 = _parse_to_sympy(first_factor), _parse_to_sympy(
            second_factor)
        result = self._function

        if prod_1 in self._parameters or prod_2 in self._parameters:
            raise ValueError('Multiplication of parameters is not allowed')

        # we multiply two variables
        if prod_1 in self._variables and prod_2 in self._variables:
            if not self._is_finite:
                marginal_l = self.marginal(first_factor)
                marginal_r = self.marginal(second_factor)
                result = sympy.Integer(0)

                if not marginal_l._is_finite and not marginal_r._is_finite:
                    if approximate is None:
                        raise ValueError(
                            f'Cannot perform the multiplication {first_factor} * {second_factor} ' \
                            'because both variables have infinite range'
                        )
                    # TODO can we choose which side to approximate in a smarter way?
                    marginal_l = marginal_l.approximate_unilaterally(
                        first_factor, approximate)

                finite, finite_var, infinite_var = (
                    marginal_l, first_factor,
                    second_factor) if marginal_l._is_finite else (
                    marginal_r, second_factor, first_factor)
                for _, state in finite:
                    result += self.filter(
                        parse_expr(f'{finite_var}={state[finite_var]}')
                    )._update_product(temp_var, state[finite_var],
                                      infinite_var, approximate)._function
            else:
                for prob, state_expr in self._iter_expr_expr():
                    state = self._monomial_to_state(state_expr)
                    term: sympy.Basic = prob * state_expr
                    result = result - term
                    term = term.subs(
                        update_var, 1) * update_var_with_assumptions ** (
                                   state[first_factor] * state[second_factor])
                    result = result + term

        # we multiply a variable with a literal
        elif prod_1 in self._variables or prod_2 in self._variables:
            if prod_1 in self._variables:
                var, lit = prod_1, prod_2
            else:
                var, lit = prod_2, prod_1
            if var == update_var:
                result = result.subs(update_var,
                                     update_var_with_assumptions ** lit)
            else:
                result = result.subs([
                    (update_var, 1),
                    (var, var * update_var_with_assumptions ** lit)
                ])

        # we multiply two literals
        else:
            result = result.subs(update_var, 1) * (update_var_with_assumptions
                                                   ** (prod_1 * prod_2))

        return GeneratingFunction(
            # remove the nonnegativity assumption from all symbols
            result.replace(lambda expr: expr.is_Symbol,
                           lambda expr: _sympy_symbol(expr.name)),
            *self._variables,
            closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _update_var(self, updated_var: str,
                    assign_var: str | int) -> GeneratingFunction:
        if _sympy_symbol(str(assign_var)) in self._parameters:
            raise ValueError('Assignment to parameters is not allowed')
        if _parse_to_sympy(str(assign_var)).is_Symbol and _parse_to_sympy(
                str(assign_var)) not in self._variables:
            raise ValueError(f"Unknown symbol: {assign_var}")

        if not updated_var == assign_var:
            if _parse_to_sympy(assign_var) in self._variables:
                result = self._function.subs([
                    (_sympy_symbol(updated_var), 1),
                    (_parse_to_sympy(assign_var),
                     _parse_to_sympy(assign_var) * _sympy_symbol(updated_var))
                ])
            else:
                result = self._function.subs(
                    _sympy_symbol(updated_var), 1) * _sympy_symbol(
                    updated_var) ** _parse_to_sympy(assign_var)
            return GeneratingFunction(
                result,
                *self._variables,
                closed=self._is_closed_form,
                finite=self._is_finite).set_parameters(*self.get_parameters())
        else:
            return self.copy()

    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> GeneratingFunction:
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
                result = result.subs([(update_var, 1),
                                      (sum_1, sum_1 * update_var),
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
                result = result.subs([(update_var, 1),
                                      (var, update_var * var)])
            result = result * (update_var ** lit)

        # we add two literals / parameters
        else:
            result = result.subs(update_var, 1) * (update_var ** (sum_1 + sum_2))

        return GeneratingFunction(
            result,
            *self._variables,
            closed=self._is_closed_form,
            finite=self._is_finite).set_parameters(*self.get_parameters())

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> Distribution:
        update_var = _sympy_symbol(temp_var)
        pow_1, pow_2 = _parse_to_sympy(base), _parse_to_sympy(exp)
        res = self._function

        if pow_1 in self._parameters or pow_2 in self._parameters:
            raise ValueError(
                "Cannot perfrom an exponentiation containing parameters")

        # variable to the power of a variable
        if pow_1 in self._variables and pow_2 in self._variables:
            assert isinstance(base, str)
            assert isinstance(exp, str)
            marginal_l, marginal_r = self.marginal(base), self.marginal(exp)

            if not marginal_l._is_finite:
                if approximate is None:
                    raise ValueError(
                        "Can only perform exponentiation of variables if both have a finite marginal"
                    )
                marginal_l = marginal_l.approximate_unilaterally(
                    base, approximate)
            if not marginal_r._is_finite:
                if approximate is None:
                    raise ValueError(
                        "Can only perform exponentiation of variables if both have a finite marginal"
                    )
                marginal_r = marginal_r.approximate_unilaterally(
                    exp, approximate)

            for _, state_l in marginal_l:
                for _, state_r in marginal_r:
                    x = self.filter(
                        parse_expr(
                            f'{base}={state_l[base]} & {exp}={state_r[exp]}')
                    )._function
                    res -= x
                    res += x.subs(update_var, 1) * update_var ** (state_l[base] **
                                                                  state_r[exp])

        # variable to the power of a literal
        elif pow_1 in self._variables:
            marginal = self.marginal(base)

            if not marginal._is_finite:
                raise ValueError(
                    "Can only perform exponentiation if the base has a finite marginal"
                )

            for _, state in marginal:
                x = self.filter(parse_expr(f'{base}={state[base]}'))._function
                res -= x
                res += x.subs(update_var, 1) * update_var ** (state[base] ** pow_2)

        # literal to the power of a variable
        elif pow_2 in self._variables:
            marginal = self.marginal(exp)

            if not marginal._is_finite:
                raise ValueError(
                    "Can only perform exponentiation if the exponent has a finite marginal"
                )

            for _, state in marginal:
                x = self.filter(parse_expr(f'{exp}={state[exp]}'))._function
                res -= x
                res += x.subs(update_var, 1) * update_var ** (pow_1 ** state[exp])

        # literal to the power of a literal
        else:
            res = res.subs(update_var, 1) * update_var ** (pow_1 ** pow_2)

        return GeneratingFunction(
            res,
            *self._variables,
            finite=self._is_finite,
            closed=self._is_closed_form).set_parameters(*self.get_parameters())

    def update_iid(self, sampling_dist: Expr, count: VarExpr,
                   variable: Union[str, VarExpr]) -> Distribution:

        subst_var = count.var

        def subs(dist_gf, subst_var, variable):
            result = self.marginal(
                variable,
                method=MarginalType.EXCLUDE) if subst_var != variable else self
            result = result.set_variables(*self.get_variables(), str(variable))
            if subst_var == variable:
                result._function = result._function.subs(
                    _sympy_symbol(subst_var), dist_gf)
            else:
                result._function = result._function.subs(
                    _sympy_symbol(subst_var),
                    _sympy_symbol(subst_var) * dist_gf)
            return result.set_parameters(*self.get_parameters())

        if not isinstance(sampling_dist, FunctionCallExpr):
            # create distribution in correct variable:
            expr = Mut.alloc(sampling_dist)
            # for ref in walk_expr(Walk.DOWN, expr):
            #    if isinstance(ref.val, VarExpr) and _sympy_symbol(ref.val.var) not in self._parameters:
            #        ref.val.var = variable
            #    elif isinstance(ref.val, FunctionCallExpr):
            #        raise ValueError(
            #            "Cannot handle nested user-defined functions in iid parameters"
            #        )
            dist_gf = _parse_to_sympy(str(expr.val))
            return subs(dist_gf, subst_var, variable)

        if sampling_dist.function == "binomial":
            n, p = map(_parse_to_sympy, sampling_dist.params[0])
            dist_gf = (1 - p + p * _sympy_symbol(variable)) ** n
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function in {"unif", "unif_d"}:
            start, end = map(_parse_to_sympy, sampling_dist.params[0])
            var = _sympy_symbol(variable)
            dist_gf = 1 / (end - start + 1) * var ** start * (
                    var ** (end - start + 1) - 1) / (var - 1)
            return subs(dist_gf, subst_var, variable)
        # All remaining distributions have only one parameter
        [param] = map(_parse_to_sympy, sampling_dist.params[0])
        if sampling_dist.function == "geometric":
            dist_gf = param / (1 - (1 - param) * _sympy_symbol(variable))
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "logdist":
            dist_gf = sympy.log(1 - param *
                                _sympy_symbol(variable)) / sympy.log(1 - param)
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "bernoulli":
            dist_gf = param * _sympy_symbol(variable) + (1 - param)
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "poisson":
            dist_gf = sympy.exp(param * (_sympy_symbol(variable) - 1))
            return subs(dist_gf, subst_var, variable)

        raise NotImplementedError(f"Unsupported distribution: {sampling_dist}")

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        expr = _parse_to_sympy(str(expression)).ratsimp().expand()
        if not expr.is_polynomial():
            raise NotImplementedError(
                "Expected Value only computable for polynomial expressions.")

        if len(expr.free_symbols & self._variables) == 0:
            return str(expr)
        if not expr.free_symbols.issubset(
                self._variables.union(self._parameters)):
            raise ValueError(
                f"Cannot compute expected value of {expression} because it contains unknown symbols"
            )

        marginal = self.marginal(*(expr.free_symbols & self._variables),
                                 method=MarginalType.INCLUDE)._function
        gen_func = GeneratingFunction(expr,
                                      *(expr.free_symbols & self._variables))
        expected_value = sympy.Integer(0)
        for prob, state in gen_func:
            tmp = marginal
            for var, val in state.items():
                var_sym = _sympy_symbol(var)
                for _ in range(val):
                    tmp = tmp.diff(var_sym, 1) * var_sym
                tmp = tmp.limit(var_sym, 1, '-')
            summand: sympy.Expr = _parse_to_sympy(prob) * tmp
            if len(summand.free_symbols) == 0 and summand < 0:
                raise ValueError(f'Intermediate result is negative: {summand}')
            expected_value += summand
        if expected_value == sympy.S('oo'):
            return str(RealLitExpr.infinity())
        else:
            return str(expected_value)

    def copy(self, deep: bool = True) -> GeneratingFunction:
        res = GeneratingFunction(self._function,
                                 *self._variables,
                                 closed=self._is_closed_form,
                                 finite=self._is_finite)
        res._parameters = self._parameters.copy()
        return res

    def set_variables(self, *variables: str) -> GeneratingFunction:
        if not variables:
            raise ValueError(
                "The free-variables of a distribution cannot be empty!")
        remove_dups = set(variables)
        return GeneratingFunction(self._function,
                                  *remove_dups,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite).set_parameters(
            *(self.get_parameters() - remove_dups))

    def set_parameters(self, *parameters: str) -> GeneratingFunction:
        remove_dups = set(parameters)
        if self._variables.intersection(remove_dups):
            raise ValueError(
                "A indeterminate cannot be variable and parameter at the same time."
            )
        result = self.copy()
        result._parameters = set(_sympy_symbol(param) for param in remove_dups)
        return result

    def _arithmetic(self, other, op: Callable) -> GeneratingFunction:
        """
        Computes the basic arithmetic of two generating functions.
        """
        logger.debug("Computing the %s of %s with %s", op, self, other)

        # other object is already a generating function.
        if isinstance(other, GeneratingFunction):
            # Checking for naming clashes in variables and parameters:
            if self._variables.intersection(
                    other._parameters) or self._parameters.intersection(
                other._variables):
                raise ArithmeticError(
                    "Name clash for parameters and variables in " \
                    f"{self._variables=} {other._variables=} \t {self._parameters=} {other._parameters=}"
                )

            # collect the meta information.
            is_closed_form = self._is_closed_form and other._is_closed_form
            is_finite = self._is_finite and other._is_finite

            # do the actual operation
            function = op(self._function, other._function)

            # simplify the result if demanded.
            if GeneratingFunction.use_simplification:
                nominator, denominator = sympy.fraction(function)
                nominator = nominator.factor()
                denominator = denominator.factor()
                function = nominator / denominator
                if not is_closed_form:
                    function = function.expand()

            # update the occuring variables and parameters
            variables = self._variables.union(other._variables)
            parameters = self._parameters.union(other._parameters)
            result = GeneratingFunction(function,
                                        *variables,
                                        closed=is_closed_form,
                                        finite=is_finite)
            result._parameters = parameters
            return result

        # other object is either an expression, or literal
        elif isinstance(other, (str, float, int)):
            # we try to convert this into a Generatingfunction and compute the arithmetic from there on.
            return self._arithmetic(
                GeneratingFunction(str(other), *self._variables), op)
        # We don't know how to do arithmetic on other types.
        else:
            raise SyntaxError(
                f"You cannot {str(op)} {type(self)} with {type(other)}.")

    def __add__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.add)

    def __sub__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other) -> GeneratingFunction:
        logger.debug("Isn't it weird to divide Distributions?")
        return self._arithmetic(other, operator.truediv)

    def __str__(self) -> str:
        if GeneratingFunction.use_latex_output:
            return sympy.latex(self._function)
        else:
            output = f"{self._function}"
        return output

    def __repr__(self):
        return repr(self._function)

    # ====== Comparison of Generating Functions ======

    def __le__(self, other) -> bool:
        if not isinstance(other, GeneratingFunction):
            raise TypeError(
                f"Incomparable types {type(self)} and {type(other)}.")

        # if one of the generating functions is finite, we can check the relation coefficient-wise.
        if self._is_finite:
            for prob, state in self._iter_expr_state():
                if prob > other._probability_of_state(state):
                    return False
            return True
        if other._is_finite:
            for prob, state in other._iter_expr_state():
                if self._probability_of_state(state) > prob:
                    return False
            return True
        # when both generating functions have infinite support we can only try to check whether we can eliminate
        # common terms and check whether the result is finite. If so we can do a coefficient-wise pass again.
        difference = self._function - other._function
        if difference.is_polynomial():
            return all(
                map(lambda x: x > 0,
                    difference.as_coefficients_dict().values()))

        # We currently do not know what to do otherwise.
        raise RuntimeError(
            "Both objects have infinite support. We cannot determine the order between them."
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeneratingFunction):
            return False
        # We rely on simplification of __sympy__ here. Thus, we cannot guarantee to detect equality when
        # simplification fails.
        return bool(self._function.equals(other._function)) \
            and self._variables == other._variables \
            and self._parameters == other._parameters

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        """ Iterates over the generating function yielding (coefficient, state) pairs.

            For implicit (closed-form) generating functions, we generate terms via tailor expansion.
            If the generating function is even multivariate this might be costly as we try to enumerate all different
            k-tuples, for which the coefficient might be zero (this we do not know upfront).
        """
        logger.debug("iterating over %s...", self)
        return map(
            lambda term: (str(term[0]), self._monomial_to_state(term[1])),
            self._iter_expr_expr())

    def _iter_expr_state(self) -> Iterator[Tuple[sympy.Expr, State]]:
        """Iterates over self and yields the (coefficient, state) pairs as a sympy expression and a `State`"""
        return map(lambda term: (term[0], self._monomial_to_state(term[1])),
                   self._iter_expr_expr())

    def _iter_expr_expr(self) -> Iterator[Tuple[sympy.Expr, sympy.Expr]]:
        """Iterates over self and yields the (coefficient, state) pairs as sympy expressions"""

        if self._is_finite:
            if self._is_closed_form:
                func = self._function.expand().ratsimp().as_poly(
                    *self._variables)
            else:
                if not self._variables:
                    logger.warning(
                        "Empty Polynomial, introducing auxilliary variable to create polynomial."
                    )
                    func = self._function.as_poly(
                        _parse_to_sympy("empty_generator"))
                else:
                    func = self._function.as_poly(*self._variables)
            return _term_generator(func)
        else:
            logger.debug("Multivariate Taylor expansion might take a while...")
            return self._mult_term_generator()

    # FIXME: It's not nice to have different behaviour depending on the variable type of `threshold`.
    def approximate(
            self,
            threshold: Union[str,
            int]) -> Generator[GeneratingFunction, None, None]:
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
                yield GeneratingFunction(approx,
                                         *self._variables,
                                         closed=False,
                                         finite=True)

        elif isinstance(threshold, str):
            s_threshold = _parse_to_sympy(threshold)
            assert s_threshold < self.coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {s_threshold}, CSum {self.coefficient_sum()}"
            for s_prob, state in self._iter_expr_expr():
                if precision >= _parse_to_sympy(threshold):
                    break
                approx += s_prob * state
                precision += s_prob
                yield GeneratingFunction(str(approx.expand()),
                                         *self._variables,
                                         closed=False,
                                         finite=True)
        else:
            raise TypeError(
                f"Parameter threshold can only be of type str or int,"
                f" not {type(threshold)}.")

    def coefficient_sum(self) -> sympy.Expr:
        logger.debug("coefficient_sum() call")
        coefficient_sum = self._function.simplify(
        ) if GeneratingFunction.use_simplification else self._function
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(
                var, 1, "-") if self._is_closed_form else coefficient_sum.subs(
                var, 1)
        return coefficient_sum

    def get_probability_mass(self):
        return str(self.coefficient_sum())

    def get_parameters(self) -> Set[str]:
        return set(map(str, self._parameters))

    def get_variables(self) -> Set[str]:
        return set(map(str, self._variables))

    def is_zero_dist(self) -> bool:
        return self._function == 0

    def normalize(self) -> GeneratingFunction:
        logger.debug("normalized() call")
        mass = self.coefficient_sum()
        if mass == 0:
            raise ZeroDivisionError
        return GeneratingFunction(self._function / mass,
                                  *self._variables,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def is_finite(self):
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
    def evaluate_condition(cls, condition: BinopExpr, state: State) -> bool:
        logger.debug("evaluate_condition() call")
        return super().evaluate_condition(condition, state)

    def marginal(
            self,
            *variables: Union[str, VarExpr],
            method: MarginalType = MarginalType.INCLUDE) -> GeneratingFunction:
        logger.debug(
            "Creating marginal for variables %s and joint probability distribution %s",
            variables, self)

        if len(variables) == 0:
            raise ValueError("No variables where provided")
        if not {_sympy_symbol(str(x))
                for x in variables}.issubset(self._variables):
            raise ValueError(
                f"Unknown variable(s): { {_sympy_symbol(str(x)) for x in variables} - self._variables}"
            )

        marginal_vars = set(
            map(_sympy_symbol, filter(lambda v: v != '', map(str, variables))))
        marginal = self.copy()
        s_var: str | VarExpr | sympy.Symbol
        if method == MarginalType.INCLUDE:
            for s_var in marginal._variables.difference(marginal_vars):
                if marginal._is_closed_form:
                    marginal._function = marginal._function.limit(
                        s_var, 1, "-")
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = marginal_vars
        else:
            for s_var in marginal_vars:
                if marginal._is_closed_form:
                    marginal._function = marginal._function.limit(
                        s_var, 1, "-")
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = marginal._variables.difference(marginal_vars)

        marginal._is_closed_form = not marginal._function.is_polynomial()
        marginal._is_finite = marginal._function.ratsimp().is_polynomial()

        return marginal

    def filter(self, condition: Expr) -> GeneratingFunction:
        logger.debug("filter(%s) call on %s", condition, self)
        res = super().filter(condition)
        assert isinstance(res, GeneratingFunction)
        return res

    def _exhaustive_search(self, condition: Expr) -> GeneratingFunction:
        res = sympy.Integer(0)
        for prob, state in self._iter_expr_expr():
            if self.evaluate_condition(condition,
                                       self._monomial_to_state(state)):
                res += prob * state
        return GeneratingFunction(res,
                                  *self._variables,
                                  closed=False,
                                  finite=True)

    def _find_symbols(self, expr: str) -> Set[str]:
        return {str(s) for s in _parse_to_sympy(expr).free_symbols}

    def get_symbols(self) -> Set[str]:
        return {str(s) for s in self._function.free_symbols}


class SympyPGF(CommonDistributionsFactory):
    """Implements PGFs of standard distributions."""

    @staticmethod
    def geometric(var: Union[str, VarExpr],
                  p: DistributionParam) -> GeneratingFunction:
        if not isinstance(p, get_args(Expr)):
            expr = parse_expr(str(p))
        else:
            expr = p
        if not has_variable(expr, None) and not 0 < _parse_to_sympy(p) < 1:
            raise ValueError(
                f"parameter of geom distr must be 0 < p <= 1, was {p}")
        return GeneratingFunction(f"({p}) / (1 - (1-({p})) * {var})",
                                  str(var),
                                  closed=True,
                                  finite=False)

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> GeneratingFunction:
        if not isinstance(lower, get_args(Expr)):
            expr_l = parse_expr(str(lower))
        else:
            expr_l = lower
        if not isinstance(upper, get_args(Expr)):
            expr_u = parse_expr(str(upper))
        else:
            expr_u = upper
        if not has_variable(expr_l, None) and (
                not 0 <= _parse_to_sympy(lower) or
                (not has_variable(expr_u, None)
                 and not _parse_to_sympy(lower) <= _parse_to_sympy(upper))):
            raise ValueError(
                "Distribution parameters must satisfy 0 <= a < b < oo")
        return GeneratingFunction(
            f"1/(({upper}) - ({lower}) + 1) * {var}**({lower}) * ({var}**(({upper}) - ({lower}) + 1) - 1) " \
            f"/ ({var} - 1)",
            str(var),
            closed=True,
            finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr],
                  p: DistributionParam) -> GeneratingFunction:
        if not isinstance(p, get_args(Expr)):
            expr = parse_expr(str(p))
        else:
            expr = p
        if not has_variable(expr, None) and not 0 <= _parse_to_sympy(p) <= 1:
            raise ValueError(
                f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}"
            )
        return GeneratingFunction(f"1 - ({p}) + ({p}) * {var}",
                                  str(var),
                                  closed=True,
                                  finite=True)

    @staticmethod
    def poisson(var: Union[str, VarExpr],
                lam: DistributionParam) -> GeneratingFunction:
        if not isinstance(lam, get_args(Expr)):
            expr = parse_expr(str(lam))
        else:
            expr = lam
        if not has_variable(expr, None) and _parse_to_sympy(lam) < 0:
            raise ValueError(
                f"Parameter of Poisson Distribution must be in [0, oo), but was {lam}"
            )
        return GeneratingFunction(f"exp(({lam}) * ({var} - 1))",
                                  str(var),
                                  closed=True,
                                  finite=False)

    @staticmethod
    def log(var: Union[str, VarExpr],
            p: DistributionParam) -> GeneratingFunction:
        if not isinstance(p, get_args(Expr)):
            expr = parse_expr(str(p))
        else:
            expr = p
        if not has_variable(expr, None) and not 0 <= _parse_to_sympy(p) <= 1:
            raise ValueError(
                f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}"
            )
        return GeneratingFunction(f"log(1-({p})*{var})/log(1-({p}))",
                                  str(var),
                                  closed=True,
                                  finite=False)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> GeneratingFunction:
        if not isinstance(p, get_args(Expr)):
            expr_p = parse_expr(str(p))
        else:
            expr_p = p
        if not has_variable(expr_p, None) and not 0 <= _parse_to_sympy(p) <= 1:
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,1], but was {p}"
            )
        if not isinstance(n, get_args(Expr)):
            expr_n = parse_expr(str(n))
        else:
            expr_n = n
        if not has_variable(expr_n, None) and not 0 <= _parse_to_sympy(n):
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,oo), but was {n}"
            )
        return GeneratingFunction(f"(1-({p})+({p})*{var})**({n})",
                                  str(var),
                                  closed=True,
                                  finite=True)

    @staticmethod
    def zero(*variables: Union[str, sympy.Symbol]) -> GeneratingFunction:
        if variables:
            return GeneratingFunction("0",
                                      *variables,
                                      closed=True,
                                      finite=True)
        return GeneratingFunction("0", closed=True, finite=True)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> GeneratingFunction:
        """ A distribution where actually no information about the states is given."""
        return GeneratingFunction("0",
                                  *variables,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        return GeneratingFunction("1",
                                  *map(str, variables),
                                  closed=True,
                                  finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> Distribution:
        return GeneratingFunction(str(expression), *map(str, variables),
                                  **kwargs)
