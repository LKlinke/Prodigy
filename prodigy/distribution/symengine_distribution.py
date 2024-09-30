from __future__ import annotations

import logging
import math
from typing import Union, Generator, Set, FrozenSet, Sequence, Iterator, Tuple, Type, List

import symengine as se
from probably.pgcl import VarExpr, Expr, FunctionCallExpr, RealLitExpr, UnopExpr, Binop, BinopExpr, Unop
from probably.util.ref import Mut

from prodigy.distribution import Distribution
from prodigy.distribution import MarginalType, State, CommonDistributionsFactory, DistributionParam
from prodigy.util.logger import log_setup
from prodigy.util.order import default_monomial_iterator

from probably.pgcl.parser import parse_expr

# todo remove once method in symengine
import sympy as sp

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG, file="GF_operations.log")


class SymengineDist(Distribution):
    @staticmethod
    def factory() -> Type[CommonDistributionsFactory]:
        return SymenginePGF

    def __init__(self, function, *variables):
        self._s_func: se.Expr = se.S(function)
        self._variables: set[str] = set(variables) if len(variables) > 0 else {str(v) for v in
                                                                               self._s_func.free_symbols}
        self._parameters: set[str] = {str(s) for s in self._s_func.free_symbols} - self._variables

    def _check_symbol_consistency(self, other: SymengineDist) -> bool:
        """
        Checks whether variables and parameters names are compliant.
        """
        variables = self._variables | other._variables
        parameters = self._parameters | other._parameters
        return variables.intersection(parameters) == set()

    def __add__(self, other) -> SymengineDist:
        variables = self._operator_prerequisites(other, self.__add__, "add")
        s_result = self._s_func + other._s_func
        return SymengineDist(s_result, *variables)

    def __sub__(self, other) -> SymengineDist:
        variables = self._operator_prerequisites(other, self.__sub__, "subtract")
        s_result = self._s_func - other._s_func
        return SymengineDist(s_result, *variables)

    def __mul__(self, other) -> SymengineDist:
        variables = self._operator_prerequisites(other, self.__mul__, "multiply")
        s_result = self._s_func * other._s_func
        return SymengineDist(s_result, *variables)

    def __truediv__(self, other) -> SymengineDist:
        # TODO is __mul__ the right method?
        variables = self._operator_prerequisites(other, self.__mul__, "divide")
        s_result = self._s_func / other._s_func
        return SymengineDist(s_result, *variables)

    def _operator_prerequisites(self, other, f_pointer, textual_descr: str):
        """
        Checks whether the operation can be applied to the given distributions.
        If other is a constant, i.e. a string / float / int, the result is directly computed
        by the given function pointer.
        If both are SymengineDistributions and don't have inconsistent variables, the union of
        their variables is returned.
        """
        if isinstance(other, (str, float, int)):
            return f_pointer(SymengineDist(other, self._variables))
        if not isinstance(other, SymengineDist):
            raise SyntaxError(f"You cannot {textual_descr} {type(self)} by {type(other)}.")

        # Actual operation
        if not self._check_symbol_consistency(other):
            clash = (self._variables | other._variables) & (self._parameters | other._parameters)
            raise SyntaxError(f"Name clash: {clash} for {self} and {other}.")
        return self._variables | other._variables

    def __eq__(self, other) -> bool:
        if not isinstance(other, SymengineDist):
            return False
        if not other._variables == self._variables:
            return False
        if not other._parameters == self._parameters:
            return False
        return other._s_func == self._s_func

    def __le__(self, other) -> bool:
        if not isinstance(other, SymengineDist):
            raise TypeError(f"Incomparable types {type(self)} and {type(other)}.")

        if self.is_finite():
            for prob, state in self:
                pass  # TODO

        if other.is_finite():
            for prob, state in other:
                pass  # TODO

        difference: se.Basic = self._s_func - other._s_func
        # fixme replace once suitable method is found
        #   cf. https://github.com/symengine/symengine.py/issues/492
        if sp.S(difference).is_polynomial():
            return all(
                map(lambda x: x > 0, difference.as_coefficients_dict().values())
            )

    def __str__(self) -> str:
        return f"{self._s_func}"

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        prob_fun = self.get_prob_by_diff  # TODO which probability function should we use
        if not self.is_finite():
            v = list(self.get_variables())
            if len(v) == 1:
                v = v[0]
                i = 0
                while True:
                    yield prob_fun(State({v: i})), State({v: i})
                    i += 1
            else:
                for tup in default_monomial_iterator(len(v)):
                    yield prob_fun(State(dict(zip(v, tup)))), State(dict(zip(v, tup)))
        else:
            for prob, vals in sp.S(self._s_func).as_terms():  # fixme replace with symengine method
                yield prob, State(vals)

    def iter_with(self, monomial_iterator: Iterator[List[int]]) -> Iterator[Tuple[str, State]]:
        """
        Iterates over the expression with a custom monomial order.
        :param monomial_iterator: An iterator that yields the monomial order.
        :return: An iterator over the expression in the given order.
        """
        prob_fun = self.get_prob_by_diff  # TODO which function here
        if not self.is_finite():
            v = list(self.get_variables())
            for tup in monomial_iterator:
                yield prob_fun(State(dict(zip(v, tup)))), State(dict(zip(v, tup)))
        else:
            for prob, vals in sp.S(self._s_func).as_terms():  # fixme replace with symengine method
                yield prob, State(vals)

    # TODO integrate these functions better / move them / replace by correct signature
    def get_prob_by_diff(self, state: State) -> se.Basic:
        """
        Get the probability of a given state by means of differentiation
        :param state: The state to get the probability of
        :return: The probability of the given state
        """
        fun = self._s_func.diff(*state.items())
        return fun.subs({v: 0 for v in self._variables}) / math.prod([math.factorial(el[1]) for el in state.items()])

    def get_prob_by_series(self, state: State) -> se.Basic:
        """
        Get the probability of a given state by means of taylor expansion
        :param state: The state to get the probability of
        :return: The probability of the given state
        """
        series = self._s_func
        args = {}
        for (v, val) in state.items():
            series: se.Basic = se.series(series, v, 0, val + 1)
            # Build the expression
            args += [f"{v} ** {val}"]
        args = se.S("*".join(args))
        # Expand series as simplifications (such as factoring out) lead to missing coefficients
        coefficient_dict = series.expand().as_coefficients_dict()
        return coefficient_dict[args] if args in coefficient_dict else se.S(0)

    def copy(self, deep: bool = True) -> SymengineDist:
        res = SymengineDist(0)
        res._s_func = self._s_func.copy()
        res._variables = self._variables.copy()
        res._parameters = self._parameters.copy()
        return res

    def get_probability_mass(self) -> str:
        fast_result: se.Expr = self._s_func.subs(tuple(self._variables), (1,) * len(self._variables))
        if fast_result == se.nan or fast_result == se.zoo:
            raise ValueError(
                f"Indeterminate expression {self._s_func} with {self._variables} mapped to {(1,) * len(self._variables)}"
            )
        return str(fast_result)

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        # FIXME replace sympy by symengine if method is implemented
        expr = sp.S(str(expression)).ratsimp().expand()
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
                                 method=MarginalType.INCLUDE)._s_func
        gen_func = SymengineDist(expr,
                                      *(expr.free_symbols & self._variables))
        expected_value = sp.Integer(0)
        for prob, state in gen_func:
            tmp = marginal
            for var, val in state.items():
                var_sym = se.S(var)
                for _ in range(val):
                    tmp = tmp.diff(var_sym, 1) * var_sym
                tmp = tmp.limit(var_sym, 1, '-')
            summand: se.Expr = se.S(prob) * tmp
            if len(summand.free_symbols) == 0 and summand < 0:
                raise ValueError(f'Intermediate result is negative: {summand}')
            expected_value += summand
        if expected_value == se.S('oo'):
            return str(RealLitExpr.infinity())
        else:
            return str(expected_value)

    def normalize(self) -> SymengineDist:
        mass = self.get_probability_mass()
        if mass == 0:
            raise ZeroDivisionError
        return SymengineDist(self._s_func / mass, *self._variables)

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        return self._parameters

    def _exhaustive_search(self, condition: Expr) -> SymengineDist:
        res = se.S("0")
        for prob, state in self:
            if self.evaluate_condition(condition, state):
                res += se.S(f"{prob} * {state.to_monomial()}")
        return SymenginePGF.from_expr(
            res,
            self._variables
        )

    def _filter_constant_condition(self, condition: Expr) -> SymengineDist:
        # Normalize the condition into the format _var_ (< | <= | =) const. I.e., having the variable on the lhs.
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

        if isinstance(condition.lhs,
                      VarExpr) and condition.lhs.var not in self._variables:
            if condition.operator == Binop.GT:
                return self.filter(
                    BinopExpr(operator=Binop.LT,
                              lhs=condition.rhs,
                              rhs=condition.lhs))
            elif condition.operator == Binop.GEQ:
                return self.filter(
                    BinopExpr(operator=Binop.LEQ,
                              lhs=condition.rhs,
                              rhs=condition.lhs))

        # Now we have an expression of the form _var_ (< | <=, =) _const_.
        variable = se.S(str(condition.lhs))
        constant = condition.rhs.value
        result = 0
        ranges = {
            Binop.LT: range(constant),
            Binop.LEQ: range(constant + 1),
            Binop.EQ: [constant]
        }

        # Compute the probabilities of the states _var_ = i where i ranges depending on the operator (< , <=, =).
        for i in ranges[condition.operator]:
            result += (self._s_func.diff(variable, i) / math.factorial(i)
                       ).limit(variable, 0, '-') * variable ** i

        return SymengineDist(result, *self._variables)

    def _arithmetic_progression(self, variable: str, modulus: str) -> Sequence[SymengineDist]:
        a = se.S(modulus)
        var = (se.S(variable))

        # This should be a faster variant for univariate distributions.
        if self._variables == {var}:
            result = []
            for remainder in range(a):
                other = SymengineDist(f"({var}^{remainder})/(1-{var}^{modulus})", *self._variables)
                result.append(self.hadamard_product(other))
            return result

        # This is the general algorithm
        primitive_uroot = se.exp(2 * se.pi * se.I / a)
        result = []
        for remainder in range(a):  # type: ignore
            psum = 0
            for m in range(a):  # type: ignore
                psum += primitive_uroot ** (-m *
                                            remainder) * self._s_func.subs(
                    var, (primitive_uroot ** m) * var)
            result.append(
                SymengineDist(f"(1/{a}) * ({psum})",*self._variables)
            )
        return result

    def hadamard_product(self, other: SymengineDist) -> SymengineDist:
        raise NotImplementedError("Hadamard Product is currently supported")  # ignore for now

    def _find_symbols(self, expr: str) -> Set[str]:
        return se.S(expr).free_symbols

    @staticmethod
    def evaluate(expression: str, state: State):
        """ Evaluates the expression in a given state. """

        s_exp = se.S(expression)
        # Iterate over the variable, value pairs in the state
        variables = []
        values = []
        for var, value in state.items():
            # Convert the variable into a sympy symbol and substitute
            variables.append(se.Symbol(var))
            values.append(se.S(value))
        s_exp = s_exp.subs(variables, values)

        # If the state did not interpret all variables in the condition, interpret the remaining variables as 0
        # Why do we do this?
        for free_var in s_exp.free_symbols:
            s_exp = s_exp.subs(free_var, 0)
        return s_exp

    def is_zero_dist(self) -> bool:
        return self._s_func.is_zero

    def is_finite(self) -> bool:
        # fixme replace once a suitable method is found within symengine
        #   cf. https://github.com/symengine/symengine.py/issues/492
        return sp.S(self._s_func).is_polynomial()

    def get_fresh_variable(self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        i = 0
        while se.S(f'_{i}') in (
                self._variables
                | self._parameters) or f'_{i}' in exclude:
            i += 1
        return f'_{i}'

    # TODO when to use se.S(*) and when se.Symbol(*)?

    def _update_var(self, updated_var: str, assign_var: str | int) -> SymengineDist:
        up_var, as_var = se.Symbol(updated_var), se.Symbol(assign_var)
        if str(assign_var) in self._parameters:
            raise ValueError("Assignment to parameters is not allowed")
        if se.S(str(assign_var)).is_symbol and se.S(str(assign_var)) not in self._variables:
            raise ValueError(f"Unknown symbol: {assign_var}")

        if not updated_var == assign_var:
            if as_var in self._variables:
                res = self._s_func.subs(up_var, 1).subs(as_var, as_var * up_var)
            else:
                res = self._s_func.subs(up_var, 1) * up_var ** as_var
            return SymengineDist(res, *self._variables)
        else:
            return self.copy()

    def _update_sum(self, temp_var: str, first_summand: str | int, second_summand: str | int) -> SymengineDist:
        update_var, sum_1, sum_2, res = se.Symbol(temp_var), se.S(first_summand), se.S(second_summand), self._s_func

        # Two variables are added
        if sum_1 in self._variables and sum_2 in self._variables:
            if sum_2 == update_var:
                sum_1, sum_2 = sum_2, sum_1
            if sum_1 == update_var:
                if sum_2 == update_var:
                    res = res.subs(update_var, update_var ** 2)
                else:
                    res = res.subs(sum_2, sum_2 * update_var)
            else:
                res = res.subs(update_var, 1).subs(sum_1, sum_1 * update_var).subs(sum_2, sum_2 * update_var)

        # One variable and one literal / parameter is added
        elif sum_1 in self._variables or sum_2 in self._variables:
            if sum_1 in self._variables:
                var, lit = sum_1, sum_2
            else:
                var, lit = sum_2, sum_1
            if not var == update_var:
                res = res.subs(update_var, 1).subs(var, update_var * var)
            res = res * (update_var ** lit)

        # Two literals / parameters are added
        else:
            res = res.subs(update_var, 1) * (update_var ** (sum_1 * sum_2))

        return SymengineDist(res, *self._variables)

    def _update_product(self, temp_var: str, first_factor: str, second_factor: str,
                        approximate: str | float | None) -> SymengineDist:
        update_var = se.S(temp_var)
        # TODO: implement assumption here if implemented in symengine
        update_var_with_assumptions = se.S(temp_var)
        prod_1, prod_2 = se.S(first_factor), se.S(second_factor)
        res = self._s_func

        if prod_1 in self._parameters or prod_2 in self._parameters:
            raise ValueError("Assignment of parameters is not allowed")

        # Multiplication of two variables
        if prod_1 in self._variables and prod_2 in self._variables:
            if not self.is_finite():
                marginal_l = self.marginal(first_factor)
                marginal_r = self.marginal(second_factor)
                res = se.Integer(0)

                if not marginal_l.is_finite() and not marginal_r.is_finite():
                    if approximate is None:
                        raise ValueError(
                            f'Cannot perform the multiplication {first_factor} * {second_factor} ' \
                            'because both variables have infinite range'
                        )
                    # TODO can we choose which side to approximate in a smarter way?
                    #   from generating_function.py
                    marginal_l = marginal_l.approximate_unilaterally(first_factor, approximate)

                finite, finite_var, infinite_var = (marginal_l, first_factor, second_factor) if marginal_l.is_finite() \
                    else (marginal_r, second_factor, first_factor)

                for _, state in finite:
                    res += self.filter(
                        parse_expr(f'{finite_var}={state[finite_var]}')
                    )._update_product(
                        temp_var, state[finite_var], infinite_var, approximate
                    )._s_func
            else:
                for prob, state in self:
                    term: se.Basic= prob * se.S(state.to_monomial())
                    res = res - term
                    term = term.subs(update_var, 1) * update_var_with_assumptions ** (state[first_factor]
                                                                                      * state[second_factor])
                    res = res + term

        # Multiplication of one variable and one literal
        elif prod_1 in self._variables or prod_2 in self._variables:
            if prod_1 in self._variables:
                var, lit = prod_1, prod_2
            else:
                var, lit = prod_2, prod_1
            if var == update_var:
                res = res.subs(update_var, update_var_with_assumptions ** lit)
            else:
                res = res.subs(update_var, 1).subs(var, var * update_var_with_assumptions ** lit)

        # Multiplication of two literals
        else:
            res = res.subs(update_var, 1) * (update_var_with_assumptions ** (prod_1 * prod_2))

        # TODO filter out assumptions over symbols once implemented
        result = SymengineDist(res, *self._variables)
        result.set_parameters(*self.get_parameters())
        return result

    def _update_subtraction(self, temp_var: str, sub_from: str | int, sub: str | int) -> SymengineDist:
        update_var, sub_1, sub_2, res = se.Symbol(temp_var), se.S(sub_from), se.S(sub), self._s_func

        # Subtraction of two variables
        if sub_1 in self._variables and sub_2 in self._variables:
            if sub_2 == update_var:
                if sub_1 == update_var:
                    res = res.subs(update_var, 1)
                else:
                    res = res.subs(update_var, update_var ** (-1)).subs(sub_1, sub_1 * update_var)
            else:
                if not sub_1 == update_var:
                    res = res.subs(update_var, 1).subs(sub_1, sub_1 * update_var)
                res = res.subs(sub_2, sub_2 * update_var ** (-1))

        # Literal subtracted from variable
        elif sub_1 in self._variables:
            if not update_var == sub_1:
                res = res.subs(update_var, 1).subs(sub_1, sub_1 * update_var)
            res = res * update_var ** (-sub_2)

        # Variable subtracted from literal
        elif sub_2 in self._variables:
            if sub_2 == update_var:
                res = res.subs(update_var, update_var ** (-1)) * update_var ** sub_1
            else:
                res = res.subs(update_var, 1) * update_var ** sub_1
                res = res.subs(sub_2, sub_2 * update_var ** (-1))

        # Two literals are subtracted from each other
        else:
            diff = sub_1 - sub_2
            if sub_1 not in self._parameters and sub_2 not in self._parameters and diff < 0:
                raise ValueError(
                    f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it is negative"
                )
            res = res.subs(update_var, 1) * update_var ** diff

        res = se.expand(res)
        expr = SymengineDist(res, *self._variables)
        expr.set_parameters(*self.get_parameters())

        test_fun: se.Basic = expr.marginal(temp_var)._s_func.subs(update_var, 0)
        if test_fun.has(se.zoo) or test_fun == se.nan:
            raise ValueError(
                f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it can be negative"
            )
        return expr

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> SymengineDist:
        left_sym, right_sym = se.Symbol(str(left)), se.Symbol(str(right))

        if left_sym in self._parameters or right_sym in self._parameters:
            raise ValueError("Cannot perform modulo operation on parameters")

        update_var = se.Symbol(temp_var)
        result = 0

        # On finite GFs, iterate over all states
        if self.is_finite():
            for prob, state_r in self:
                if left_sym in self._variables:
                    left_var = state_r[left]
                else:
                    left_var = se.S(left)
                if right_sym in self._variables:
                    right_var = state_r[right]
                else:
                    right_var = se.S(right)
                result += prob * se.S(state_r.to_monomial()).subs(update_var, 1) * update_var ** (left_var % right_var)

        # If GF is infinite and right is variable, it needs to have finite range
        elif right_sym in self._variables:
            assert isinstance(right, str)
            marginal_r = self.marginal(right)
            if not marginal_r.is_finite():
                if approximate is None:
                    raise ValueError(
                        f'Cannot perform modulo operation with infinite right hand side {right}'
                    )
                marginal_r = marginal_r.approximate_unilaterally(right, approximate)
                for _, state_r in marginal_r:
                    result += self.filter(
                        parse_expr(f'{right}={state_r[right]}')
                    )._update_modulo(
                        temp_var, left, state_r[right], None
                    )._s_func

        # If left is a variable, it doesn't have to have finite range
        elif left_sym in self._variables:
            assert isinstance(left, str)
            marginal_l = self.marginal(left)
            if not marginal_l.is_finite():
                for _, state_l in marginal_l:
                    result += self.filter(
                        parse_expr(f'{left}={state_l[left]}')
                    )._update_modulo(
                        temp_var, state_l[left], right, None
                    )._s_func

        # Both are not variables, simply compute the result
        else:
            return self._update_var(temp_var, int(left) % int(right))

        return SymengineDist(result, *self._variables)

    def _update_division(self, temp_var: str, numerator: str | int, denominator: str | int,
                         approximate: str | float | None) -> SymengineDist:
        update_var = se.Symbol(temp_var)
        div_1, div_2 = se.S(numerator), se.S(denominator)

        if div_1 in self._parameters or div_2 in self._parameters:
            raise ValueError('Division containing parameters is not allowed')

        marginal_l = self.marginal(numerator) if div_1 in self._variables else div_1
        marginal_r = self.marginal(denominator) if div_2 in self._variables else div_2

        if isinstance(marginal_l, SymengineDist) and not marginal_l.is_finite():
            if approximate is None:
                raise ValueError(
                    f'Cannot perform the division {numerator} / {denominator} because at least one has infinite range'
                )
            assert isinstance(numerator, str)
            marginal_l = marginal_l.approximate_unilaterally(numerator, approximate)

        if isinstance(marginal_r, SymengineDist) and not marginal_r.is_finite():
            if approximate is None:
                raise ValueError(
                    f'Cannot perform the division {numerator} / {denominator} because at least one has infinite range'
                )
            assert isinstance(denominator, str)
            marginal_r = marginal_r.approximate_unilaterally(denominator, approximate)

        res = 0
        if isinstance(marginal_l, SymengineDist):
            for _, state_l in marginal_l:
                if isinstance(marginal_r, SymengineDist):
                    for _, state_r in marginal_r:
                        val_l, val_r = state_l[numerator], state_r[denominator]
                        x = self.filter(
                            parse_expr(f'{numerator}={val_l} & {denominator}={val_r}')
                        )._s_func
                        if val_l % val_r != 0 and x != 0:
                            raise ValueError(
                                f"Cannot assign {numerator} / {denominator} to {temp_var} "
                                "because it is not always an integer"
                            )
                        res += x.subs(update_var, 1) * update_var ** (val_l / val_r)
                else:
                    val_l, val_r = state_l[numerator], div_2
                    x = self.filter(
                        parse_expr(f'{numerator}={val_l}'))._s_func
                    if val_l % val_r != 0 and x != 0:
                        raise ValueError(
                            f"Cannot assign {numerator} / {denominator} to {temp_var} "
                            "because it is not always an integer"
                        )
                    res += x.subs(update_var, 1) * update_var ** (val_l / val_r)
        else:
            if isinstance(marginal_r, SymengineDist):
                for _, state_r in marginal_r:
                    val_l, val_r = div_1, state_r[denominator]
                    x = self.filter(
                        parse_expr(f'{denominator}={val_r}'))._s_func
                    if val_l % val_r != 0 and x != 0:
                        raise ValueError(
                            f"Cannot assign {numerator} / {denominator} to {temp_var} "
                            "because it is not always an integer"
                        )
                    res += x.subs(update_var,
                                  1) * update_var ** (val_l / val_r)
            else:
                if div_1 % div_2 != 0:
                    raise ValueError(
                        f"Cannot assign {numerator} / {denominator} to {temp_var} because it is not always an integer"
                    )
                res = self._s_func.subs(update_var,
                                           1) * update_var ** (div_1 / div_2)

        res = SymengineDist(res, *self._variables)
        res.set_parameters(*self.get_parameters())
        return res

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> SymengineDist:
        update_var = se.S(temp_var)
        pow_1, pow_2 = se.S(base), se.S(exp)
        res = self._s_func

        if pow_1 in self._parameters or pow_2 in self._parameters:
            raise ValueError(
                "Cannot perfrom an exponentiation containing parameters")

        # variable to the power of a variable
        if pow_1 in self._variables and pow_2 in self._variables:
            assert isinstance(base, str)
            assert isinstance(exp, str)
            marginal_l, marginal_r = self.marginal(base), self.marginal(exp)

            if not marginal_l.is_finite():
                if approximate is None:
                    raise ValueError(
                        "Can only perform exponentiation of variables if both have a finite marginal"
                    )
                marginal_l = marginal_l.approximate_unilaterally(
                    base, approximate)
            if not marginal_r.is_finite():
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
                    )._s_func
                    res -= x
                    res += x.subs(update_var, 1) * update_var ** (state_l[base] **
                                                                  state_r[exp])

        # variable to the power of a literal
        elif pow_1 in self._variables:
            marginal = self.marginal(base)

            if not marginal.is_finite():
                raise ValueError(
                    "Can only perform exponentiation if the base has a finite marginal"
                )

            for _, state in marginal:
                x = self.filter(parse_expr(f'{base}={state[base]}'))._s_func
                res -= x
                res += x.subs(update_var, 1) * update_var ** (state[base] ** pow_2)

        # literal to the power of a variable
        elif pow_2 in self._variables:
            marginal = self.marginal(exp)

            if not marginal.is_finite():
                raise ValueError(
                    "Can only perform exponentiation if the exponent has a finite marginal"
                )

            for _, state in marginal:
                x = self.filter(parse_expr(f'{exp}={state[exp]}'))._s_func
                res -= x
                res += x.subs(update_var, 1) * update_var ** (pow_1 ** state[exp])

        # literal to the power of a literal
        else:
            res = res.subs(update_var, 1) * update_var ** (pow_1 ** pow_2)

        res = SymengineDist(res, *self._variables)
        res.set_parameters(*self.get_parameters())
        return res

    def update_iid(self, sampling_dist: Expr, count: VarExpr, variable: Union[str, VarExpr]) -> SymengineDist:
        subst_var = count.var

        def subs(dist_gf, subst_var, variable) -> SymengineDist:
            result = self.marginal(
                variable,
                method=MarginalType.EXCLUDE) if subst_var != variable else self
            result.set_variables(*self.get_variables(), str(variable))
            if subst_var == variable:
                result._s_func = result._s_func.subs(
                    se.S(subst_var), dist_gf)
            else:
                result._s_func = result._s_func.subs(
                    se.S(subst_var),
                    se.S(subst_var) * dist_gf)
                result = result.set_parameters(*self.get_parameters())
            return result

        if not isinstance(sampling_dist, FunctionCallExpr):
            # create distribution in correct variable:
            expr = Mut.alloc(sampling_dist)
            dist_gf = se.S(str(expr.val))
            return subs(dist_gf, subst_var, variable)

        if sampling_dist.function == "binomial":
            n, p = map(se.S, sampling_dist.params[0])
            dist_gf = (1 - p + p * se.S(variable)) ** n
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function in {"unif", "unif_d"}:
            start, end = map(se.S, sampling_dist.params[0])
            var = se.S(variable)
            dist_gf = 1 / (end - start + 1) * var ** start * (
                    var ** (end - start + 1) - 1) / (var - 1)
            return subs(dist_gf, subst_var, variable)
        # All remaining distributions have only one parameter
        [param] = map(se.S, sampling_dist.params[0])
        if sampling_dist.function == "geometric":
            dist_gf = param / (1 - (1 - param) * se.S(variable))
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "logdist":
            dist_gf = se.log(1 - param *
                             se.S(variable)) / se.log(1 - param)
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "bernoulli":
            dist_gf = param * se.S(variable) + (1 - param)
            return subs(dist_gf, subst_var, variable)
        if sampling_dist.function == "poisson":
            dist_gf = se.exp(param * (se.S(variable) - 1))
            return subs(dist_gf, subst_var, variable)

        raise NotImplementedError(f"Unsupported distribution: {sampling_dist}")

    def marginal(self, *variables: Union[str, VarExpr], method: MarginalType = MarginalType.INCLUDE) -> SymengineDist:
        result = self._s_func
        remove_vars = {
            MarginalType.EXCLUDE: {str(var)
                                   for var in variables},
            MarginalType.INCLUDE:
                self._variables - {str(var)
                                   for var in variables}
        }
        for var in remove_vars[method]:
            result = result.update_var(str(var), "0")
        return SymenginePGF.from_expr(result, self._variables - remove_vars[method],
                                      self._parameters)

    def set_variables(self, *variables: str):
        new_variables = set(variables)
        if self._parameters & new_variables:
            raise ValueError(
                f"At least one variable is already known as a parameter. {self._parameters=}, {new_variables=}")
        if not (self._parameters | new_variables).issuperset({str(s) for s in self._s_func.free_symbols}):
            raise ValueError(f"There are unknown symbols which are neither variables nor parameters.")
        self._variables = new_variables

    def set_parameters(self, *parameters: str):
        new_params = set(parameters)
        if self._variables & new_params:
            raise ValueError(
                f"At least one parameter is already known as a variable. {self._variables=}, {new_params=}")
        if not (self._variables | new_params).issuperset({str(s) for s in self._s_func.free_symbols}):
            raise ValueError(f"There are unknown symbols which are neither variables nor parameters.")
        self._parameters = new_params

    def approximate(self, threshold: Union[str, int]) -> Generator[SymengineDist, None, None]:
        logger.debug("expand_until() call")
        approx = se.Integer(0)
        precision = se.Integer(0)

        if isinstance(threshold, int):
            assert threshold > 0, "Expanding to less than 0 terms is not valid."
            for n, (prob, state) in enumerate(self):
                if n >= threshold:
                    break
                approx += prob * se.S(state.to_monomial())
                precision += prob
                yield SymengineDist(approx,*self._variables)

    def approximate_unilaterally(self, variable: str, probability_mass: str | float) -> SymengineDist:
        logger.debug("approximate_unilaterally(%s, %s) call on %s", variable,
                     probability_mass, self)
        mass = se.S(probability_mass)
        if mass == 0:
            return SymengineDist('0',*self._variables)
        elif mass > self.coefficient_sum():     # TODO implement coefficient_sum
            raise ValueError("Given probability mass is too large")
        elif mass < 0:
            raise ValueError("Given probability mass must be non-negative")
        var = se.S(variable)
        if var not in self._variables:
            raise ValueError(f'Not a variable: {variable}')
        result = 0
        mass_res = 0

        for element in se.series(self._s_func, var):
            result += element
            mass_res += element.subs([(sym, 1)
                                      for sym in element.free_symbols])
            if mass_res >= mass:
                return SymengineDist(result, *self._variables)

        raise NotImplementedError("unreachable")


class SymenginePGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"{p}/(1-(1-({p})) * {var})", str(var))

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> SymengineDist:
        return SymengineDist(f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/" +
                             "({var} - 1))", str(var))

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"({p}) * {var} + 1-({p})", str(var))

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> SymengineDist:
        return SymengineDist(f"exp(({lam}) * ({var} - 1))", str(var))

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"log(1-({p})*{var})/log(1-({p}))", str(var))

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam, p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"(({p})*{var} + (1-({p})))^({n})", str(var))

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> SymengineDist:
        return SymengineDist("0", *map(str, variables))

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> SymengineDist:
        return SymengineDist("1", *map(str, variables))

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> SymengineDist:
        return SymengineDist(expression, *map(str, variables))
