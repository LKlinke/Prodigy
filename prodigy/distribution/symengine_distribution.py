from __future__ import annotations

import logging
import operator
from typing import Union, Generator, Sequence, Iterator, Tuple, Type, List, get_args

import symengine as se
from probably.pgcl import VarExpr, Expr, FunctionCallExpr, RealLitExpr, UnopExpr, Binop, BinopExpr, Unop, NatLitExpr
from probably.util.ref import Mut

from prodigy.distribution import Distribution
from prodigy.distribution import MarginalType, State, CommonDistributionsFactory, DistributionParam
from prodigy.util.logger import log_setup
from prodigy.util.order import default_monomial_iterator

from probably.pgcl.parser import parse_expr

# todo remove once methods in symengine
import sympy as sp

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG, file="GF_operations.log")


class SymengineDist(Distribution):
    @staticmethod
    def factory() -> Type[CommonDistributionsFactory]:
        return SymenginePGF

    def __init__(self, function, *variables: Union[se.Symbol, str]):
        self._s_func: se.Expr = _parse_to_symengine(function)
        self._variables: set[se.Symbol] = self._s_func.free_symbols
        self._parameters: set[se.Symbol] = set()
        if variables:
            self._variables = {se.Symbol(str(v)) for v in variables}
            self._parameters: set[se.Symbol] = self._s_func.free_symbols - self._variables

    def _check_symbol_consistency(self, other: SymengineDist) -> bool:
        """
        Checks whether variables and parameters names are compliant.
        """
        variables = self._variables | other._variables
        parameters = self._parameters | other._parameters
        return variables.intersection(parameters) == set()

    def __add__(self, other) -> SymengineDist:
        return self._arithmetic(other, "add", operator.add)

    def __sub__(self, other) -> SymengineDist:
        return self._arithmetic(other, "subtract", operator.sub)

    def __mul__(self, other) -> SymengineDist:
        return self._arithmetic(other, "multiply", operator.mul)

    def __truediv__(self, other) -> SymengineDist:
        return self._arithmetic(other, "divide", operator.truediv)

    def _arithmetic(self, other: str | int | float | SymengineDist, textual_descr: str, op: operator) -> SymengineDist:
        """
        Applies an arithmetic operator to self and one unknown object.

        :param other: The object for which the operator should be applied with self.
            Supported types: str, int, float, SymengineDist
        :param textual_descr: The text description of the operator to be applied with self. Used for the error message
        :param op: The operator to be applied with self.
        """
        if isinstance(other, (str, int, float)):
            # FIXME
            #   not sure how this should look like
            other_syms = self._find_symbols(str(other))
            res = op(self._s_func, se.S(other))
            # If other has no symbols, simply return the current symbols as variables / parameters
            if other_syms == set():
                return SymengineDist(res).set_variables_and_parameters(
                    self.get_variables(), self.get_parameters()
                )
            params = other_syms.intersection(self.get_parameters())
            if params != set():
                return SymengineDist(res).set_variables_and_parameters(
                    self.get_variables().union(other_syms.difference(params)),
                    self.get_parameters(),
                )
            else:
                return SymengineDist(res).set_variables_and_parameters(
                    self.get_variables().union(other_syms),
                    self.get_parameters()
                )

        if not isinstance(other, SymengineDist):
            raise SyntaxError(f"You cannot {textual_descr} {type(self)} by {type(other)}.")

        # Actual operation
        if not self._check_symbol_consistency(other):
            clash = (self._variables | other._variables) & (self._parameters | other._parameters)
            raise SyntaxError(f"Name clash: {clash} for {self} and {other}.")
        new_vars, new_params = (self.get_variables() | other.get_variables()), (self.get_parameters() | other.get_parameters())
        res = op(self._s_func, other._s_func)
        return SymengineDist(res).set_variables_and_parameters(new_vars, new_params)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SymengineDist):
            return False
        if not other._variables == self._variables:
            return False
        if not other._parameters == self._parameters:
            return False
        return self._s_func == other._s_func

    def __le__(self, other: SymengineDist) -> bool:
        if not isinstance(other, SymengineDist):
            raise TypeError(f"Incomparable types {type(self)} and {type(other)}.")

        if self.is_finite():
            for prob, state in self:
                s, o = se.S(prob), other.get_prob_by_diff(state)
                if s > o:
                    return False
                return True

        if other.is_finite():
            for prob, state in other:
                if self.get_prob_by_diff(state) > se.S(prob):
                    return False
                return True

        difference: se.Basic = self._s_func - other._s_func
        # todo replace once suitable method is found
        #   cf. https://github.com/symengine/symengine.py/issues/492
        logger.info("Falling back to sympy")
        if sp.S(difference).is_polynomial():
            return all(
                map(lambda x: x > 0, difference.as_coefficients_dict().values())
            )
        raise RuntimeError(
            "Both objects have infinite support. We cannot determine the order between them."
        )

    def coefficient_sum(self) -> se.Expr:
        # TODO Limits seem to not be present in SE
        logger.info("Falling back to sympy")
        coefficient_sum: sp.Expr = sp.S(self._s_func).simplify()
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(
                var, 1, "-"
            )
        return se.S(coefficient_sum)

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
                    yield str(prob_fun(State({v: i}))), State({v: i})
                    i += 1
            else:
                for tup in default_monomial_iterator(len(v)):
                    state = State(dict(zip(v, tup)))
                    yield str(prob_fun(state)), state
        else:
            # todo replace with symengine method
            logger.info("Falling back to pygin")
            import pygin
            # FIXME this method does not return the terms in a fixed order, sometimes leads to wrong results, e.g.
            #   First test:
            #       1/2, {'x': 2, 'y': 3}
            #       1/2, {'x': 3, 'y': 2}
            #   Second test:
            #       1/2, {'x': 2, 'y': 3}
            #       1/2, {'x': 2, 'y': 3}
            for prob, vals in (pygin.Dist(str(self._s_func).replace("**", "^"), list(self.get_parameters()))
                    .get_terms(self.get_variables())):
                yield prob, State(vals)
            # TODO replace with sympy method for now
            # logger.info("Falling back to sympy")
            # fun: sp.Expr = sp.S(self._s_func)
            #
            # if not fun.is_polynomial(*self._variables):
            #     fun: sp.Poly = fun.expand().ratsimp().as_poly(
            #         *[sp.Symbol(x) for x in self._variables])
            # else:
            #     fun: sp.Poly = fun.as_poly(*[sp.Symbol(x) for x in self._variables])
            # while fun.as_expr() != 0:
            #     yield str(fun.EC()), State(fun.EM().as_expr().as_powers_dict())
            #     fun -= fun.EC() * fun.EM().as_expr()

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
                yield str(prob_fun(State(dict(zip(v, tup))))), State(dict(zip(v, tup)))
        else:
            logger.info("Falling back to pygin")
            import pygin
            # FIXME this method does not return the terms in a fixed order, sometimes leads to wrong results, e.g.
            #   First test:
            #       1/2, {'x': 2, 'y': 3}
            #       1/2, {'x': 3, 'y': 2}
            #   Second test:
            #       1/2, {'x': 2, 'y': 3}
            #       1/2, {'x': 2, 'y': 3}
            for prob, vals in (pygin.Dist(str(self._s_func).replace("**", "^"), list(self.get_parameters()))
                    .get_terms(self.get_variables())):
                yield prob, State(vals)

    # TODO integrate these functions better / move them / replace by correct signature
    def get_prob_by_diff(self, state: State) -> se.Basic:
        """
        Get the probability of a given state by means of differentiation
        :param state: The state to get the probability of
        :return: The probability of the given state
        """
        fun = self._s_func.diff(*[el for tup in state.items() for el in tup])
        # se.gamma(n + 1) == factorial(n) (for natural numbers)
        return (self.safe_subs(*zip(list(self._variables), [0] * len(self._variables)), fun=fun) /
                se.S("*".join([str(se.gamma(el + 1)) for _, el in state.items()])))

    def get_prob_by_series(self, state: State) -> se.Basic:
        """
        Get the probability of a given state by means of taylor expansion
        :param state: The state to get the probability of
        :return: The probability of the given state
        """
        series = self._s_func
        args = []
        for (v, val) in state.items():
            series: se.Basic = se.series(series, v, 0, val + 1)
            # Build the expression
            args += [f"{v} ** {val}"]
        # Use simplify to convert "v ** 0" to 1 in order to extract the probability correctly
        args = se.S("*".join(args)).simplify()
        # Expand series as simplifications (such as factoring out) lead to missing coefficients
        coefficient_dict = series.expand().as_coefficients_dict()

        if args != se.S(1):
            res = coefficient_dict[args] if args in coefficient_dict else se.S(0)
        else:
            # If we want the probability of v ** 0, the order is not {monomial: coefficient} but {coefficient: 1},
            # hence we need to return the key rather than the value
            res = list(coefficient_dict.keys())[0]
        return res

    def copy(self, deep: bool = True) -> SymengineDist:
        res = SymengineDist(0)
        res._s_func = self._s_func.copy()
        res._variables = self._variables.copy()
        res._parameters = self._parameters.copy()
        return res

    def get_probability_mass(self) -> str:
        fast_result: se.Expr = self.safe_subs(*zip(self._variables, [1] * len(self._variables)))
        if fast_result == se.nan or fast_result == se.zoo:
            raise ValueError(
                f"Indeterminate expression {self._s_func} with {self._variables} mapped to {(1,) * len(self._variables)}"
            )
        return str(fast_result)

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        # todo replace sympy by symengine if method is implemented
        logger.info("Falling back to sympy")
        expr = sp.S(str(expression)).ratsimp().expand()
        if not expr.is_polynomial():
            raise NotImplementedError(
                "Expected Value only computable for polynomial expressions.")

        if len(self._find_symbols(expr) & self.get_variables()) == 0:
            return str(expr)
        if not self._find_symbols(expr).issubset(
                self.get_variables().union(self.get_parameters())):
            raise ValueError(
                f"Cannot compute expected value of {expression} because it contains unknown symbols"
            )

        marginal = self.marginal(*(self._find_symbols(expr) & self.get_variables()),
                                 method=MarginalType.INCLUDE)._s_func
        gen_func = SymengineDist(expr,
                                 *(self._find_symbols(expr) & self.get_variables()))
        expected_value = sp.Integer(0)
        for prob, state in gen_func:
            tmp = sp.S(marginal)
            for var, val in state.items():
                var_sym = se.S(var)
                for _ in range(val):
                    tmp = tmp.diff(var_sym, 1) * var_sym
                tmp = tmp.limit(var_sym, 1, '-')
            summand: se.Expr = se.S(prob) * tmp
            if len(summand.free_symbols) == 0 and summand < 0:
                raise ValueError(f'Intermediate result is negative: {summand}')
            expected_value += summand
        if expected_value == sp.S('oo'):
            return str(RealLitExpr.infinity())
        else:
            return str(expected_value)

    def normalize(self) -> SymengineDist:
        mass = se.S(self.get_probability_mass())
        if mass == 0:
            raise ZeroDivisionError
        return SymengineDist(self._s_func / mass, *self._variables)

    def get_variables(self) -> set[str]:
        return set(map(str, self._variables))

    def get_parameters(self) -> set[str]:
        return set(map(str, self._parameters))

    def _exhaustive_search(self, condition: Expr) -> SymengineDist:
        res = se.S("0")
        for prob, state in self:
            if self.evaluate_condition(condition, state):
                res += se.S(f"{prob} * {state.to_monomial()}")
        return SymenginePGF.from_expr(
            res,
            *self._variables
        )

    def _filter_constant_condition(self, condition: Expr) -> SymengineDist:
        # FIXME
        #   calling
        #       python prodigy/cli.py --engine symengine main pgfexamples/sequential_loops_second_inv.pgcl
        #   leads to an error with the expression "m > 0", where m \in self._variables

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
            result += (self.safe_subs(variable, 0, fun=self._s_func.diff(variable, i) / se.gamma(i + 1))
                       * variable ** i)
        return SymengineDist(result).set_variables_and_parameters(self.get_variables(), self.get_parameters())

    def _arithmetic_progression(self, variable: str, modulus: str) -> Sequence[SymengineDist]:
        a = se.S(modulus)
        var = (se.S(variable))

        # This should be a faster variant for univariate distributions.
        if self._variables == {var}:
            result = []
            for remainder in range(int(modulus)):
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
                                            remainder) * self.safe_subs(
                    var, (primitive_uroot ** m) * var)
            result.append(
                SymengineDist(f"(1/{a}) * ({psum})", *self._variables)
            )
        return result

    def hadamard_product(self, other: SymengineDist) -> SymengineDist:
        raise NotImplementedError("Hadamard Product is currently unsupported")  # ignore for now

    def _find_symbols(self, expr: str) -> set[str]:
        # FIXME Symengine does not recognize the character '%', i.e.
        #       se.S("c % 2")   # Fails
        #   replace this workaround!
        if "%" in str(expr):
            import re
            expr = re.sub(r'(\w+)\s*%\s*(\w+)', r'mod(\1, \2)', expr)
        return {str(sym) for sym in se.S(expr).free_symbols}

    @staticmethod
    def evaluate(expression: str, state: State):
        """ Evaluates the expression in a given state. """

        s_exp = se.S(expression)
        # Iterate over the variable, value pairs in the state
        variables = []
        values = []
        for var, value in state.items():
            # Convert the variable into a sympy symbol and substitute
            variables.append(se.S(var))
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
        # todo replace once a suitable method is found within symengine
        #   cf. https://github.com/symengine/symengine.py/issues/492
        logger.info("Falling back to sympy")
        # cancel() is used because is_polynomial() does not attempt simplification
        #   cf. https://docs.sympy.org/latest/modules/core.html#sympy.core.expr.Expr.is_polynomial
        return sp.S(self._s_func).cancel().is_polynomial()

    def get_fresh_variable(self, exclude: set[str] | frozenset[str] = frozenset()) -> str:
        i = 0
        while f'x_{i}' in (
                self.get_variables()
                | self.get_parameters()) or f'x_{i}' in exclude:
            i += 1
        return f'x_{i}'

    def _update_var(self, updated_var: str, assign_var: str | int) -> SymengineDist:
        up_var, as_var = se.Symbol(updated_var), se.S(assign_var)
        if as_var in self._parameters:
            raise ValueError("Assignment to parameters is not allowed")
        if as_var.is_symbol and as_var not in self._variables:
            raise ValueError(f"Unknown symbol: {assign_var}")

        if not updated_var == assign_var:
            if as_var in self._variables:
                res = self.safe_subs((up_var, 1), (as_var, as_var * up_var))
            else:
                res = self.safe_subs(up_var, 1) * up_var ** as_var
            return SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())
        else:
            return self.copy()

    def _update_sum(self, temp_var: str, first_summand: str | int, second_summand: str | int) -> SymengineDist:
        update_var, sum_1, sum_2, res = se.S(temp_var), se.S(first_summand), se.S(second_summand), self._s_func

        # Two variables are added
        if sum_1 in self._variables and sum_2 in self._variables:
            if sum_2 == update_var:
                sum_1, sum_2 = sum_2, sum_1
            if sum_1 == update_var:
                if sum_2 == update_var:
                    res = self.safe_subs(update_var, update_var ** 2, fun=res)
                else:
                    res = self.safe_subs(sum_2, sum_2 * update_var, fun=res)
            else:
                res = self.safe_subs((update_var, 1), (sum_1, sum_1 * update_var), (sum_2, sum_2 * update_var), fun=res)

        # One variable and one literal / parameter is added
        elif sum_1 in self._variables or sum_2 in self._variables:
            if sum_1 in self._variables:
                var, lit = sum_1, sum_2
            else:
                var, lit = sum_2, sum_1
            if not var == update_var:
                res = self.safe_subs((update_var, 1), (var, update_var * var), fun=res)
            res = res * (update_var ** lit)

        # Two literals / parameters are added
        else:
            res = self.safe_subs(update_var, 1, fun=res) * (update_var ** (sum_1 + sum_2))
        return SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())

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
                            f'Cannot perform the multiplication {first_factor} * {second_factor} '
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
                    term: se.Expr = se.S(prob) * se.S(state.to_monomial())
                    res = res - term
                    term = (self.safe_subs(update_var, 1, fun=term) * update_var_with_assumptions
                            ** (state[first_factor] * state[second_factor]))
                    res = res + term

        # Multiplication of one variable and one literal
        elif prod_1 in self._variables or prod_2 in self._variables:
            if prod_1 in self._variables:
                var, lit = prod_1, prod_2
            else:
                var, lit = prod_2, prod_1
            if var == update_var:
                res = self.safe_subs(update_var, update_var_with_assumptions ** lit, fun=res)
            else:
                res = self.safe_subs(update_var, 1, var, var * update_var_with_assumptions ** lit, fun=res)

        # Multiplication of two literals
        else:
            res = self.safe_subs(update_var, 1, fun=res) * (update_var_with_assumptions ** (prod_1 * prod_2))

        # TODO filter out assumptions over symbols once implemented
        return SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())

    def _update_subtraction(self, temp_var: str, sub_from: str | int, sub: str | int) -> SymengineDist:
        update_var, sub_1, sub_2, res = se.S(temp_var), se.S(sub_from), se.S(sub), self._s_func

        # Subtraction of two variables
        if sub_1 in self._variables and sub_2 in self._variables:
            if sub_2 == update_var:
                if sub_1 == update_var:
                    res = self.safe_subs(update_var, 1, fun=res)
                else:
                    res = self.safe_subs(update_var, update_var ** (-1), sub_1, sub_1 * update_var, fun=res)
            else:
                if not sub_1 == update_var:
                    res = self.safe_subs(update_var, 1, sub_1, sub_1 * update_var, fun=res)
                res = self.safe_subs(sub_2, sub_2 * update_var ** (-1), fun=res)

        # Literal subtracted from variable
        elif sub_1 in self._variables:
            if not update_var == sub_1:
                res = self.safe_subs(update_var, 1, sub_1, sub_1 * update_var, fun=res)
            res = res * update_var ** (-sub_2)

        # Variable subtracted from literal
        elif sub_2 in self._variables:
            if sub_2 == update_var:
                res = self.safe_subs(update_var, update_var ** (-1), fun=res) * update_var ** sub_1
            else:
                res = self.safe_subs(update_var, 1, fun=res) * update_var ** sub_1
                res = self.safe_subs(sub_2, sub_2 * update_var ** (-1), fun=res)

        # Two literals are subtracted from each other
        else:
            diff = sub_1 - sub_2
            if sub_from not in self._parameters and sub not in self._parameters and diff < 0:
                raise ValueError(
                    f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it is negative"
                )
            res = self.safe_subs(update_var, 1, fun=res) * update_var ** diff

        res = se.expand(res)
        expr = SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())

        test_fun: se.Basic = expr.marginal(temp_var).safe_subs(update_var, 0)
        if test_fun.simplify() in [se.nan, se.zoo]:
            raise ValueError(
                f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it can be negative"
            )
        return expr

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> SymengineDist:
        left_sym, right_sym = se.Symbol(str(left)), se.Symbol(
            str(right))
        if left_sym in self._parameters or right_sym in self._parameters:
            raise ValueError('Cannot perform modulo operation on parameters')

        update_var = se.S(temp_var)
        result = 0

        # On finite GFs, iterate over all states
        if self.is_finite():
            for prob, state_r in self:
                prob = se.S(prob)
                if left_sym in self._variables:
                    left_var = state_r[left]
                else:
                    left_var = _parse_to_symengine(left)
                if right_sym in self._variables:
                    right_var = state_r[right]
                else:
                    right_var = _parse_to_symengine(right)
                result += prob * self.safe_subs(
                    update_var, 1, fun=_parse_to_symengine(state_r.to_monomial())) * update_var ** (left_var % right_var)

        # If the GF is infinite and right is a variable, it needs to have finite range
        elif right_sym in self._variables:
            assert isinstance(right, str)
            marginal_r = self.marginal(right)
            if not marginal_r.is_finite():
                if approximate is None:
                    raise ValueError(
                        f'Cannot perform modulo operation with infinite right hand side {right}'
                    )
                marginal_r = marginal_r.approximate_unilaterally(
                    right, approximate)
            for _, state_r in marginal_r:
                result += self.filter(
                    parse_expr(f'{right}={state_r[right]}'))._update_modulo(
                    temp_var, left, state_r[right], None)._s_func

        # If left is a variable, it doesn't have to have finite range
        elif left_sym in self._variables:
            assert isinstance(left, str)
            marginal_l = self.marginal(left)
            if marginal_l.is_finite():
                for _, state_l in marginal_l:
                    result += self.filter(
                        parse_expr(f'{left}={state_l[left]}'))._update_modulo(
                        temp_var, state_l[left], right, None)._s_func
            else:
                for index, gf in enumerate(
                        self._arithmetic_progression(left, str(right))):
                    # TODO this seems to compute the correct result, but it can't always be simplified to 0
                    #   from generating_function.py
                    result = result + self.safe_subs(update_var, 1, fun=gf._s_func) * update_var ** index

        # If both are not variables, simply compute the result
        else:
            return self._update_var(temp_var, int(left) % int(right))

        return SymengineDist(result).set_variables_and_parameters(self.get_variables(), self.get_parameters())

    def _update_division(self, temp_var: str, numerator: str | int, denominator: str | int,
                         approximate: str | float | None) -> SymengineDist:
        update_var = se.S(temp_var)
        div_1, div_2 = _parse_to_symengine(numerator), _parse_to_symengine(denominator)

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
                        res += self.safe_subs(update_var, 1, fun=x) * update_var ** (val_l / val_r)
                else:
                    val_l, val_r = state_l[numerator], div_2
                    x = self.filter(
                        parse_expr(f'{numerator}={val_l}'))._s_func
                    if val_l % val_r != 0 and x != 0:
                        raise ValueError(
                            f"Cannot assign {numerator} / {denominator} to {temp_var} "
                            "because it is not always an integer"
                        )
                    res += self.safe_subs(update_var, 1, fun=x) * update_var ** (val_l / val_r)
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
                    res += self.safe_subs(update_var,
                                          1, fun=x) * update_var ** (val_l / val_r)
            else:
                if div_1 % div_2 != 0:
                    raise ValueError(
                        f"Cannot assign {numerator} / {denominator} to {temp_var} because it is not always an integer"
                    )
                res = self.safe_subs(update_var,
                                     1) * update_var ** (div_1 / div_2)

        return SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> SymengineDist:
        update_var = se.Symbol(temp_var)
        pow_1, pow_2 = se.S(base), se.S(exp)
        res = self._s_func

        if pow_1 in self._parameters or pow_2 in self._parameters:
            raise ValueError(
                "Cannot perform an exponentiation containing parameters")

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
                    res += self.safe_subs(update_var, 1, fun=x) * update_var ** (state_l[base] **
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
                res += self.safe_subs(update_var, 1, fun=x) * update_var ** (state[base] ** pow_2)

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
                res += self.safe_subs(update_var, 1, fun=x) * update_var ** (pow_1 ** state[exp])

        # literal to the power of a literal
        else:
            res = self.safe_subs(update_var, 1, fun=res) * update_var ** (pow_1 ** pow_2)

        return SymengineDist(res).set_variables_and_parameters(self.get_variables(), self.get_parameters())

    def update_iid(self, sampling_dist: Expr, count: VarExpr, variable: Union[str, VarExpr]) -> SymengineDist:
        subst_var = count.var

        def subs(dist_gf, subst_var, variable) -> SymengineDist:
            result = self.marginal(
                variable,
                method=MarginalType.EXCLUDE) if subst_var != variable else self
            if subst_var == variable:
                result._s_func = result.safe_subs(
                    se.S(subst_var), dist_gf)
            else:
                result._s_func = result.safe_subs(
                    se.S(subst_var),
                    se.S(subst_var) * dist_gf)
            return result.set_variables_and_parameters(self.get_variables(), self.get_parameters())

        if not isinstance(sampling_dist, FunctionCallExpr):
            # create distribution in correct variable:
            expr = Mut.alloc(sampling_dist)
            dist_gf = se.S(str(expr.val))
            return subs(dist_gf, subst_var, variable)

        if sampling_dist.function == "binomial":
            n, p = map(_parse_to_symengine, sampling_dist.params[0])
            dist_gf = (1 - p + p * se.S(variable)) ** n
            return subs(dist_gf, subst_var, variable)

        if sampling_dist.function in {"unif", "unif_d"}:
            start, end = map(_parse_to_symengine, sampling_dist.params[0])
            var = se.S(variable)
            dist_gf = 1 / (end - start + 1) * var ** start * (
                    var ** (end - start + 1) - 1) / (var - 1)
            return subs(dist_gf, subst_var, variable)
        # All remaining distributions have only one parameter
        [param] = map(_parse_to_symengine, sampling_dist.params[0])
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
        if len(variables) == 0:
            raise ValueError("No variables were provided")
        if not {se.Symbol(str(x))
                for x in variables}.issubset(self._variables):
            raise ValueError(
                f"Unknown variable(s): { {se.S(str(x)) for x in variables} - self._variables}"
            )

        marginal_vars = set(
            map(se.Symbol, filter(lambda v: v != '', map(str, variables))))
        marginal = self.copy()
        s_var: str | VarExpr | se.Symbol
        if method == MarginalType.INCLUDE:
            for s_var in marginal._variables.difference(marginal_vars):
                marginal._s_func = marginal.safe_subs(s_var, 1)
            marginal._variables = marginal_vars
        else:
            for s_var in marginal_vars:
                marginal._s_func = marginal.safe_subs(s_var, 1)
            marginal._variables = marginal._variables.difference(marginal_vars)

        return marginal

    def set_variables(self, *variables: str) -> SymengineDist:
        new_variables = set(variables)
        if self.get_parameters() & new_variables:
            raise ValueError(
                f"At least one variable is already known as a parameter. {self._parameters=}, {new_variables=}")
        if not (self.get_parameters() | new_variables).issuperset({str(s) for s in self._s_func.free_symbols}):
            raise ValueError(f"There are unknown symbols which are neither variables nor parameters.")
        return SymenginePGF.from_expr(str(self._s_func), *new_variables)

    def set_parameters(self, *parameters: str) -> SymengineDist:
        new_params = {se.Symbol(p) for p in parameters}
        if self._variables & new_params:
            raise ValueError(
                f"At least one parameter is already known as a variable. {self._variables=}, {new_params=}")
        if not (self._variables | new_params).issuperset(self._s_func.free_symbols):
            raise ValueError(f"There are unknown symbols which are neither variables nor parameters.")
        new_gf = SymenginePGF.from_expr(str(self._s_func), *self._variables)
        new_gf._parameters = new_params
        return new_gf

    def set_variables_and_parameters(self, variables: set[str], parameters: set[str]):
        var_sym = {se.Symbol(v) for v in variables}
        param_sym = {se.Symbol(p) for p in parameters}
        if not (var_sym | param_sym).issuperset(self._s_func.free_symbols):
            raise ValueError(f"There are unknown symbols which are neither variables nor parameters: {set(self._s_func.free_symbols).difference((var_sym | param_sym))}")
        new_gf = SymenginePGF.from_expr(str(self._s_func))
        new_gf._variables = var_sym
        new_gf._parameters = param_sym
        return new_gf

    def approximate(self, threshold: Union[str, int]) -> Generator[SymengineDist, None, None]:
        logger.debug("expand_until() call")
        approx = se.Integer(0)
        precision = se.Integer(0)

        if isinstance(threshold, int):
            assert threshold > 0, "Expanding to less than 0 terms is not valid."
            for n, (prob, state) in enumerate(self):
                prob = se.S(prob)
                if n >= threshold:
                    break
                approx += prob * se.S(state.to_monomial())
                precision += prob
                yield SymengineDist(approx, *self._variables)
        elif isinstance(threshold, str):
            s_threshold = se.S(threshold)
            assert s_threshold < self.coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {s_threshold}, CSum {self.coefficient_sum()}"
            for prob, state in self:
                prob = se.S(prob)
                state = se.S(state.to_monomial())
                if precision >= s_threshold:
                    break
                approx += prob * state
                precision += prob
                yield SymengineDist(str(approx.expand()), *self._variables)
        else:
            raise TypeError(
                f"Parameter threshold can only be of type str or int,"
                f" not {type(threshold)}.")

    # TODO fix signature of parameters
    def safe_subs(self, *parameters: Union[Tuple[str, str | int], str, se.Expr, int], fun: se.Expr = None) -> se.Expr:
        """
        Substitution with checks for divisions by zero.
        First tries to use simple substitution, if result is NaN, convert to sympy,
        then check whether it is rational and depending on that, cancel the fraction or
        take the limit.

        :param parameters: The substitution parameters of form variable, value.
        :param fun: The function the substitution should be performed on. If not given, self._s_func is used.

        :returns: The substituted se.Expr
        """
        # todo make method more flexible? does it matter?
        all_tuples = all(isinstance(param, tuple) for param in parameters)
        if len(parameters) % 2 != 0 and not all_tuples:
            raise ValueError("There has to be an equal amount of variables and values")
        if all_tuples:
            pairs = parameters
        else:
            pairs = [(parameters[i], parameters[i + 1]) for i in range(0, len(parameters), 2)]
        if fun is None:
            fun = self._s_func
        for (variable, value) in pairs:
            f = fun
            # FIXME subs with dict instead of stepwise
            if f.subs(variable, value).simplify() in [se.nan, se.zoo]:
                logger.info("Falling back to sympy")
                sympy_func: sp.Expr = sp.S(fun)
                if sympy_func.is_Rational:
                    fun = se.S(sympy_func.cancel().simplify())
                else:
                    fun = se.S(sympy_func.limit(variable, value, "-").simplify())
            else:
                fun = fun.subs(variable, value)

        return fun

    def approximate_unilaterally(self, variable: str, probability_mass: str | float) -> SymengineDist:
        logger.debug("approximate_unilaterally(%s, %s) call on %s", variable,
                     probability_mass, self)
        mass = se.S(probability_mass)
        if mass == 0:
            return SymengineDist('0', *self._variables)
        elif mass > self.coefficient_sum():
            raise ValueError("Given probability mass is too large")
        elif mass < 0:
            raise ValueError("Given probability mass must be non-negative")
        var = se.S(variable)
        if var not in self._variables:
            raise ValueError(f'Not a variable: {variable}')
        result = 0
        mass_res = 0

        logger.info("Falling back to sympy")
        sympy_f: sp.Expr = sp.S(self._s_func)
        # todo remove with symengine method
        for element in sympy_f.series(sp.S(var), n=None):
            result += element
            mass_res += self.safe_subs(*[(sym, 1)
                                         for sym in element.free_symbols], fun=element)
            if mass_res >= mass:
                return SymengineDist(result, *self._variables)

        raise NotImplementedError("unreachable")


# TODO check which types of expressions one needs

def _parse_to_symengine(expression) -> se.Expr:
    """
    Parses an arbitrary expression to symengine.

    :param expression: The expression to parse.
    :returns: The parsed expression.
    """
    def probably_to_symengine(expr: Expr):
        if isinstance(expr, NatLitExpr):
            return se.Integer(expr.value)
        elif isinstance(expr, VarExpr):
            return se.Symbol(expr.var)
        elif isinstance(expr, RealLitExpr):
            return se.S(str(expr.to_fraction()))
        elif isinstance(expr, BinopExpr):
            supported_operators = {
                Binop.PLUS: operator.add,
                Binop.MINUS: operator.sub,
                Binop.TIMES: operator.mul,
                Binop.POWER: operator.pow,
                Binop.DIVIDE: operator.truediv,
                Binop.MODULO: operator.mod
            }
            op = expr.operator
            if op not in supported_operators:
                raise ValueError(f'Unsupported operand: {op}')
            return supported_operators[op](
                probably_to_symengine(expr.lhs),
                probably_to_symengine(expr.rhs)
            )
        raise ValueError(f"Unsupported type: {type(expression)}")

    if isinstance(expression, get_args(Expr)):
        return probably_to_symengine(expression)

    return se.S(str(expression))


class SymenginePGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"{p}/(1-(1-({p})) * {var})", str(var))

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> SymengineDist:
        return SymengineDist(f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/" +
                             f"({var} - 1))", str(var))

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
