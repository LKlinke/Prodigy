from __future__ import annotations

import functools
from typing import Generator, Iterator, List, Set, Tuple, Union, get_args

import pygin  # type: ignore
from probably.pgcl import (BernoulliExpr, Binop, BinopExpr, BoolLitExpr,
                           DistrExpr, DUniformExpr, Expr, GeometricExpr,
                           IidSampleExpr, NatLitExpr, PoissonExpr, RealLitExpr,
                           Unop, UnopExpr, VarExpr)
from sympy import sympify

from prodigy.distribution.distribution import (CommonDistributionsFactory,
                                               Distribution, DistributionParam,
                                               MarginalType, State)
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.pgcl.pgcl_checks import (check_is_constant_constraint,
                                      check_is_modulus_condition, has_variable)


class FPS(Distribution):
    """
    This class models a probability distribution in terms of a formal power series.
    These formal powerseries are itself provided by `prodigy` a python binding to GiNaC,
    something similar to a computer algebra system implemented in C++.
    """
    def __init__(self,
                 expression: str,
                 *variables: str | VarExpr,
                 finite: bool = None):
        self._variables = set(str(var) for var in variables if str(var) != "")
        self._parameters = set()

        for var in pygin.find_symbols(expression):
            if var not in self._variables:
                if len(variables) > 0:
                    self._parameters.add(var)
                else:
                    self._variables.add(var)
        self._dist = pygin.Dist(expression, list(self._parameters))

        self._finite = finite if finite is not None else self._dist.is_polynomial(
            self._variables) == pygin.troolean.true

    @classmethod
    def from_dist(cls,
                  dist: pygin.Dist,
                  variables: Set[str],
                  parameters: Set[str],
                  finite: bool = None):
        result = cls("0")
        result._dist = dist
        result._variables = variables
        result._parameters = parameters
        result._finite = finite if finite is not None else dist.is_polynomial(
            variables) == pygin.troolean.true
        return result

    def __add__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(
                self._dist + pygin.Dist(
                    str(other),
                    list(
                        set(pygin.find_symbols(str(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist + other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Addition of {self._dist} and {other} not supported.")

    def __sub__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(
                self._dist - pygin.Dist(
                    str(other),
                    list(
                        set(pygin.find_symbols(str(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist - other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Subtraction of {self._dist} and {other} not supported.")

    def __mul__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(
                self._dist * pygin.Dist(
                    str(other),
                    list(
                        set(pygin.find_symbols(str(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist * other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Multiplication of {type(self._dist)} and {type(other)} not supported."
            )

    def __truediv__(self, other) -> FPS:
        if isinstance(other, str):
            other = FPS(other)
        if isinstance(other, FPS):
            return FPS.from_dist(self._dist * pygin.Dist(f"1/({str(other)})"),
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        raise NotImplementedError(
            f"Division of {type(self._dist)} and {type(other)} not supported.")

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            other = FPS(other)
        if isinstance(other, FPS):
            if not self._variables == other._variables:
                return False
            if not self._parameters == other._parameters:
                return False
            return self._dist == other._dist
        else:
            return False

    def __le__(self, other) -> bool:
        raise NotImplementedError(__name__)

    def __str__(self) -> str:
        return str(self._dist)

    def __repr__(self):
        return self._dist.__repr__()

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        if not self._finite:
            if len(self._variables) == 1:
                variable = list(self._variables)[0]
                it = self._dist.coefficient_iterator(variable)
                i = 0
                while it.rest() != pygin.Dist("0"):
                    yield it.next(), State({variable: i})
                    i += 1
            else:
                raise NotImplementedError(
                    "Currently only one-dimensional iteration is supported on infinite FPS"
                )
        else:
            terms = self._dist.get_terms(self._variables)
            for res in [(prob, State(vals)) for prob, vals in terms]:
                yield res

    def copy(self, deep: bool = True) -> Distribution:
        return FPS.from_dist(self._dist, self._variables, self._parameters,
                             self._finite)

    def get_probability_of(self, condition: Union[Expr, str]):
        raise NotImplementedError(__name__)

    def get_probability_mass(self) -> Union[Expr, str]:
        return self._dist.mass()

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        return self._dist.E(str(expression))

    def normalize(self) -> FPS:
        return FPS(self._dist.normalize(), *self._variables)

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        return self._parameters

    def filter(self, condition: Expr) -> FPS:
        # Boolean literals
        if isinstance(condition, BoolLitExpr):
            return self if condition.value else FPS("0", *self._variables)

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
            return self.filter(BoolLitExpr(sympify(
                str(condition))))  # TODO somehow handle this without sympy?

        if isinstance(condition, BinopExpr) and not set(
                pygin.find_symbols(str(condition.lhs))) | set(
                    pygin.find_symbols(str(condition.rhs))) <= (
                        self._variables | self._parameters):
            raise ValueError(
                f"Cannot filter based on the expression {str(condition)} because it contains unknown variables"
            )

        # Modulo extractions
        if check_is_modulus_condition(condition):
            return self._arithmetic_progression(
                str(condition.lhs.lhs),
                str(condition.lhs.rhs))[condition.rhs.value]

        # Constant expressions
        if check_is_constant_constraint(condition, self):
            return self._filter_constant_condition(condition)

        # all other conditions given that the Generating Function is finite (exhaustive search)
        if self._finite:
            res = pygin.Dist('0')
            for prob, state in self:
                # TODO implement condition evaluation in C++
                if GeneratingFunction.evaluate_condition(condition, state):
                    res += pygin.Dist(f"{prob} * {state.to_monomial()}")
            return FPS.from_dist(res,
                                 self._variables,
                                 self._parameters,
                                 finite=True)

        # Worst case: infinite Generating function and  non-standard condition.
        # Here we try marginalization and hope that the marginal is finite so we can do
        # exhaustive search again. If this is not possible, we raise an NotComputableException
        expression = self._explicit_state_unfolding(condition)
        return self.filter(expression)

    def _filter_constant_condition(self, condition: Expr) -> FPS:
        # Normalize the conditional to variables on the lhs from the relation symbol.
        if isinstance(condition.rhs, VarExpr):
            switch_comparison = {
                Binop.EQ: Binop.EQ,
                Binop.LEQ: Binop.GEQ,
                Binop.LT: Binop.GT,
                Binop.GEQ: Binop.LEQ,
                Binop.GT: Binop.LT
            }
            return self._filter_constant_condition(
                BinopExpr(operator=switch_comparison[condition.operator],
                          lhs=condition.rhs,
                          rhs=condition.lhs))

        # is normalized conditional
        if isinstance(condition.lhs, VarExpr):
            if condition.operator == Binop.EQ:
                return FPS.from_dist(
                    self._dist.filterEq(str(condition.lhs),
                                        str(condition.rhs)), self._variables,
                    self._parameters)
            elif condition.operator == Binop.LT:
                return FPS.from_dist(
                    self._dist.filterLess(str(condition.lhs),
                                          str(condition.rhs)), self._variables,
                    self._parameters)
            elif condition.operator == Binop.LEQ:
                return FPS.from_dist(
                    self._dist.filterLeq(str(condition.lhs),
                                         str(condition.rhs)), self._variables,
                    self._parameters)
            elif condition.operator == Binop.GT:
                return FPS.from_dist(
                    self._dist.filterGreater(str(condition.lhs),
                                             str(condition.rhs)),
                    self._variables, self._parameters)
            elif condition.operator == Binop.GEQ:
                return FPS.from_dist(
                    self._dist.filterGeq(str(condition.lhs),
                                         str(condition.rhs)), self._variables,
                    self._parameters)

        raise ValueError("Parameter is not a constant condition")

    def _explicit_state_unfolding(self, condition: Expr) -> BinopExpr:
        """
        Checks whether one side of the condition has only finitely many valuations and explicitly creates a new
        condition which is the disjunction of each individual evaluations.
        :param condition: The condition to unfold.
        :return: The disjunction condition of explicitly encoded state conditions.
        """
        expr = pygin.Dist(str(condition.rhs))
        syms = expr.get_symbols()
        if not len(syms) == 0:
            marginal = self.marginal(*syms)

        # Marker to express which side of the equation has only finitely many interpretations.
        left_side_original = True

        # Check whether left hand side has only finitely many interpretations.
        if len(syms) == 0 or not marginal.is_finite():
            # Failed, so we have to check the right hand side
            left_side_original = False
            expr = pygin.Dist(str(condition.lhs))
            syms = expr.get_symbols()
            marginal = self.marginal(*syms)

            if not marginal.is_finite():
                # We are not able to marginalize into a finite amount of states! -> FAIL filtering.
                raise NotImplementedError(
                    f"Instruction {condition} is not computable on infinite generating function"
                    f" {self._dist}")

        # Now we know that `expr` can be instantiated with finitely many states.
        # We generate these explicit state.
        state_expressions: List[BinopExpr] = []

        # Compute for all states the explicit condition checking that specific valuation.
        for _, state in marginal:

            # Evaluate the current expression
            # TODO implement evaluation in GiNaC
            evaluated_expr = GeneratingFunction.evaluate(str(expr), state)

            # create the equalities for each variable, value pair in a given state
            # i.e., {x:1, y:0, c:3} -> [x=1, y=0, c=3]
            # TODO extract this function from the GeneratingFunction class
            encoded_state = GeneratingFunction._state_to_equality_expression(
                state)

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

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[FPS]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        return [
            FPS.from_dist(dist, self._variables, self._parameters)
            for dist in self._dist.arithmetic_progression(variable, modulus)
        ]

    def is_zero_dist(self) -> bool:
        res = self._dist.is_zero()
        if res == pygin.troolean.false:
            return False
        elif res == pygin.troolean.true:
            return True
        else:
            raise ValueError('Cannot determine whether this FPS is zero')

    def is_finite(self) -> bool:
        return self._finite

    def old_update(self, expression: Expr) -> FPS:
        return FPS.from_dist(
            self._dist.update(str(expression.lhs), str(expression.rhs)),
            self._variables, self._parameters, self._finite)

    def update(self, expression: Expr) -> FPS:
        """ Updates the current distribution by applying the expression to itself.

            Some operations are illegal and will cause this function to raise an error. These operations include subtraction
            that may cause a variable to have a negative value, division that may cause a variable to have a value that is
            not an integer, and certain operations on infinite generating functions if the variables involved have an infinite
            marginal (such as multiplication of two variables).

            Parameters are not allowed in an update expression.
        """

        # TODO add some useful form of approximation support

        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."

        variable = expression.lhs.var
        if variable not in self._variables:
            raise ValueError(
                f"Cannot assign to variable {variable} because it does not exist"
            )

        # pylint: disable=protected-access
        def evaluate(function: FPS, expression: Expr,
                     temp_var: str) -> Tuple[FPS, str]:
            # TODO handle reals in every case
            if isinstance(expression, BinopExpr):
                xl = pygin.get_fresh_variable()
                xr = pygin.get_fresh_variable()
                f = function.set_variables(*(function.get_variables()
                                             | {xl, xr}))
                f, t_1 = evaluate(f, expression.lhs, xl)
                f, t_2 = evaluate(f, expression.rhs, xr)
                if expression.operator == Binop.PLUS:
                    f = FPS.from_dist(f._dist.update_sum(temp_var, t_1, t_2),
                                      f._variables, f._parameters)
                elif expression.operator == Binop.TIMES:
                    f = FPS.from_dist(
                        f._dist.update_product(temp_var, t_1, t_2,
                                               f._variables, f._finite),
                        f._variables, f._parameters)
                elif expression.operator == Binop.MINUS:
                    f = FPS.from_dist(
                        f._dist.update_subtraction(temp_var, t_1, t_2),
                        f._variables, f._parameters)
                elif expression.operator == Binop.MODULO:
                    f = FPS.from_dist(
                        f._dist.update_modulo(temp_var, t_1, t_2, f._variables,
                                              f._finite), f._variables,
                        f._parameters)
                elif expression.operator == Binop.DIVIDE:
                    f = FPS.from_dist(
                        f._dist.update_division(temp_var, t_1, t_2),
                        f._variables, f._parameters)
                # TODO handle power
                else:
                    raise ValueError(
                        f"Unsupported binary operator: {expression.operator}")

                f = f.marginal(xl, xr, method=MarginalType.EXCLUDE)
                return f, temp_var

            if isinstance(expression, VarExpr):
                f = FPS.from_dist(
                    function._dist.update_var(temp_var, expression.var),
                    function._variables, function._parameters,
                    function._finite)
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
            result = FPS.from_dist(
                self.marginal(variable, method=MarginalType.EXCLUDE)._dist *
                pygin.Dist(f'pow({variable}, {value})'), self._variables,
                self._parameters, self._finite)
        else:
            result, _ = evaluate(self, expression.rhs, variable)
        return result

    def update_iid(self, sampling_exp: IidSampleExpr,
                   variable: Union[str, VarExpr]) -> FPS:

        sample_dist = sampling_exp.sampling_dist
        if isinstance(sample_dist, GeometricExpr):
            result = self._dist.updateIid(
                str(variable), pygin.geometric("test", str(sample_dist.param)),
                str(sampling_exp.variable))
            return FPS.from_dist(result, self._variables, self._parameters)
        if isinstance(sample_dist, BernoulliExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(
                    f"{sample_dist.param} * test + (1-{sample_dist.param})"),
                str(sampling_exp.variable))
            return FPS.from_dist(result, self._variables, self._parameters)

        elif isinstance(sample_dist, PoissonExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(f"exp({sample_dist.param} * (test - 1))"),
                str(sampling_exp.variable))
            return FPS.from_dist(result, self._variables, self._parameters)

        elif isinstance(sample_dist, DUniformExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(
                    f"1/(({sample_dist.end}) - ({sample_dist.start}) + 1) * test^({sample_dist.start}) "
                    f"* (test^(({sample_dist.end}) - ({sample_dist.start}) + 1) - 1) / (test - 1)"
                ), str(sampling_exp.variable))
            return FPS.from_dist(result, self._variables, self._parameters)

        elif isinstance(sample_dist, get_args(Expr)) and not isinstance(
                sample_dist, get_args(DistrExpr)):
            result = FPS.from_dist(
                self._dist.updateIid(str(variable),
                                     pygin.Dist(str(sample_dist)),
                                     str(sampling_exp.variable)),
                self._variables, self._parameters)
            return result
        else:
            raise NotImplementedError(
                "Iid Distribution type currently not supported.")

    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> FPS:
        result = self._dist
        remove_vars = {
            MarginalType.EXCLUDE: {str(var)
                                   for var in variables},
            MarginalType.INCLUDE:
            self._variables - {str(var)
                               for var in variables}
        }
        for var in remove_vars[method]:
            result = result.update_var(str(var), "0")
        return FPS.from_dist(result, self._variables - remove_vars[method],
                             self._parameters)

    def set_variables(self, *variables: str) -> FPS:
        new_variables = set(variables)
        if new_variables.intersection(self._parameters):
            raise ValueError(
                f"Indeterminate(s) {new_variables.intersection(self._parameters)} cannot be parameters and"
                f" variables at the same time.")
        self._parameters |= self._variables - new_variables
        return FPS.from_dist(self._dist, new_variables, self._parameters,
                             self._finite)

    def set_parameters(self, *parameters: str) -> FPS:
        new_parameters = set(parameters)
        if new_parameters.intersection(self._variables):
            raise ValueError(
                f"Indeterminate(s) {new_parameters.intersection(self._variables)} cannot be parameters and"
                f" variables at the same time.")
        self._variables |= self._parameters - new_parameters
        return FPS.from_dist(self._dist, self._variables, new_parameters,
                             self._finite)

    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        raise NotImplementedError(__name__)


class ProdigyPGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        return FPS.from_dist(pygin.geometric(var, str(p)), {str(var)},
                             {str(p)})

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> FPS:
        function = f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/({var} - 1))"
        return FPS(function, str(var))

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        function = f"({p}) * {var} + 1-({p})"
        return FPS(function, str(var))

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> FPS:
        function = f"exp(({lam}) * ({var} - 1))"
        return FPS(function, str(var))

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        function = f"log(1-({p})*{var})/log(1-({p}))"
        return FPS(function, str(var))

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> FPS:
        function = f"(({p})*{var} + (1-({p})))^({n})"
        return FPS(function, str(var))

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> FPS:
        raise NotImplementedError(__name__)

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("1", *variables)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> FPS:
        return FPS(expression, *variables)

    @staticmethod
    def zero(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("0", *variables)
