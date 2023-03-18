from __future__ import annotations

from typing import (FrozenSet, Generator, Iterator, List, Set, Tuple, Type,
                    Union)

import pygin  # type: ignore
from probably.pgcl import Binop, BinopExpr, Expr, FunctionCallExpr, VarExpr
from sympy import nan, sympify

from prodigy.distribution.distribution import (CommonDistributionsFactory,
                                               Distribution, DistributionParam,
                                               MarginalType, State)


class FPS(Distribution):
    """
    This class models a probability distribution in terms of a formal power series.
    These formal powerseries are itself provided by `prodigy` a python binding to GiNaC,
    something similar to a computer algebra system implemented in C++.
    """
    def __init__(self,
                 expression: str,
                 *variables: str | VarExpr,
                 finite: bool | None = None):
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
                  finite: bool | None = None):
        result = cls("0")
        result._dist = dist
        result._variables = variables
        result._parameters = parameters
        result._finite = finite if finite is not None else dist.is_polynomial(
            variables) == pygin.troolean.true
        return result

    @staticmethod
    def factory() -> Type[ProdigyPGF]:
        return ProdigyPGF

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
                while True:
                    yield str(it.next()), State({variable: i})
                    i += 1
            else:
                # TODO this is just a placeholder until we have proper multivariate iteration
                variables = list(self.get_variables())

                def n_tuples(n):
                    """Generates all `n`-tuples of the natural numbers"""
                    if n < 1:
                        raise ValueError("n is too small")
                    if n == 1:
                        num = 0
                        while True:
                            yield [num]
                            num += 1
                    else:
                        index = 0
                        gen = n_tuples(n - 1)
                        vals = []
                        while True:
                            # This is absolutely unreadable, so just another reason to delete this asap
                            while len(vals) < index + 1:
                                # pylint: disable=stop-iteration-return
                                vals.append(next(gen))
                                # pylint: enable=stop-iteration-return
                            for i in range(index, -1, -1):
                                yield [i] + vals[index - i]
                            index += 1

                for vals in n_tuples(len(variables)):
                    s = f'{variables[0]}={vals[0]}'
                    for i in range(1, len(variables)):
                        s += f' & {variables[i]}={vals[i]}'
                    mass = str(self.get_probability_of(s))
                    if mass != '0':
                        yield mass, State(dict(zip(variables, vals)))
        else:
            terms = self._dist.get_terms(self._variables)
            for prob, vals in terms:
                yield prob, State(vals)

    def copy(self, deep: bool = True) -> Distribution:
        return FPS.from_dist(self._dist, self._variables, self._parameters,
                             self._finite)

    def get_probability_mass(self) -> Union[Expr, str]:
        return self._dist.mass(self._variables & set(self._dist.get_symbols()))

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        return self._dist.E(str(expression))

    def normalize(self) -> FPS:
        return FPS(self._dist.normalize(), *self._variables)

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        return self._parameters

    def _find_symbols(self, expr: str) -> Set[str]:
        return set(pygin.find_symbols(expr))

    def get_symbols(self) -> Set[str]:
        return set(self._dist.get_symbols())

    @staticmethod
    def evaluate(expression: str, state: State) -> int:
        return pygin.evaluate(expression, state.valuations)

    def _exhaustive_search(self, condition: Expr) -> Distribution:
        res = pygin.Dist('0')
        for prob, state in self:
            if self.evaluate_condition(condition, state):
                res += pygin.Dist(f"{prob} * {state.to_monomial()}")
        return FPS.from_dist(res,
                             self._variables,
                             self._parameters,
                             finite=True)

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

    def get_fresh_variable(
        self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        res: str = pygin.get_fresh_variable()
        while res in exclude:
            res = pygin.get_fresh_variable()
        return res

    def _update_var(self, updated_var: str, assign_var: str | int) -> FPS:
        return FPS.from_dist(
            self._dist.update_var(updated_var, str(assign_var)),
            self._variables, self._parameters, self._finite)

    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> FPS:
        return FPS.from_dist(
            self._dist.update_sum(temp_var, str(first_summand),
                                  str(second_summand)), self._variables,
            self._parameters)

    def _update_product(self, temp_var: str, first_factor: str,
                        second_factor: str,
                        approximate: str | float | None) -> FPS:
        return FPS.from_dist(
            self._dist.update_product(temp_var, first_factor, second_factor,
                                      self._variables, self._finite,
                                      approximate), self._variables,
            self._parameters)

    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> Distribution:
        res = self._dist.update_subtraction(temp_var, str(sub_from), str(sub))

        variables = self.get_variables()
        if sub_from in variables and sub in variables and not self.is_finite():
            # TODO this should be implemented in GiNaC, which might not be possible
            test = sympify(str(res))
            for var in self._variables - {temp_var}:
                test = test.limit(sympify(var), 1)
            test = test.subs(sympify(temp_var), 0)
            if test.has(sympify("zoo")) or test == nan:
                raise ValueError(
                    f"Cannot assign '{sub_from} - {sub}' to '{temp_var}' because it can be negative"
                )

        return FPS.from_dist(res, self._variables, self._parameters)

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> FPS:
        return FPS.from_dist(
            self._dist.update_modulo(temp_var, str(left), str(right),
                                     self._variables, self._finite,
                                     approximate), self._variables,
            self._parameters)

    def _update_division(self, temp_var: str, numerator: str | int,
                         denominator: str | int,
                         approximate: str | float | None) -> FPS:
        return FPS.from_dist(
            self._dist.update_division(temp_var, str(numerator),
                                       str(denominator), approximate),
            self._variables, self._parameters)

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> Distribution:
        return FPS.from_dist(
            self._dist.update_power(temp_var, base, exp, approximate),
            self._variables, self._parameters)

    def update_iid(self, sampling_dist: Expr, count: VarExpr,
                   variable: Union[str, VarExpr]) -> FPS:
        if not isinstance(sampling_dist, FunctionCallExpr):
            result = FPS.from_dist(
                self._dist.updateIid(str(variable),
                                     pygin.Dist(str(sampling_dist)),
                                     str(count)), self._variables,
                self._parameters)
            return result

        if sampling_dist.function in {"unif", "unif_d"}:
            start, end = sampling_dist.params[0]
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(
                    f"1/(({end}) - ({start}) + 1) * test^({start}) "
                    f"* (test^(({end}) - ({start}) + 1) - 1) / (test - 1)"),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        [param] = sampling_dist.params[0]
        if sampling_dist.function == "geometric":
            result = self._dist.updateIid(str(variable),
                                          pygin.geometric("test", str(param)),
                                          str(count))
            return FPS.from_dist(result, self._variables, self._parameters)
        if sampling_dist.function == "bernoulli":
            result = self._dist.updateIid(
                str(variable), pygin.Dist(f"{param} * test + (1-{param})"),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == "poisson":
            result = self._dist.updateIid(
                str(variable), pygin.Dist(f"exp({param} * (test - 1))"),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        raise NotImplementedError(f"Unsupported distribution: {sampling_dist}")

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

    def approximate_unilaterally(
            self, variable: str,
            probability_mass: str | float) -> Distribution:
        return FPS.from_dist(
            self._dist.approximate_unilaterally(variable,
                                                str(probability_mass)),
            self._variables, self._parameters)


class ProdigyPGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        return FPS.from_dist(pygin.geometric(var, str(p)), {str(var)},
                             {*pygin.find_symbols(str(p))})

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> FPS:
        function = f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/({var} - 1))"
        return FPS(function, str(var), finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        function = f"({p}) * {var} + 1-({p})"
        return FPS(function, str(var), finite=True)

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
        return FPS("1", *variables, finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> FPS:
        return FPS(expression, *variables)

    @staticmethod
    def zero(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("0", *variables, finite=True)
