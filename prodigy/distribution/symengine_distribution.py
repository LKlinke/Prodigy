from __future__ import annotations

import logging
from typing import Union, Generator, Set, FrozenSet, Sequence, Iterator, Tuple, Type

import symengine as se
from probably.pgcl import VarExpr, Expr

from prodigy.distribution import Distribution
from prodigy.distribution import MarginalType, State, CommonDistributionsFactory, DistributionParam
from prodigy.util.logger import log_setup

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
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self._s_func}"

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def normalize(self) -> SymengineDist:
        raise NotImplementedError()

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        raise self._parameters

    def _exhaustive_search(self, condition: Expr) -> SymengineDist:
        raise NotImplementedError()

    def _filter_constant_condition(self, condition: Expr) -> SymengineDist:
        raise NotImplementedError()

    def _arithmetic_progression(self, variable: str, modulus: str) -> Sequence[SymengineDist]:
        raise NotImplementedError()

    def hadamard_product(self, other: SymengineDist) -> SymengineDist:
        raise NotImplementedError()

    def _find_symbols(self, expr: str) -> Set[str]:
        raise NotImplementedError()

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
        return self._s_func.is_finite

    def get_fresh_variable(self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        raise NotImplementedError()

    def _update_var(self, updated_var: str, assign_var: str | int) -> SymengineDist:
        raise NotImplementedError()

    def _update_sum(self, temp_var: str, first_summand: str | int, second_summand: str | int) -> SymengineDist:
        raise NotImplementedError()

    def _update_product(self, temp_var: str, first_factor: str, second_factor: str,
                        approximate: str | float | None) -> SymengineDist:
        raise NotImplementedError()

    def _update_subtraction(self, temp_var: str, sub_from: str | int, sub: str | int) -> SymengineDist:
        raise NotImplementedError()

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> SymengineDist:
        raise NotImplementedError()

    def _update_division(self, temp_var: str, numerator: str | int, denominator: str | int,
                         approximate: str | float | None) -> SymengineDist:
        raise NotImplementedError()

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> SymengineDist:
        raise NotImplementedError()

    def update_iid(self, sampling_dist: Expr, count: VarExpr, variable: Union[str, VarExpr]) -> SymengineDist:
        raise NotImplementedError()

    def marginal(self, *variables: Union[str, VarExpr], method: MarginalType = MarginalType.INCLUDE) -> SymengineDist:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def approximate_unilaterally(self, variable: str, probability_mass: str | float) -> SymengineDist:
        raise NotImplementedError()


class SymenginePGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"{p}/(1-(1-({p})) * {var})", str(var))

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> SymengineDist:
        return SymengineDist(f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/"+
                             "({var} - 1))", str(var))

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist(f"({p}) * {var} + 1-({p})", str(var))

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> SymengineDist:
        return SymengineDist(f"exp(({lam}) * ({var} - 1))", str(var))

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> SymengineDist:
        return SymengineDist( f"log(1-({p})*{var})/log(1-({p}))", str(var))

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
        return SymengineDist(expression, *map(str,variables))
