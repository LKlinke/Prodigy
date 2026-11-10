from __future__ import annotations

import re

from typing import (Dict, FrozenSet, Generator, Iterator, List, Set, Tuple, Type,
                    Union)

import prodigy_ginac  # type: ignore
from sympy import Expr as Expr_, Symbol, parse_expr
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

from prodigy.distribution.exact_division import divide_exponents
from prodigy.distribution.negative_values import can_be_negative, operand
from prodigy.distribution.roots_of_unity import require_single_valued
from prodigy.distribution.distribution import (CommonDistributionsFactory,
                                               Distribution, DistributionParam,
                                               MarginalType, State)
from prodigy.pgcl import Binop, BinopExpr, Expr, FunctionCallExpr, NatLitExpr, VarExpr

from prodigy.util.order import default_monomial_iterator


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
        for var in prodigy_ginac.find_symbols(expression):
            if var not in self._variables:
                if len(variables) > 0:
                    self._parameters.add(var)
                else:
                    self._variables.add(var)
        self._dist = prodigy_ginac.Dist(expression, list(self._parameters))
        self._finite = finite if finite is not None else self._dist.is_polynomial(
            self._variables) == prodigy_ginac.troolean.true

    @classmethod
    def from_dist(cls,
                  dist: prodigy_ginac.Dist,
                  variables: Set[str],
                  parameters: Set[str],
                  finite: bool | None = None):
        result = cls("0")
        result._dist = dist
        result._variables = variables
        result._parameters = parameters
        result._finite = finite if finite is not None else dist.is_polynomial(variables) == prodigy_ginac.troolean.true
        return result

    @staticmethod
    def factory() -> Type[ProdigyPGF]:
        return ProdigyPGF

    def __add__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(
                self._dist + prodigy_ginac.Dist(
                    str(other),
                    list(
                        set(prodigy_ginac.find_symbols(str(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist + other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Addition of {self._dist} (type {type(self._dist)}) and {other} (type {type(other)} not supported.")

    def __sub__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(
                self._dist - prodigy_ginac.Dist(
                    str(other),
                    list(
                        set(prodigy_ginac.find_symbols(str(other))) -
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
                self._dist * prodigy_ginac.Dist(
                    str(other),
                    list(
                        set(prodigy_ginac.find_symbols(str(other))) -
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
            other = FPS(other, *self._variables)
        if isinstance(other, FPS):
            parameters = self._parameters | other._parameters
            return FPS.from_dist(self._dist * prodigy_ginac.Dist(f"1/({str(other)})", sorted(parameters)),
                                 self._variables | other._variables,
                                 parameters)
        raise NotImplementedError(
            f"Division of {type(self._dist)} and {type(other)} not supported.")

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            other = FPS(other)
        if isinstance(other, FPS):
            #print(self._variables, other._variables)
            #print(self._parameters, other._parameters)
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
                for vals in default_monomial_iterator(len(variables)):
                    s = f'{variables[0]}={vals[0]}'
                    for i in range(1, len(variables)):
                        s += f' & {variables[i]}={vals[i]}'
                    mass = str(self.get_probability_of(s))
                    if mass != '0':
                        yield mass, State(dict(zip(variables, vals)))
        else:
            terms: List[Tuple[str, Dict[str, int]]] = self._dist.get_terms(self._variables)
            for prob, vals in terms:        # type: ignore
                yield prob, State(vals)     # type: ignore

    def copy(self, deep: bool = True) -> Distribution:
        return FPS.from_dist(self._dist, self._variables, self._parameters,
                             self._finite)

    def get_probability_mass(self) -> str:
        return self._dist.mass(self._variables & set(self._dist.get_symbols()))

    def _expected_value_of(self, expression: Union[Expr, str]) -> str:
        # Symbols that are neither variables nor parameters have no value: reject them, as the SymPy engine
        # does (the bindings reject them as well since prodigy-ppl-ginac 0.0.5).
        symbols = self._find_symbols(str(expression))
        if not symbols & self._variables:
            return str(expression)
        if not symbols <= self._variables | self._parameters:
            raise ValueError(f"Cannot compute expected value of {expression} because it contains unknown symbols")
        # A variable that does not occur in the generating function is 0 in every state, but the bindings
        # do not know it as a variable; substitute its value.
        absent = (symbols & self._variables) - set(self._dist.get_symbols())
        text = str(expression)
        if absent:
            text = re.sub(r"\b(" + "|".join(sorted(map(re.escape, absent))) + r")\b", "0", text)
        return self._dist.E(text)

    def normalize(self) -> FPS:
        return FPS.from_dist(self._dist.normalize(), self._variables, self._parameters)

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        return self._parameters

    def _find_symbols(self, expr: str) -> Set[str]:
        return set(prodigy_ginac.find_symbols(expr))

    def get_symbols(self) -> Set[str]:
        return set(self._dist.get_symbols())

    @staticmethod
    def evaluate(expression: str, state: State) -> int:
        return prodigy_ginac.evaluate(expression, state.valuations)

    def _exhaustive_search(self, condition: Expr) -> Distribution:
        res = prodigy_ginac.Dist('0')
        for prob, state in self:
            if self.evaluate_condition(condition, state):
                res += prodigy_ginac.Dist(f"{prob} * {state.to_monomial()}", sorted(self._parameters))
        return FPS.from_dist(res,
                             self._variables,
                             self._parameters,
                             finite=True)

    def _filter_constant_condition(self, condition: BinopExpr) -> FPS:
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
        if isinstance(condition.lhs, VarExpr) and condition.operator in (
                Binop.EQ, Binop.LT, Binop.LEQ, Binop.GT, Binop.GEQ):
            variable, bound = str(condition.lhs), str(condition.rhs)
            filters = {
                Binop.EQ: self._dist.filterEq,
                Binop.LT: self._dist.filterLess,
                Binop.LEQ: self._dist.filterLeq,
                Binop.GT: self._dist.filterGreater,
                Binop.GEQ: self._dist.filterGeq,
            }
            result = filters[condition.operator](variable, bound)
            return FPS.from_dist(result, self._variables, self._parameters)

        raise ValueError("Parameter is not a constant condition")

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[FPS]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        if not self._finite:
            require_single_valued(self._sympy_function(), variable, f"{variable} % {modulus}")
        return [
            FPS.from_dist(dist, self._variables, self._parameters)
            for dist in self._dist.arithmetic_progression(variable, modulus)
        ]

    def hadamard_product(self, other: Distribution) -> Distribution:
        raise NotImplementedError("Hadamard product currently not implemented in GiNaC")

    def is_zero_dist(self) -> bool:
        res = self._dist.is_zero()
        if res == prodigy_ginac.troolean.false:
            return False
        elif res == prodigy_ginac.troolean.true:
            return True
        else:
            raise ValueError('Cannot determine whether this FPS is zero')

    def is_finite(self) -> bool:
        return self._finite

    def old_update(self, expression: BinopExpr) -> FPS:
        return FPS.from_dist(
            self._dist.update(str(expression.lhs), str(expression.rhs)),
            self._variables, self._parameters, self._finite)

    def get_fresh_variable(
            self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        res: str = self._dist.get_fresh_variable()
        while res in exclude:
            res = self._dist.get_fresh_variable()
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
        return FPS.from_dist(res, self._variables, self._parameters)

    def _subtraction_never_negative(self, minuend: str, subtrahend: str) -> bool:
        variables = self.get_variables()
        return can_be_negative(self._sympy_function(), {Symbol(var) for var in variables},
                               operand(minuend, variables), operand(subtrahend, variables)) is False

    def _update_modulo(self, temp_var: str, left: str | int, right: str | int,
                       approximate: str | float | None) -> FPS:
        # With an infinite marginal of `left`, GiNaC extracts the residue classes of `left` with roots
        # of unity (a finite marginal makes the function polynomial in `left`).
        if not self._finite and str(left) in self._variables:
            require_single_valued(self._sympy_function(), str(left), f"{left} % {right}")
        return FPS.from_dist(
            self._dist.update_modulo(temp_var, str(left), str(right),
                                     self._variables, self._finite,
                                     approximate), self._variables,
            self._parameters)

    def _update_division(self, temp_var: str, numerator: str | int,
                         denominator: str | int,
                         approximate: str | float | None) -> FPS:
        if str(numerator) in self._variables and str(denominator).isdigit() \
                and not self.marginal(str(numerator)).is_finite():
            divided = self._divide_exponents(temp_var, str(numerator), int(str(denominator)))
            if divided is not None:
                return divided
        return FPS.from_dist(
            self._dist.update_division(temp_var, str(numerator),
                                       str(denominator), approximate),
            self._variables, self._parameters)

    def _sympy_function(self) -> Expr_:
        """This generating function as a SymPy expression (all symbols without assumptions)."""
        text = str(self._dist)
        symbols = {name: Symbol(name) for name in prodigy_ginac.find_symbols(text)}
        return parse_expr(text, local_dict=symbols,
                          transformations=standard_transformations + (convert_xor,))

    def _divide_exponents(self, temp_var: str, numerator: str, divisor: int) -> FPS | None:
        """
        The update `temp_var := numerator / divisor` for an infinite marginal of `numerator`, or None
        if it is not proven that `numerator` is always a multiple of `divisor` (see
        :mod:`prodigy.distribution.exact_division`).
        """
        divided = divide_exponents(self._sympy_function(), numerator, temp_var, divisor)
        if divided is None:
            return None
        return FPS(str(divided).replace('**', '^'), *self._variables, finite=False)

    def _update_power(self, temp_var: str, base: str | int, exp: str | int,
                      approximate: str | float | None) -> Distribution:
        return FPS.from_dist(
            self._dist.update_power(temp_var, base, exp, approximate),
            self._variables, self._parameters)

    def update_iid(self, sampling_dist: Expr, count: Union[VarExpr, NatLitExpr],
                   variable: Union[str, VarExpr]) -> FPS:
        if not isinstance(sampling_dist, FunctionCallExpr):
            result = FPS.from_dist(
                self._dist.updateIid(str(variable),
                                     prodigy_ginac.Dist(str(sampling_dist), list(self._parameters)),
                                     str(count)), self._variables,
                self._parameters)
            return result

        if sampling_dist.function in {"unif", "unif_d"}:
            [start, end] = sampling_dist.params[0]
            # Note: add parameters in constructor call of prodigy_ginac.Dist
            result = self._dist.updateIid(
                str(variable),
                prodigy_ginac.Dist(
                    f"1/(({end}) - ({start}) + 1) * test^({start}) "
                    f"* (test^(({end}) - ({start}) + 1) - 1) / (test - 1)", list(self._parameters)),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == "binomial":
            [n, p] = sampling_dist.params[0]
            # Note: add parameters in constructor call of prodigy_ginac.Dist
            result = self._dist.updateIid(
                str(variable),
                prodigy_ginac.Dist(f'(1 - ({p}) + ({p}) * {variable})^({n})', list(self._parameters)),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == "negative_binomial":
            [r, p] = sampling_dist.params[0]
            result = self._dist.updateIid(
                str(variable),
                prodigy_ginac.Dist(f'(({p}) / (1 - (1 - ({p})) * {variable}))^({r})', list(self._parameters)),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        [param] = sampling_dist.params[0]
        if sampling_dist.function == "geometric":
            result = self._dist.updateIid(
                str(variable), prodigy_ginac.geometric(str(variable), str(param)),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == "bernoulli":
            symbolic_param = prodigy_ginac.find_symbols(str(param))
            result = self._dist.updateIid(
                str(variable),
                prodigy_ginac.Dist(f"{param} * {variable} + (1-{param})", symbolic_param), str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == "poisson":
            result = self._dist.updateIid(
                str(variable), prodigy_ginac.Dist(f"exp({param} * ({variable} - 1))", list(self._parameters)),
                str(count))
            return FPS.from_dist(result, self._variables, self._parameters)

        if sampling_dist.function == 'logdist':
            result = self._dist.updateIid(
                str(variable),
                prodigy_ginac.Dist(
                    f'log(1 - ({param}) * {variable}) / log(1 - ({param}))', list(self._parameters)),
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

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"Static class {str(cls)} cannot be instantiated.")

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        return FPS.from_dist(prodigy_ginac.geometric(var, str(p)), {str(var)},
                             {*prodigy_ginac.find_symbols(str(p))})

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
    def negative_binomial(var: Union[str, VarExpr], r: DistributionParam,
                          p: DistributionParam) -> FPS:
        function = f"(({p}) / (1 - (1-({p})) * {var}))^({r})"
        return FPS(function, str(var))

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("1", *variables, finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> FPS:
        return FPS(str(expression), *variables)

    @staticmethod
    def zero(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("0", *variables, finite=True)
