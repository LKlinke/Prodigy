from __future__ import annotations

from typing import Generator, Iterator, Set, Tuple, Union, get_args

import probably.util.ref
import pygin  # type: ignore
from probably.pgcl import (BernoulliExpr, Binop, BinopExpr, DistrExpr,
                           DUniformExpr, Expr, GeometricExpr, IidSampleExpr,
                           PoissonExpr, Unop, UnopExpr, VarExpr, Walk,
                           walk_expr)
from probably.pgcl.parser import parse_expr

from prodigy.distribution.distribution import (CommonDistributionsFactory,
                                               Distribution, DistributionParam,
                                               MarginalType, State)


class FPS(Distribution):
    """
    This class models a probability distribution in terms of a formal power series.
    These formal powerseries are itself provided by `prodigy` a python binding to GiNaC,
    something similar to a computer algebra system implemented in C++.
    """
    def __init__(self, expression: str, *variables: str | VarExpr):
        self._variables = set(str(var) for var in variables)
        self._parameters = set()
        parsed_expression = parse_expr(expression)

        for expr in walk_expr(Walk.DOWN,
                              probably.util.ref.Mut.alloc(parsed_expression)):
            if isinstance(expr, VarExpr):
                self._parameters |= expr.var if expr.var not in self._variables else {}
            self._dist = pygin.Dist(expression, list(self._parameters))

    @classmethod
    def from_dist(cls, dist: pygin.Dist) -> FPS:
        result = FPS("0")
        result.dist = dist
        return result

    def __add__(self, other) -> FPS:
        if isinstance(other, str):
            return FPS.from_dist(self._dist + other)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist + other._dist)
        else:
            raise NotImplementedError(
                f"Addition of {self._dist} and {other} not supported.")

    def __sub__(self, other) -> FPS:
        if isinstance(other, str):
            return FPS.from_dist(self._dist - other)
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist - other._dist)
        else:
            raise NotImplementedError(
                f"Subtraction of {self._dist} and {other} not supported.")

    def __mul__(self, other) -> FPS:
        if isinstance(other, (str, int)):
            return FPS.from_dist(self._dist * str(other))
        elif isinstance(other, FPS):
            return FPS.from_dist(self._dist * other._dist)
        else:
            raise NotImplementedError(
                f"Multiplication of {type(self._dist)} and {type(other)} not supported."
            )

    def __truediv__(self, other) -> FPS:
        if isinstance(other, (str, FPS)):
            return FPS.from_dist(self._dist * f"1/{str(other)}")
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
        raise NotImplementedError(__name__)

    def copy(self, deep: bool = True) -> Distribution:
        return FPS.from_dist(self._dist)

    def get_probability_of(self, condition: Union[Expr, str]):
        raise NotImplementedError(__name__)

    def get_probability_mass(self) -> Union[Expr, str]:
        return self._dist.mass()

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        return self._dist.E(str(expression))

    def normalize(self) -> FPS:
        return FPS.from_dist(self._dist.normalize())

    def get_variables(self) -> Set[str]:
        return self._variables

    def get_parameters(self) -> Set[str]:
        return self._parameters

    def filter(self, condition: Union[Expr, str]) -> FPS:
        if isinstance(condition, BinopExpr):
            if condition.operator == Binop.AND:
                return self.filter(condition.lhs).filter(condition.rhs)
            if condition.operator == Binop.OR:
                filtered_left = self.filter(condition.lhs)
                return filtered_left + self.filter(
                    condition.rhs) - filtered_left.filter(condition.lhs)

            # Normalize the conditional to variables on the lhs from the relation symbol.
            if isinstance(condition.rhs, VarExpr):
                switch_comparison = {
                    Binop.EQ: Binop.EQ,
                    Binop.LEQ: Binop.GEQ,
                    Binop.LT: Binop.GT,
                    Binop.GEQ: Binop.LEQ,
                    Binop.GT: Binop.LT
                }
                return self.filter(
                    BinopExpr(operator=switch_comparison[condition.operator],
                              lhs=condition.rhs,
                              rhs=condition.lhs))

            # is normalized conditional
            if isinstance(condition.lhs, VarExpr):
                if condition.operator == Binop.EQ:
                    return FPS.from_dist(
                        self._dist.filterEq(str(condition.lhs),
                                            str(condition.rhs)))
                elif condition.operator == Binop.LT:
                    return FPS.from_dist(
                        self._dist.filterLess(str(condition.lhs),
                                              str(condition.rhs)))
                elif condition.operator == Binop.LEQ:
                    return FPS.from_dist(
                        self._dist.filterLeq(str(condition.lhs),
                                             str(condition.rhs)))
                elif condition.operator == Binop.GT:
                    return FPS.from_dist(
                        self._dist.filterGreater(str(condition.lhs),
                                                 str(condition.rhs)))
                elif condition.operator == Binop.GEQ:
                    return FPS.from_dist(
                        self._dist.filterGeq(str(condition.lhs),
                                             str(condition.rhs)))
        if isinstance(condition, UnopExpr):
            # unary relation
            if condition.operator == Unop.NEG:
                return self - self.filter(condition.expr)
            raise SyntaxError(
                f"We do not support filtering for {type(Unop.IVERSON)} expressions."
            )

        raise SyntaxError(
            f"Filtering Condition has unknown format {condition}.")

    def is_zero_dist(self) -> bool:
        return self._dist.isZero()

    def is_finite(self) -> bool:
        raise NotImplementedError(__name__)

    def update(self, expression: Expr) -> FPS:
        return FPS.from_dist(
            self._dist.update(str(expression.lhs), str(expression.rhs)))

    def update_iid(self, sampling_exp: IidSampleExpr,
                   variable: Union[str, VarExpr]) -> FPS:

        sample_dist = sampling_exp.sampling_dist
        if isinstance(sample_dist, GeometricExpr):
            result = self._dist.updateIid(
                str(variable), pygin.geometric("test", str(sample_dist.param)),
                str(sampling_exp.variable))
            return FPS.from_dist(result)
        if isinstance(sample_dist, BernoulliExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(
                    f"{sample_dist.param} * test + (1-{sample_dist.param})"),
                str(sampling_exp.variable))
            return FPS.from_dist(result)

        elif isinstance(sample_dist, PoissonExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(f"exp({sample_dist.param} * (test - 1))"),
                str(sampling_exp.variable))
            return FPS.from_dist(result)

        elif isinstance(sample_dist, DUniformExpr):
            result = self._dist.updateIid(
                str(variable),
                pygin.Dist(
                    f"1/(({sample_dist.end}) - ({sample_dist.start}) + 1) * test^({sample_dist.start}) "
                    f"* (test^(({sample_dist.end}) - ({sample_dist.start}) + 1) - 1) / (test - 1)"
                ), str(sampling_exp.variable))
            return FPS.from_dist(result)

        elif isinstance(sample_dist, get_args(Expr)) and not isinstance(
                sample_dist, get_args(DistrExpr)):
            result = FPS.from_dist(
                self._dist.updateIid(str(variable),
                                     pygin.Dist(str(sample_dist)),
                                     str(sampling_exp.variable)))
            return result
        else:
            raise NotImplementedError(
                "Iid Distribution type currently not supported.")

    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> Distribution:
        # TODO: Make this work with an arbitrary number of variables to marginalize. <-- Is this still a point?
        if len(variables) > 1:
            raise NotImplementedError(__name__)
        if method == MarginalType.EXCLUDE:
            result = self._dist
            for var in variables:
                result = result.update(str(var), "0")
            return FPS.from_dist(result)
        if method == MarginalType.INCLUDE:
            for var in variables:
                return FPS.from_dist(self._dist.marginal(str(var)))
        raise AttributeError(
            f"`method`-argument can only be of type {type(MarginalType)} -- was {type(method)}"
        )

    def set_variables(self, *variables: str) -> FPS:
        return FPS(str(self._dist), *variables)

    def set_parameters(self, *parameters: str) -> FPS:
        result = FPS(str(self._dist), *self._variables)
        result._parameters = {str(param) for param in parameters}
        return result

    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        raise NotImplementedError(__name__)


class ProdigyPGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        return FPS(str(pygin.geometric(var, str(p))))

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> FPS:
        function = f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/({var} - 1))"
        return FPS(function)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        function = f"({p}) * {var} + 1-({p})"
        return FPS(function, var)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> FPS:
        function = f"exp(({lam}) * ({var} - 1))"
        return FPS(function, var)

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> FPS:
        function = f"log(1-({p})*{var})/log(1-({p}))"
        return FPS(function, var)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> FPS:
        function = f"(({p})*{var} + (1-({p})))^({n})"
        return FPS(function, var)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> FPS:
        raise NotImplementedError(__name__)

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> FPS:
        return FPS("1", *variables)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> FPS:
        return FPS(expression, *variables)
