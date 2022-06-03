from typing import Union

import pygin
import sympy as sp
from probably.pgcl import VarExpr, Expr

from prodigy.distribution import Distribution, DistributionParam
from prodigy.distribution.distribution import CommonDistributionsFactory
from prodigy.distribution.fast_generating_function import FPS
from prodigy.distribution.generating_function import GeneratingFunction
# We need to be able to get this import from probably
from prodigy.pgcl.pgcl_checks import has_variable


class SympyPGF(CommonDistributionsFactory):
    """Implements PGFs of standard distributions."""

    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 < sp.S(p) < 1:
            raise ValueError(
                f"parameter of geom distr must be >0 and <=1, was {p}")
        if isinstance(p, VarExpr) and has_variable(p):
            raise ValueError(
                "Parameter for geometric distribution cannot depend on a program variable."
            )
        return GeneratingFunction(f"({p}) / (1 - (1-({p})) * {var})",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> Distribution:
        if isinstance(lower, str) and isinstance(
                upper, str) and not 0 <= sp.S(lower) <= sp.S(upper):
            raise ValueError(
                "Distribution parameters must satisfy 0 <= a < b < oo")
        if (isinstance(lower, VarExpr)
              and has_variable(lower)) or (isinstance(upper, VarExpr)
                                           and has_variable(upper)):
            raise ValueError(
                "Parameter for uniform distribution cannot depend on a program variable."
            )
        return GeneratingFunction(
            f"1/(({upper}) - ({lower}) + 1) * {var}**({lower}) * ({var}**(({upper}) - ({lower}) + 1) - 1) / ({var} - 1)",
            var,
            closed=True,
            finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sp.S(p) <= 1:
            raise ValueError(
                f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}"
            )
        if isinstance(p, VarExpr) and has_variable(p):
            raise ValueError(
                "Parameter for geometric distribution cannot depend on a program variable."
            )
        return GeneratingFunction(f"1 - ({p}) + ({p}) * {var}",
                                  var,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> Distribution:
        if isinstance(lam, str) and sp.S(lam) < 0:
            raise ValueError(
                f"Parameter of Poisson Distribution must be in [0, oo), but was {lam}"
            )
        if isinstance(lam, VarExpr) and has_variable(lam):
            raise ValueError(
                "Parameter for geometric distribution cannot depend on a program variable."
            )
        return GeneratingFunction(f"exp(({lam}) * ({var} - 1))",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sp.S(p) <= 1:
            raise ValueError(
                f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}"
            )
        if isinstance(p, VarExpr) and has_variable(p):
            raise ValueError(
                "Parameter for geometric distribution cannot depend on a program variable."
            )
        return GeneratingFunction(f"log(1-({p})*{var})/log(1-({p}))",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam, p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sp.S(p) <= 1:
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,1], but was {p}"
            )
        if isinstance(n, str) and not 0 <= sp.S(n):
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,oo), but was {n}"
            )
        if (isinstance(n, VarExpr)
            and has_variable(n)) or (isinstance(p, VarExpr)
                                     and has_variable(p)):
            raise ValueError(
                "Parameter for geometric distribution cannot depend on a program variable."
            )
        return GeneratingFunction(f"(1-({p})+({p})*{var})**({n})",
                                  var,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def zero(*variables: Union[str, sp.Symbol]):
        if variables:
            return GeneratingFunction("0",
                                      *variables,
                                      preciseness=1,
                                      closed=True,
                                      finite=True)
        return GeneratingFunction("0", preciseness=1, closed=True, finite=True)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""
        return SympyPGF.zero(*map(str, variables))

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> 'Distribution':
        return GeneratingFunction("1",
                                  *variables,
                                  preciseness=1,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> 'Distribution':
        return GeneratingFunction(str(expression), *variables, **kwargs)


class ProdigyPGF(CommonDistributionsFactory):
    @staticmethod
    def geometric(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        return FPS(str(pygin.geometric(var, str(p))))

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> Distribution:
        function = f"1/({upper} - {lower} + 1) * ({var}^{lower}) * (({var}^({upper} - {lower} + 1) - 1)/({var} - 1))"
        return FPS(function)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        function = f"({p}) * {var} + 1-({p})"
        return FPS(function)

    @staticmethod
    def poisson(var: Union[str, VarExpr], lam: DistributionParam) -> Distribution:
        function = f"exp(({lam}) * ({var} - 1))"
        return FPS(function)

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        function = f"log(1-({p})*{var})/log(1-({p}))"
        return FPS(function)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam, p: DistributionParam) -> Distribution:
        function = f"(({p})*{var} + (1-({p})))^({n})"
        return FPS(function)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        raise NotImplementedError(__name__)

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> 'Distribution':
        return FPS("1")

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> 'Distribution':
        return FPS(expression, **kwargs)
