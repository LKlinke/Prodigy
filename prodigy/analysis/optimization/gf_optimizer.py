import sympy
from typing import Union, List, get_args

from probably.pgcl import Expr, VarExpr, OptimizationType
from probably.pgcl.parser import parse_expr
from prodigy.analysis.exceptions import ParameterError
from prodigy.distribution.generating_function import GeneratingFunction
from .optimizer import Optimizer




class GFOptimizer(Optimizer):
    @staticmethod
    def optimize(
        condition: Union[str, Expr],
        dist: GeneratingFunction,
        *parameters: Union[str, VarExpr],
        method: OptimizationType = OptimizationType.MINIMIZE
    ) -> List[Union[str, Expr]]:

        if len(parameters) > 1:
            raise NotImplementedError(
                "We currently do not support multivariate optimization.")

        # Make sure, that the condition is a parsed expression.
        if not isinstance(condition, get_args(Expr)):
            expr = parse_expr(condition)
        else:
            expr = condition

        # Compute the target function that needs to be optimized:
        target_function = sympy.S(dist.filter(expr).get_probability_mass())

        # invoke sympy to get the optimization.

        if method == OptimizationType.MAXIMIZE:
            maximum = sympy.maximum(
                target_function, sympy.Symbol(str(parameters[0])),
                sympy.sets.Interval(0, sympy.S("oo"), False, True))
            param = sympy.solve((target_function - maximum).simplify(),
                                sympy.S(str(parameters[0])))
            param = [
                real_sol for real_sol in param
                if real_sol.is_real and real_sol.is_nonnegative
            ]
        elif method == OptimizationType.MINIMIZE:
            minimum = sympy.minimum(
                target_function, sympy.Symbol(str(parameters[0])),
                sympy.sets.Interval(0, sympy.S("oo"), False, True))
            param = sympy.solve(target_function - minimum,
                                sympy.S(str(parameters[0])))
            param = [
                real_sol for real_sol in param
                if real_sol.is_real and real_sol.is_nonnegative
            ]
        else:
            raise ParameterError(f"Unknown optimization method. {method}")

        return param
