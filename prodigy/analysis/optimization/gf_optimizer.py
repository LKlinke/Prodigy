from typing import List, Union

import sympy

from prodigy.distribution.equality import engine_text, sympify_engine_string
from prodigy.pgcl import Expr, OptimizationType, VarExpr
from prodigy.pgcl.ppy_parser import parse_ppy_expr

from prodigy.distribution.distribution import Distribution

from .optimizer import Optimizer


class GFOptimizer(Optimizer):
    @staticmethod
    def optimize(
            condition: Union[str, Expr],
            dist: Distribution,
            *parameters: Union[str, VarExpr],
            method: OptimizationType = OptimizationType.MINIMIZE
    ) -> List[Union[str, Expr]]:

        if len(parameters) > 1:
            raise NotImplementedError(
                "We currently do not support multivariate optimization.")

        # Make sure, that the condition is a parsed expression (text is in the .ppy syntax).
        if isinstance(condition, str):
            expr = parse_ppy_expr(condition)
        else:
            expr = condition

        # Compute the target function that needs to be optimized:
        target_function = sympify_engine_string(engine_text(dist.filter(expr).get_probability_mass()),
                                           dist.get_variables() | dist.get_parameters())

        # invoke sympy to get the optimization.

        if method == OptimizationType.MAXIMIZE:
            maximum = sympy.maximum(
                target_function, sympy.Symbol(str(parameters[0])),
                sympy.sets.Interval(0, sympy.S("oo"), False, True))
            param = sympy.solve(target_function - maximum,
                                sympy.Symbol(str(parameters[0])))
            param = [
                real_sol for real_sol in param
                if real_sol.is_real and real_sol.is_nonnegative
            ]
        elif method == OptimizationType.MINIMIZE:
            minimum = sympy.minimum(
                target_function, sympy.Symbol(str(parameters[0])),
                sympy.sets.Interval(0, sympy.S("oo"), False, True))
            param = sympy.solve(target_function - minimum,
                                sympy.Symbol(str(parameters[0])))
            param = [
                real_sol for real_sol in param
                if real_sol.is_real and real_sol.is_nonnegative
            ]
        else:
            raise TypeError(f"Unknown optimization method. {method}")

        return param
