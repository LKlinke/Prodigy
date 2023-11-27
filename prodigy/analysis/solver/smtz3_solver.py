import logging
import time
from typing import Tuple, Optional, List, Dict

import pysmt.solvers.solver
import sympy
from probably.pgcl.parser import parse_expr
from probably.pysmt.context import TranslationContext
from probably.pysmt.expr import expr_to_pysmt
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.logics import QF_NRA
from pysmt.shortcuts import Symbol, EqualsOrIff, Not, And
from pysmt.typing import REAL

from prodigy.analysis.solver.solver import Solver
from prodigy.distribution import Distribution, CommonDistributionsFactory
from prodigy.util.logger import log_setup


class SMTZ3Solver(Solver):
    logger = log_setup("Z3Solver", logging.DEBUG)

    def __init__(self, dist_factory: CommonDistributionsFactory):
        self._factory = dist_factory

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        self.logger.debug("Check %s == %s", f, g)
        s_f, s_g = sympy.S(str(f)), sympy.S(str(g))
        if not (s_f.is_rational_function() and s_g.is_rational_function()):
            raise NotImplementedError("Non-rational functions not supported yet.")

        # write the functions as numerators and denominators
        f_num, f_denom = s_f.as_numer_denom()
        g_num, g_denom = s_g.as_numer_denom()

        # cross-multiply and get the coefficients to compare.
        lhs = f_num * g_denom
        rhs = g_num * f_denom

        goal_function = self._factory.from_expr(str(lhs - rhs).replace("**", "^"),
                                                f.get_variables() | g.get_variables())
        formula = parse_expr(" & ".join([f"( 0 <= {prob.replace('**', '^')})" for prob, _ in goal_function]))
        self.logger.debug("Generated formula %s", formula)

        # create a (variable, type) context and parse these expressions to pysmt.
        context = TranslationContext({var: Symbol(var, REAL) for var in f.get_symbols() | g.get_symbols()})
        smt_formula = expr_to_pysmt(context, formula, is_expectation=True)
        self.logger.debug("Converted to SMT %s", smt_formula)

        start = time.perf_counter()
        sol = {}
        with pysmt.shortcuts.Solver(logic=QF_NRA) as solver:
            solver.add_assertion(smt_formula)
            self.logger.debug("Solve call to Z3")
            is_sat = True
            while is_sat:
                try:
                    is_sat = solver.solve()
                    if is_sat:
                        partial_model = []
                        sol = {}
                        for k in context.variables.values():
                            sol[sympy.S(str(k))] = sympy.S(str(solver.get_value(k)))
                            partial_model.append(EqualsOrIff(k, solver.get_value(k)))
                        if any(sol.values()):
                            break
                        solver.add_assertion(Not(And(partial_model)))
                        self.logger.debug("Generated model %s", sol)
                except SolverReturnedUnknownResultError:
                    self.logger.debug("Z3 was unable to produce a result.")
                    is_sat = None
            self.logger.debug(f"Solving time: {time.perf_counter() - start:04f}")

        if is_sat is None:
            self.logger.debug("Could not determine equality of %s and %s", str(f), str(g))
            return None, []
        elif is_sat is False:
            self.logger.debug("no solutions exist for %s == %s ", str(f), str(g))
            return False, []
        elif is_sat is True:
            return True, [sol]
