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
        """
        WARNING: This method currently just tries to find models which are positive FPS by design. We hence do not have
        completeness of the algorithm for a given template.
        """
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

        numerator_symbols = list(f_num.free_symbols - {sympy.S(var) for var in f.get_variables()})
        denominator_symbols = sorted(list(f_denom.free_symbols - {sympy.S(var) for var in f.get_variables()}),
                                     key=str)

        # create a (variable, type) context for parsing to pySMT
        context = TranslationContext({var: Symbol(var, REAL) for var in f.get_parameters() | g.get_parameters()})

        # encode positivity (sufficient condition)
        positivity_of_num = " & ".join(f"0 <= {sym}" for sym in numerator_symbols)
        positivity_of_denom = " & ".join(
            f"0 < {sym}" if i == 0 else f"{sym} <= 0" for i, sym in enumerate(denominator_symbols))
        positivity_of_solution = " & ".join([positivity_of_num, positivity_of_denom])
        positivity_smt = expr_to_pysmt(context, parse_expr(positivity_of_solution), is_expectation=True)

        # encode invariance condition
        goal_function = self._factory.from_expr(str(lhs - rhs).replace("**", "^"),
                                                *f.get_variables() | g.get_variables())

        coefficient_at_state = {}
        for prob, state in goal_function:
            if prob.startswith("-"):
                prob = "(-1)*" + prob[1:]
            if coefficient_at_state.get(state):
                coefficient_at_state[state].append(f"{prob}")
            else:
                coefficient_at_state[state] = [f"{prob}"]
        coefficients = [" + ".join(prob).replace('**', '^') for prob in coefficient_at_state.values()]
        coefficients = list(
            map(lambda x: expr_to_pysmt(context, parse_expr(f"0 = {x}"), is_expectation=True), coefficients))
        self.logger.debug("Generated formula %s", coefficients)

        smt_formula = And(*coefficients, positivity_smt)
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

                        # Generate solution mapping
                        for k in context.variables.values():
                            sol[sympy.S(str(k))] = sympy.S(str(solver.get_value(k)), rational=True)
                            partial_model.append(EqualsOrIff(k, solver.get_value(k)))
                        self.logger.debug("Solution found: %s", sol)

                        # exclude division by zero solutioms
                        if f_denom.subs(sol) == 0:
                            self.logger.debug("denominator is zero, exclude this solution.")
                            denom_model = []
                            for k in context.variables.values():
                                if sympy.S(str(k)) in denominator_symbols:
                                    denom_model.append(EqualsOrIff(k, solver.get_value(k)))
                            solver.add_assertion(Not(And(denom_model)))

                        # exclude numerator is 0 solutions.
                        elif f_num.subs(sol) == 0:
                            self.logger.debug("numerator is zero, exclude this solution.")
                            num_model = []
                            for k in context.variables.values():
                                if sympy.S(str(k)) in numerator_symbols:
                                    num_model.append(EqualsOrIff(k, solver.get_value(k)))
                            solver.add_assertion(Not(And(num_model)))

                        # might be a good solution
                        else:
                            self.logger.debug("Generated model %s", sol)
                            break
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
