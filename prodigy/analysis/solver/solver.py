import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional, List, Dict

import sympy

from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup


class Solver(ABC):

    @abstractmethod
    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        ...


class SympySolver(Solver):
    logger = log_setup("SympySolver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        self.logger.debug("Check %s == %s", f, g)
        s_variables = {sympy.sympify(var) for var in f.get_variables() | g.get_variables()}
        s_parameters = {sympy.sympify(param) for param in f.get_parameters() | g.get_parameters()}
        self.logger.debug("Parameters: %s \t Variables: %s", s_parameters, s_variables)
        s_equation = sympy.sympify(str(f - g))

        if s_parameters:
            solutions = sympy.solve_undetermined_coeffs(s_equation, s_parameters, *s_variables, dict=True,
                                                        particular=True)
            # validate solutions:

            # no solutions or infinitely many found.
            if not len(solutions) > 0:
                if s_equation.equals(0):
                    self.logger.debug("All parameter value combinations are valid.")
                    return True, []
                self.logger.debug("No solutions exist.")
                return False, []

            # at least one solution found
            for sol in solutions:
                for key, val in sol.items():
                    if not val.free_symbols <= s_parameters:
                        self.logger.info("SympySolver produced the invalid result %s.", sol)
                        return None, []
            self.logger.debug("solutions found: %s", solutions)
            return True, solutions

        else:
            is_equal = s_equation.equals(0)
            if is_equal is None:
                self.logger.debug("Could not determine equality of %s and %s", str(f), str(g))
                return None, []
            if is_equal is False:
                self.logger.debug("no solutions exist for %s == %s ", str(f), str(g))
                return False, []
            if is_equal:
                self.logger.debug("Validated %s == %s", str(f), str(g))
                return True, []

        raise Exception("This should not be reachable")

    def __str__(self):
        return f"SympySolver"


class SMTZ3Solver(Solver):
    logger = log_setup("Z3Solver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        raise NotImplementedError()


class SolverType(Enum):
    SYMPY = SympySolver
    Z3 = SMTZ3Solver

    @staticmethod
    def make(st: 'SolverType', *args, **kwargs) -> Solver:
        return st.value(*args, **kwargs)
