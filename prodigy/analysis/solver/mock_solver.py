import logging
from typing import Tuple, Optional, List, Dict

import sympy

from prodigy.analysis.solver.solver import Solver
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup


class MockTrueSolver(Solver):
    logger = log_setup("MockTrueSolver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        return True, []


class MockFalseSolver(Solver):
    logger = log_setup("MockFalseSolver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        return False, []


class MockNoneSolver(Solver):
    logger = log_setup("MockNoneSolver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        return None, []
