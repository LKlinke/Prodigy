import logging
from typing import Tuple, Optional, List, Dict

import sympy

from prodigy.analysis.solver.solver import Solver
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup


class SMTZ3Solver(Solver):
    logger = log_setup("Z3Solver", logging.DEBUG)

    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        raise NotImplementedError()
