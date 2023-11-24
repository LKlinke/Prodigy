from enum import Enum

from prodigy.analysis.solver.mock_solver import MockTrueSolver, MockFalseSolver, MockNoneSolver
from prodigy.analysis.solver.smtz3_solver import SMTZ3Solver
from prodigy.analysis.solver.solver import Solver
from prodigy.analysis.solver.sympy_solver import SympySolver


class SolverType(Enum):
    SYMPY = SympySolver
    Z3 = SMTZ3Solver
    TRUE = MockTrueSolver
    FALSE = MockFalseSolver
    NONE = MockNoneSolver

    @staticmethod
    def make(st: 'SolverType', *args, **kwargs) -> Solver:
        return st.value(*args, **kwargs)
