from enum import Enum
from typing import Any

from prodigy.analysis.solver.mock_solver import MockTrueSolver, MockFalseSolver, MockNoneSolver
from prodigy.analysis.solver.solver import Solver
from prodigy.analysis.solver.sympy_solver import SympySolver

Z3_REBUILT = "the Z3 solver is being rebuilt (#474); use sympy"
"""The answer to choosing ``z3``: the old solver went with the ``.pgcl`` parser (#433)."""


class SolverType(Enum):
    SYMPY = SympySolver
    TRUE = MockTrueSolver
    FALSE = MockFalseSolver
    NONE = MockNoneSolver

    @staticmethod
    def make(st: 'SolverType', *args: Any, **kwargs: Any) -> Solver:
        return st.value(*args, **kwargs)

    @staticmethod
    def named(name: str) -> 'SolverType':
        """The solver called ``name`` (any case); raises ``ValueError`` for an unknown one and for
        ``z3``, whose solver is being rebuilt."""
        if name.lower() == "z3":
            raise ValueError(Z3_REBUILT)
        if name.upper() not in SolverType.__members__:
            raise ValueError(f"Solver {name} is not known.")
        return SolverType[name.upper()]
