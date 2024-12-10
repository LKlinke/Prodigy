from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from probably.pgcl import Program, Var


@dataclass(frozen=True)
class ProgramInfo():
    """
    Contains a program and some information about it, such as which variables are
    considered second order variables, and which are independent.

    All attributes of the wrapped program can be accessed directly (e.g., by writing
    `prog_info.variables` instead of `prog_info.program.variables`).
    """

    program: Program
    so_vars: frozenset[Var] = frozenset()
    """Variables that are to be considered second order variables, used in the equivalence check"""
    independents_vars: frozenset[frozenset[Var]] = frozenset()
    """Pairs of variables that are independent from each other; see 
    `prodigy.analysis.static.independence.independent_vars`.
    """

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.program, __name)
