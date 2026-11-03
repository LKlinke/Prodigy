from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prodigy.analysis.exceptions import UnsupportedFeatureError
from prodigy.pgcl import Program, Var
from prodigy.pgcl.pgcl_checks import (NESTED_INFERENCE_UNSUPPORTED,
                                      find_nested_inference)


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

    def __post_init__(self) -> None:
        # Every analysis runs on a ProgramInfo, so this rejects such programs before any
        # instruction is executed (instead of failing deep inside Distribution.update).
        found = find_nested_inference(self.program)
        if found is not None:
            raise UnsupportedFeatureError(f"{NESTED_INFERENCE_UNSUPPORTED} (found `{found}`).")

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.program, __name)
