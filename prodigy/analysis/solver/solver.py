from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict

import sympy

from prodigy.distribution import Distribution


class Solver(ABC):

    @abstractmethod
    def solve(self, f: Distribution, g: Distribution) -> Tuple[Optional[bool], List[Dict[sympy.Expr, sympy.Expr]]]:
        ...
