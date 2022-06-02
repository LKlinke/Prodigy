"""
-----------------------
Forward Analysis Config
-----------------------
"""
from enum import Enum, auto
from typing import Type

import attr

from prodigy.distribution.distribution_factory import CommonDistributionsFactory
from .exceptions import ConfigurationError
from prodigy.distribution.pgfs import ProdigyPGF
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.analysis.optimization.gf_optimizer import GFOptimizer
from prodigy.analysis.optimization import Optimizer
from prodigy.distribution.pgfs import SympyPGF


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""
    class Engine(Enum):
        """
        This enumeration specifies the type of backend used for distribution encodings and mathematical operations.
        """
        SYMPY = auto()
        GINAC = auto()

    """Enables the printing of results after each instruction."""
    show_intermediate_steps: bool = attr.ib(default=False)
    """Displays the probabilities in rational form. This does not change the preciseness of computation."""
    show_rational_probabilities: bool = attr.ib(default=False)
    """Enables simplification heuristics for expressions."""
    use_simplification: bool = attr.ib(default=False)
    """Toggle to print LaTeX-Code instead of ASCII expressions."""
    use_latex: bool = attr.ib(default=False)
    """Selects the distribution backend."""
    engine: Engine = attr.ib(default=Engine.SYMPY)

    @property
    def optimizer(self) -> Type[Optimizer]:
        if self.engine == ForwardAnalysisConfig.Engine.SYMPY:
            return GFOptimizer
        else:
            raise ConfigurationError(
                "The configured engine does not implement an optimizer.")

    @property
    def factory(self) -> Type[CommonDistributionsFactory]:
        if self.engine == self.Engine.SYMPY:
            return SympyPGF
        elif self.engine == self.Engine.GINAC:
            return ProdigyPGF
        else:
            return CommonDistributionsFactory

    def __attrs_post_init__(self):
        GeneratingFunction.use_latex_output = self.use_latex
        GeneratingFunction.rational_preciseness = self.show_rational_probabilities
        GeneratingFunction.use_simplification = self.use_simplification
        GeneratingFunction.intermediate_results = self.show_intermediate_steps
