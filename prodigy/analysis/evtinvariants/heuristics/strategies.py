from dataclasses import dataclass, field
from enum import Enum
from typing import Collection

from prodigy.analysis.evtinvariants.heuristics.positivity import FPSPositivityHeuristic, FPSPositivityHeuristicFactory
from prodigy.analysis.evtinvariants.heuristics.templates import TemplateHeuristic, TemplateHeuristicsFactory
from prodigy.distribution import CommonDistributionsFactory


@dataclass
class SynthesisStrategy:
    template_heuristics: TemplateHeuristic = field(init=False)
    positivity_heuristics: FPSPositivityHeuristic = field(init=False)

    def __str__(self) -> str:
        return f"Strategy {self.__class__.__name__} ({self.template_heuristics=}, {self.positivity_heuristics=})"


@dataclass
class DefaultStrategy(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = 10

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristicsFactory.rational_functions_total_deg(self.max_degree,
                                                                                          self.variables,
                                                                                          self.dist_factory)
        self.positivity_heuristics = FPSPositivityHeuristicFactory.rational_function_denominator_sign(self.dist_factory)


class SynthesisStrategies(Enum):
    DEFAULT = DefaultStrategy

    @classmethod
    def make(cls, st: 'SynthesisStrategies', *args, **kwargs) -> SynthesisStrategy:
        return st.value(*args, **kwargs)
