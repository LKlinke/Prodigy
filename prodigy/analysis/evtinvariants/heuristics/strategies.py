from dataclasses import dataclass, field
from enum import Enum
from typing import Collection

from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.distribution import CommonDistributionsFactory


@dataclass
class SynthesisStrategy:
    template_heuristics: TemplateHeuristic = field(init=False)
    positivity_heuristics: PositivityHeuristic = field(init=False)

    def __str__(self) -> str:
        return f"Strategy {self.__class__.__name__} ({self.template_heuristics=}, {self.positivity_heuristics=})"


@dataclass
class DefaultStrategy(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = 10

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )
        self.positivity_heuristics = PositivityHeuristics.create(
            PositivityHeuristics.MIXED_RAT_FUNC, self.dist_factory
        )


@dataclass
class PartialRational(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = 10

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )
        self.positivity_heuristics = PositivityHeuristics.create(
            PositivityHeuristics.INDIVIDUAL_SUM,
            PositivityHeuristics.create(PositivityHeuristics.TRUE_RAT_FUNC)
        )


@dataclass
class AllOrPartialRational(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = 10

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )

        iterate_heur = PositivityHeuristics.create(PositivityHeuristics.ITERATE_TERMS, self.dist_factory)
        rat_func_heur = PositivityHeuristics.create(PositivityHeuristics.TRUE_RAT_FUNC)
        mixed_rat_func = PositivityHeuristics.create(PositivityHeuristics.MIXED_RAT_FUNC, self.dist_factory,
                                                     iterate_heur)
        sum_heur = PositivityHeuristics.create(PositivityHeuristics.INDIVIDUAL_SUM, rat_func_heur)

        self.positivity_heuristics = PositivityHeuristics.create(PositivityHeuristics.OR, sum_heur, mixed_rat_func)


class SynthesisStrategies(Enum):
    DEFAULT = DefaultStrategy
    SUM_THEN_RATIONAL = PartialRational
    APART_OR_TOGETHER = AllOrPartialRational
    FROM_HEURISTICS = SynthesisStrategy

    @classmethod
    def make(cls, st: 'SynthesisStrategies', *args, **kwargs) -> SynthesisStrategy:
        return st.value(*args, **kwargs)
