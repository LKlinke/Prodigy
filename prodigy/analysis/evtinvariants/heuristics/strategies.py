from dataclasses import dataclass, field
from enum import Enum
from typing import Collection, Optional

from prodigy.analysis.exceptions import HeuristicsError
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.distribution import CommonDistributionsFactory

DEFAULT_MAX_DEGREE = 4
"""Default degree bound of the templates (strategy parameter ``max_degree`` of ``evt_synthesis``). The
templates of the rational heuristic have a numerator of at most this degree in each variable."""


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
    max_degree: int = DEFAULT_MAX_DEGREE

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )
        self.positivity_heuristics = PositivityHeuristics.create(
            PositivityHeuristics.MIXED_RAT_FUNC, self.dist_factory, variables=self.variables
        )


@dataclass
class PartialRational(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = DEFAULT_MAX_DEGREE

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )
        self.positivity_heuristics = PositivityHeuristics.create(
            PositivityHeuristics.INDIVIDUAL_SUM,
            PositivityHeuristics.create(PositivityHeuristics.TRUE_RAT_FUNC, variables=self.variables)
        )


@dataclass
class InfinitePolynomial(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory

    def __post_init__(self):
        self.template_heuristics = (
            TemplateHeuristics.create(TemplateHeuristics.POLY, self.variables, self.dist_factory))
        self.positivity_heuristics = PositivityHeuristics.create(PositivityHeuristics.POLYNOMIAL,
                                                                 variables=self.variables)


@dataclass
class FinitePolynomial(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = DEFAULT_MAX_DEGREE

    def __post_init__(self):
        self.template_heuristics = (
            TemplateHeuristics.create(TemplateHeuristics.POLY, self.variables, self.dist_factory, self.max_degree))
        self.positivity_heuristics = PositivityHeuristics.create(PositivityHeuristics.POLYNOMIAL,
                                                                 variables=self.variables)


@dataclass
class AllOrPartialRational(SynthesisStrategy):
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    max_degree: int = DEFAULT_MAX_DEGREE

    def __post_init__(self):
        self.template_heuristics = TemplateHeuristics.create(
            TemplateHeuristics.DEFAULT,
            self.max_degree,
            self.variables,
            self.dist_factory
        )

        iterate_heur = PositivityHeuristics.create(PositivityHeuristics.ITERATE_TERMS, self.dist_factory,
                                                   variables=self.variables)
        rat_func_heur = PositivityHeuristics.create(PositivityHeuristics.TRUE_RAT_FUNC, variables=self.variables)
        mixed_rat_func = PositivityHeuristics.create(PositivityHeuristics.MIXED_RAT_FUNC, self.dist_factory,
                                                     iterate_heur, variables=self.variables)
        sum_heur = PositivityHeuristics.create(PositivityHeuristics.INDIVIDUAL_SUM, rat_func_heur)

        self.positivity_heuristics = (
            PositivityHeuristics.create(PositivityHeuristics.OR, sum_heur, mixed_rat_func))


# Positivity heuristics that combine other heuristics; they can only be used inside a predefined strategy.
COMBINING_POSITIVITY_HEURISTICS = frozenset({
    PositivityHeuristics.OR, PositivityHeuristics.AND, PositivityHeuristics.NOT, PositivityHeuristics.INDIVIDUAL_SUM
})

SELECTABLE_POSITIVITY_HEURISTICS = tuple(
    name.lower() for name, heuristic in PositivityHeuristics.__members__.items()
    if heuristic not in COMBINING_POSITIVITY_HEURISTICS)
"""Names (lower case, aliases included) of the positivity heuristics that ``FROM_HEURISTICS`` accepts."""


@dataclass
class FromHeuristics(SynthesisStrategy):
    """A strategy made of one template heuristic and one positivity heuristic chosen by the user."""
    variables: Collection[str]
    dist_factory: CommonDistributionsFactory
    templ_heuristic: TemplateHeuristics
    positivity_heuristic: PositivityHeuristics
    max_degree: int = DEFAULT_MAX_DEGREE

    def __post_init__(self):
        if self.templ_heuristic == TemplateHeuristics.POLY:
            self.template_heuristics = TemplateHeuristics.create(
                TemplateHeuristics.POLY, self.variables, self.dist_factory, self.max_degree)
        else:
            self.template_heuristics = TemplateHeuristics.create(
                self.templ_heuristic, self.max_degree, self.variables, self.dist_factory)

        if self.positivity_heuristic in COMBINING_POSITIVITY_HEURISTICS:
            raise HeuristicsError(
                f"The positivity heuristic {self.positivity_heuristic.name.lower()} combines other heuristics and "
                f"cannot be used on its own. Choose one of {', '.join(SELECTABLE_POSITIVITY_HEURISTICS)}.")
        # every heuristic gets the variables of the series; other symbols are parameters (#295)
        if self.positivity_heuristic in (PositivityHeuristics.POLYNOMIAL, PositivityHeuristics.TRUE_RAT_FUNC):
            self.positivity_heuristics = PositivityHeuristics.create(self.positivity_heuristic,
                                                                     variables=self.variables)
        else:
            self.positivity_heuristics = PositivityHeuristics.create(self.positivity_heuristic, self.dist_factory,
                                                                     variables=self.variables)


class SynthesisStrategies(Enum):
    DEFAULT = DefaultStrategy
    SUM_THEN_RATIONAL = PartialRational
    APART_OR_TOGETHER = AllOrPartialRational
    INF_POLY = InfinitePolynomial
    FIN_POLY = FinitePolynomial
    FROM_HEURISTICS = FromHeuristics

    @classmethod
    def make(cls, st: 'SynthesisStrategies', variables: Collection[str], dist_factory: CommonDistributionsFactory,
             templ_heuristic: Optional[TemplateHeuristics] = None,
             positivity_heuristic: Optional[PositivityHeuristics] = None,
             max_degree: int = DEFAULT_MAX_DEGREE) -> SynthesisStrategy:
        """Builds the strategy ``st``.

        The heuristics are only used by ``FROM_HEURISTICS``, which needs both of them; the other strategies
        fix their own heuristics. ``max_degree`` bounds the degree of the templates; ``INF_POLY`` enumerates
        polynomials of every degree and ignores it."""
        if st == SynthesisStrategies.INF_POLY:
            return st.value(variables, dist_factory)
        if st != SynthesisStrategies.FROM_HEURISTICS:
            return st.value(variables, dist_factory, max_degree)
        if templ_heuristic is None or positivity_heuristic is None:
            missing = [name for name, value in (("template heuristic", templ_heuristic),
                                                ("positivity heuristic", positivity_heuristic)) if value is None]
            raise HeuristicsError(f"The synthesis strategy from_heuristics needs a {' and a '.join(missing)}.")
        return FromHeuristics(variables, dist_factory, templ_heuristic, positivity_heuristic, max_degree)

    @classmethod
    def from_config(cls, config, variables: Collection[str], max_degree: int = DEFAULT_MAX_DEGREE) -> SynthesisStrategy:
        """Builds the strategy chosen in a :class:`ForwardAnalysisConfig` with templates up to ``max_degree``."""
        return cls.make(config.strategy, variables, config.factory,
                        templ_heuristic=config.templ_heuristic, positivity_heuristic=config.positivity_heuristic,
                        max_degree=max_degree)
