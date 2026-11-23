"""
-----------------------
Forward Analysis Config
-----------------------
"""
from contextlib import contextmanager
from enum import Enum, EnumMeta, auto
from typing import Iterator, Type

import attr

from prodigy.analysis.optimization import Optimizer
from prodigy.analysis.optimization.gf_optimizer import GFOptimizer
from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      GeneratingFunctionSettings,
                                                      SympyPGF)
from prodigy.util.integers import unlimited_int_digits
from .evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from .evtinvariants.heuristics.strategies import SynthesisStrategies
from .evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from .deprecations import REMOVED_ENGINES
from .exceptions import ConfigurationError
from .solver.solver_type import SolverType
from ..distribution.distribution import CommonDistributionsFactory


class _EngineMeta(EnumMeta):
    """Direct access to a removed engine (``Engine.SYMENGINE``) raises an ``AttributeError`` with the removal
    message (:data:`REMOVED_ENGINES`) instead of a bare name, like :meth:`Engine.from_name` (#464)."""

    def __getattr__(cls, name: str) -> object:
        if name.lower() in REMOVED_ENGINES:
            raise AttributeError(REMOVED_ENGINES[name.lower()])
        raise AttributeError(name)


@attr.s
class ForwardAnalysisConfig:
    """Global configurable options for forward analysis."""

    class Engine(Enum, metaclass=_EngineMeta):
        """
        This enumeration specifies the type of backend used for distribution encodings and mathematical operations.
        """
        SYMPY = auto()
        GINAC = auto()

        @classmethod
        def from_name(cls, name: str) -> "ForwardAnalysisConfig.Engine":
            """The engine called ``name`` (case-insensitive; the empty name is the default, SymPy), as the CLI
            option ``--engine`` and the GUI's requests give it. A removed engine and an unknown name raise a
            :class:`ConfigurationError` that names the engines there are."""
            key = name.strip().lower() or "sympy"
            if key in REMOVED_ENGINES:
                raise ConfigurationError(REMOVED_ENGINES[key])
            if key.upper() not in cls.__members__:
                raise ConfigurationError(f"Unknown engine {name!r}; use sympy or ginac.")
            return cls[key.upper()]

    show_intermediate_steps: bool = attr.ib(default=False)
    """Enables the printing of results after each instruction."""

    step_wise: bool = attr.ib(default=False)
    """Enables step-by-step computation"""

    use_simplification: bool = attr.ib(default=False)
    """Simplifies the shown output distribution, bounded in size and time (:mod:`prodigy.distribution.display`).
    It changes neither the computation nor the result's value, only the form in which it is shown (#404)."""

    use_latex: bool = attr.ib(default=False)
    """Toggle to print LaTeX-Code instead of ASCII expressions (SymPy engine; see :meth:`activate`)."""

    engine: Engine = attr.ib(default=Engine.SYMPY)
    """Selects the distribution backend."""

    normalize: bool = attr.ib(default=True)
    """Switch to compute the normalized distribution"""

    strategy: SynthesisStrategies = attr.ib(default=SynthesisStrategies.DEFAULT)
    """The Strategy for dealing with synthesis."""

    templ_heuristic: TemplateHeuristics = attr.ib(default=TemplateHeuristics.DEFAULT)
    """ The heuristics to use for template genesis"""

    positivity_heuristic: PositivityHeuristics = attr.ib(default=PositivityHeuristics.DEFAULT)
    """ The heuristics to validate positivity"""

    solver_type: SolverType = attr.ib(default=SolverType.SYMPY)
    """The solver to use for equation systems."""

    show_all_invs: bool = attr.ib(default=False)
    """Toggels printing of spurious invariants"""

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
            # prodigy-ppl-ginac is only installed where it has wheels; only the GiNaC engine imports it
            try:
                # pylint: disable=import-outside-toplevel
                from prodigy.distribution.fast_generating_function import ProdigyPGF
            except ImportError as error:
                raise ConfigurationError(
                    "The GiNaC engine needs prodigy-ppl-ginac: pip install prodigy-ppl-ginac") from error
            return ProdigyPGF
        else:
            return CommonDistributionsFactory

    @contextmanager
    def activate(self) -> Iterator[GeneratingFunctionSettings]:
        """
        Applies ``use_latex`` within a ``with`` block only, and lifts
        Python's limit on the digits of integer strings (exact coefficients can be longer, #235).

        :func:`~prodigy.analysis.analyzer.compute_discrete_distribution` and
        :meth:`~prodigy.analysis.session.AnalysisSession.run` do this themselves; code that works
        with the distributions outside of them (e.g. printing a result) enters it as well.
        """
        with unlimited_int_digits(), GeneratingFunction.configured(latex_output=self.use_latex) as settings:
            yield settings
