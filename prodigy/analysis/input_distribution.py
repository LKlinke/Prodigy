"""
Validation of input distributions given in closed form (#296).

An input distribution is a generating function typed in by a person: the CLI argument of
``prodigy analyze`` (and ``main``, ``invariant_synthesis``, ``evt_lower_bound``), the field of the
GUI's Setup, the ``input_distribution`` of an :class:`~prodigy.analysis.session.AnalysisSession`,
or the distribution passed to :func:`~prodigy.analysis.analyzer.compute_discrete_distribution`.
The analysis treats it as a (sub-)probability distribution: residual masses and lower bounds
assume non-negative coefficients, and masses are read off the closed form at 1. So an input is
accepted only when Prodigy *proves* that it is

* a power series at 0 in the program variables,
* with non-negative coefficients,
* and finite mass (the sum of all coefficients).

An input for which one of these is refuted, or cannot be proven, is refused with
:class:`InputDistributionError`, naming the input and the reason. Refusing is preferred over
running the analysis with every guarantee downgraded to ``unknown``: a result without any
guarantee is of little use, and the message tells how to state the input so that it can be
checked.

The proofs are in :mod:`prodigy.distribution.finite_measure`
(:func:`~prodigy.distribution.finite_measure.check_input_expression`); the distribution package uses them as
well, for the convergence of marginals and the arguments of ``sample_pgf`` and ``iid``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import sympy

from prodigy.analysis.exceptions import ForwardAnalysisError
from prodigy.distribution.finite_measure import check_input_expression
from prodigy.language.expression_allowlist import MATH_FUNCTIONS, InadmissibleExpression, admissible_expression

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.distribution import Distribution
    from prodigy.analysis.instructionhandler.program_info import ProgramInfo
    from prodigy.language import Program


class InputDistributionError(ForwardAnalysisError):
    """The input distribution is refused: it is not proven to be a power series at 0 with
    non-negative coefficients and finite mass. ``details["proof"]`` says whether this was refuted
    (``refuted``) or could not be proven (``not_proven``). It is not called ``verdict``, which the
    result views explain as the verdict on a loop invariant."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


REQUIREMENT = ("Prodigy accepts an input distribution only when it can prove that it is a power series at 0 "
               "in the program variables with non-negative coefficients and finite mass.")


def input_distribution_from_text(config: ForwardAnalysisConfig, text: str, program: Program,
                                 syntax: str = "ppy") -> Distribution:
    """The distribution of the closed form ``text`` for ``program``, after checking that ``text``
    is a generating function that is safe to give to the engines (SymPy evaluates Python code)
    and that it is a distribution (:func:`validate_input_distribution`)."""
    names = [str(v) for v in program.variables] + [str(p) for p in program.parameters]
    try:
        spelled = admissible_expression(text, names=names, syntax=syntax, functions=MATH_FUNCTIONS)
    except InadmissibleExpression as exc:
        raise InputDistributionError(f"The input distribution is refused: {exc}",
                                     details={"type": "input_distribution", "proof": "refuted",
                                              "input": text}) from exc
    dist = config.factory.from_expr(spelled, *program.variables.keys())
    validate_input_distribution(dist, program)
    return dist


def validate_input_distribution(dist: Distribution, program: Union[Program, ProgramInfo]) -> None:
    """Raises :class:`InputDistributionError` unless ``dist`` is proven to be a power series at 0
    in its variables (the program's and its own) with non-negative coefficients and finite mass.
    Every symbol that is not a variable is a parameter; ``program`` gives their types."""
    variables = sorted({str(v) for v in program.variables} | {str(v) for v in dist.get_variables()})
    parameter_types = {str(p): t for p, t in program.parameters.items()}
    text = _plain_text(dist)
    verdict, reason = check_input_expression(text, variables, parameter_types)
    if verdict is True:
        return
    if verdict is False:
        message = f"The input distribution {text} is not a distribution: {reason}."
    else:
        message = f"The input distribution {text} is refused: {reason}. {REQUIREMENT}"
    raise InputDistributionError(message, details={"type": "input_distribution", "input": text,
                                                   "proof": "refuted" if verdict is False else "not_proven",
                                                   "reason": reason})


def _plain_text(dist: Distribution) -> str:
    """The closed form of ``dist`` as plain text (the SymPy engine prints LaTeX under ``use_latex``)."""
    function = getattr(dist, "_function", None)
    return str(function) if isinstance(function, sympy.Basic) else str(dist)
