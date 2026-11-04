from typing import Any, Dict, Optional


class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""


class ObserveZeroEventError(ForwardAnalysisError):
    """ Observing an event that has zero probablity."""


class UnsupportedFeatureError(ForwardAnalysisError):
    """The program uses a language feature that the analysis has no semantics for."""


class DistributionParameterError(ForwardAnalysisError):
    """The Parameter of a Distribution does not typematch"""


class VerificationError(Exception):
    """A check could not establish what it was asked to. ``details`` holds the evidence, if any (e.g. the
    negative coefficient that refutes a sub-invariant)."""

    def __init__(self, message: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


class SolveError(Exception):
    """The solver has encountered any issue."""


class HeuristicsError(Exception):
    """ Representing errors occuring in positivity heuristics."""


class ConfigurationError(Exception):
    pass
