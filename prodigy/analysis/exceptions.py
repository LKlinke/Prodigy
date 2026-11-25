from typing import Any, Dict, Optional


class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""


class ObserveZeroEventError(ForwardAnalysisError):
    """ Observing an event that has zero probablity."""


class UnsupportedFeatureError(ForwardAnalysisError):
    """The program uses a language feature that the analysis has no semantics for."""


class UndefinedDistributionError(ForwardAnalysisError):
    """A value of type ``dist`` is undefined where the program needs it: ``infer(f, a)`` with
    normalising constant 0 (no run of ``f(a)`` terminates and passes its observations), or a
    ``dist`` variable that is read before it was assigned."""


class InvalidProbabilityError(ForwardAnalysisError, ValueError):
    """The probability of a probabilistic choice or of a counter operator is undefined or outside of
    [0, 1] in a reachable state (e.g. ``{x := 1} [5/k] {x := 2}`` with ``k = 3``)."""


class ParameterSynthesisRequired(ForwardAnalysisError):
    """The invariant chosen for a loop has template parameters, or the check of it fixes parameters to
    values: that is parameter synthesis, a comparison of the loop with the invariant program, not an
    analysis step (#154). ``details`` holds the loop, the invariant, the parameters and a program with
    just the loop, so that a front end can open the pair for comparison."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


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
