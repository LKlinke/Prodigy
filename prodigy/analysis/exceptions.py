class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""


class ObserveZeroEventError(ForwardAnalysisError):
    """ Observing an event that has zero probablity."""


class DistributionParameterError(ForwardAnalysisError):
    """The Parameter of a Distribution does not typematch"""


class VerificationError(Exception):
    pass


class SolveError(Exception):
    """The solver has encountered any issue."""


class HeuristicsError(Exception):
    """ Representing errors occuring in positivity heuristics."""


class ConfigurationError(Exception):
    pass
