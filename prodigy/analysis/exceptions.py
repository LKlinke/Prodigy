class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""


class ObserveZeroEventError(ForwardAnalysisError):
    """ Observing an event that has zero probablity."""


class DistributionParameterError(ForwardAnalysisError):
    """The Parameter of a Distribution does not typematch"""


class VerificationError(Exception):
    pass


class ConfigurationError(Exception):
    pass
