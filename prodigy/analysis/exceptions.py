class ForwardAnalysisError(Exception):
    """Base class for forward analysis-related exceptions."""


class ObserveZeroEventError(ForwardAnalysisError):
    pass


class DistributionParameterError(ForwardAnalysisError):
    pass


class IncomparableTypesException(Exception):
    pass


class ComparisonException(Exception):
    pass


class NotComputableException(Exception):
    pass


class ParameterError(Exception):
    pass


class ExpressionError(Exception):
    pass


class VerificationError(Exception):
    pass


class ConfigurationError(Exception):
    pass
