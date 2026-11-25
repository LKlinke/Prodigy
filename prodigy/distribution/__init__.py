"""
General Classes
###############

.. automodule:: prodigy.distribution.distribution
   :members:

Sympy Implementation
####################

.. automodule:: prodigy.distribution.generating_function

GiNaC Implementation
####################

.. automodule:: prodigy.distribution.fast_generating_function

Convergence of Marginals
########################

.. automodule:: prodigy.distribution.convergence
   :members:

Proof Rules on Generating Functions
###################################

.. automodule:: prodigy.distribution.finite_measure
   :members:

.. automodule:: prodigy.distribution.product_form
   :members: ProductFormSeries, Denominator, to_product_form, rational_mass

.. automodule:: prodigy.distribution.coefficient_signs
   :members:

Order
#####

.. automodule:: prodigy.distribution.ordering
   :members:

Exact Division
##############

.. automodule:: prodigy.distribution.exact_division
   :members:
"""

from .distribution import (CommonDistributionsFactory, Distribution,
                           DistributionParam, MarginalType, NonFiniteMarginalError, ZeroDivisorError,
                           State, known_convergent_series)
from .ordering import UndecidedComparisonError
