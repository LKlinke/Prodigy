"""
================
Forward Analysis
================

The semantics transformation of the forwards analysis is mainly specified in the instruction handlers. They provide
the actual semantics on a generic level without knowing any specific encoding of a probability distribution.

The Semantics
#############

The semantics of a probabilistic program is defined as a distribution transformer. Each individual program instruction
is handled via a dedicated instruction handler. These handlers describe on a generic level, how the input distribution
is changed after executing the program instruction. This is oftentimes not computable, i.e., when the input distribution
has infinite support, or in case of while-loop instructions.

.. automodule:: prodigy.analysis.forward.instruction_handler

Distribution
############

Distributions are the core objects to transform by a probabilistic program. This class specifies a generaic interface
with desired convenience methods needed, in order to abstract away from an actual distribution encoding.
In fact, we deliver two types of distribution backends:

#. A `sympy` implementation of generating functions
#. A `GiNaC` C++ implementation shipped as a `pybind11` module.

.. automodule:: prodigy.analysis.forward.distribution

Analysis Configuration
######################

Sometimes, intermediate steps of the computation are desired. This and more configurations can be setup in a
`ForwardAnalysisConfig` object. Here also different backend Engines can be chosen.

.. automodule:: prodigy.analysis.forward.config
"""
from . import equivalence, optimization
from .config import ForwardAnalysisConfig
from .instruction_handler import compute_discrete_distribution
