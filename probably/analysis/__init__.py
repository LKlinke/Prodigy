"""
=====================
``probably.analysis``
=====================

The analysis module provides functionalities to analyze probablistic programs written in pGCL language.

There are two main types of analysis:
    * Forwards Analysis
    * Backwards Analysis

The *forward analysis* module considers a distribution transformer semantics, i.e., given a program and an initial distribution,
it computes (if possible) the exact output distribution. Sometimes, when this is not computable, approximations can be
derived.

The *backwards semantics* module considers expectation transoformer semantics a la weakest preexpectation calculus [Kam19].

.. automodule:: probably.analysis.forward
.. automodule:: probably.analysis.backward
"""

from probably.analysis.forward.config import ForwardAnalysisConfig
from probably.analysis.forward.distribution import Distribution
from probably.analysis.forward.instruction_handler import compute_discrete_distribution
from .backward.wp import loopfree_wp_transformer
