"""
===============
Static Analysis
===============

This module contains implementation of static analysis methods for probabilistic programs.

"""

from .independence import independent_vars

# the API of the package (#379, step 1)
__all__ = ["independent_vars"]
