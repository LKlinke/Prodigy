from __future__ import annotations

import logging

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)

def independent_vars(program: Program,
                     config: ForwardAnalysisConfig) -> Set[Set[Var, Var]]:
    """
    This method under-approximates the pairwise stochastic independence relation using the d-separation on a simple
    dependence graph.

    .. param config: Some configuration.
    .. param program: The program
    .. returns: Set of variable pairs which are surely independent.
    """

    logger.debug("start.")

    logger.debug(" result:\t%s", "")

    return {set()}
