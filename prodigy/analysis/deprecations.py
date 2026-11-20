"""
Deprecations of v0.5.0 that end in v0.6.0: the ``.pgcl`` syntax (#432, removal #433). Reading such a
program is still possible; each use gives the warning below, in the CLI on stderr and in the ``warnings``
of the result (and so in the GUI). The SymEngine engine, deprecated in v0.5.0 as well, was removed in
v0.6.0 (#416; :data:`REMOVED_ENGINES`).
"""
from typing import Dict, List

REMOVAL_RELEASE = "v0.6.0"
"""The release that removes the deprecated features."""

REMOVED_ENGINES: Dict[str, str] = {
    "symengine": f"the SymEngine engine was removed in {REMOVAL_RELEASE}; use sympy or ginac",
}
"""The engines that Prodigy had and removed, with the message that choosing one of them gives (#416): in the
CLI (``--engine``), the GUI's requests and :meth:`prodigy.analysis.config.ForwardAnalysisConfig.Engine.from_name`."""

PGCL_DEPRECATION = (
    f"The .pgcl syntax is deprecated and will be removed in {REMOVAL_RELEASE}. Convert the program to .ppy "
    f"while v0.5.x is installed (`prodigy convert program.pgcl`, or Convert to .ppy in the GUI); "
    f"{REMOVAL_RELEASE} removes the converter as well.")


def deprecation_warnings(syntax: str) -> List[str]:
    """The deprecation warnings for a program in ``syntax`` (``pgcl`` or ``ppy``)."""
    return [PGCL_DEPRECATION] if syntax == "pgcl" else []
