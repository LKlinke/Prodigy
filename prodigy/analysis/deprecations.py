"""
Deprecations of v0.5.0 that end in v0.6.0: the ``.pgcl`` syntax (#432, removal #433) and the SymEngine
engine (#416). Reading such a program or choosing such an engine is still possible; each use gives
the warning below, in the CLI on stderr and in the ``warnings`` of the result (and so in the GUI).
"""
from typing import List

REMOVAL_RELEASE = "v0.6.0"
"""The release that removes the deprecated features."""

PGCL_DEPRECATION = (
    f"The .pgcl syntax is deprecated and will be removed in {REMOVAL_RELEASE}. Convert the program to .ppy "
    f"while v0.5.x is installed (`prodigy convert program.pgcl`, or Convert to .ppy in the GUI); "
    f"{REMOVAL_RELEASE} removes the converter as well.")

SYMENGINE_DEPRECATION = (
    f"The SymEngine engine is deprecated and will be removed in {REMOVAL_RELEASE}. Use the SymPy engine "
    "(the reference) or the GiNaC engine (the fast one) instead.")


def deprecation_warnings(syntax: str, engine: str) -> List[str]:
    """The deprecation warnings for a program in ``syntax`` (``pgcl`` or ``ppy``) analysed with ``engine``
    (``sympy``, ``ginac`` or ``symengine``, case-insensitive)."""
    found = []
    if syntax == "pgcl":
        found.append(PGCL_DEPRECATION)
    if engine.lower() == "symengine":
        found.append(SYMENGINE_DEPRECATION)
    return found
