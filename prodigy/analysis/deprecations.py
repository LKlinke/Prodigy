"""
Features that Prodigy had and removed in v0.6.0, with the messages that using one of them gives: the
``.pgcl`` syntax (#433, deprecated in v0.5.0 by #432) and the SymEngine engine (#416).
"""
import os
from pathlib import Path
from typing import Dict, Optional, Union

REMOVAL_RELEASE = "v0.6.0"
"""The release that removed the deprecated features."""

REMOVED_ENGINES: Dict[str, str] = {
    "symengine": f"the SymEngine engine was removed in {REMOVAL_RELEASE}; use sympy or ginac",
}
"""The engines that Prodigy had and removed, with the message that choosing one of them gives (#416): in the
CLI (``--engine``), the GUI's requests and :meth:`prodigy.analysis.config.ForwardAnalysisConfig.Engine.from_name`."""

PGCL_REMOVED = (
    f"The .pgcl syntax was removed in {REMOVAL_RELEASE}. Convert the program to .ppy with Prodigy v0.5.x "
    "(`prodigy convert program.pgcl`, or Convert to .ppy in its GUI).")
"""The answer to a ``.pgcl`` program in the CLI, the GUI and the API (#433)."""


class PgclRemovedError(ValueError):
    """A ``.pgcl`` program was given; its message is :data:`PGCL_REMOVED`."""

    def __init__(self) -> None:
        super().__init__(PGCL_REMOVED)


def program_syntax(syntax: Optional[str] = None, path: Union[str, os.PathLike, None] = None) -> str:
    """
    The syntax of a program given from outside (CLI, GUI, API): always ``ppy``. Raises
    :class:`PgclRemovedError` if ``syntax`` is ``pgcl`` or ``path`` ends in ``.pgcl``, and ``ValueError``
    for an unknown ``syntax``.

    .. doctest::

        >>> program_syntax(), program_syntax("ppy"), program_syntax(path="dice.ppy")
        ('ppy', 'ppy', 'ppy')
        >>> program_syntax(path="dice.pgcl")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        prodigy.analysis.deprecations.PgclRemovedError: The .pgcl syntax was removed in v0.6.0. ...
    """
    if syntax == "pgcl" or (path is not None and Path(path).suffix == ".pgcl"):
        raise PgclRemovedError()
    if syntax not in (None, "ppy"):
        raise ValueError(f"Unknown syntax {syntax!r}, expected ppy.")
    return "ppy"
