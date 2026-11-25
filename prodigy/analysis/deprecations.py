"""
Features that Prodigy had and removed in v0.6.0, with the messages that using one of them gives: the
``.pgcl`` syntax (#433, deprecated in v0.5.0 by #432; the message and its check are
:func:`prodigy.language.parser.check_syntax`) and the SymEngine engine (#416).
"""
from typing import Dict

# the one removal message of .pgcl and its check live in the language (#489); re-exported here
from prodigy.language.parser import PGCL_REMOVED, PgclRemovedError

__all__ = ["PGCL_REMOVED", "REMOVAL_RELEASE", "REMOVED_ENGINES", "PgclRemovedError"]

REMOVAL_RELEASE = "v0.6.0"
"""The release that removed the deprecated features."""

REMOVED_ENGINES: Dict[str, str] = {
    "symengine": f"the SymEngine engine was removed in {REMOVAL_RELEASE}; use sympy or ginac",
}
"""The engines that Prodigy had and removed, with the message that choosing one of them gives (#416): in the
CLI (``--engine``), the GUI's requests and :meth:`prodigy.analysis.config.ForwardAnalysisConfig.Engine.from_name`."""
