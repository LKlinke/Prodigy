"""
prodigy also offers a simple command-line interface for quick program inspection.

If you do not have prodigy installed globally, you can use `uv run prodigy INPUT` in the project folder.

The package is a front end of the analysis (``docs/source/architecture.md``): ``group`` holds the command group
``prodigy`` and its global options (the analysis configuration), ``common`` reading programs and options,
``output`` the text output of results, ``commands`` the analysis commands and ``web``, ``proofs`` the commands for
single proof rules (equivalence, independence, invariants, sandwich bounds).

.. click:: prodigy.cli:cli
   :prog: prodigy
   :show-nested:
"""
from prodigy.cli.group import cli
# the command modules register their commands with the group when they are imported
from prodigy.cli import commands, proofs  # noqa: F401  pylint: disable=unused-import
from prodigy.cli.proofs import EXIT_EQUIVALENT, EXIT_NOT_EQUIVALENT, EXIT_UNDECIDED

__all__ = ["cli", "EXIT_EQUIVALENT", "EXIT_NOT_EQUIVALENT", "EXIT_UNDECIDED"]
