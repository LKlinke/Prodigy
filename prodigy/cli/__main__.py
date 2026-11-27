"""``python -m prodigy.cli``: the same as the command ``prodigy``."""
from prodigy.cli import cli

cli()  # pylint: disable=no-value-for-parameter,missing-kwoa  # click passes the options
