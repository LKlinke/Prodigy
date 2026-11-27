"""``python -m prodigy.cli``: the same as the command ``prodigy``."""
from prodigy.cli import cli

# only when run, not when imported (the API reference of the docs imports every module)
if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter,missing-kwoa  # click passes the options
