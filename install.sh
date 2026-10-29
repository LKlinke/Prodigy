#!/bin/bash
set -e

# Install Prodigy and the dependencies locked in uv.lock into .venv (needs uv: https://docs.astral.sh/uv/)
uv sync

# Install z3 solver
uv run pysmt-install --z3 --confirm-agreement
