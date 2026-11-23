#!/bin/bash
set -e

# Install Prodigy and the dependencies locked in uv.lock into .venv (needs uv: https://docs.astral.sh/uv/)
uv sync
