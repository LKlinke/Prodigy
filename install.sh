#!/bin/bash

# Install poetry packages
poetry install

# Install z3 solver
poetry run pysmt-install --z3 --confirm-agreement
