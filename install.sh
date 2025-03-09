#!/bin/bash

# Install poetry packages
poetry install

# Install z3 solver
pysmt-install --z3 --confirm-agreement