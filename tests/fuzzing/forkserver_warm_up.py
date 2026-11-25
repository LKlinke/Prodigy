"""
Preloaded by the fork server of :func:`~tests.fuzzing.oracles.isolated`: one tiny program through every oracle,
so that the lazy imports (GiNaC), the parser and the first-use caches are set up once in the server and every
child, a fork of it, starts warm. Without it each program paid about 1.5 s of warm-up, ten times the analysis of
a typical program, and the smoke run of ``make fuzz`` did not fit its budget (#486).
"""
from .oracles import check_program

_PROGRAM = "x: nat;\ny: nat;\nx = unif(0, 2);\nobserve(x < 2);\ny = poisson(1/2);\nquery Ex[x + y];\nprint;\n"

try:
    check_program(_PROGRAM, "ppy", {})
except Exception:  # pylint: disable=broad-exception-caught
    # the warm-up is an optimisation: a failure here must not stop the fork server; the programs report it
    pass
