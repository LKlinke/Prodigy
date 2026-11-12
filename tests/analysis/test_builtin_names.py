"""
Variables named like Python builtins (`sum`, `max`, `abs`, ...) stay variables when an engine's string form is
read back into SymPy (#321): `sympy.S` without local names reads `sum` as Python's builtin function.
"""
from pathlib import Path

import pytest
import sympy
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.distribution.equality import sympify_engine_string
from prodigy.pgcl import parse_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

EXAMPLE = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "skip_exceeding_sum.ppy"

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

GEOMETRIC = """
NAME: nat;
c: nat;
c = 1;
while (c == 1) {
    { c = 0; } [1/2] { NAME = NAME + 1; }
}
query Ex[NAME];
"""


def _run(source: str, decision: str, engine) -> dict:
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"0": decision}, config=ForwardAnalysisConfig(engine=engine))
    return session.run().to_json()


def test_sympify_engine_string_keeps_names_as_symbols():
    expr = sympify_engine_string("sum^2 + max*S/(1 - abs)", ["sum", "max", "S", "abs"])
    names = {"sum", "max", "S", "abs"}
    assert {str(symbol) for symbol in expr.free_symbols} == names
    assert expr == sympy.Symbol("sum") ** 2 + sympy.Symbol("max") * sympy.Symbol("S") / (1 - sympy.Symbol("abs"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["sum", "max", "abs", "pow"])
def test_synthesis_with_a_variable_named_like_a_builtin(engine, name):
    result = _run(GEOMETRIC.replace("NAME", name), "evt_synthesis", engine)
    assert result["status"] == "ok", result.get("error")
    assert result["bounds"]["kind"] == "exact"
    assert result["queries"][0]["value"] == "1"


@pytest.mark.parametrize("engine", ENGINES)
def test_skip_exceeding_sum_ends_with_the_synthesis_bound(engine):
    result = _run(EXAMPLE.read_text(encoding="utf-8"), "evt_synthesis:max_degree=0", engine)
    assert result["status"] == "error"
    assert result["error"]["type"] == "SynthesisBoundReached", result["error"]
    assert result["bounds"]["kind"] == "unknown"


def test_refuted_subinvariant_is_a_plain_cli_error(tmp_path):
    program = tmp_path / "geometric.ppy"
    program.write_text(GEOMETRIC.replace("NAME", "x"))
    result = CliRunner().invoke(cli, ["evt_lower_bound", str(program), "--invariant", "c*x"])
    assert result.exit_code == 1, result.output
    assert "Error: Could not establish nu <= Phi(nu)" in result.output
    assert "Traceback" not in result.output
