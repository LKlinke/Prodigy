"""
A loop that runs several times (e.g. inside another loop) reports one summary of its executions instead of
the messages of all of them (#229).
"""
from pathlib import Path

import sympy

from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession, LoopRecord
from prodigy.pgcl import parse_pgcl

YULE_SIMON = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "hard_distributions" / \
    "d_finite" / "yule_simon_coins.ppy"


def _record() -> LoopRecord:
    return LoopRecord(loop=None, cond="c == 1", strategy="unroll", params={})


def test_one_execution_keeps_its_messages():
    record = _record()
    record.add_outcome(Bound.LOWER, ["Lower bound after 3 iterations; mass 1/8 is still inside the loop."])
    assert record.to_json()["messages"] == ["Lower bound after 3 iterations; mass 1/8 is still inside the loop."]


def test_identical_messages_of_several_executions_are_kept_once():
    record = _record()
    for _ in range(3):
        record.add_outcome(Bound.EXACT, ["Exact: I is a fixed point."])
    assert record.to_json()["messages"] == ["Exact: I is a fixed point."]


def test_differing_messages_of_several_executions_become_one_summary():
    record = _record()
    record.add_outcome(Bound.EXACT, ["Exact: no mass is left inside the loop after 3 iterations.", "Note."])
    for mass in ("1/8", "1/16"):
        record.add_outcome(Bound.LOWER, [f"Lower bound after 3 iterations; mass {mass} is still inside the loop.",
                                         "Note."])
    record.residual = sympy.Rational(3, 16)
    record.add_message("A note of the session.")
    assert record.to_json()["messages"] == [
        "Note.",
        "3 executions: lower bound in 2, exact in 1; in total, mass 3/16 is still inside the loop.",
        "A note of the session.",
    ]


def test_inner_loop_of_yule_simon_reports_one_summary():
    source = YULE_SIMON.read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"0": "unroll:iterations=8", "1": "unroll:iterations=30"})
    result = session.run().to_json()
    inner = next(loop for loop in result["loops"] if str(loop["id"]) == "1")
    assert inner["kind"] == Bound.LOWER
    assert inner["invocations"] == 8
    [message] = inner["messages"]
    assert message.startswith("8 executions: lower bound in 7, exact in 1; in total, mass ")
    assert message.endswith(" is still inside the loop.")
    total = sympy.Rational(message.split("mass ")[1].split(" ")[0])
    assert total == sympy.Rational(inner["residual_mass"])


def test_cli_loop_line_without_messages_has_no_dangling_colon():
    # #333: a failed loop without messages printed "loop 0 (line 8) [evt_synthesis] unknown: "
    # pylint: disable=import-outside-toplevel
    from click.testing import CliRunner
    from prodigy.cli import cli
    example = Path(__file__).parents[2] / "pgfexamples" / "inference" / "loopy" / "skip_exceeding_sum.ppy"
    result = CliRunner().invoke(cli, ["analyze", str(example), "--loop", "*=evt_synthesis:max_degree=0"])
    [line] = [line for line in result.output.splitlines() if line.strip().startswith("loop 0")]
    assert line.strip() == "loop 0 (line 8) [evt_synthesis] unknown (the strategy failed, see the error above)"
    assert not line.rstrip().endswith(":")
