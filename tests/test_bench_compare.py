"""Tests of benchmarks/compare.py (the comparison of two benchmark runs)."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "bench_compare", Path(__file__).resolve().parent.parent / "benchmarks" / "compare.py")
assert SPEC is not None and SPEC.loader is not None
bench_compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_compare)


def entry(seconds, status="ok", total=None):
    """One measurement."""
    return {"status": status, "seconds": seconds, "total": total, "wall": seconds}


def run(**cases):
    """A results file with the engine `sympy` for each case."""
    return {"calibration": 0.1, "results": {name: {"sympy": value} for name, value in cases.items()}}


def problems(base, head, **kwargs):
    """The names of the flagged cases."""
    return [name for name, _, _ in bench_compare.compare(base, head, **kwargs)[1]]


def test_factor_and_minimum_time():
    base = run(a=entry(1.0), b=entry(1.0), c=entry(0.1), d=entry(1.0))
    head = run(a=entry(2.5), b=entry(1.9), c=entry(0.4), d=entry(0.2))
    # a: more than 2x and above 0.5 s; b: below 2x; c: 4x but below the minimum time; d: faster
    assert problems(base, head) == ["a"]
    assert problems(base, head, factor=1.5) == ["a", "b"]
    assert problems(base, head, min_seconds=0.1) == ["a", "c"]


def test_missing_entries_are_not_slower():
    base = run(a=entry(1.0), gone=entry(1.0))
    head = run(a=entry(1.0), new=entry(9.0))
    table, flagged = bench_compare.compare(base, head)
    assert not flagged
    assert "new" in table and "removed" in table


def test_missing_engine_on_one_side():
    base = {"results": {"a": {"sympy": entry(1.0)}}}
    head = {"results": {"a": {"sympy": entry(1.0), "ginac": entry(30.0)}}}
    assert problems(base, head) == []


def test_status_change_is_flagged():
    assert problems(run(a=entry(1.0)), run(a=entry(None, "timeout"))) == ["a"]
    assert problems(run(a=entry(None, "timeout")), run(a=entry(1.0))) == []


def test_zero_base_time():
    assert problems(run(a=entry(0.0)), run(a=entry(0.6))) == ["a"]
    assert problems(run(a=entry(0.0)), run(a=entry(0.2))) == []


def test_main_writes_summary_and_exit_codes(tmp_path):
    base, head, summary = tmp_path / "base.json", tmp_path / "head.json", tmp_path / "summary.md"
    base.write_text(json.dumps(run(a=entry(1.0))), encoding="utf-8")
    head.write_text(json.dumps(run(a=entry(5.0))), encoding="utf-8")
    assert bench_compare.main([str(base), str(head), "--summary", str(summary)]) == 0
    assert "**slower**" in summary.read_text(encoding="utf-8")
    assert bench_compare.main([str(base), str(head), "--fail-on-slower"]) == 1
    assert bench_compare.main([str(base), str(tmp_path / "missing.json")]) == 2


def test_total_time_is_compared_where_both_runs_have_it():
    # the analysis is as fast as before, but the result summary (in the total time) got slower (#479)
    base = run(a=entry(1.0, total=1.5), b=entry(1.0, total=1.5), c=entry(1.0))
    head = run(a=entry(1.0, total=4.0), b=entry(1.0, total=2.0), c=entry(1.0, total=9.0))
    assert problems(base, head) == ["a"]
    table, _ = bench_compare.compare(base, head)
    assert "total 1.50 → 4.00 (2.7×)" in table
