"""
Runs the benchmark set (benchmarks/suite.json) on each engine and compares the times with a baseline.

Each case runs ``prodigy --engine E analyze --json`` in a fresh process (so no cache survives from one case
to the next) with a time limit, and its time is the analysis time from the JSON result (without the start of
Python and the imports). Shared CI runners differ in speed, so the baseline also stores the time of a fixed
calibration workload, and the baseline times are scaled by the ratio of the calibration times before they
are compared. A case is a regression if it takes more than ``--threshold`` times its scaled baseline time
(and more than ``--min-seconds``), or if its status differs from the baseline (e.g. a time-out or an error).

    uv run python benchmarks/run.py                     # all engines, table on stdout
    uv run python benchmarks/run.py --engines sympy     # one engine
    uv run python benchmarks/run.py --write-baseline    # store the times as the new baseline

See docs/source/testing.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "suite.json"
BASELINE = ROOT / "benchmarks" / "baseline.json"
ENGINES = ("sympy", "ginac", "symengine")
RUNNER = "import sys; from prodigy.cli import cli; sys.exit(cli())"


def calibrate(repetitions: int = 5) -> float:
    """The fastest of ``repetitions`` runs of a fixed workload of Python and SymPy (seconds)."""
    import sympy  # pylint: disable=import-outside-toplevel
    x, y = sympy.symbols("x y")
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        sympy.core.cache.clear_cache()
        expression = sympy.expand((1 + x + y) ** 14 * (1 - x / 2) ** 8)
        sympy.factor(expression)
        sympy.cancel(((x ** 8 - 1) / (x - 1) + y / (1 - x / 2)) * (1 - x))
        sum(i * i % 7 for i in range(500_000))
        best = min(best, time.perf_counter() - start)
    return best


def run_case(case: Dict[str, Any], engine: str, timeout: float) -> Dict[str, Any]:
    """Runs one case on one engine; returns its status and analysis time in seconds."""
    command = [sys.executable, "-c", RUNNER, "--engine", engine, *case.get("options", []),
               "analyze", "--json", "--coefficients", "0"]
    for loop in case.get("loops", []):
        command += ["--loop", loop]
    command.append(case["program"])
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": None, "wall": round(time.perf_counter() - started, 3)}
    wall = round(time.perf_counter() - started, 3)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"status": "crash", "seconds": None, "wall": wall, "stderr": process.stderr[-2000:]}
    return {"status": result["status"], "seconds": round(result["timing"]["analysis"], 3), "wall": wall}


def compare(measured: Dict[str, Any], baseline: Optional[Dict[str, Any]], scale: float, threshold: float,
            min_seconds: float) -> Optional[str]:
    """A description of the regression of `measured` against `baseline`, or None."""
    if baseline is None:
        return None
    if measured["status"] != baseline["status"]:
        return f"status {measured['status']} (baseline {baseline['status']})"
    if measured["seconds"] is None or baseline["seconds"] is None:
        return None
    expected = baseline["seconds"] * scale
    if measured["seconds"] > max(threshold * expected, min_seconds):
        return f"{measured['seconds'] / expected:.1f}× the baseline"
    return None


def markdown(cases: List[Dict[str, Any]], engines: List[str], results: Dict[str, Dict[str, Any]],
             baseline: Dict[str, Any], scale: float, regressions: Dict[Tuple[str, str], str],
             threshold: float) -> str:
    """The results as a Markdown table (program × engine, seconds) with the baseline times."""
    lines = ["## Benchmarks", "",
             f"Analysis time in seconds; in brackets the baseline scaled by the calibration "
             f"(this runner is {1 / scale:.2f}× as fast as the one of the baseline). "
             f"A regression is more than {threshold:g}× the scaled baseline or another status.", "",
             "| Program | " + " | ".join(engines) + " |", "|---|" + "---|" * len(engines)]
    for case in cases:
        cells = []
        for engine in engines:
            measured = results[case["name"]][engine]
            base = baseline.get("results", {}).get(case["name"], {}).get(engine)
            text = f"{measured['seconds']:.2f}" if measured["seconds"] is not None else measured["status"]
            if base is not None and base["seconds"] is not None:
                text += f" ({base['seconds'] * scale:.2f})"
            if measured["status"] != "ok":
                text += f" {measured['status']}"
            if (case["name"], engine) in regressions:
                text += " **regression**"
            cells.append(text)
        lines.append(f"| `{case['name']}` | " + " | ".join(cells) + " |")
    lines.append("")
    if regressions:
        lines += ["**Regressions:**", ""] + [f"- `{name}` on {engine}: {problem}"
                                             for (name, engine), problem in regressions.items()]
    else:
        lines.append("No regressions.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", default=",".join(ENGINES), help="Comma-separated engines (default: all).")
    parser.add_argument("--cases", default="", help="Comma-separated case names (default: all).")
    parser.add_argument("--timeout", type=float, default=60, help="Time limit per case and engine (seconds).")
    parser.add_argument("--threshold", type=float, default=3.0, help="Allowed factor over the scaled baseline.")
    parser.add_argument("--min-seconds", type=float, default=1.0,
                        help="Times below this are never a regression (noise of short runs).")
    parser.add_argument("--output", type=Path, help="Write the results as JSON to this file.")
    parser.add_argument("--summary", type=Path, help="Append the Markdown table to this file "
                                                     "(e.g. $GITHUB_STEP_SUMMARY).")
    parser.add_argument("--write-baseline", action="store_true",
                        help="Replace the baseline by these results (run it on all cases and engines).")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with 1 if there is a regression.")
    args = parser.parse_args(argv)

    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    names = [name for name in args.cases.split(",") if name]
    cases = [case for case in suite["cases"] if not names or case["name"] in names]
    engines = [engine for engine in args.engines.split(",") if engine]
    baseline: Dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}

    calibration = calibrate()
    scale = calibration / baseline["calibration"] if baseline.get("calibration") else 1.0
    print(f"calibration {calibration:.3f} s (scale {scale:.2f})", flush=True)
    results: Dict[str, Dict[str, Any]] = {}
    regressions: Dict[Tuple[str, str], str] = {}
    for case in cases:
        results[case["name"]] = {}
        for engine in engines:
            measured = run_case(case, engine, args.timeout)
            results[case["name"]][engine] = measured
            base = baseline.get("results", {}).get(case["name"], {}).get(engine)
            problem = compare(measured, base, scale, args.threshold, args.min_seconds)
            if problem is not None:
                regressions[case["name"], engine] = problem
            seconds = "-" if measured["seconds"] is None else f"{measured['seconds']:.2f} s"
            print(f"{case['name']:<28} {engine:<10} {measured['status']:<8} {seconds:>9}"
                  f"{'  ' + problem if problem else ''}", flush=True)

    table = markdown(cases, engines, results, baseline, scale, regressions, args.threshold)
    print(table)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(table)
    if args.output is not None:
        args.output.write_text(json.dumps({"calibration": calibration, "results": results}, indent=2) + "\n",
                               encoding="utf-8")
    if args.write_baseline:
        # the whole baseline is replaced: its times only make sense together with its calibration
        stored = {name: {engine: {"status": value["status"], "seconds": value["seconds"]}
                         for engine, value in per_engine.items()} for name, per_engine in results.items()}
        BASELINE.write_text(json.dumps({"calibration": round(calibration, 4), "results": stored}, indent=2) + "\n",
                            encoding="utf-8")
        print(f"Baseline written to {BASELINE.relative_to(ROOT)}")
    return 1 if regressions and args.fail_on_regression else 0


if __name__ == "__main__":
    sys.exit(main())
