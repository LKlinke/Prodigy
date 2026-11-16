"""
Compares two benchmark results (the JSON of ``benchmarks/run.py --output``) and prints a Markdown table.

A case on an engine is slower if it takes more than ``--factor`` times the base time and more than
``--min-seconds`` (shorter times are noise). Cases that exist on one side only, and cases whose status
changed (e.g. a time-out), are listed but never counted as slower, except that a status change to a
non-``ok`` status is reported as a problem. The script always exits with 0 unless ``--fail-on-slower`` is
given, so it can be used in a report-only CI job.

    uv run python benchmarks/compare.py BASE.json HEAD.json [--summary FILE]
    make bench-compare BASE=base.json HEAD=head.json

See docs/source/testing.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Entry = Dict[str, Any]
Results = Dict[str, Dict[str, Entry]]


def classify(base: Optional[Entry], head: Optional[Entry], factor: float, min_seconds: float) -> Tuple[str, str]:
    """The verdict (``slower``, ``status``, ``ok``, ``new``, ``removed`` or ``n/a``) and a short description."""
    if base is None and head is None:
        return "n/a", "-"
    if base is None:
        return "new", "new"
    if head is None:
        return "removed", "removed"
    if head["status"] != base["status"]:
        if head["status"] != "ok":
            return "status", f"{base['status']} → {head['status']}"
        return "ok", f"{base['status']} → ok"
    if base["seconds"] is None or head["seconds"] is None:
        return "n/a", "-"
    ratio = head["seconds"] / base["seconds"] if base["seconds"] > 0 else float("inf")
    text = f"{base['seconds']:.2f} → {head['seconds']:.2f} ({ratio:.1f}×)"
    if head["seconds"] > max(factor * base["seconds"], min_seconds):
        return "slower", text
    return "ok", text


def compare(base: Dict[str, Any], head: Dict[str, Any], factor: float = 2.0,
            min_seconds: float = 0.5) -> Tuple[str, List[Tuple[str, str, str]]]:
    """The Markdown report and the list of (case, engine, description) that are slower or changed status."""
    base_results: Results = base.get("results", {})
    head_results: Results = head.get("results", {})
    cases = list(head_results) + [name for name in base_results if name not in head_results]
    engines: List[str] = []
    for results in (head_results, base_results):
        for per_engine in results.values():
            engines += [engine for engine in per_engine if engine not in engines]
    problems: List[Tuple[str, str, str]] = []
    lines = ["## Benchmarks against the base", "",
             f"Analysis time in seconds, base → head (ratio). Slower means more than {factor:g}× the base time "
             f"and more than {min_seconds:g} s. Both runs were measured on the same runner, "
             f"but shared runners are noisy: a single flagged case is a hint, not a proof.", "",
             "| Program | " + " | ".join(engines) + " |", "|---|" + "---|" * len(engines)]
    for name in cases:
        cells = []
        for engine in engines:
            verdict, text = classify(base_results.get(name, {}).get(engine), head_results.get(name, {}).get(engine),
                                     factor, min_seconds)
            if verdict in ("slower", "status"):
                problems.append((name, engine, text))
                text += " **slower**" if verdict == "slower" else " **status**"
            cells.append(text)
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("")
    if problems:
        lines += ["**Slower than the base:**", ""] + [f"- `{name}` on {engine}: {text}"
                                                      for name, engine, text in problems]
    else:
        lines.append("Nothing is slower than the base.")
    return "\n".join(lines) + "\n", problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="Results (JSON) of the base.")
    parser.add_argument("head", type=Path, help="Results (JSON) of the head.")
    parser.add_argument("--factor", type=float, default=2.0, help="Allowed factor over the base time (default 2).")
    parser.add_argument("--min-seconds", type=float, default=0.5,
                        help="Head times below this are never slower (noise of short runs, default 0.5).")
    parser.add_argument("--summary", type=Path, help="Append the table to this file (e.g. $GITHUB_STEP_SUMMARY).")
    parser.add_argument("--fail-on-slower", action="store_true", help="Exit with 1 if a case is slower.")
    args = parser.parse_args(argv)
    try:
        base = json.loads(args.base.read_text(encoding="utf-8"))
        head = json.loads(args.head.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Cannot read the results: {error}", file=sys.stderr)
        return 2
    table, problems = compare(base, head, args.factor, args.min_seconds)
    print(table)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(table)
    return 1 if problems and args.fail_on_slower else 0


if __name__ == "__main__":
    sys.exit(main())
