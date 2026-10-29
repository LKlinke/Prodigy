"""
The example library of the web GUI: the programs in ``pgfexamples/``.

The first line of every example says how it is meant to be run (see
``pgfexamples/README.md``), e.g. ``# check_equality prog.pgcl invariants/prog_invariant.pgcl``.
From that line the library derives a suggested setup for the GUI: an invariant for the loop,
invariant synthesis, or plain analysis.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prodigy.pgcl import parse_pgcl

_CATEGORY_TITLES = {
    "inference": "Inference",
    "equivalence": "Equivalence",
    "invariant_synthesis": "Invariant synthesis",
    "template_parameter_synthesis": "Parameter synthesis",
    "independence": "Independence",
    "loop_free": "loop-free",
    "loopy": "with loops",
    "conditioning": "conditioning",
}


def default_examples_dir() -> Optional[Path]:
    """``pgfexamples`` of the source checkout this package lives in, if present."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pgfexamples"
        if candidate.is_dir():
            return candidate
    return None


def _title(stem: str) -> str:
    text = stem.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


@dataclass
class Example:
    id: str  # pylint: disable=invalid-name
    title: str
    category: List[str]
    command: str
    args: List[str] = field(default_factory=list)
    known_issue: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "category_title": " · ".join(_CATEGORY_TITLES.get(c, c.replace("_", " ")) for c in self.category),
            "command": self.command,
            "known_issue": self.known_issue,
        }


class ExampleLibrary:
    def __init__(self, root: Optional[Union[str, os.PathLike]]):
        self.root = Path(root).resolve() if root else None
        self._examples: Optional[List[Example]] = None

    def _scan(self) -> List[Example]:
        if self.root is None or not self.root.is_dir():
            return []
        examples = []
        for path in sorted(self.root.rglob("*.pgcl")):
            relative = path.relative_to(self.root)
            parts = list(relative.parts[:-1])
            if "invariants" in parts:
                continue
            # second programs of loop-free equivalence pairs
            if "loop_free" in parts and "equivalence" in parts and path.stem.endswith("2"):
                continue
            try:
                first = path.read_text(encoding="utf-8").split("\n", 1)[0].strip()
            except OSError:
                continue
            words = shlex.split(first.lstrip("#").strip()) if first.startswith("#") else []
            command = words[0] if words else "main"
            known_issue = command == "skip" or path.stem.startswith("skip_")
            examples.append(Example(id=relative.as_posix(), title=_title(path.stem.removeprefix("skip_")),
                                    category=parts, command=command, args=words[2:] if len(words) > 2 else [],
                                    known_issue=known_issue))
        return examples

    @property
    def examples(self) -> List[Example]:
        if self._examples is None:
            self._examples = self._scan()
        return self._examples

    def list(self) -> List[Dict[str, Any]]:
        return [example.to_json() for example in self.examples]

    def _resolve(self, example_id: str) -> Path:
        if self.root is None:
            raise FileNotFoundError(example_id)
        path = (self.root / example_id).resolve()
        if self.root not in path.parents or path.suffix != ".pgcl" or not path.is_file():
            raise FileNotFoundError(example_id)
        return path

    def load(self, example_id: str) -> Dict[str, Any]:
        path = self._resolve(example_id)
        source = path.read_text(encoding="utf-8")
        example = next((e for e in self.examples if e.id == example_id), None)
        suggestion = self._suggest(path, source, example)
        return {"id": example_id, "source": source,
                "example": example.to_json() if example else None, "suggestion": suggestion}

    def _resolve_arg(self, arg: str) -> Optional[Path]:
        """Paths in the first line are relative to the repository root (``pgfexamples/...``)."""
        if self.root is None:
            return None
        for base in (self.root, self.root.parent):
            candidate = (base / arg).resolve()
            if candidate.is_file() and self.root in candidate.parents:
                return candidate
        return None

    @staticmethod
    def _numbered_invariants(path: Path, loops: int) -> List[Path]:
        """``invariants/<name>_invariant1.pgcl`` … for programs with several loops (pgfexamples/README.md)."""
        if loops < 2:
            return []
        files = [path.parent / "invariants" / f"{path.stem}_invariant{k}.pgcl" for k in range(1, loops + 1)]
        return files if all(f.is_file() for f in files) else []

    def _suggest(self, path: Path, source: str, example: Optional[Example]) -> Dict[str, Any]:
        suggestion: Dict[str, Any] = {"decisions": {}, "input_distribution": None, "note": ""}
        if example is None:
            return suggestion
        try:
            loops = _count_loops(source)
        except Exception:  # pylint: disable=broad-except
            loops = 0
        numbered = self._numbered_invariants(path, loops) if example.command == "check_equality" else []
        if numbered:
            suggestion["decisions"] = {
                str(i): {"strategy": "invariant", "params": {"invariant": inv.read_text(encoding="utf-8")}}
                for i, inv in enumerate(numbered)}
            suggestion["note"] = (f"Invariants from {', '.join(inv.name for inv in numbered)} (one per loop, "
                                  "in textual order). With the default input every variable is 0, so the loops "
                                  "may not run: set an initial distribution under Input.")
        elif example.command == "check_equality" and example.args and self.root is not None:
            other = self._resolve_arg(example.args[0])
            if other is not None:
                other_source = other.read_text(encoding="utf-8")
                if loops == 1:
                    suggestion["decisions"] = {"0": {"strategy": "invariant",
                                                     "params": {"invariant": other_source}}}
                    suggestion["note"] = (f"Invariant from {other.relative_to(self.root).as_posix()}; it is checked "
                                          "for all inputs. With the default input every variable is 0, so the "
                                          "loop may not run: set an initial distribution under Input.")
                elif loops == 0:
                    suggestion["note"] = ("Equivalence example: the GUI analyses the first program; "
                                          f"the second one is {other.relative_to(self.root).as_posix()}.")
        elif example.command == "invariant_synthesis":
            suggestion["decisions"] = {"*": {"strategy": "evt_synthesis", "params": {}}}
            suggestion["note"] = "Expected-visiting-time invariant synthesis."
            if example.args:
                suggestion["input_distribution"] = example.args[-1] if not example.args[-1].startswith("-") \
                    else None
        elif example.command == "main" and example.args:
            suggestion["input_distribution"] = example.args[0]
        if loops and not suggestion["decisions"]:
            suggestion["decisions"] = {"*": {"strategy": "unroll", "params": {"iterations": 25}}}
        if example.known_issue:
            suggestion["note"] = (suggestion["note"] + " Marked as skipped in the test suite (known issue)."
                                  ).strip()
        return suggestion


def _count_loops(source: str) -> int:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import LoopRegistry
    return len(LoopRegistry(parse_pgcl(source), source))
