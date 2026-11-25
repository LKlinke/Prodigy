"""
The example library of the web GUI: the programs in ``pgfexamples/``.

The first line of every example says how it is meant to be run (see
``pgfexamples/README.md``), e.g. ``# check_equality prog.ppy invariants/prog_invariant.ppy``.
From that line the library derives a suggested setup for the GUI: an invariant for the loop,
invariant synthesis, or plain analysis. For ``check_equality`` it also loads the second program
(``other``), which the GUI compares with the first.

The examples are ``.ppy`` files; ``.pgcl`` files in a folder given with ``prodigy web --examples``
are not listed (the syntax was removed in v0.6.0).
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prodigy.language import parse_program

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

EXTENSIONS = (".ppy",)
"""File extensions of example programs."""


def default_examples_dir() -> Optional[Path]:
    """
    ``pgfexamples`` of the source checkout this package lives in, or the copy inside the
    installed package (``prodigy/pgfexamples``, see ``pyproject.toml``), if present.
    """
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
        for path in sorted(p for p in self.root.rglob("*") if p.suffix in EXTENSIONS):
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
            words = _command_words(first)
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
        if self.root not in path.parents or path.suffix not in EXTENSIONS or not path.is_file():
            raise FileNotFoundError(example_id)
        return path

    def load(self, example_id: str) -> Dict[str, Any]:
        """Source, example data and suggested setup of an example. Raises ``FileNotFoundError`` for an
        unknown example."""
        path = self._resolve(example_id)
        source = path.read_text(encoding="utf-8")
        file_syntax = "ppy"
        example = next((e for e in self.examples if e.id == example_id), None)
        suggestion = self._suggest(path, source, example, file_syntax)
        data: Dict[str, Any] = {"id": example_id, "syntax": file_syntax, "source": source,
                                "example": example.to_json() if example else None, "suggestion": suggestion,
                                "file_name": path.name}
        other = self._other_program(example)
        if other is not None:
            data["other"] = other
        if data["syntax"] == "ppy" and suggestion["input_distribution"]:
            suggestion["input_distribution"] = _to_ppy_expression(suggestion["input_distribution"])
        return data

    def _other_program(self, example: Optional[Example]) -> Optional[Dict[str, Any]]:
        """The second program of an equivalence example (``# check_equality prog other``): the GUI
        loads it into its second editor and compares the two."""
        if example is None or example.command != "check_equality" or not example.args or self.root is None:
            return None
        other = self._resolve_arg(example.args[0])
        if other is None or other.suffix not in EXTENSIONS:
            return None
        return {"id": other.relative_to(self.root).as_posix(), "source": other.read_text(encoding="utf-8"),
                "syntax": "ppy", "file_name": other.name}

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
        """``invariants/<name>_invariant1.ppy`` … for programs with several loops (pgfexamples/README.md)."""
        if loops < 2:
            return []
        files = [path.parent / "invariants" / f"{path.stem}_invariant{k}{path.suffix}" for k in range(1, loops + 1)]
        return files if all(f.is_file() for f in files) else []

    def _suggest(self, path: Path, source: str, example: Optional[Example],
                 syntax: str = "ppy") -> Dict[str, Any]:
        suggestion: Dict[str, Any] = {"decisions": {}, "input_distribution": None, "note": ""}
        if example is None:
            return suggestion
        try:
            loops = _count_loops(source, syntax)
        except Exception:  # pylint: disable=broad-except
            loops = 0
        numbered = self._numbered_invariants(path, loops) if example.command == "check_equality" else []
        if numbered:
            suggestion["decisions"] = {
                str(i): {"strategy": "invariant", "params": {"invariant": inv.read_text(encoding="utf-8")}}
                for i, inv in enumerate(numbered)}
            suggestion["note"] = (f"Equivalence example: both programs are loaded, and Compare checks them for every "
                                  f"input. The loops use the invariants {', '.join(inv.name for inv in numbered)} "
                                  "(one per loop, in textual order).")
        elif example.command == "check_equality" and example.args and self.root is not None:
            other = self._resolve_arg(example.args[0])
            if other is not None:
                other_source = other.read_text(encoding="utf-8")
                name = other.relative_to(self.root).as_posix()
                if loops == 1:
                    suggestion["decisions"] = {"0": {"strategy": "invariant",
                                                     "params": {"invariant": other_source}}}
                    suggestion["note"] = (f"Equivalence example: both programs are loaded ({name} is the second), "
                                          "and Compare checks them for every input. The loop uses the second "
                                          "program as its invariant.")
                elif loops == 0:
                    suggestion["note"] = (f"Equivalence example: both programs are loaded ({name} is the second), "
                                          "and Compare checks them for every input.")
        elif example.command == "invariant_synthesis":
            suggestion["decisions"] = {"*": {"strategy": "evt_synthesis", "params": {}}}
            suggestion["note"] = "Expected-visiting-time invariant synthesis."
            if example.args:
                suggestion["input_distribution"] = example.args[-1] if not example.args[-1].startswith("-") \
                    else None
        elif example.command == "evt_sandwich":
            suggestion["decisions"] = {"*": {"strategy": "evt_sandwich", "params": {}}}
            suggestion["note"] = "Sandwich: a lower and an upper bound from EVT sub- and super-invariants."
        elif example.command == "main" and example.args:
            suggestion["input_distribution"] = example.args[0]
        if loops and not suggestion["decisions"]:
            suggestion["decisions"] = {"*": {"strategy": "unroll", "params": {"iterations": 25}}}
        if example.known_issue:
            suggestion["note"] = (suggestion["note"] + " Marked as skipped in the test suite (known issue)."
                                  ).strip()
        return suggestion


_COMMANDS = {"main", "skip", "check_equality", "invariant_synthesis", "independent_vars", "evt_sandwich"}


def _command_words(first_line: str) -> List[str]:
    """The command in the first line of an example (``# check_equality prog.ppy inv.ppy``), or no
    words if the first line is an ordinary comment."""
    if not first_line.startswith("#"):
        return []
    text = first_line.lstrip("#").strip()
    if text.split(" ", 1)[0] not in _COMMANDS:
        return []
    try:
        return shlex.split(text)
    except ValueError:  # e.g. an unmatched quote
        return text.split()


def _count_loops(source: str, syntax: str) -> int:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import LoopRegistry
    return len(LoopRegistry(parse_program(source, syntax), source, syntax))


def _to_ppy_expression(expression: str) -> str:
    return expression.replace("^", "**")
