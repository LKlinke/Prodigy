"""
The packages of Prodigy form layers (#423): ``prodigy.util`` and ``prodigy.pgcl`` (the language) are below ``prodigy.distribution``
(generating functions and their proof rules), which is below ``prodigy.analysis``. A lower layer imports nothing
from a higher one, also not lazily inside a function: the analysis may use the rules of the distributions, not
the other way round.
"""
import ast
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

import prodigy

ROOT = Path(prodigy.__file__).parent


def _imports(package: str) -> Iterator[Tuple[str, int, str]]:
    """Every module imported by a module of ``package`` (file, line, module), including imports inside
    functions and ``TYPE_CHECKING`` blocks."""
    for path in sorted((ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield str(path.relative_to(ROOT.parent)), node.lineno, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                yield str(path.relative_to(ROOT.parent)), node.lineno, node.module


@pytest.mark.parametrize("package, forbidden", [
    ("distribution", ("prodigy.analysis",)),
    ("pgcl", ("prodigy.analysis",)),
    ("util", ("prodigy.analysis", "prodigy.distribution")),
])
def test_lower_layers_do_not_import_higher_ones(package, forbidden):
    violations = [f"{path}:{line} imports {module}" for path, line, module in _imports(package)
                  if any(module == name or module.startswith(name + ".") for name in forbidden)]
    assert not violations, "\n".join(violations)


def test_relative_imports_are_not_used_to_cross_layers():
    """The check above reads absolute imports; a relative import that leaves its package would escape it."""
    for package in ("distribution", "pgcl", "util"):
        for path in sorted((ROOT / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            depth = len(path.relative_to(ROOT / package).parts)  # level 1 is the own package, level k leaves k - 1
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > depth:
                    raise AssertionError(f"{path}:{node.lineno} leaves prodigy.{package} with a relative import")


def test_importing_the_distributions_loads_no_analysis():
    code = ("import sys, prodigy.distribution, prodigy.pgcl; "
            "print(sorted(m for m in sys.modules if m.startswith('prodigy.analysis')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]", out.stdout
