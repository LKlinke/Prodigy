"""
The dependency rules of ``docs/source/architecture.md`` (#423, #493).

The packages of Prodigy form layers: ``prodigy.util`` below ``prodigy.language`` (the language; ``prodigy.pgcl`` before #446), below ``prodigy.distribution`` (generating functions and their proof rules), below
``prodigy.analysis``, below the front ends (``prodigy.cli``, ``prodigy.web``, ``prodigy.plotting``). A layer
imports nothing from a higher one, also not lazily inside a function or under ``TYPE_CHECKING``; there are no
import cycles; only the analysis core chooses an engine; the front ends use the front-end API of the analysis.

A rule that the code still breaks is an ``xfail(strict=True)`` naming the item of the architecture page
(``T1`` …) and the work package that removes it; that package turns the expected failure into a pass and
removes the marker.
"""
import ast
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple

import pytest

import prodigy

ROOT = Path(prodigy.__file__).parent


class Import(NamedTuple):
    path: str
    line: int
    source: str  # importing module
    target: str  # imported module (absolute)
    runtime: bool  # executed when the code runs, i.e. not only under TYPE_CHECKING


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _type_checking_nodes(tree: ast.Module) -> Set[int]:
    nodes: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            nodes.update(id(inner) for inner in ast.walk(node))
    return nodes


@lru_cache(maxsize=None)
def _imports() -> Tuple[Import, ...]:
    """Every import of every module of ``prodigy``, also inside functions, with relative imports resolved. A
    ``from package import name`` where ``name`` is a submodule counts as an import of that submodule."""
    modules = {_module_name(path): path for path in ROOT.rglob("*.py")}
    found: List[Import] = []
    for module, path in sorted(modules.items()):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        type_checking = _type_checking_nodes(tree)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        where = str(path.relative_to(ROOT.parent))
        for node in ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names if f"{base}.{alias.name}" in modules]
            for target in targets:
                found.append(Import(where, node.lineno, module, target, id(node) not in type_checking))
    return tuple(found)


def _within(module: str, prefixes: Tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _violations(sources: Tuple[str, ...], forbidden: Tuple[str, ...], allowed: Tuple[str, ...] = ()) -> List[str]:
    return [f"{imp.path}:{imp.line} imports {imp.target}" for imp in _imports()
            if _within(imp.source, sources) and _within(imp.target, forbidden) and not _within(imp.target, allowed)]


def _expected(item: str, package: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"{item} of docs/source/architecture.md, removed by {package}")


FRONT_ENDS = ("prodigy.cli", "prodigy.web", "prodigy.plotting")


@pytest.mark.parametrize("package, forbidden", [
    ("prodigy.util", ("prodigy.language", "prodigy.distribution", "prodigy.analysis", "prodigy.simulator") + FRONT_ENDS),
    pytest.param("prodigy.language", ("prodigy.distribution", "prodigy.analysis", "prodigy.simulator") + FRONT_ENDS,
                 marks=_expected("T3 (the type checker imports a proof rule)", "P5a")),
    ("prodigy.distribution", ("prodigy.analysis", "prodigy.simulator") + FRONT_ENDS),
    ("prodigy.simulator", ("prodigy.distribution", "prodigy.analysis") + FRONT_ENDS),
    pytest.param("prodigy.analysis", FRONT_ENDS,
                 marks=_expected("T2/T7 (result imports web.diagnostics, query_handler imports plotting)",
                                 "P5a and P7")),
])
def test_lower_layers_do_not_import_higher_ones(package, forbidden):
    violations = _violations((package,), forbidden)
    assert not violations, "\n".join(violations)


def _cycles(scope: Tuple[str, ...]) -> List[List[str]]:
    """The strongly connected components (more than one module) of the runtime imports between modules in
    ``scope``; an import inside a function counts, as for pylint's ``cyclic-import``."""
    graph: Dict[str, Set[str]] = {}
    for imp in _imports():
        if imp.runtime and imp.source != imp.target and _within(imp.source, scope) and _within(imp.target, scope):
            graph.setdefault(imp.source, set()).add(imp.target)
            graph.setdefault(imp.target, set())
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    stack: List[str] = []
    components: List[List[str]] = []

    def visit(node: str) -> None:  # Tarjan
        index[node] = low[node] = len(index)
        stack.append(node)
        for succ in graph[node]:
            if succ not in index:
                visit(succ)
                low[node] = min(low[node], low[succ])
            elif succ in stack:
                low[node] = min(low[node], index[succ])
        if low[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return components


@pytest.mark.parametrize("scope", [
    ("prodigy.util",),
    ("prodigy.language",),
    ("prodigy.distribution",),
    pytest.param(("prodigy.analysis",), marks=_expected("T1/T4 (session, result, strategies and handlers)", "P5a")),
    ("prodigy.web",),
    pytest.param(("prodigy",), marks=_expected("T1–T4", "P5a")),
])
def test_no_import_cycles(scope):
    cycles = _cycles(scope)
    assert not cycles, "\n".join(" <-> ".join(cycle) for cycle in cycles)


ENGINES = ("prodigy.distribution.generating_function", "prodigy.distribution.fast_generating_function",
           "prodigy.distribution.kernel")
ENGINE_CHOICE = ("prodigy.analysis.config",)


@pytest.mark.parametrize("package", [
    pytest.param("prodigy.analysis", marks=_expected("T5/T6 (while_handler uses SympyPGF, result.sym the SymPy "
                                                     "engine's parser)", "P5a and #479")),
    "prodigy.cli",
    "prodigy.web",
    "prodigy.plotting",
    "prodigy.simulator",
])
def test_only_the_analysis_core_chooses_an_engine(package):
    violations = [v for v in _violations((package,), ENGINES)
                  if not any(v.startswith(choice.replace(".", "/")) for choice in ENGINE_CHOICE)]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("package", [
    "prodigy.analysis", "prodigy.language", "prodigy.util", "prodigy.simulator", "prodigy.cli",
    pytest.param("prodigy.web", marks=_expected("T8 (web/app.py imports prodigy_ginac)", "P7")),
])
def test_only_the_engine_imports_the_bindings(package):
    violations = _violations((package,), ("prodigy_ginac",))
    assert not violations, "\n".join(violations)


# The front-end API of the analysis (architecture page, "Interfaces between the layers").
FRONT_END_API = ("prodigy.analysis", "prodigy.analysis.session", "prodigy.analysis.result",
                 "prodigy.analysis.trace", "prodigy.analysis.config", "prodigy.analysis.loop_strategies",
                 "prodigy.analysis.loops.api", "prodigy.analysis.values",
                 "prodigy.analysis.exceptions", "prodigy.analysis.deprecations", "prodigy.analysis.equivalence")


def _front_end_violations(package: str) -> List[str]:
    return [f"{imp.path}:{imp.line} imports {imp.target}" for imp in _imports()
            if _within(imp.source, (package,)) and _within(imp.target, ("prodigy.analysis",))
            and imp.target not in FRONT_END_API]


@pytest.mark.parametrize("package", [
    pytest.param("prodigy.cli", marks=_expected("T8 (lowerbounds, evtinvariants, program_info, independence)",
                                                "P3 and P7")),
    pytest.param("prodigy.web", marks=_expected("T8 (ai_candidates, joint_grid, heuristics factories)",
                                                "P3 and P7")),
    "prodigy.plotting",
])
def test_front_ends_use_the_front_end_api_of_the_analysis(package):
    violations = _front_end_violations(package)
    assert not violations, "\n".join(violations)


def _no_lines(violations: List[str]) -> Set[str]:
    return {f"{v.split(':')[0]} -> {v.split(' imports ')[1]}" for v in violations}


_API = "prodigy.analysis."
# The violations behind the expected failures above, as of #493. They may only shrink: a new violation of a rule
# that is still an expected failure fails here instead of hiding behind the xfail.
KNOWN_VIOLATIONS: Dict[str, FrozenSet[str]] = {
    "T3 language imports a proof rule": frozenset({
        "prodigy/language/check.py -> prodigy.distribution.finite_measure"}),
    "T2/T7 analysis imports front ends": frozenset({
        "prodigy/analysis/instructionhandler/query_handler.py -> prodigy.plotting.plotter",
        "prodigy/analysis/result.py -> prodigy.web.diagnostics"}),
    "T5/T6 analysis imports an engine": frozenset({
        "prodigy/analysis/instructionhandler/while_handler.py -> prodigy.distribution.generating_function",
        "prodigy/analysis/values.py -> prodigy.distribution.generating_function"}),
    "T8 web imports the bindings": frozenset({"prodigy/web/app.py -> prodigy_ginac"}),
    "T8 CLI uses analysis internals": frozenset(f"prodigy/cli.py -> {_API}{module}" for module in (
        "analyzer", "equivalence.verdict", "evtinvariants.heuristics.positivity.heuristics_factory",
        "evtinvariants.heuristics.strategies", "evtinvariants.heuristics.templates.templates_factory",
        "evtinvariants.invariant_synthesis", "evtinvariants.invariant_verification", "independence.independence",
        "input_distribution", "instructionhandler.program_info", "lowerbounds.parameters", "lowerbounds.problem",
        "lowerbounds.sandwich", "lowerbounds.subinvariant", "solver.solver_type")),
    "T8 web uses analysis internals": frozenset(f"prodigy/web/{path} -> {_API}{module}" for path, module in (
        ("app.py", "evtinvariants.heuristics.strategies"),
        ("app.py", "evtinvariants.heuristics.templates.templates_factory"),
        ("app.py", "joint_grid"), ("diagnostics.py", "nested_inference"), ("grid_cache.py", "joint_grid"),
        ("requests.py", "ai_candidates"), ("requests.py", "analyzer"), ("requests.py", "equivalence.verdict"),
        ("requests.py", "evtinvariants.heuristics.positivity.heuristics_factory"),
        ("requests.py", "evtinvariants.heuristics.strategies"),
        ("requests.py", "evtinvariants.heuristics.templates.templates_factory"),
        ("requests.py", "input_distribution"), ("requests.py", "joint_grid"), ("requests.py", "solver.solver_type"))),
    "T1/T2/T4 modules in import cycles": frozenset(_API + module for module in (
        "analyzer", "equivalence.verdict", "evt_exploration", "evtinvariants.series_identity",
        "instructionhandler.assignment_handler", "instructionhandler.function_handler",
        "instructionhandler.query_handler", "instructionhandler.queryblock_handler",
        "instructionhandler.sample_handler", "instructionhandler.while_handler", "loops.api",
        "lowerbounds.problem", "lowerbounds.subinvariant", "nested_inference", "result", "session", "trace",
        "upper_track", "values")) | {"prodigy.web.diagnostics"},
}


def _current_violations(rule: str) -> Set[str]:
    language = ("prodigy.distribution", "prodigy.analysis", "prodigy.simulator") + FRONT_ENDS
    current = {
        "T3 language imports a proof rule": lambda: _no_lines(_violations(("prodigy.language",), language)),
        "T2/T7 analysis imports front ends": lambda: _no_lines(_violations(("prodigy.analysis",), FRONT_ENDS)),
        "T5/T6 analysis imports an engine": lambda: {
            v for v in _no_lines(_violations(("prodigy.analysis",), ENGINES))
            if not any(v.startswith(choice.replace(".", "/")) for choice in ENGINE_CHOICE)},
        "T8 web imports the bindings": lambda: _no_lines(_violations(("prodigy.web",), ("prodigy_ginac",))),
        "T8 CLI uses analysis internals": lambda: _no_lines(_front_end_violations("prodigy.cli")),
        "T8 web uses analysis internals": lambda: _no_lines(_front_end_violations("prodigy.web")),
        "T1/T2/T4 modules in import cycles": lambda: {m for cycle in _cycles(("prodigy",)) for m in cycle},
    }
    return current[rule]()


@pytest.mark.parametrize("rule", sorted(KNOWN_VIOLATIONS))
def test_known_violations_do_not_grow(rule):
    new = _current_violations(rule) - KNOWN_VIOLATIONS[rule]
    assert not new, "new violations of a rule that is still an expected failure:\n" + "\n".join(sorted(new))


def test_the_import_walk_sees_every_layer():
    """The rules above are vacuous for a package the walk does not see (as ``prodigy.ppy`` was, T17)."""
    sources: FrozenSet[str] = frozenset(imp.source.split(".")[1] for imp in _imports() if "." in imp.source)
    assert {"util", "language", "distribution", "analysis", "web", "cli", "plotting", "simulator"} <= sources


def test_importing_the_distributions_loads_no_analysis():
    code = ("import sys, prodigy.distribution, prodigy.language; "
            "print(sorted(m for m in sys.modules if m.startswith('prodigy.analysis')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]", out.stdout
