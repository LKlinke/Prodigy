"""
The layout of ``prodigy.distribution`` (#511, ``docs/source/architecture.md``): the ``Distribution`` interface and
its values import no generating-function logic, the proof rules import no engine and no layer of ``PGF``, every
engine implements every hook, and the old module paths only re-export their names until the end of v0.7.0.
"""
import ast
import importlib
import inspect

import pytest

from prodigy.distribution.pgf.hooks import PGFHooks
from prodigy.distribution.pgf.pgf import PGF
from tests.distribution.engines import ENGINES
from tests.test_layering import ROOT, _imports, _within

INTERFACE = ("prodigy.distribution.distribution", "prodigy.distribution.values", "prodigy.distribution.errors",
             "prodigy.distribution.terms", "prodigy.distribution.engine_text")
LAYERS = ("prodigy.distribution.pgf.base", "prodigy.distribution.pgf.filtering", "prodigy.distribution.pgf.updating",
          "prodigy.distribution.pgf.pgf")
ENGINE_PACKAGES = ("prodigy.distribution.pgf.sympy", "prodigy.distribution.pgf.ginac")

# old path -> new module (#511)
MOVED = {f"prodigy.distribution.{name}": f"prodigy.distribution.pgf.rules.{name}" for name in (
    "algebraic_comparison", "coefficient_signs", "convergence", "dominant_pole", "equality", "exact_division",
    "exponential_series", "finite_measure", "independent_comparison", "linear_guards", "multisection",
    "negative_values", "ordering", "product_form", "rational_series", "roots_of_unity", "series_transforms")}
MOVED.update({"prodigy.distribution.coefficient_stream": "prodigy.distribution.pgf.coefficient_stream",
              "prodigy.distribution.generating_function": "prodigy.distribution.pgf.sympy.engine",
              "prodigy.distribution.fast_generating_function": "prodigy.distribution.pgf.ginac.engine"})


def _violations(sources, forbidden):
    return [f"{imp.path}:{imp.line} imports {imp.target}" for imp in _imports()
            if _within(imp.source, sources) and _within(imp.target, forbidden)]


def test_the_interface_imports_no_generating_function_logic():
    assert not _violations(INTERFACE, ("prodigy.distribution.pgf",))


def test_the_rules_import_no_engine_and_no_layer():
    assert not _violations(("prodigy.distribution.pgf.rules",), LAYERS + ENGINE_PACKAGES)


def test_no_engine_imports_another_engine():
    for engine in ENGINE_PACKAGES:
        others = tuple(package for package in ENGINE_PACKAGES if package != engine)
        assert not _violations((engine,), others)


def test_the_layers_import_no_engine():
    assert not _violations(LAYERS + ("prodigy.distribution.pgf.hooks",), ENGINE_PACKAGES)


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_implements_every_hook(engine):
    factory = engine()
    assert issubclass(factory, PGF)
    assert not inspect.isabstract(factory), sorted(factory.__abstractmethods__)
    # the engine is its own factory (#511)
    assert factory.factory() is factory
    assert isinstance(factory.from_expr("x", "x"), PGFHooks)


# the distributions of the language, built from the shared table (pgf/sampling.py, #511 P4b)
SAMPLING_FACTORIES = ("geometric", "uniform", "bernoulli", "poisson", "log", "binomial", "negative_binomial")


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_take_iid_and_the_factories_from_the_shared_logic(engine):
    factory = engine()
    for name in SAMPLING_FACTORIES + ("update_iid",):
        assert name not in vars(factory), f"{factory.__name__} defines {name} itself"
        assert getattr(factory, name).__qualname__.startswith(("PGF.", "PGFUpdating.")), name


@pytest.mark.parametrize("old, new", sorted(MOVED.items()))
def test_old_paths_only_re_export(old, new):
    if new.startswith("prodigy.distribution.pgf.ginac"):
        pytest.importorskip("prodigy_ginac")
    module, target = importlib.import_module(old), importlib.import_module(new)
    tree = ast.parse((ROOT.parent / (old.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    statements = [type(node).__name__ for node in tree.body]
    assert set(statements) <= {"Expr", "ImportFrom", "Assign"}, statements
    assert all(isinstance(node.value, ast.Constant) for node in tree.body if isinstance(node, ast.Expr))
    assert all(node.targets[0].id == "__all__" for node in tree.body if isinstance(node, ast.Assign))
    assert module.__all__
    for name in module.__all__:
        # the same object as in the new module (helpers of the SymPy engine come from its expressions module)
        assert not hasattr(target, name) or getattr(module, name) is getattr(target, name), name


def test_deprecated_names_are_the_engines():
    from prodigy.distribution.generating_function import GeneratingFunction, SympyPGF  # pylint: disable=import-outside-toplevel
    assert GeneratingFunction is SympyPGF
    pytest.importorskip("prodigy_ginac")
    from prodigy.distribution.fast_generating_function import FPS, GinacPGF, ProdigyPGF  # pylint: disable=import-outside-toplevel
    assert FPS is GinacPGF and ProdigyPGF is GinacPGF
