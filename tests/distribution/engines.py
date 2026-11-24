"""
The engines of the conformance suite (#361): every test that takes the fixture ``factory`` (see ``conftest.py``)
runs once per engine. GiNaC is skipped where its wheel is missing.

Results are compared as expressions, never as strings: GiNaC's order of terms in ``str()`` changes between runs
(#418), and the engines spell powers differently (``**`` and ``^``).
"""
import re
from typing import Callable, List, Type, Union

import pytest
import sympy

from prodigy.distribution.distribution import CommonDistributionsFactory, Distribution
from prodigy.distribution.equality import decide_equal
from prodigy.distribution.generating_function import SympyPGF

Factory = Type[CommonDistributionsFactory]


def sympy_factory() -> Factory:
    """The factory of the SymPy engine (the default)."""
    return SympyPGF


def ginac_factory() -> Factory:
    """The factory of the GiNaC engine; skips the test if the bindings are not installed."""
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


ENGINES: List[Callable[[], Factory]] = [sympy_factory, ginac_factory]
FACTORIES = [pytest.param(sympy_factory, id="sympy"), pytest.param(ginac_factory, id="ginac")]
"""The engines as parameters: each one a function that returns the factory (or skips)."""


def symbolic(value: Union[Distribution, str, sympy.Expr]) -> sympy.Expr:
    """A distribution, a result string of an engine or an expression as a SymPy expression with plain symbols."""
    if isinstance(value, Distribution):
        value = value._sympy_expression()  # pylint: disable=protected-access
    expression = sympy.sympify(str(value).replace("^", "**")) if isinstance(value, str) else value
    return expression.xreplace({s: sympy.Symbol(s.name) for s in expression.free_symbols})


def same(left: Union[Distribution, str, sympy.Expr], right: Union[Distribution, str, sympy.Expr]) -> bool:
    """Whether the two values are provably the same expression."""
    return decide_equal(symbolic(left), symbolic(right)) is True


def in_common_words(message: str) -> str:
    """An error message without the engine's names of temporary variables (``_0`` for SymPy, ``sym_0`` for
    GiNaC); the analysis replaces them by the terms of the program (#349)."""
    return re.sub(r"\b(?:sym)?_\d+\b", "<temporary>", message)
