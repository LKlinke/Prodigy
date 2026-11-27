"""The names an expression reads (`ast.walk.names_in`, review L7) and the API of `prodigy.language`."""
import pytest

import prodigy.language
from prodigy.language import ast
from prodigy.language.ast import AsgnInstr, Expr, SampleExpr, VarExpr, name_uses, names_in
from prodigy.language.ppy_parser import parse_ppy, parse_ppy_expr


def _value(text: str) -> Expr:
    """``text`` as the whole right-hand side of an assignment (where categorical, sample and infer may stand)."""
    assignment = parse_ppy(f"v = {text};").instructions[0]
    assert isinstance(assignment, AsgnInstr)
    return assignment.rhs


@pytest.mark.parametrize("text, names", [
    ("1 + 2", set()),
    ("x", {"x"}),
    ("x + y * x", {"x", "y"}),
    ("not (a and b) or x < n", {"a", "b", "x", "n"}),
    ("binomial(n, p)", {"n", "p"}),
    ("binomial(n, p = q)", {"n", "q"}),
    ("categorical({x: 1/2, y + 1: 1/2})", {"x", "y"}),
    ("sample(d)", {"d"}),
    ("Pr[d == k] + Ex[e]", {"d", "k", "e"}),
    ("infer(f, x, m = y)", {"x", "y"}),
    ("f(x + 1)", {"x"}),
])
def test_names_in_reads_variables_parameters_constants_and_sampled_dists(text: str, names: set) -> None:
    assert names_in(_value(text)) == names


def test_names_in_takes_several_expressions() -> None:
    assert names_in() == set()
    assert names_in(parse_ppy_expr("x + 1"), _value("sample(d)"), parse_ppy_expr("y")) == {"x", "d", "y"}


def test_name_uses_gives_the_nodes_top_down() -> None:
    uses = list(name_uses(parse_ppy_expr("x + y * x"), _value("sample(d)")))
    assert uses == [VarExpr("x"), VarExpr("y"), VarExpr("x"), SampleExpr("d")]


def test_language_exports_the_ast() -> None:
    assert set(ast.__all__) <= set(prodigy.language.__all__)
    for module in (ast, prodigy.language):
        assert len(module.__all__) == len(set(module.__all__))
        for name in module.__all__:
            assert getattr(module, name) is getattr(ast, name, getattr(module, name)), name
    assert prodigy.language.DistributionType is ast.types.DistributionType
