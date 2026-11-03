"""
Direct tests for :mod:`prodigy.pgcl.pgcl_operations`, independent of the equivalence check.
"""
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl.ast import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, IfInstr, NatLitExpr,
                              Node, Program, SkipInstr, VarExpr, WhileInstr)
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.pgcl.pgcl_operations import (cav_phi, make_else_if,
                                          state_to_equality_expression)

LOOP = """
    nat x;
    nat c;

    while (x >= 1) {
        {x := 0} [1/2] {c := c + 1}
    }
"""

INVARIANT = """
    nat x;
    nat c;

    if (x >= 1) {
        c := c + 1
        x := 0
    } else {skip}
"""


def _compile(source: str) -> Program:
    prog = compile_pgcl(source)
    assert isinstance(prog, Program)
    return prog


def _walk_ids(instructions) -> set:
    """Ids of all AST nodes (and lists of them) reachable from ``instructions``."""
    seen: set = {id(instructions)}
    stack = list(instructions)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for value in vars(node).values():
            if isinstance(value, list):
                seen.add(id(value))
                stack.extend(value)
            elif isinstance(value, dict):
                stack.extend(v for v in value.values() if isinstance(v, Node))
            elif isinstance(value, Node):
                stack.append(value)
    return seen


def test_cav_phi_structure():
    prog, inv = _compile(LOOP), _compile(INVARIANT)
    loop = prog.instructions[0]
    assert isinstance(loop, WhileInstr)

    result = cav_phi(prog, inv)

    assert len(result.instructions) == 1
    guarded = result.instructions[0]
    assert isinstance(guarded, IfInstr)
    assert guarded.cond == loop.cond
    assert guarded.true == loop.body + inv.instructions
    assert guarded.false == [SkipInstr()]


def test_cav_phi_printed():
    result = cav_phi(_compile(LOOP), _compile(INVARIANT))
    expected = _compile("""
        nat x;
        nat c;

        if (x >= 1) {
            {x := 0} [1/2] {c := c + 1}
            if (x >= 1) {
                c := c + 1
                x := 0
            } else {skip}
        } else {skip}
    """)
    assert str(result) == str(expected)


def test_cav_phi_shares_no_nodes():
    prog, inv = _compile(LOOP), _compile(INVARIANT)
    prog_before, inv_before = str(prog), str(inv)

    result = cav_phi(prog, inv)
    assert not _walk_ids(result.instructions) & _walk_ids(prog.instructions + inv.instructions)

    # Mutating the result must leave both inputs untouched.
    guarded = result.instructions[0]
    assert isinstance(guarded, IfInstr)
    guarded.true.append(AsgnInstr(lhs=VarExpr("c"), rhs=NatLitExpr(42)))
    guarded.true[0].prob = NatLitExpr(1)
    guarded.true[1].true.clear()
    assert str(prog) == prog_before
    assert str(inv) == inv_before


def test_cav_phi_takes_context_from_invariant():
    prog = _compile("""
        nat x;
        nat unused;
        rparam q;

        while (x >= 1) { x := x - 1 }
    """)
    inv = _compile("""
        nat x;
        rparam p;

        if (x >= 1) { x := 0 } else {skip}
    """)

    result = cav_phi(prog, inv)

    assert result.declarations is inv.declarations
    assert result.variables is inv.variables
    assert result.constants is inv.constants
    assert result.parameters is inv.parameters
    assert result.functions is inv.functions
    assert set(result.variables) == {"x"}
    assert set(result.parameters) == {"p"}


def test_cav_phi_keeps_nested_instructions_in_order():
    prog = _compile("""
        nat x;
        nat y;

        while (x >= 1) {
            while (y >= 1) { y := y - 1 }
            if (y = 0) { x := x - 1 } else { skip }
        }
    """)
    inv = _compile("""
        nat x;
        nat y;

        y := 0
        while (x >= 1) { x := x - 1 }
        if (x = 0) { skip } else { y := 1 }
    """)
    loop = prog.instructions[0]
    assert isinstance(loop, WhileInstr)

    guarded = cav_phi(prog, inv).instructions[0]

    assert isinstance(guarded, IfInstr)
    assert [type(instr) for instr in guarded.true] == \
        [WhileInstr, IfInstr, AsgnInstr, WhileInstr, IfInstr]
    assert guarded.true == loop.body + inv.instructions


@pytest.mark.parametrize("source", [
    "nat x; x := 1",
    "nat x; skip",
    "nat x; while (x >= 1) { x := x - 1 } x := 2",
    "nat x; x := 2 while (x >= 1) { x := x - 1 }",
])
def test_cav_phi_rejects_non_loops(source):
    with pytest.raises(ValueError, match="one big loop"):
        cav_phi(_compile(source), _compile(INVARIANT))


def test_cav_phi_rejects_empty_program():
    empty = Program(declarations=[], variables={}, constants={}, parameters={},
                    instructions=[], functions={})
    with pytest.raises(ValueError, match="one big loop"):
        cav_phi(empty, _compile(INVARIANT))


@pytest.mark.parametrize("engine", [
    ForwardAnalysisConfig.Engine.SYMPY,
    ForwardAnalysisConfig.Engine.GINAC,
    ForwardAnalysisConfig.Engine.SYMENGINE,
])
@pytest.mark.parametrize("initial", ["x * c^2", "x^3", "1"])
def test_cav_phi_semantics_match_hand_written_unrolling(engine, initial):
    """Φ(I) and a hand-written unrolling have the same output distribution on concrete inputs."""
    config = ForwardAnalysisConfig(engine=engine)
    phi_inv = cav_phi(_compile(LOOP), _compile(INVARIANT))
    unrolled = _compile("""
        nat x;
        nat c;

        if (x >= 1) {
            {x := 0} [1/2] {c := c + 1}
            if (x >= 1) {
                c := c + 1
                x := 0
            } else {skip}
        } else {skip}
    """)

    dist = config.factory.from_expr(initial, "x", "c")
    result, error = compute_discrete_distribution(phi_inv, dist, config)
    expected, expected_error = compute_discrete_distribution(unrolled, dist, config)

    assert result == expected
    assert error == expected_error


def test_make_else_if():
    branches = [
        (BinopExpr(Binop.EQ, VarExpr("x"), NatLitExpr(0)), [SkipInstr()]),
        (BinopExpr(Binop.EQ, VarExpr("x"), NatLitExpr(1)),
         [AsgnInstr(lhs=VarExpr("x"), rhs=NatLitExpr(5))]),
    ]

    chain = make_else_if(branches)

    assert chain == IfInstr(
        cond=branches[0][0], true=branches[0][1],
        false=[IfInstr(cond=branches[1][0], true=branches[1][1], false=[SkipInstr()])])


def test_make_else_if_single_branch():
    cond = BinopExpr(Binop.LEQ, VarExpr("x"), NatLitExpr(3))
    assert make_else_if([(cond, [SkipInstr()])]) == \
        IfInstr(cond=cond, true=[SkipInstr()], false=[SkipInstr()])


def test_state_to_equality_expression():
    expr = state_to_equality_expression({"x": 1, "y": 2})
    assert expr == BinopExpr(
        Binop.AND,
        BinopExpr(Binop.AND, BoolLitExpr(True),
                  BinopExpr(Binop.EQ, VarExpr("x"), NatLitExpr(1))),
        BinopExpr(Binop.EQ, VarExpr("y"), NatLitExpr(2)))


def test_state_to_equality_expression_empty_state():
    assert state_to_equality_expression({}) == BoolLitExpr(True)
