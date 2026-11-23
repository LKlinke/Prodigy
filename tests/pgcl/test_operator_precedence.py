from decimal import Decimal
from fractions import Fraction

from pytest import mark, raises

from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.pgcl.ast import *
from prodigy.pgcl.ppy_printer import program_to_ppy
from prodigy.pgcl.parser import parse_expr, parse_pgcl
from prodigy.pgcl.ppy_parser import parse_ppy, parse_ppy_expr
from tests.pgcl.legacy import parse_pgcl  # noqa: E402  # the old syntax, see legacy.py


def test_all_operators():
    program = parse_pgcl("""
        const pi := 3.14;
        bool b;
        nat x;
        real r;
        b := true || false & true
        b := 6.0 > pi & not (pi = pi*3)
        n := 1 : 0.5 + 2 : 1/2
        n := 1 + 2 + 3
        n := 1^2^3
        n := 1 % 2^3
    """)
    instr: List[AsgnInstr] = program.instructions

    assert instr[0].rhs == BinopExpr(
        Binop.OR, BoolLitExpr(True),
        BinopExpr(Binop.AND, BoolLitExpr(False), BoolLitExpr(True)))

    assert instr[1].rhs == BinopExpr(Binop.AND, BinopExpr(Binop.GT, RealLitExpr(Decimal(6.0)), VarExpr("pi")),\
         UnopExpr(Unop.NEG, BinopExpr(Binop.EQ, VarExpr("pi"), BinopExpr(Binop.TIMES, VarExpr("pi"), NatLitExpr(3)))))

    exprs = instr[2].rhs.exprs
    assert exprs[0] == (NatLitExpr(1), RealLitExpr(Decimal(0.5)))
    assert exprs[1] == (NatLitExpr(2), RealLitExpr(Fraction(1, 2)))

    assert instr[3].rhs == BinopExpr(
        Binop.PLUS, BinopExpr(Binop.PLUS, NatLitExpr(1), NatLitExpr(2)),
        NatLitExpr(3))

    assert instr[4].rhs == BinopExpr(
        Binop.POWER, NatLitExpr(1),
        BinopExpr(Binop.POWER, NatLitExpr(2), NatLitExpr(3)))

    assert instr[5].rhs == BinopExpr(
        Binop.MODULO, NatLitExpr(1),
        BinopExpr(Binop.POWER, NatLitExpr(2), NatLitExpr(3)))


def test_likely_minus_error():
    with raises(Exception) as e:
        parse_pgcl("""
            nat x
            x := 1 : 0.5 - 2 : 0.5
        """)
    assert "Illegal place for a probability annotation" in str(e)


# --- unary minus (#430) ---------------------------------------------------------------------------
# The unary minus binds tighter than `*`, `/`, `+` and `-` but weaker than the power, in both syntaxes.

_ENV = {"x": Fraction(3), "a": Fraction(5), "b": Fraction(7)}


def _value(expr: Expr) -> Fraction:
    """The value of an arithmetic expression over the variables in ``_ENV``."""
    if isinstance(expr, (NatLitExpr, RealLitExpr)):
        return Fraction(expr.value)
    if isinstance(expr, VarExpr):
        return _ENV[expr.var]
    assert isinstance(expr, BinopExpr), expr
    lhs, rhs = _value(expr.lhs), _value(expr.rhs)
    operations = {Binop.PLUS: lambda: lhs + rhs, Binop.MINUS: lambda: lhs - rhs,
                  Binop.TIMES: lambda: lhs * rhs, Binop.DIVIDE: lambda: lhs / rhs,
                  Binop.POWER: lambda: lhs ** rhs}
    return operations[expr.operator]()


_UMINUS_CASES = [
    # (.pgcl, value with x = 3, a = 5, b = 7)
    ("-1 + 3", 2),
    ("-(1) + 3", 2),
    ("-x + 2*x", 3),
    ("-x^2", -9),
    ("-x^2 + 1", -8),
    ("2 * -x", -6),
    ("-x * 2", -6),
    ("-x / 2", Fraction(-3, 2)),
    ("a - -b", 12),
    ("1 - -1 + 2", 4),
    ("--x", 3),
    ("2^-x", Fraction(1, 8)),
    ("2^-x^2", Fraction(1, 512)),
    ("a * -b + x", -32),
]


@mark.parametrize("pgcl,value", _UMINUS_CASES)
def test_unary_minus_precedence_is_the_same_in_both_syntaxes(pgcl, value):
    parsed = parse_expr(pgcl)
    assert parsed == parse_ppy_expr(pgcl.replace("^", "**"))
    assert _value(parsed) == value


def test_unary_minus_in_a_program_and_its_conversion():
    """A leading minus of a sum is not the minus of the whole sum, and printing as ``.ppy`` keeps
    the meaning (#430)."""
    source = "nat x; nat y; x := 3; y := -x + 2*x + (-1 + 3) * 2"
    program = parse_pgcl(source)
    assert _value(program.instructions[1].rhs) == 7
    printed = program_to_ppy(program)
    assert "y = -x + 2 * x + (-1 + 3) * 2;" in printed
    converted = parse_ppy(printed)
    assert converted == program
    assert _value(converted.instructions[1].rhs) == 7


def test_unary_minus_in_a_query_string():
    """Strings given to the distribution API are read with the .ppy expression parser."""
    dirac = GeneratingFunction("x^3", "x")
    assert dirac.get_expected_value_of("-x + 3") == "0"
    assert dirac.get_probability_of("-x + 2*x == 3") == "1"
