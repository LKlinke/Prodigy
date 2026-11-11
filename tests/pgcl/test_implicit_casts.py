"""
Implicit casts of the type checker: widening ``bool ⊑ nat ⊑ real`` is always allowed, narrowing to
``bool`` only for values that are provably 0 or 1, and parameters of distributions are real.
"""
import glob

import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl.ast import Program
from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl


@pytest.mark.parametrize("code", [
    # bool as 0/1 in comparisons and arithmetic
    "bool b; nat x; if (b = 1) { x := 2 } else { skip }",
    "bool b; bool c; nat x; x := b + c",
    "bool b; nat x; x := b * 3 + 1",
    "bool b; nat x; x := b",
    "bool b; nat x; x := true",
    "bool b; nat x; observe(b < x)",
    # nat where real is expected
    "nat c; nat t; t := iid(1 - (1 - c)^(1/2), c)",
    "nat x; real y; y := x",
    "rparam a; nat t; t := bernoulli((1 - a) / 2)",
    "nat x; x := poisson(1/2)",
    "nat x; x := bernoulli(1)",
    # categorical choices take the widest type
    "nat x; bool b; x := b : 1/2 + 3 : 1/2",
])
def test_widening_is_accepted(code):
    assert isinstance(compile_pgcl(code), Program)


@pytest.mark.parametrize("code", [
    "bool b; b := bernoulli(1/2)",
    "bool b; rparam p; b := bernoulli(p)",
    "bool b; b := 0",
    "bool b; b := 1",
    "bool b; bool c; b := c",
    "bool b; nat x; b := x < 3",
    "bool b; b := unif(0, 1)",
    "bool b; bool c; b := c : 1/3 + 1 : 2/3",
])
def test_provable_narrowing_is_accepted(code):
    assert isinstance(compile_pgcl(code), Program)


@pytest.mark.parametrize("code", [
    "bool b; nat x; b := x + 1",
    "bool b; nat x; b := x",
    "bool b; b := 2",
    "bool b; bool c; b := c + 1",
    # arithmetic on bools yields nat, it is never reduced back to a bool
    "bool b; bool b1; bool b2; b := b1 + b2",
    "bool b; bool b1; bool b2; b := b1 * b2",
    # the bounds of a nat are not enforced (`nat x [0,1]; x := 5` type-checks), so they prove nothing
    "bool b; nat x [0, 1]; b := x",
    "bool b; b := unif(0, 2)",
    "bool b; b := geometric(1/2)",
    "bool b; b := binomial(2, 1/2)",
    "bool b; nat x; b := x : 1/2 + 0 : 1/2",
    # a user-defined function is not the predefined distribution
    "fun f := { nat x; x := 2; return x; } bool b; b := f()",
])
def test_unprovable_narrowing_is_rejected(code):
    result = compile_pgcl(code)
    assert isinstance(result, CheckFail)
    assert "not provably 0 or 1" in result.message


@pytest.mark.parametrize("code", [
    # real to nat loses values
    "nat x; x := 1/2",
    "nat x; rparam p; x := p * x",
    # conditions stay boolean (a guard that is no bool is rejected by the parser, see
    # tests/pgcl/test_guard_types.py)
    "nat x; bool b; b := not x",
    "nat x; bool b; b := b & x",
])
def test_other_narrowing_is_rejected(code):
    assert isinstance(compile_pgcl(code), CheckFail)


def test_bool_is_zero_or_one_in_the_generating_function():
    """The analysis treats a bool variable as a number that is 0 or 1, as the checker assumes."""
    program = compile_pgcl("""
        bool b
        nat x
        b := bernoulli(1/3)
        x := b + 1
        if (b = 1) { x := x + 5 } else { skip }
    """)
    assert isinstance(program, Program)
    config = ForwardAnalysisConfig()
    dist, _ = compute_discrete_distribution(program, config.factory.one('b', 'x'), config)
    expected = config.factory.from_expr("1/3 * b * x^7 + 2/3 * x", 'b', 'x')
    assert dist == expected


# Examples the strict checker rejects on purpose, because the value assigned is not provably
# of the variable's type (the analysis itself does not run the checker).
KNOWN_UNTYPED_EXAMPLES = {
    # `x := 1/2 * x` assigns a real to a nat; it is a natural number only under the guard x%2 = 0
    "pgfexamples/inference/loop_free/prob_collatz.ppy",
    # `flag := flag + 1` assigns a nat to a bool; it is 1 only under the loop guard flag = 0
    "pgfexamples/invariant_synthesis/fast_dice_roller.ppy",
}


@pytest.mark.parametrize("path", sorted(glob.glob("pgfexamples/**/*.ppy", recursive=True))
                         + sorted(glob.glob("tests/pgcl/former_examples/**/*.pgcl", recursive=True)))
def test_examples_type_check(path):
    with open(path, encoding="utf-8") as file:
        result = compile_pgcl(file.read(), path=path)
    path = path.replace("tests/pgcl/former_examples/", "pgfexamples/").replace(".pgcl", ".ppy")
    if path in KNOWN_UNTYPED_EXAMPLES:
        assert isinstance(result, CheckFail)
    else:
        assert isinstance(result, Program), result
