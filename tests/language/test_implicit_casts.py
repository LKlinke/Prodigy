"""
Implicit casts of the type checker: widening ``bool ⊑ nat ⊑ real`` is always allowed, narrowing to
``bool`` only for values that are provably 0 or 1, and parameters of distributions are real.
"""
import glob

import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.language.ast import Program
from prodigy.language.check import CheckFail
from prodigy.language.compiler import compile_program


@pytest.mark.parametrize("code", [
    # bool as 0/1 in comparisons and arithmetic
    """b: bool;
x: nat;
if (b == 1) {
    x = 2;
} else {
    skip;
}""",
    """b: bool;
c: bool;
x: nat;
x = b + c;""",
    """b: bool;
x: nat;
x = b * 3 + 1;""",
    """b: bool;
x: nat;
x = b;""",
    """b: bool;
x: nat;
x = true;""",
    """b: bool;
x: nat;
observe(b < x);""",
    # nat where real is expected
    """c: nat;
t: nat;
t = iid(1 - (1 - c) ** (1/2), c);""",
    """x: nat;
y: real;
y = x;""",
    """param a: real;
t: nat;
t = bernoulli((1 - a) / 2);""",
    """x: nat;
x = poisson(1/2);""",
    """x: nat;
x = bernoulli(1);""",
    # categorical choices take the widest type
    """x: nat;
b: bool;
x = categorical({b: 1/2, 3: 1/2});""",
])
def test_widening_is_accepted(code):
    assert isinstance(compile_program(code), Program)


@pytest.mark.parametrize("code", [
    """b: bool;
b = bernoulli(1/2);""",
    """b: bool;
param p: real;
b = bernoulli(p);""",
    """b: bool;
b = 0;""",
    """b: bool;
b = 1;""",
    """b: bool;
c: bool;
b = c;""",
    """b: bool;
x: nat;
b = x < 3;""",
    """b: bool;
b = unif(0, 1);""",
    """b: bool;
c: bool;
b = categorical({c: 1/3, 1: 2/3});""",
])
def test_provable_narrowing_is_accepted(code):
    assert isinstance(compile_program(code), Program)


@pytest.mark.parametrize("code", [
    """b: bool;
x: nat;
b = x + 1;""",
    """b: bool;
x: nat;
b = x;""",
    """b: bool;
b = 2;""",
    """b: bool;
c: bool;
b = c + 1;""",
    # arithmetic on bools yields nat, it is never reduced back to a bool
    """b: bool;
b1: bool;
b2: bool;
b = b1 + b2;""",
    """b: bool;
b1: bool;
b2: bool;
b = b1 * b2;""",
    # the bounds of a nat are not enforced (`nat x [0,1]; x := 5` type-checks), so they prove nothing
    """b: bool;
x: nat[0, 1];
b = x;""",
    """b: bool;
b = unif(0, 2);""",
    """b: bool;
b = geometric(1/2);""",
    """b: bool;
b = binomial(2, 1/2);""",
    """b: bool;
x: nat;
b = categorical({x: 1/2, 0: 1/2});""",
    # a user-defined function is not the predefined distribution
    """def f(x: nat) -> nat {
    x = 2;
    return x;
}
b: bool;
b = f();""",
])
def test_unprovable_narrowing_is_rejected(code):
    result = compile_program(code)
    assert isinstance(result, CheckFail)
    assert "not provably 0 or 1" in result.message


@pytest.mark.parametrize("code", [
    # real to nat loses values
    """x: nat;
x = 1/2;""",
    """x: nat;
param p: real;
x = p * x;""",
    # conditions stay boolean (a guard that is no bool is rejected by the parser, see
    # tests/language/test_guard_types.py)
    """x: nat;
b: bool;
b = not x;""",
    """x: nat;
b: bool;
b = b and x;""",
])
def test_other_narrowing_is_rejected(code):
    assert isinstance(compile_program(code), CheckFail)


def test_bool_is_zero_or_one_in_the_generating_function():
    """The analysis treats a bool variable as a number that is 0 or 1, as the checker assumes."""
    program = compile_program("""
b: bool;
x: nat;
b = bernoulli(1/3);
x = b + 1;
if (b == 1) {
    x = x + 5;
} else {
    skip;
}""")
    assert isinstance(program, Program)
    config = ForwardAnalysisConfig()
    dist, _ = compute_discrete_distribution(program, config.factory.one('b', 'x'), config)
    expected = config.factory.from_expr("1/3 * b * x^7 + 2/3 * x", 'b', 'x')
    assert dist == expected


@pytest.mark.parametrize("path", sorted(glob.glob("pgfexamples/**/*.ppy", recursive=True))
                         + sorted(glob.glob("tests/language/former_examples/**/*.ppy", recursive=True)))
def test_examples_type_check(path):
    with open(path, encoding="utf-8") as file:
        result = compile_program(file.read(), path=path)
    assert isinstance(result, Program), result
