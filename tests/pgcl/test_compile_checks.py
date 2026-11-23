import pytest
from lark.exceptions import UnexpectedCharacters

from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl


def test_uniform_checks():
    program = compile_pgcl("""x: nat;
x = unif(13, 6.0);""")
    assert isinstance(program, CheckFail)


def test_categorical_confusion():
    res = compile_pgcl("""x: bool;
y: nat;
x = categorical({x: 0.5, y: 0.5});""")
    assert isinstance(res, CheckFail)
    assert res.message == 'Expected value of type BoolType(), got NatType(bounds=None) that is not provably 0 or 1.'
