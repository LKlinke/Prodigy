import pytest
from lark.exceptions import UnexpectedCharacters

from prodigy.language.check import CheckFail
from prodigy.language.compiler import compile_program


def test_uniform_checks():
    program = compile_program("""x: nat;
x = unif(13, 6.0);""")
    assert isinstance(program, CheckFail)


def test_categorical_confusion():
    res = compile_program("""x: bool;
y: nat;
x = categorical({x: 0.5, y: 0.5});""")
    assert isinstance(res, CheckFail)
    assert res.message == 'Expected value of type BoolType(), got NatType(bounds=None) that is not provably 0 or 1.'
