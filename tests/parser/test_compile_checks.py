import pytest
from lark.exceptions import UnexpectedCharacters

from probably.pgcl.typechecker.check import CheckFail
from probably.pgcl.compiler import compile_pgcl


def test_uniform_checks():
    program = compile_pgcl("nat x; x := unif(13, 6.0);")
    assert isinstance(program, CheckFail)

    with pytest.raises(UnexpectedCharacters) as excinfo:
        compile_pgcl("nat x; x := x + unif(13, 6);")


def test_categorical_confusion():
    res = compile_pgcl("bool x; nat y; x := x : 0.5 + y : 0.5;")
    assert isinstance(res, CheckFail)
    assert res.message == 'Expected value of type BoolType(), got NatType(bounds=None).'

    with pytest.raises(ValueError) as excinfo:
        compile_pgcl("bool x; x := x : 0.9 + x : 0.5;")
    assert "Probabilities need to sum up to 1" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        compile_pgcl("nat x; x := 13 + (1 : 0.5 + 0 : 0.5)")
