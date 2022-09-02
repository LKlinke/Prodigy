import pygin
import pytest
from probably.pgcl.parser import parse_expr

from prodigy.distribution.fast_generating_function import ProdigyPGF


def test_geometric():
    dist = ProdigyPGF.geometric("x", "1/2")
    print(dist)
    assert True


def test_conditions():
    dist = ProdigyPGF.from_expr("1/2 * x*y + 1/2*y")
    with pytest.raises(SyntaxError):
        dist.filter(parse_expr("y*x < 5"))
