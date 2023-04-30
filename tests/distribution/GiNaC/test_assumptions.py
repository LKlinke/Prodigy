from probably.pgcl.parser import parse_expr
from pytest import raises
from sympy import sqrt

from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF, _sympy_symbol


def test_divisible():
    gf = ProdigyPGF.geometric('x', '1/2')
    with raises(ValueError, match='numerator must have a finite marginal'):
        gf.update(parse_expr('x = x / 2'))
    gf = gf.filter(parse_expr('x % 2 = 0'))
    gf._update_division('x', 'x', 2, None)
