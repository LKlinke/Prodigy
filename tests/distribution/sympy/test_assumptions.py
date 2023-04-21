from probably.pgcl.parser import parse_expr
from pytest import raises
from sympy import sqrt

from prodigy.distribution.generating_function import SympyPGF, _sympy_symbol


def test_divisible():
    gf = SympyPGF.geometric('x', '1/2')
    with raises(ValueError, match='because at least one has infinite range'):
        gf.update(parse_expr('x = x / 2'))
    gf = gf.filter(parse_expr('x % 2 = 0'))
    assert gf._update_division('x', 'x', 2,
                               None)._function == gf._function.subs(
                                   _sympy_symbol('x'),
                                   sqrt(_sympy_symbol('x')))
