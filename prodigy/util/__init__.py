"""
=================
``prodigy.util``
=================

.. autofunction:: prodigy.util.dedup_list
"""
import itertools
from typing import List, TypeVar, Collection

import sympy

T = TypeVar("T")


def dedup_list(data: List[T]) -> List[T]:
    """
    Deduplicate a list using a set, preserving the ordering of the list.

    .. doctest::

        >>> dedup_list([1,2,3,3])
        [1, 2, 3]
    """
    data_set = set()
    res = []
    for element in data:
        if element not in data_set:
            res.append(element)
            data_set.add(element)
    return res


def make_poly_clause(coef: str, variables: Collection[str], powers: Collection[int]) -> str:
    """
        Generates a polynomial clause _alpha_ * X^a * Y^b * Z^c ...
    """
    c = [coef]
    vp = (['', '{}', '({}^{})'][min(p, 2)].format(v, p) for v, p in zip(variables, powers))
    return '*'.join(c + [s for s in vp if s])


def compute_hadamard_product(f: str, g: str) -> str:
    """
    Computes the Hadamard product of two univariate rational functions, i.e. the point-wise product of the coefficients
    in their formal power series representation.
    TODO: To be moved into the distribution interface.
    """
    s_f, s_g = sympy.S(f), sympy.S(g)
    s_f_num, s_f_denom = s_f.as_numer_denom()
    s_g_num, s_g_denom = s_g.as_numer_denom()
    f_roots = s_f_denom.as_poly().all_roots()
    g_roots = s_g_denom.as_poly().all_roots()
    print(f"Degree f: {len(f_roots)}")
    print(f"Degree g: {len(g_roots)}")
    s_result_denom = sympy.Mul(*[1 - (fr * gr) * sympy.S("x") for fr, gr in itertools.product(f_roots, g_roots)])
    print(f"Denominator: {s_result_denom}")

    s_compm_f = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
        (sympy.S(f"x^{len(f_roots)}") * s_f_denom.subs("x", "1/x")).as_poly().monic()))
    s_compm_g = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
        (sympy.S(f"x^{len(g_roots)}") * s_g_denom.subs("x", "1/x")).as_poly().monic()))

    s_result_denom_2 = sympy.det(
        sympy.eye(len(f_roots) * len(g_roots)) - (sympy.matrices.kronecker_product(s_compm_f, s_compm_g)) * sympy.S(
            "x"))
    print(f"Denominator: {s_result_denom_2}")

    print(f"Comparing {len(f_roots) * len(g_roots)} terms")

    print(s_f.series(n=len(f_roots) * len(g_roots)))
    print(s_g.series(n=len(f_roots) * len(g_roots)))
    f_const, f_rest = s_f.series(n=len(f_roots) * len(g_roots)).removeO().as_coeff_Add()
    g_const, g_rest = s_g.series(n=len(f_roots) * len(g_roots)).removeO().as_coeff_Add()

    f_rest_dict = f_rest.as_coefficients_dict()
    g_rest_dict = g_rest.as_coefficients_dict()
    res = {sympy.S("1"): f_const * g_const}
    for power in range(1, len(f_roots) * len(g_roots) + 1):
        s_power = sympy.S(f"x^{power}")
        if (s_power in g_rest_dict) and (s_power in f_rest_dict):
            res[s_power] = f_rest_dict[s_power] * g_rest_dict[s_power]
        else:
            res[s_power] = 0
    print(res)
    s_num = sympy.Add(*[coef * monom for monom, coef in res.items()]) * s_result_denom_2
    # truncate the polynomial for terms lower than m*n
    s_num = s_num.series(n=len(f_roots) * len(g_roots)).removeO()
    return str(s_num / s_result_denom_2)
