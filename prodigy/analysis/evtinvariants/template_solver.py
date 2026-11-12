"""
Solving the fixed-point system of a rational template N / D (#281).

The EVT operator Ψ(I) = G + ⟦body⟧(⟨I⟩_guard) is linear in I over the coefficients, so the equation
N/D − Ψ(N/D) = 0 is, after clearing denominators, *linear* in the unknown coefficients ``n`` of the numerator
N; only the unknown coefficients ``d`` of the denominator enter non-linearly. Comparing the coefficients of the
monomials in the program variables gives a system ``A(d) · n = b(d)``. A Gröbner basis of this bilinear system
in all unknowns (what :func:`sympy.solve_undetermined_coeffs` computes) can take minutes even for small
templates; here instead

1. ``A(d) · n = b(d)`` is row-reduced over the field of rational functions in ``d`` (and in the program's
   parameters). The rows without a pivot are the consistency conditions ``r(d) = 0``, polynomial equations
   in the few denominator unknowns only;
2. these conditions are solved for ``d`` (without denominator unknowns there are none: the system is linear);
3. for each solution ``d`` the then linear system in ``n`` is solved exactly.

The result is only a proposal: the synthesis checks every solution exactly (``is_evt_fixed_point``), so
a solution missed here (e.g. one for which a pivot chosen in step 1 vanishes) costs completeness, never
soundness.
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import sympy
from sympy.polys.matrices import DomainMatrix

from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Solution = Dict[sympy.Symbol, sympy.Expr]


def _system(equation: sympy.Expr, numerator_unknowns: Sequence[sympy.Symbol],
            variables: Sequence[sympy.Symbol]) -> Optional[DomainMatrix]:
    """The coefficient comparison of ``equation = 0`` in the program ``variables``: the augmented matrix
    ``[A | c]`` with one row per monomial, ``A · n + c = 0``, over the field of fractions of the remaining
    symbols (denominator unknowns, parameters). ``None`` if the equation holds identically."""
    numerator = sympy.numer(sympy.together(equation))
    if numerator == 0:
        return None
    gens = list(variables) + list(numerator_unknowns)
    if not gens:
        gens = [sympy.Dummy("v")]
    poly = sympy.Poly(numerator, *gens)
    field = poly.domain.get_field()
    count, width = len(variables), len(numerator_unknowns)
    rows: Dict[Tuple[int, ...], Dict[int, object]] = {}
    for monomial, coefficient in poly.as_dict(native=True).items():
        powers = monomial[count:count + width]
        degree = sum(powers)
        if degree > 1:
            raise ValueError("The template is not linear in its numerator coefficients.")
        column = powers.index(1) if degree else width
        rows.setdefault(monomial[:count], {})[column] = field.convert_from(coefficient, poly.domain)
    return DomainMatrix(dict(enumerate(rows.values())), (len(rows), width + 1), field)


def _conditions(system: DomainMatrix, denominator_unknowns: Sequence[sympy.Symbol]) -> Optional[List[sympy.Expr]]:
    """The consistency conditions of ``A · n + c = 0`` on the denominator unknowns: ``y · c = 0`` for every
    ``y`` with ``y · A = 0`` (a basis of the left kernel of ``A`` over the field of rational functions).
    ``None`` if the system is inconsistent whatever the denominator."""
    height, width = system.shape
    matrix = system.extract(range(height), range(width - 1))
    constant = system.extract(range(height), [width - 1])
    if width == 1:
        kernel = DomainMatrix.eye(height, system.domain)
    else:
        kernel = matrix.transpose().nullspace()
        if kernel.shape[0] == 0:
            return []
    conditions = []
    for (entry,) in (kernel * constant).to_Matrix().tolist():
        condition = sympy.expand(sympy.numer(sympy.together(entry)))
        if condition == 0:
            continue
        if not condition.free_symbols & set(denominator_unknowns):
            # a non-zero constant (possibly in the program's parameters): no solution
            return None
        conditions.append(condition)
    return conditions


def _solve_linear(system: Optional[DomainMatrix], numerator_unknowns: Sequence[sympy.Symbol]) -> Optional[Solution]:
    """One solution of ``A · n + c = 0`` (free unknowns set to 0), or ``None`` if it has none."""
    if system is None:
        return {n: sympy.S.Zero for n in numerator_unknowns}
    width = len(numerator_unknowns)
    reduced, pivots = system.rref()
    if width in pivots:
        return None
    solution = {n: sympy.S.Zero for n in numerator_unknowns}
    values = reduced.to_Matrix()
    for row, column in enumerate(pivots):
        solution[numerator_unknowns[column]] = -values[row, width]
    return solution


def solve_template(equation: sympy.Expr, numerator_unknowns: Sequence[sympy.Symbol],
                   denominator_unknowns: Sequence[sympy.Symbol], variables: Sequence[sympy.Symbol],
                   parameters: Sequence[sympy.Symbol] = ()) -> Optional[List[Solution]]:
    """Solutions of ``equation = 0`` for all program ``variables`` (identity of rational functions), where
    ``equation`` is linear in ``numerator_unknowns``. The program's ``parameters`` stay symbolic.

    Returns the solutions as dictionaries over all unknowns (an empty list: none), or ``None`` if the
    conditions on the denominator unknowns could not be solved (the template is undecided)."""
    numerator_unknowns = list(numerator_unknowns)
    system = _system(equation, numerator_unknowns, variables)
    if not denominator_unknowns or system is None:
        solution = _solve_linear(system, numerator_unknowns)
        if solution is None:
            return []
        solution.update({d: sympy.S.Zero for d in denominator_unknowns})
        return [solution]

    conditions = _conditions(system, denominator_unknowns)
    if conditions is None:
        return []
    logger.debug("%d condition(s) on the denominator: %s", len(conditions), conditions)
    if conditions:
        try:
            denominator_solutions = sympy.solve(conditions, list(denominator_unknowns), dict=True)
        except NotImplementedError as exc:
            logger.info("Could not solve the denominator conditions %s: %s", conditions, exc)
            return None
    else:
        denominator_solutions = [{}]
    results: List[Solution] = []
    for d_solution in denominator_solutions:
        if any(value.is_real is False for value in d_solution.values()):
            continue  # a generating function has real coefficients
        if not all(value.free_symbols <= set(parameters) for value in d_solution.values()):
            # a family of denominators: take the member with the free unknowns set to 0
            free = set().union(*(value.free_symbols for value in d_solution.values())) & set(denominator_unknowns)
            d_solution = {d: value.subs({f: 0 for f in free}) for d, value in d_solution.items()}
        d_values = {d: d_solution.get(d, sympy.S.Zero) for d in denominator_unknowns}
        solution = _solve_linear(_system(equation.subs(d_values), numerator_unknowns, variables),
                                 numerator_unknowns)
        if solution is not None:
            solution.update(d_values)
            if solution not in results:
                results.append(solution)
    return results
