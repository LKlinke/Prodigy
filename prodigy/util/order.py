from typing import Iterator, List
import sympy

def default_monomial_iterator(n: int) -> Iterator[List[int]]:
    """
    Generates all `n`-tuples of the natural numbers, i.e. iterates over all possible pairs of natural
    numbers in n dimensions.
    :param n: Length of tuple
    :return: Iterator of all possible pairs of natural numbers in n dimensions
    """
    if n < 1:
        raise ValueError("n is too small")
    if n == 1:
        num = 0
        while True:
            yield [num]
            num += 1
    else:
        index = 0
        gen = default_monomial_iterator(n - 1)
        vals: list[list[int]] = []
        while True:
            # This is absolutely unreadable, so just another reason to delete this asap
            while len(vals) < index + 1:
                # pylint: disable=stop-iteration-return
                vals.append(next(gen))
                # pylint: enable=stop-iteration-return
            for i in range(index, -1, -1):
                yield [i] + vals[index - i]
            index += 1


def all_coeffs_multivariate(expr, *free):
    x = sympy.IndexedBase('x')
    expr = expr.expand()
    f = expr.free_symbols
    free = set(free) & f if free else f
    if not free:
        return {1: expr}  # XXX {S(1): expr} might be needed?
    pows = [p.as_base_exp() for p in expr.atoms(sympy.Pow, sympy.Symbol)]
    P = {}
    for p, e in pows:
        if p not in free:
            continue
        elif p not in P:
            P[p] = e
        elif e > P[p]:
            P[p] = e
    reps = dict([(f, x[i]) for i, f in enumerate(free)])
    xzero = dict([(v, 0) for k, v in reps.items()])
    e = expr.xreplace(reps);
    reps = {v: k for k, v in reps.items()}
    return dict([(m.xreplace(reps), e.coeff(m).xreplace(xzero) if m != 1 else e.xreplace(xzero)) for m in
                 _monoms(*[P[f] for f in free])])

def _monoms(*o):
    x = sympy.IndexedBase('x')
    f = []
    for i, o in enumerate(o):
        f.append(sympy.Poly([1] * (o + 1), x[i]).as_expr())
    return sympy.Mul(*f).expand().args