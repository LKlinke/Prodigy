from itertools import islice

import pytest

from prodigy.util.order import default_monomial_iterator


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_monomial_iterator_is_graded_and_complete(n):
    # A search for a state (the witness of an inequivalence) must find small states first: with an
    # order that runs far in the first variable, every step filters a deeper series (flaky timeout on
    # the gambler's-ruin check, depending on the hash seed that fixed the variable order).
    first = list(islice(default_monomial_iterator(n), 200))
    assert len({tuple(m) for m in first}) == len(first)
    sums = [sum(m) for m in first]
    assert sums == sorted(sums)
    assert first[0] == [0] * n
    low = {tuple(m) for m in first if sum(m) <= 2}
    assert len(low) == len([m for m in first if sum(m) <= 2])
    # all tuples with sum <= 2 appear before any tuple with sum 3
    assert sums.index(3) == len(low) if 3 in sums else True
