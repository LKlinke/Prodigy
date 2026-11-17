from itertools import count
from typing import Iterator, List


def _compositions(total: int, parts: int) -> Iterator[List[int]]:
    """All ways to write ``total`` as an ordered sum of ``parts`` natural numbers."""
    if parts == 1:
        yield [total]
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield [first] + rest


def default_monomial_iterator(n: int) -> Iterator[List[int]]:
    """
    Generates all `n`-tuples of the natural numbers in graded order: first by their sum, so that states with
    small values come first in every dimension, then by the position of the largest entries in the first
    components. Every tuple appears exactly once.
    :param n: Length of tuple
    :return: Iterator of all possible tuples of natural numbers in n dimensions
    """
    if n < 1:
        raise ValueError("n is too small")
    for total in count():
        yield from _compositions(total, n)
