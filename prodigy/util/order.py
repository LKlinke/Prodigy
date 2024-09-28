from typing import Iterator, List


def default_monomial_iterator(n: int) -> Iterator[List[int]]:
    """
    Generates all `n`-tuples of the natural numbers, i.e. iterates over all possible pairs of natural
    numbers in n dimensions.
    :param n: Length of tuple
    :return: Iterator of all possible pairs of natural numbers in n dimensions

    Example:
        >>> list(default_monomial_iterator(2))
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], ...]
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
        vals = []
        while True:
            # This is absolutely unreadable, so just another reason to delete this asap
            while len(vals) < index + 1:
                # pylint: disable=stop-iteration-return
                vals.append(next(gen))
                # pylint: enable=stop-iteration-return
            for i in range(index, -1, -1):
                yield [i] + vals[index - i]
            index += 1