"""
=================
``prodigy.util``
=================

.. autofunction:: prodigy.util.dedup_list
"""

from typing import List, TypeVar, Collection

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
