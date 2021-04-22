"""
------------------
Mutable References
------------------
"""
from typing import (Callable, Dict, Generic, Iterable, List, Tuple, TypeVar,
                    Union)

import attr

M = TypeVar("M")
Key = TypeVar("Key")


@attr.s(repr=False)
class Mut(Generic[M]):
    """
    A mutable reference to a value of type M, represented a getter and a setter.
    Use :attr:`val` to read and write the value.

    Mut is useful to transform an AST while walking it.
    """

    read: Callable[[], M] = attr.ib()
    write: Callable[[M], None] = attr.ib()

    @property
    def val(self) -> M:
        """Property that wraps ``read`` and ``write``."""
        return self.read()

    @val.setter
    def val(self, value: M):
        self.write(value)

    def __repr__(self):
        return f'Mut(val={self.val!r})'

    @staticmethod
    def alloc(value: M) -> "Mut[M]":
        """Create a new Mut in a new anonmyous location."""
        mut = Mut(lambda: value, lambda v: None)

        def setter(new_value: M):
            nonlocal value
            value = new_value

        mut.write = setter
        return mut

    @staticmethod
    def wrap(ref: Union["Mut[M]", M]) -> "Mut[M]":
        """If ``ref`` isn' already a Mut, wrap it with :meth:`alloc`."""
        if isinstance(ref, Mut):
            return ref
        return Mut.alloc(ref)

    @staticmethod
    def list(values: List[M]) -> Iterable["Mut[M]"]:
        """
        Create a Mut for each element in the list.

        .. doctest::

            >>> my_list = [1, 2, 3]
            >>> ref_list = list(Mut.list(my_list))
            >>> ref_list[0].val = 15
            >>> my_list
            [15, 2, 3]
        """
        for i in range(len(values)):

            def getter(ii=i):
                return values[ii]

            def setter(new_value: M, ii=i):
                values[ii] = new_value

            yield Mut(getter, setter)

    @staticmethod
    def dict_items(values: Dict[Key, M]) -> Iterable[Tuple[Key, "Mut[M]"]]:
        """
        Create a Mut for each key-value pair in the dict.

        .. doctest::

            >>> my_dict = {'hello': 42}
            >>> refs = list(Mut.dict_items(my_dict))
            >>> refs[0][1].val = 43
            >>> my_dict
            {'hello': 43}
        """
        for key in values.keys():

            def getter(keyy=key):
                return values[keyy]

            def setter(new_value: M, keyy=key):
                values[keyy] = new_value

            yield (key, Mut(getter, setter))

    @staticmethod
    def dict_values(values: Dict[Key, M]) -> Iterable["Mut[M]"]:
        """
        A Mut reference to each value in the map.
        See :meth:`dict_items`.
        """
        for _, value_ref in Mut.dict_items(values):
            yield value_ref

    @staticmethod
    def iterate(
            reference: Union["Mut[M]", "Mut[List[M]]"]) -> Iterable["Mut[M]"]:
        """
        If this is a reference to a list of elements, apply :meth:`list`.
        Otherwise return only the reference itself.

        .. doctest::

            >>> list(Mut.iterate(Mut.alloc('a')))
            [Mut(val='a')]

            >>> list(Mut.iterate(Mut.alloc(['a', 'b'])))
            [Mut(val='a'), Mut(val='b')]
        """
        if isinstance(reference.val, list):
            yield from Mut.list(reference.val)
        else:
            # mypy can't yet infer the type of reference based on the isinstance check above
            res: "Mut[M]" = reference  # type:ignore
            yield res


Ref = Union[Mut, M]
"""A Ref is either a Mut or a value."""
