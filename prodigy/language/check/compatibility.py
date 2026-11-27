"""
The implicit type coercions of the language (:func:`is_compatible`) and the order of the numeric types
``bool ⊑ nat ⊑ real``.
"""
# pylint: disable=line-too-long  # style as taken over from probably
from __future__ import annotations

from typing import Dict

from prodigy.language.ast import BoolType, DistributionType, NatType, RealType, Type


def is_compatible(lhs: Type, rhs: Type) -> bool:
    """
    Check if the right-hand side type is compatible with the left-hand side type.

    This function essentially defines the implicit type coercions of the language.
    Compatibility is not only important for which values can be used as the right-hand side of an assignment, but compatibility
    rules are also checked when operators are used.

    The rules for compatibility are:

    * Natural numbers are compatible with variables of every other type of number (bounded or unbounded).
    * Widening along ``bool ⊑ nat ⊑ real`` never loses a value: a boolean is the number 0 or 1, and a
      natural number is a real number. So ``bool`` is compatible with ``nat`` and ``real``, and ``nat``
      with ``real``.
    * Otherwise the types must be exactly equal for values to be compatible.

    Narrowing (e.g. a natural number where a boolean is expected) is not compatible. Assignments to
    ``bool`` variables additionally accept values that are provably 0 or 1, see
    :func:`~prodigy.language.check.instructions.check_instr`.

    .. doctest::

        >>> from prodigy.language.ast import Bounds, NatLitExpr

        >>> is_compatible(BoolType(), BoolType())
        True
        >>> is_compatible(NatType(bounds=None), NatType(bounds=None))
        True
        >>> is_compatible(NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))), NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))))
        True
        >>> is_compatible(NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))), NatType(bounds=None))
        True

        >>> is_compatible(NatType(bounds=None), BoolType())
        True
        >>> is_compatible(RealType(), NatType(bounds=None))
        True
        >>> is_compatible(BoolType(), NatType(bounds=None))
        False
        >>> is_compatible(NatType(bounds=None), RealType())
        False
    """
    if is_numeric(lhs) and is_numeric(rhs):
        return _NUMERIC_RANK[type(lhs)] >= _NUMERIC_RANK[type(rhs)]
    return lhs == rhs


# the widening order bool ⊑ nat ⊑ real
_NUMERIC_RANK: Dict[type, int] = {BoolType: 0, NatType: 1, RealType: 2}


def is_numeric(typ: Type) -> bool:
    """Whether a value of this type can be used as a number (a bool counts as 0/1)."""
    return type(typ) in _NUMERIC_RANK


def join(*types: Type) -> Type:
    """
    The widest of the given numeric types, i.e., their least upper bound in ``bool ⊑ nat ⊑ real``.
    Bounds of natural numbers are dropped (see :class:`~prodigy.language.ast.NatType`).

    .. doctest::

        >>> join(BoolType(), NatType(bounds=None), RealType())
        RealType()
        >>> join(NatType(bounds=None), BoolType())
        NatType(bounds=None)
    """
    widest = max(types, key=lambda typ: _NUMERIC_RANK[type(typ)])
    if isinstance(widest, NatType):
        return NatType(bounds=None)
    return widest


_TYPE_NAMES = {NatType: "nat", BoolType: "bool", RealType: "real", DistributionType: "dist"}


def type_name(typ: Type) -> str:
    """The name of a type as the programs write it (``nat``, ``bool``, ``real``, ``dist``)."""
    return _TYPE_NAMES.get(type(typ), str(typ))
