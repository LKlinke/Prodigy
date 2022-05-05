"""
-------------------
Translation Context
-------------------

The translation of pGCL programs to pySMT requires a few details to be known:
How to translate variables to pySMT, how to handle infinities, and so on. The
:py:class:`TranslationContext` handles all that.
"""

from typing import Dict, Optional

import attr
from pysmt.fnode import FNode
from pysmt.shortcuts import Symbol
from pysmt.typing import BOOL, INT, REAL

from prodigy.pgcl.ast import BoolType, RealType, NatType, Program, Type, Var


def _translate_type(typ: Type):
    if isinstance(typ, BoolType):
        return BOOL
    elif isinstance(typ, NatType):
        # pylint: disable=fixme
        return INT  # TODO: return bounds?
    elif isinstance(typ, RealType):
        return REAL

    raise Exception("unreachable")


@attr.s(init=False)
class TranslationContext:
    """
    Context used for translation of pGCL parts.
    """

    variables: Dict[Var, FNode] = attr.ib()
    infinity: FNode = attr.ib()
    id_subst: Dict[FNode, FNode] = attr.ib()

    def __init__(self,
                 variables: Dict[Var, FNode],
                 infinity: Optional[FNode] = None,
                 id_subst: Optional[Dict[FNode, FNode]] = None):
        if infinity is None:
            infinity = Symbol("_infinity", REAL)
        if id_subst is None:
            id_subst = {var: var for var in variables.values()}
        self.variables = variables
        self.infinity = infinity
        self.id_subst = id_subst

    @staticmethod
    def from_program(program: Program) -> "TranslationContext":
        """
        Create a new TranslationContext from a pGCL program's declared
        variables.
        """
        variables = {
            var: Symbol(var, _translate_type(typ))
            for var, typ in program.variables.items()
        }
        return TranslationContext(variables)
