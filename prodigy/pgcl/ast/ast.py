from abc import ABC

import attr

Var = str


@attr.s
class Node(ABC):
    """Superclass for all node types in the AST."""
