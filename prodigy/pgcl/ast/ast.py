from abc import ABC
import attr


@attr.s
class Node(ABC):
    """Superclass for all node types in the AST."""
