from __future__ import annotations

from typing import Sequence, Tuple, Type, TypeVar, Union

from prodigy.pgcl import Instr

T = TypeVar("T")


def _assume(instruction: Union[Instr, Sequence[Instr]], instr_type: Union[Type[T], Tuple[Type[T], ...]],
            clsname: str) -> T:
    """
    Checks that the instruction is given as the right type and returns it with that type, so
    that the caller can continue with the narrowed type.
    """
    assert isinstance(instruction, instr_type), f"The Instruction handled by a {clsname} must be of type" \
                                                f" {instr_type} got {type(instruction)}"
    return instruction
