from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Tuple

from probably.pgcl import Binop, BinopExpr, NatLitExpr, RealLitExpr, VarExpr


class Assumption(ABC):
    @abstractmethod
    def implies(self, other: Assumption) -> bool | None:
        """
        Tries to determine whether this assumptions implies the other one. Returns `None` if this couldn't
        be determined.
        """

    def apply_update(self, updated_var: str,
                     update: BinopExpr | NatLitExpr | VarExpr,
                     *others: Assumption) -> Set[Assumption]:
        """
        Returns the assumptions that hold after the specified update, given that this assumption held before.
        All assumptions given via the `others` parameter are also assumed to have held before the update.

        It is assumed that the given update is simple, i.e., it is either a `VarExpr`, a `NatLitExpr`,
        or a `BinopExpr` combining these two or a `RealLitExpr`.
        """
        if __debug__ and isinstance(update, BinopExpr):
            assert not update.operator.returns_boolean()
            assert isinstance(
                update.lhs, (VarExpr, NatLitExpr, RealLitExpr)) and isinstance(
                    update.rhs, (NatLitExpr, VarExpr, RealLitExpr))
        return self._apply_update(updated_var, update, *others)

    @abstractmethod
    def _apply_update(self, updated_var: str,
                      update: BinopExpr | VarExpr | NatLitExpr,
                      *others: Assumption) -> Set[Assumption]:
        return set()


@dataclass(frozen=True)
class Divisible(Assumption):
    """Represents the assumption `self.lhs % self.rhs = 0`, or that `lhs` is divisible by `rhs`."""
    lhs: str
    rhs: int

    def implies(self, other: Assumption) -> bool | None:
        if not isinstance(other, Divisible):
            return None
        return other.lhs == self.lhs and self.rhs % other.rhs == 0

    def _apply_update(self, updated_var: str,
                      update: BinopExpr | VarExpr | NatLitExpr,
                      *others: Assumption) -> Set[Assumption]:
        if isinstance(update, VarExpr) and update.var == self.lhs:
            if not updated_var == self.lhs:
                return {self, Divisible(updated_var, self.rhs)}
            return {self}

        if updated_var == self.lhs and isinstance(update, BinopExpr):
            if update.operator == Binop.TIMES:
                if isinstance(update.lhs, NatLitExpr) and isinstance(
                        update.rhs, NatLitExpr):
                    return {
                        Divisible(self.lhs,
                                  update.lhs.value * update.rhs.value)
                    }
                if isinstance(update.lhs, NatLitExpr) or isinstance(
                        update.rhs, NatLitExpr):
                    if isinstance(update.lhs, NatLitExpr):
                        nat, var = update.lhs, update.rhs
                    else:
                        nat, var = update.rhs, update.lhs
                    assert isinstance(nat, NatLitExpr) and isinstance(
                        var, VarExpr)
                    if var.var == self.lhs:
                        return {Divisible(self.lhs, self.rhs * nat.value)}
                    res: Set[Assumption] = set()
                    for assumption in others:
                        if isinstance(assumption,
                                      Divisible) and assumption.lhs == var.var:
                            res.add(
                                Divisible(self.lhs,
                                          assumption.rhs * nat.value))
                    return res

                assert isinstance(update.lhs, VarExpr) and isinstance(
                    update.rhs, VarExpr)
                divisors: Tuple[Set[int], Set[int]] = (set(), set())
                for assumption in others:
                    if isinstance(assumption, Divisible):
                        if assumption.lhs == update.lhs.var:
                            divisors[0].add(assumption.rhs * self.rhs)
                        elif assumption.lhs == update.rhs.var:
                            divisors[1].add(assumption.rhs * self.rhs)
                res = set()
                for a, b in itertools.product(*divisors):
                    res.add(Divisible(self.lhs, a * b))
                return res

            # TODO implement update for division and potentially more

        return set()


def from_condition(condition: BinopExpr) -> Set[Assumption]:
    """Creates a list of assumptions given a condition that is known to hold (e.g., inside an if-block or
    while-loop)."""

    op = condition.operator
    if op == Binop.AND:
        res = set()
        if isinstance(condition.lhs, BinopExpr):
            res |= from_condition(condition.lhs)
        if isinstance(condition.rhs, BinopExpr):
            res |= from_condition(condition.rhs)
        return res
    if op == Binop.EQ:
        expr, eq = condition.lhs, condition.rhs
        if isinstance(expr,
                      (VarExpr, NatLitExpr)) and isinstance(eq, BinopExpr):
            expr, eq = eq, expr
        elif not isinstance(eq, (VarExpr, NatLitExpr)) or not isinstance(
                expr, BinopExpr):
            return set()
        if expr.operator == Binop.MODULO and isinstance(
                expr.lhs, VarExpr) and isinstance(
                    expr.rhs, NatLitExpr) and eq == NatLitExpr(0):
            return {Divisible(expr.lhs.var, expr.rhs.value)}
    return set()
