"""Optional harmony filtering placeholder."""

from __future__ import annotations

from typing import Mapping

from .base import Constraint


class HarmonyFilterConstraint(Constraint):
    def apply(self, events: Mapping[str, object]) -> Mapping[str, object]:
        return events


def build_constraint(*_: object, **__: object) -> Constraint:
    return HarmonyFilterConstraint()


__all__ = ["HarmonyFilterConstraint", "build_constraint"]

