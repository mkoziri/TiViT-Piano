"""Inference-time hand gate placeholder."""

from __future__ import annotations

from typing import Mapping

from .base import Constraint


class HandGateConstraint(Constraint):
    def apply(self, events: Mapping[str, object]) -> Mapping[str, object]:
        # No-op placeholder; actual gating handled in decoder utilities.
        return events


def build_constraint(*_: object, **__: object) -> Constraint:
    return HandGateConstraint()


__all__ = ["HandGateConstraint", "build_constraint"]

