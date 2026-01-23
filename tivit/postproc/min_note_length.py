"""Remove too-short notes placeholder."""

from __future__ import annotations

from typing import Mapping

from .base import Constraint


class MinNoteLengthConstraint(Constraint):
    def __init__(self, min_frames: int = 1) -> None:
        self.min_frames = max(1, int(min_frames))

    def apply(self, events: Mapping[str, object]) -> Mapping[str, object]:
        return events


def build_constraint(cfg: Mapping[str, object] | None = None) -> Constraint:
    cfg = cfg or {}
    return MinNoteLengthConstraint(min_frames=int(cfg.get("min_frames", 1)))


__all__ = ["MinNoteLengthConstraint", "build_constraint"]

