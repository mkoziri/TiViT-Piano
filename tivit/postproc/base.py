"""Post-processing constraint interface."""

from __future__ import annotations

from typing import Mapping


class Constraint:
    def apply(self, events: Mapping[str, object]) -> Mapping[str, object]:
        return events


__all__ = ["Constraint"]
