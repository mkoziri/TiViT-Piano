"""Shared dataclasses used across pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass
class Sample:
    video: Any
    path: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Targets:
    pitch: Any | None = None
    onset: Any | None = None
    offset: Any | None = None
    hand: Any | None = None
    clef: Any | None = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    samples: Sequence[Sample]
    targets: Targets | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Predictions:
    logits: Mapping[str, Any]
    probs: Optional[Mapping[str, Any]] = None
    loss: Optional[float] = None
    aux: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MetricsResult:
    summary: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["Sample", "Targets", "Batch", "Predictions", "MetricsResult"]

