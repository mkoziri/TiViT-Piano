"""Shared dataclasses used across pipelines.

Purpose:
- Provide typed containers for samples, targets, batches, predictions, and metrics.
- Standardise the data exchanged between datasets, models, and postprocessing.

Key Functions/Classes:
- Sample, Targets, Batch, Predictions, MetricsResult dataclasses.

CLI Arguments:
- (none; import-only utilities).

Usage:
- Import typed containers: ``from tivit.core.types import Batch, Predictions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass
class Sample:
    """Single clip/frame sample with optional path and metadata."""
    video: Any
    path: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Targets:
    """Container for multitask supervision tensors."""
    pitch: Any | None = None
    onset: Any | None = None
    offset: Any | None = None
    hand: Any | None = None
    clef: Any | None = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """Batch wrapper combining samples, optional targets, and metadata."""
    samples: Sequence[Sample]
    targets: Targets | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Predictions:
    """Model outputs, optional probabilities/loss, and auxiliary payloads."""
    logits: Mapping[str, Any]
    probs: Optional[Mapping[str, Any]] = None
    loss: Optional[float] = None
    aux: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MetricsResult:
    """Metrics summary and detail blobs."""
    summary: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["Sample", "Targets", "Batch", "Predictions", "MetricsResult"]
