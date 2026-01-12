"""Purpose:
    Grid-aware snapping of onset events using a lightweight tempo tracker.

Key Functions/Classes:
    - SnapOptions: Normalized configuration bundle for the snap pass.
    - TempoEstimate: Result of the IOI-based tempo search used downstream.
    - estimate_tempo(): Score candidate BPMs by IOI agreement and report
      the best tempo along with a confidence score.
    - apply(): Mutate onset events by blending them toward the inferred grid
      while respecting confidence thresholds and displacement guards.

CLI:
    Not a standalone entry-point; invoked from decoder post-processing flows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from . import DecoderEventSet, NoteEvent

_MIN_CONFIDENCE = 0.6
_GRID_DIVISIONS = 4  # 16th-note resolution by default
_MAX_IOI_EVENTS = 2048
_SIGMA = 0.3  # histogram tolerance (in beat multiples)


@dataclass
class SnapOptions:
    enabled: bool = False
    bpm_min: float = 40.0
    bpm_max: float = 240.0
    bpm_step: float = 1.0
    local_window_beats: int = 4
    max_dev_ms: float = 35.0
    blend: float = 0.5
    min_support_notes: int = 12
    confidence_floor: float = 0.15


@dataclass
class TempoEstimate:
    bpm: float | None
    confidence: float
    scores: List[Tuple[float, float]]
    support: int
    reason: str | None = None

    def ok(self) -> bool:
        return self.bpm is not None and self.confidence > 0.0


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    x = start
    while x <= stop + 1e-9:
        yield round(x, 6)
        x += step


def _event_subset(events: List[NoteEvent], max_events: int) -> List[NoteEvent]:
    if len(events) <= max_events:
        return events
    # Keep the strongest events (confidence desc) for tempo estimation.
    ordered = sorted(events, key=lambda ev: ev.confidence, reverse=True)
    return ordered[:max_events]


def _build_intervals(events: List[NoteEvent], max_interval_ms: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if not events:
        return intervals
    sorted_events = sorted(events, key=lambda ev: ev.onset_ms)
    n = len(sorted_events)
    for i in range(n):
        base_event = sorted_events[i]
        for j in range(i + 1, n):
            delta = sorted_events[j].onset_ms - base_event.onset_ms
            if delta <= 0:
                continue
            if delta > max_interval_ms:
                break
            weight = 0.5 * (base_event.confidence + sorted_events[j].confidence)
            intervals.append((delta, max(weight, 1e-3)))
    return intervals


def estimate_tempo(events: List[NoteEvent], options: SnapOptions) -> TempoEstimate:
    if not events:
        return TempoEstimate(None, 0.0, [], 0, reason="no_events")
    filtered = [ev for ev in events if ev.confidence >= options.confidence_floor]
    support = len(filtered)
    if support < max(1, options.min_support_notes):
        return TempoEstimate(None, 0.0, [], support, reason="insufficient_support")
    subset = _event_subset(filtered, _MAX_IOI_EVENTS)
    max_interval_ms = (60_000.0 / max(options.bpm_min, 1.0)) * max(options.local_window_beats, 1)
    intervals = _build_intervals(subset, max_interval_ms)
    if not intervals:
        return TempoEstimate(None, 0.0, [], support, reason="no_intervals")
    scores: List[Tuple[float, float]] = []
    total_weight = max(sum(weight for _, weight in intervals), 1e-6)
    best_score = -float("inf")
    best_bpm: float | None = None
    for bpm in _frange(options.bpm_min, options.bpm_max, options.bpm_step):
        beat_ms = 60_000.0 / max(bpm, 1e-6)
        sigma = max(_SIGMA, options.bpm_step * 0.5 / 60.0)
        var = sigma * sigma
        score = 0.0
        for delta, weight in intervals:
            approx = delta / beat_ms
            nearest = round(approx)
            if nearest < 1 or nearest > options.local_window_beats:
                continue
            error = approx - nearest
            score += weight * math.exp(-0.5 * (error * error) / max(var, 1e-6))
        scores.append((bpm, score))
        if score > best_score:
            best_score = score
            best_bpm = bpm
    confidence = max(0.0, best_score) / total_weight if total_weight > 0 else 0.0
    return TempoEstimate(best_bpm, confidence, scores, support)


def _alpha(conf: float, floor: float, blend: float) -> float:
    if conf <= floor:
        return 0.0
    return min(blend, max(0.0, 2.0 * (conf - floor)))


def apply(event_set: DecoderEventSet, cfg: Mapping[str, Any]) -> DecoderEventSet:
    options = _parse_options(cfg)
    if not options.enabled:
        return event_set
    ms_per_frame = 1000.0 / max(event_set.fps, 1e-6)
    candidate_events = [ev for ev in event_set.events if ev.confidence >= options.confidence_floor]
    tempo = estimate_tempo(candidate_events, options)
    summary: Dict[str, Any] = {
        "enabled": True,
        "tempo": {
            "bpm": tempo.bpm,
            "confidence": tempo.confidence,
            "support": tempo.support,
            "reason": tempo.reason,
            "curve": tempo.scores,
        },
        "max_dev_ms": options.max_dev_ms,
        "blend": options.blend,
    }
    if not tempo.ok() or tempo.confidence < _MIN_CONFIDENCE:
        summary["applied"] = False
        event_set.stats["snap"] = summary
        return event_set
    assert tempo.bpm is not None
    beat_ms = 60_000.0 / tempo.bpm
    grid_ms = beat_ms / _GRID_DIVISIONS
    total = len(event_set.events)
    shifted = 0
    abs_shift = 0.0
    per_key: Dict[int, Dict[str, float]] = {}
    for ev in event_set.events:
        alpha = _alpha(ev.confidence, options.confidence_floor, options.blend)
        if alpha <= 1e-6:
            continue
        grid_idx = round(ev.onset_ms / grid_ms)
        target_ms = grid_idx * grid_ms
        delta = target_ms - ev.onset_ms
        delta = max(-options.max_dev_ms, min(options.max_dev_ms, delta))
        shift = alpha * delta
        if abs(shift) <= 1e-3:
            continue
        shifted += 1
        abs_shift += abs(shift)
        ev.onset_ms += shift
        ev.onset_frame = int(round(ev.onset_ms / ms_per_frame))
        ev.clamp_frames(event_set.frames)
        stats = per_key.setdefault(ev.key, {"sum": 0.0, "abs": 0.0, "count": 0})
        stats["sum"] += shift
        stats["abs"] += abs(shift)
        stats["count"] += 1
    if shifted:
        event_set.rebuild_mask()
    for key, payload in per_key.items():
        count = max(payload["count"], 1)
        payload["mean_shift_ms"] = payload["sum"] / count
        payload["mean_abs_shift_ms"] = payload["abs"] / count
        payload.pop("sum", None)
        payload.pop("abs", None)
    summary.update(
        {
            "applied": bool(shifted),
            "shifted_pct": shifted / total if total else 0.0,
            "mean_abs_shift_ms": abs_shift / shifted if shifted else 0.0,
            "per_key": per_key,
            "grid_ms": grid_ms,
            "events_total": total,
            "events_shifted": shifted,
        }
    )
    event_set.stats["snap"] = summary
    return event_set


def _parse_options(cfg: Mapping[str, Any] | None) -> SnapOptions:
    if not isinstance(cfg, Mapping):
        return SnapOptions()
    return SnapOptions(
        enabled=bool(cfg.get("enabled", False)),
        bpm_min=float(cfg.get("bpm_min", 40.0)),
        bpm_max=float(cfg.get("bpm_max", 240.0)),
        bpm_step=float(cfg.get("bpm_step", 1.0)),
        local_window_beats=int(cfg.get("local_window_beats", 4)),
        max_dev_ms=float(cfg.get("max_dev_ms", 35.0)),
        blend=float(cfg.get("blend", 0.5)),
        min_support_notes=int(cfg.get("min_support_notes", 12)),
        confidence_floor=float(cfg.get("confidence_floor", 0.15)),
    )

__all__ = [
    "SnapOptions",
    "TempoEstimate",
    "estimate_tempo",
    "apply",
]
