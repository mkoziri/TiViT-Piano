"""Dataset-only stats for threshold initialization.

Purpose:
    - Stream dataset labels to estimate densities, positive rates, durations, and polyphony.
    - Summarize key coverage per tile for search-bound constraints.

Key Functions/Classes:
    - SplitStats: Streaming accumulator for per-split statistics.
    - Histogram: Lightweight histogram with percentile estimates.
    - IntHistogram: Integer histogram for polyphony counts.

CLI Arguments:
    (none)

Usage:
    stats = SplitStats(...)
    stats.record_events(events, clip_start=0.0)
    summary = stats.summary()
"""

from __future__ import annotations

import bisect
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def build_log_bins(low: float, high: float, bins: int) -> List[float]:
    """Return log-spaced bin edges from ``low`` to ``high`` (inclusive)."""
    low_val = max(float(low), 1e-6)
    high_val = max(float(high), low_val * 1.001)
    bins = max(int(bins), 1)
    log_low = math.log(low_val)
    log_high = math.log(high_val)
    edges = [math.exp(log_low + (log_high - log_low) * i / bins) for i in range(bins + 1)]
    edges[0] = float(low)
    edges[-1] = float(high)
    return edges


class Histogram:
    """Histogram with fixed bins and approximate percentiles."""

    def __init__(self, edges: Sequence[float]) -> None:
        if len(edges) < 2:
            raise ValueError("Histogram requires at least 2 bin edges")
        self.edges = [float(v) for v in edges]
        self.counts = [0 for _ in range(len(self.edges) - 1)]
        self.underflow = 0
        self.overflow = 0
        self.total = 0

    def add(self, value: float, *, count: int = 1) -> None:
        if count <= 0:
            return
        val = float(value)
        if not math.isfinite(val):
            return
        if val < self.edges[0]:
            self.underflow += count
            self.total += count
            return
        if val >= self.edges[-1]:
            self.overflow += count
            self.total += count
            return
        idx = bisect.bisect_right(self.edges, val) - 1
        idx = max(0, min(idx, len(self.counts) - 1))
        self.counts[idx] += count
        self.total += count

    def percentile(self, pct: float) -> Optional[float]:
        if self.total <= 0:
            return None
        pct = max(0.0, min(float(pct), 100.0))
        target = (pct / 100.0) * self.total
        cumulative = self.underflow
        if cumulative >= target:
            return self.edges[0]
        for idx, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                return self.edges[idx + 1]
        return self.edges[-1]

    def to_dict(self, *, digits: int = 6) -> Dict[str, object]:
        return {
            "bins": [round(float(v), digits) for v in self.edges],
            "counts": list(self.counts),
            "underflow": int(self.underflow),
            "overflow": int(self.overflow),
            "total": int(self.total),
        }


class IntHistogram:
    """Histogram for integer values with percentile support."""

    def __init__(self) -> None:
        self.counts: List[int] = []
        self.total = 0
        self.sum = 0

    def add(self, value: int, *, count: int = 1) -> None:
        if count <= 0:
            return
        val = int(value)
        if val < 0:
            return
        if val >= len(self.counts):
            self.counts.extend([0] * (val + 1 - len(self.counts)))
        self.counts[val] += count
        self.total += count
        self.sum += val * count

    def add_many(self, values: Iterable[int]) -> None:
        for val in values:
            self.add(int(val))

    def percentile(self, pct: float) -> Optional[int]:
        if self.total <= 0:
            return None
        pct = max(0.0, min(float(pct), 100.0))
        target = (pct / 100.0) * self.total
        cumulative = 0
        for idx, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                return idx
        return len(self.counts) - 1 if self.counts else None

    def mean(self) -> Optional[float]:
        if self.total <= 0:
            return None
        return self.sum / self.total

    def to_dict(self) -> Dict[str, object]:
        return {"counts": list(self.counts), "total": int(self.total)}


class SplitStats:
    """Streaming accumulator for dataset-only threshold stats."""

    def __init__(
        self,
        *,
        split: str,
        frames: int,
        stride: int,
        fps: float,
        note_min: int,
        note_max: int,
        num_tiles: int,
        duration_bins: Sequence[float],
        ioi_bins: Sequence[float],
    ) -> None:
        self.split = str(split)
        self.frames = int(frames)
        self.stride = int(stride)
        self.fps = float(fps)
        self.hop_seconds = self.stride / max(self.fps, 1e-6)
        self.duration_sec = self.hop_seconds * max(self.frames - 1, 0)
        self.note_min = int(note_min)
        self.note_max = int(note_max)
        self.n_keys = int(self.note_max - self.note_min + 1)
        self.num_tiles = int(num_tiles)

        self.clip_count = 0
        self.clips_with_events = 0
        self.onset_count = 0
        self.offset_count = 0
        self.offset_in_window = 0
        self.onset_counts_by_key = [0 for _ in range(self.n_keys)]
        self.offset_counts_by_key = [0 for _ in range(self.n_keys)]
        self.offset_in_window_by_key = [0 for _ in range(self.n_keys)]

        self.duration_hist = Histogram(duration_bins)
        self.duration_full_count = 0
        self.duration_truncated_count = 0
        self.ioi_hist = Histogram(ioi_bins)

        self.pos_counts = {"pitch": 0, "onset": 0, "offset": 0}
        self.total_cells = {"pitch": 0, "onset": 0, "offset": 0}
        self.pos_counts_by_key = {
            "pitch": [0 for _ in range(self.n_keys)],
            "onset": [0 for _ in range(self.n_keys)],
            "offset": [0 for _ in range(self.n_keys)],
        }
        self.polyphony_hist = IntHistogram()

        self.coverage_counts = (
            np.zeros((self.num_tiles, self.n_keys), dtype=np.int64) if self.num_tiles > 0 else None
        )
        self.coverage_clips = 0
        self.coverage_fallbacks = 0

    def record_events(self, events: Sequence[Sequence[float]], *, clip_start: float) -> None:
        """Accumulate event-level stats from ``events`` in the clip window."""
        self.clip_count += 1
        if not events:
            return
        self.clips_with_events += 1

        window_end = float(self.duration_sec)
        clip_start = float(clip_start or 0.0)
        onsets_by_key: Dict[int, List[float]] = {}

        for evt in events:
            if not isinstance(evt, (list, tuple)) or len(evt) < 3:
                continue
            try:
                onset = float(evt[0])
                offset = float(evt[1])
                pitch = int(evt[2])
            except (TypeError, ValueError):
                continue
            local_onset = onset - clip_start
            if local_onset < 0.0 or local_onset >= window_end:
                continue
            if pitch < self.note_min or pitch > self.note_max:
                continue

            key_idx = pitch - self.note_min
            self.onset_count += 1
            self.onset_counts_by_key[key_idx] += 1
            onsets_by_key.setdefault(key_idx, []).append(local_onset)

            local_offset = offset - clip_start
            self.offset_count += 1
            self.offset_counts_by_key[key_idx] += 1
            if 0.0 <= local_offset < window_end:
                self.offset_in_window += 1
                self.offset_in_window_by_key[key_idx] += 1
            if local_offset > local_onset:
                duration = max(0.0, min(local_offset, window_end) - local_onset)
                self.duration_hist.add(duration)
                if local_offset <= window_end:
                    self.duration_full_count += 1
                else:
                    self.duration_truncated_count += 1

        for key_idx, onsets in onsets_by_key.items():
            if len(onsets) < 2:
                continue
            onsets.sort()
            prev = onsets[0]
            for current in onsets[1:]:
                delta = current - prev
                if delta > 0.0:
                    self.ioi_hist.add(delta)
                prev = current

    def record_rolls(
        self,
        *,
        pitch_roll: Optional[torch.Tensor],
        onset_roll: Optional[torch.Tensor],
        offset_roll: Optional[torch.Tensor],
    ) -> None:
        """Accumulate frame-target stats from roll tensors."""
        if pitch_roll is None or onset_roll is None or offset_roll is None:
            return
        for head, roll in (("pitch", pitch_roll), ("onset", onset_roll), ("offset", offset_roll)):
            if roll.numel() == 0:
                continue
            self.total_cells[head] += int(roll.numel())
            self.pos_counts[head] += int(torch.count_nonzero(roll).item())
            per_key = roll.sum(dim=0).to(torch.int64).tolist()
            target = self.pos_counts_by_key[head]
            for idx, val in enumerate(per_key):
                target[idx] += int(val)

        polyphony = pitch_roll.sum(dim=1).to(torch.int64)
        self.polyphony_hist.add_many(polyphony.tolist())

    def record_empty_rolls(self) -> None:
        """Accumulate totals for an empty clip without building rolls."""
        total = int(self.frames * self.n_keys)
        for head in self.total_cells:
            self.total_cells[head] += total
        self.polyphony_hist.add(0, count=self.frames)

    def record_tile_mask(self, mask: Optional[np.ndarray], *, registration_based: bool) -> None:
        """Track key visibility coverage per tile."""
        if self.coverage_counts is None or mask is None:
            return
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != self.coverage_counts.shape:
            return
        self.coverage_counts += mask_arr.astype(np.int64)
        self.coverage_clips += 1
        if not registration_based:
            self.coverage_fallbacks += 1

    def _coverage_fraction(self) -> Optional[np.ndarray]:
        if self.coverage_counts is None or self.coverage_clips <= 0:
            return None
        return self.coverage_counts / float(self.coverage_clips)

    def _coverage_distribution(self) -> Optional[np.ndarray]:
        if self.coverage_counts is None or self.coverage_clips <= 0:
            return None
        totals = self.coverage_counts.sum(axis=0)
        denom = np.where(totals > 0, totals, 1)
        return self.coverage_counts / denom

    def _tile_counts_from_keys(self, counts: Sequence[int]) -> Optional[List[float]]:
        fractions = self._coverage_distribution()
        if fractions is None:
            return None
        per_key = np.asarray(counts, dtype=np.float32)
        tile_counts = fractions @ per_key
        return tile_counts.astype(np.float32).tolist()

    def summary(self, *, digits: int = 6) -> Dict[str, object]:
        total_minutes = (self.clip_count * self.duration_sec) / 60.0 if self.clip_count > 0 else 0.0
        onset_per_min = self.onset_count / total_minutes if total_minutes > 0 else 0.0
        offset_per_min = self.offset_in_window / total_minutes if total_minutes > 0 else 0.0
        offset_total_per_min = self.offset_count / total_minutes if total_minutes > 0 else 0.0

        per_key_onset = [
            (count / total_minutes if total_minutes > 0 else 0.0) for count in self.onset_counts_by_key
        ]
        per_key_offset = [
            (count / total_minutes if total_minutes > 0 else 0.0) for count in self.offset_in_window_by_key
        ]
        tile_onset_counts = self._tile_counts_from_keys(self.onset_counts_by_key)
        tile_offset_counts = self._tile_counts_from_keys(self.offset_in_window_by_key)
        per_tile_onset = (
            [(count / total_minutes if total_minutes > 0 else 0.0) for count in tile_onset_counts]
            if tile_onset_counts is not None
            else None
        )
        per_tile_offset = (
            [(count / total_minutes if total_minutes > 0 else 0.0) for count in tile_offset_counts]
            if tile_offset_counts is not None
            else None
        )

        positive_rate = {}
        for head in ("pitch", "onset", "offset"):
            total_cells = self.total_cells.get(head, 0)
            pos_counts = self.pos_counts.get(head, 0)
            positive_rate[head] = pos_counts / total_cells if total_cells > 0 else 0.0

        duration_total = self.duration_full_count + self.duration_truncated_count
        duration_truncated_frac = (
            self.duration_truncated_count / duration_total if duration_total > 0 else 0.0
        )
        ioi_p10 = self.ioi_hist.percentile(10)
        ioi_p50 = self.ioi_hist.percentile(50)

        poly_mean = self.polyphony_hist.mean()
        poly_p50 = self.polyphony_hist.percentile(50)
        poly_p90 = self.polyphony_hist.percentile(90)
        poly_p95 = self.polyphony_hist.percentile(95)

        coverage_fraction = self._coverage_fraction()
        coverage_summary = None
        if coverage_fraction is not None:
            mins = coverage_fraction.min(axis=1)
            means = coverage_fraction.mean(axis=1)
            maxes = coverage_fraction.max(axis=1)
            coverage_summary = {
                "per_tile_min": [round(float(v), digits) for v in mins.tolist()],
                "per_tile_mean": [round(float(v), digits) for v in means.tolist()],
                "per_tile_max": [round(float(v), digits) for v in maxes.tolist()],
                "fallback_fraction": round(
                    (self.coverage_fallbacks / self.coverage_clips) if self.coverage_clips > 0 else 0.0,
                    digits,
                ),
            }

        return {
            "split": self.split,
            "clip_count": int(self.clip_count),
            "clips_with_events": int(self.clips_with_events),
            "frames": int(self.frames),
            "stride": int(self.stride),
            "fps": round(float(self.fps), digits),
            "hop_seconds": round(float(self.hop_seconds), digits),
            "clip_duration_sec": round(float(self.duration_sec), digits),
            "note_min": int(self.note_min),
            "note_max": int(self.note_max),
            "keys": int(self.n_keys),
            "tiles": int(self.num_tiles),
            "event_density": {
                "total_minutes": round(float(total_minutes), digits),
                "onset_per_min": round(float(onset_per_min), digits),
                "offset_per_min": round(float(offset_per_min), digits),
                "offset_total_per_min": round(float(offset_total_per_min), digits),
                "onset_count": int(self.onset_count),
                "offset_in_window_count": int(self.offset_in_window),
                "offset_total_count": int(self.offset_count),
                "per_key": {
                    "onset_per_min": [round(float(v), digits) for v in per_key_onset],
                    "offset_per_min": [round(float(v), digits) for v in per_key_offset],
                },
                "per_tile": {
                    "onset_per_min": [round(float(v), digits) for v in per_tile_onset] if per_tile_onset else None,
                    "offset_per_min": [round(float(v), digits) for v in per_tile_offset] if per_tile_offset else None,
                },
            },
            "positive_rate": {k: round(float(v), digits) for k, v in positive_rate.items()},
            "note_duration": {
                "histogram_sec": self.duration_hist.to_dict(digits=digits),
                "percentiles_sec": {
                    "p10": round(float(self.duration_hist.percentile(10) or 0.0), digits),
                    "p50": round(float(self.duration_hist.percentile(50) or 0.0), digits),
                    "p90": round(float(self.duration_hist.percentile(90) or 0.0), digits),
                    "p95": round(float(self.duration_hist.percentile(95) or 0.0), digits),
                },
                "truncated_fraction": round(float(duration_truncated_frac), digits),
                "total": int(duration_total),
            },
            "inter_onset_interval": {
                "histogram_sec": self.ioi_hist.to_dict(digits=digits),
                "percentiles_sec": {
                    "p10": round(float(ioi_p10 or 0.0), digits),
                    "p50": round(float(ioi_p50 or 0.0), digits),
                },
            },
            "polyphony": {
                "histogram": self.polyphony_hist.to_dict(),
                "mean": round(float(poly_mean or 0.0), digits),
                "p50": int(poly_p50 or 0),
                "p90": int(poly_p90 or 0),
                "p95": int(poly_p95 or 0),
            },
            "key_visibility": {
                "coverage_fraction": coverage_fraction.round(digits).tolist() if coverage_fraction is not None else None,
                "coverage_summary": coverage_summary,
            },
        }


__all__ = ["SplitStats", "Histogram", "IntHistogram", "build_log_bins"]
