"""Purpose:
    Duration-aware decoding via a tiny two-state DP/Viterbi smoother.

Key Functions/Classes:
    - DPOptions: Configuration bundle for the duration DP stage.
    - apply(): Run the duration DP over each pitch track, enforcing minimum
      ON/OFF dwell times and reporting per-key diagnostics.

CLI:
    Not a standalone module; invoked from decoder post-processing flows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from . import DecoderEventSet, NoteEvent
from .snap import SnapOptions, TempoEstimate, estimate_tempo as snap_estimate_tempo

_MIN_BPM_CONF = 0.5
_DEFAULT_BPM = 120.0
_MERGE_THRESHOLD = 0.30
_EPS = 1e-4


@dataclass
class DPOptions:
    enabled: bool = False
    min_on_beats: float = 0.25
    min_off_beats: float = 0.25
    penalty_on: float = 0.5
    penalty_off: float = 0.5
    keep_gates: bool = False
    per_key: bool = True
    beam: int = 5
    emission_mode: str = "logit"
    emission_gain: float = 4.0
    emission_bias: float = 0.0


def apply(event_set: DecoderEventSet, cfg: Mapping[str, Any], snap_cfg: Mapping[str, Any] | None = None) -> DecoderEventSet:
    options = _parse_options(cfg)
    if not options.enabled or not options.per_key:
        return event_set
    baseline_events = event_set.clone_events()
    baseline_mask = event_set.mask.clone()
    tempo = _resolve_tempo(event_set, snap_cfg or {})
    bpm = tempo.bpm or _DEFAULT_BPM
    frames_per_beat = (event_set.fps * 60.0) / max(bpm, 1e-6)
    min_on_frames = max(1, int(round(options.min_on_beats * frames_per_beat)))
    min_off_frames = max(1, int(round(options.min_off_beats * frames_per_beat)))
    new_mask = torch.zeros_like(event_set.mask)
    per_key: Dict[int, Dict[str, float]] = {}
    penalty_adjustments = 0
    for clip_idx in range(event_set.clip_count):
        for key in range(event_set.pitches):
            prob_seq = event_set.probs[clip_idx, :, key]
            base_count = _count_events(baseline_events, clip_idx, key)
            dp_mask, merged_ratio, adjusted = _dp_with_guard(
                prob_seq,
                min_on_frames,
                min_off_frames,
                options.penalty_on,
                options.penalty_off,
                options.beam,
                base_count,
                options.emission_mode,
                options.emission_gain,
                options.emission_bias,
            )
            if adjusted:
                penalty_adjustments += 1
            if options.keep_gates:
                gated = torch.logical_and(dp_mask, baseline_mask[clip_idx, :, key])
            else:
                gated = dp_mask
            new_mask[clip_idx, :, key] = gated
            post_count = _count_mask_events(dp_mask)
            stats = per_key.setdefault(key, {
                "baseline_events": 0,
                "post_events": 0,
                "max_merge_ratio": 0.0,
                "clips": 0,
            })
            stats["baseline_events"] += base_count
            stats["post_events"] += post_count
            stats["clips"] += 1
            stats["max_merge_ratio"] = max(stats["max_merge_ratio"], merged_ratio)
    event_set.mask = new_mask
    event_set.refresh_events()
    latency = _mean_latency_shift(baseline_events, event_set.events)
    summary = {
        "enabled": True,
        "keep_gates": options.keep_gates,
        "min_on_frames": min_on_frames,
        "min_off_frames": min_off_frames,
        "penalty_on": options.penalty_on,
        "penalty_off": options.penalty_off,
        "beam": options.beam,
        "bpm": bpm,
        "tempo_confidence": tempo.confidence,
        "emission_mode": options.emission_mode,
        "emission_gain": options.emission_gain,
        "emission_bias": options.emission_bias,
        "latency_mean_abs_ms": latency,
        "per_key": per_key,
        "penalty_reductions": penalty_adjustments,
    }
    event_set.stats["dp"] = summary
    return event_set


def _dp_with_guard(
    prob_seq: torch.Tensor,
    min_on_frames: int,
    min_off_frames: int,
    penalty_on: float,
    penalty_off: float,
    beam: int,
    base_count: int,
    emission_mode: str,
    emission_gain: float,
    emission_bias: float,
) -> Tuple[torch.Tensor, float, bool]:
    best_mask = _run_viterbi(
        prob_seq,
        min_on_frames,
        min_off_frames,
        penalty_on,
        penalty_off,
        beam,
        emission_mode,
        emission_gain,
        emission_bias,
    )
    base_count = max(base_count, 0)
    merged_ratio = _merge_ratio(base_count, _count_mask_events(best_mask))
    adjusted = False
    if merged_ratio > _MERGE_THRESHOLD:
        best_mask = _run_viterbi(
            prob_seq,
            min_on_frames,
            min_off_frames,
            penalty_on * 0.5,
            penalty_off * 0.5,
            beam,
            emission_mode,
            emission_gain,
            emission_bias,
        )
        new_ratio = _merge_ratio(base_count, _count_mask_events(best_mask))
        if new_ratio < merged_ratio:
            merged_ratio = new_ratio
            adjusted = True
    return best_mask, merged_ratio, adjusted


def _merge_ratio(base_count: int, post_count: int) -> float:
    if base_count <= 0:
        return 0.0
    return max(0.0, (base_count - post_count) / base_count)


def _count_mask_events(mask: torch.Tensor) -> int:
    seq = mask.to(torch.bool).cpu().numpy().tolist()
    active = False
    count = 0
    for val in seq:
        if val and not active:
            count += 1
            active = True
        elif not val:
            active = False
    return count


def _count_events(events: List[NoteEvent], clip_idx: int, key: int) -> int:
    return sum(1 for ev in events if ev.clip_idx == clip_idx and ev.key == key)


def _run_viterbi(
    prob_seq: torch.Tensor,
    min_on: int,
    min_off: int,
    penalty_on: float,
    penalty_off: float,
    beam: int,
    emission_mode: str,
    emission_gain: float,
    emission_bias: float,
) -> torch.Tensor:
    probs = prob_seq.detach().cpu().to(torch.float32)
    T = probs.shape[0]
    if T == 0:
        return torch.zeros_like(probs, dtype=torch.bool)
    off_states = max(min_off, 1)
    on_states = max(min_on, 1)
    total_states = off_states + on_states

    def is_off(state: int) -> bool:
        return state < off_states

    def dwell_idx(state: int) -> int:
        return state if is_off(state) else state - off_states

    def stay_state(state: int) -> int:
        cap = off_states if is_off(state) else on_states
        idx = dwell_idx(state)
        next_idx = min(idx + 1, cap - 1)
        return next_idx if is_off(state) else next_idx + off_states

    emission_mode_norm = (emission_mode or "logit").strip().lower()
    if emission_mode_norm not in {"logit", "log_prob", "log"}:
        emission_mode_norm = "logit"
    gain = emission_gain if math.isfinite(emission_gain) else 1.0
    bias = emission_bias if math.isfinite(emission_bias) else 0.0

    def emission(state: int, t: int) -> float:
        p = float(probs[t].item())
        p = min(max(p, _EPS), 1.0 - _EPS)
        is_on_state = not is_off(state)
        if emission_mode_norm == "logit":
            logit = math.log(p + _EPS) - math.log(1.0 - p + _EPS)
            signed = logit if is_on_state else -logit
            return gain * signed + bias
        log_val = math.log(p) if is_on_state else math.log(1.0 - p)
        return gain * log_val + (bias if is_on_state else -bias)

    def transitions(state: int) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = [(stay_state(state), 0.0)]
        if is_off(state) and dwell_idx(state) >= off_states - 1:
            out.append((off_states, penalty_on))
        elif not is_off(state) and dwell_idx(state) >= on_states - 1:
            out.append((0, penalty_off))
        return out

    prev_scores = {state: emission(state, 0) for state in range(total_states)}
    backrefs: List[Dict[int, Optional[int]]] = [{state: None for state in prev_scores}]

    for t in range(1, T):
        current: Dict[int, float] = {}
        current_back: Dict[int, Optional[int]] = {}
        for state, score in prev_scores.items():
            for nxt, penalty in transitions(state):
                new_score = score - penalty + emission(nxt, t)
                if nxt not in current or new_score > current[nxt]:
                    current[nxt] = new_score
                    current_back[nxt] = state
        if beam > 0 and len(current) > beam:
            top = sorted(current.items(), key=lambda item: item[1], reverse=True)[:beam]
            allowed = {state for state, _ in top}
            current = {state: current[state] for state in allowed}
            current_back = {state: current_back[state] for state in allowed}
        backrefs.append(current_back)
        prev_scores = current
        if not prev_scores:
            break

    if not prev_scores:
        return torch.zeros(T, dtype=torch.bool)
    best_state = max(prev_scores, key=lambda state: prev_scores[state])
    path = [best_state]
    for t in range(T - 1, 0, -1):
        prev = backrefs[t].get(best_state)
        if prev is None:
            break
        path.append(prev)
        best_state = prev
    path = list(reversed(path))
    if len(path) < T:
        path.extend([path[-1]] * (T - len(path)))
    mask = torch.zeros(T, dtype=torch.bool)
    for idx, state in enumerate(path[:T]):
        mask[idx] = not is_off(state)
    return mask


def _mean_latency_shift(before: List[NoteEvent], after: List[NoteEvent]) -> float:
    def _group(events: List[NoteEvent]) -> Dict[Tuple[int, int], List[NoteEvent]]:
        grouped: Dict[Tuple[int, int], List[NoteEvent]] = {}
        for ev in events:
            grouped.setdefault((ev.clip_idx, ev.key), []).append(ev)
        for pack in grouped.values():
            pack.sort(key=lambda e: e.onset_frame)
        return grouped

    before_map = _group(before)
    after_map = _group(after)
    total = 0.0
    count = 0
    for key, old_events in before_map.items():
        new_events = after_map.get(key)
        if not new_events:
            continue
        pairs = min(len(old_events), len(new_events))
        for idx in range(pairs):
            total += abs(new_events[idx].onset_ms - old_events[idx].onset_ms)
            count += 1
    return total / count if count else 0.0


def _resolve_tempo(event_set: DecoderEventSet, snap_cfg: Mapping[str, Any]) -> TempoEstimate:
    tempo_summary = (event_set.stats.get("snap", {}) or {}).get("tempo", {})
    bpm = tempo_summary.get("bpm") if isinstance(tempo_summary, Mapping) else None
    confidence = tempo_summary.get("confidence") if isinstance(tempo_summary, Mapping) else 0.0
    if bpm and confidence and confidence >= _MIN_BPM_CONF:
        support = tempo_summary.get("support", 0) if isinstance(tempo_summary, Mapping) else 0
        return TempoEstimate(bpm, confidence, [], support, None)
    snap_opts = SnapOptions(
        enabled=True,
        bpm_min=float(snap_cfg.get("bpm_min", 40.0)),
        bpm_max=float(snap_cfg.get("bpm_max", 240.0)),
        bpm_step=float(snap_cfg.get("bpm_step", 1.0)),
        local_window_beats=int(snap_cfg.get("local_window_beats", 4)),
        max_dev_ms=float(snap_cfg.get("max_dev_ms", 35.0)),
        blend=float(snap_cfg.get("blend", 0.5)),
        min_support_notes=int(snap_cfg.get("min_support_notes", 12)),
        confidence_floor=float(snap_cfg.get("confidence_floor", 0.15)),
    )
    return snap_estimate_tempo(event_set.events, snap_opts)


def _parse_options(cfg: Mapping[str, Any] | None) -> DPOptions:
    if not isinstance(cfg, Mapping):
        return DPOptions()
    mode = str(cfg.get("emission_mode", "logit")).strip().lower()
    if not mode:
        mode = "logit"
    return DPOptions(
        enabled=bool(cfg.get("enabled", False)),
        min_on_beats=float(cfg.get("min_on_beats", 0.25)),
        min_off_beats=float(cfg.get("min_off_beats", 0.25)),
        penalty_on=float(cfg.get("penalty_on", 0.5)),
        penalty_off=float(cfg.get("penalty_off", 0.5)),
        keep_gates=bool(cfg.get("keep_gates", False)),
        per_key=bool(cfg.get("per_key", True)),
        beam=int(cfg.get("beam", 5)),
        emission_mode=mode,
        emission_gain=float(cfg.get("emission_gain", 4.0)),
        emission_bias=float(cfg.get("emission_bias", 0.0)),
    )


__all__ = ["DPOptions", "apply"]
