"""Purpose:
    Build and cache frame-level piano targets aligned to sampled video clips,
    including configuration helpers shared by PianoYT and OMAPS datasets.

Key Functions/Classes:
    - FrameTargetSpec: Immutable configuration describing target generation.
    - prepare_frame_targets(): Loads cached targets or builds them on demand.
    - FRAME_TARGET_KEYS: Canonical set of tensors expected by downstream code.

CLI:
    Not a standalone CLI; utilities are imported by dataset loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple, TypedDict

import logging
import time

import torch

from .av_sync import AVLagResult
from .identifiers import canonical_video_id, id_aliases, log_legacy_id_hit
from .time_grid import sec_to_frame
from tivit.data.cache.frame_target_cache import FrameTargetCache, FrameTargetMeta, make_target_cache_key

LOGGER = logging.getLogger(__name__)

FRAME_TARGET_KEYS: Tuple[str, ...] = (
    "pitch_roll",
    "onset_roll",
    "offset_roll",
    "hand_frame",
    "clef_frame",
)


def _normalise_clef_thresholds(cfg: Any) -> Tuple[int, int]:
    """Coerce the clef threshold configuration into a 2-tuple of ints."""

    if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):
        values = [int(v) for v in list(cfg)[:2]]
        if len(values) == 1:
            values.append(values[0])
        if len(values) >= 2:
            return int(values[0]), int(values[1])
    return 60, 64


@dataclass(frozen=True)
class FrameTargetSpec:
    """Immutable configuration for frame-level target construction."""

    frames: int
    stride: int
    fps: float
    canonical_hw: Tuple[int, int]
    tolerance: float
    dilation: int
    note_min: int
    note_max: int
    fill_mode: str
    hand_from_pitch: bool
    clef_thresholds: Tuple[int, int]
    targets_sparse: bool

    @property
    def cache_key_prefix(self) -> str:
        """Return a stable hash prefix derived from the cache configuration."""

        key_hash, _ = make_target_cache_key(
            split="__spec__",
            video_id="__spec__",
            lag_ms=0.0,
            fps=self.fps,
            frames=self.frames,
            tolerance=self.tolerance,
            dilation=self.dilation,
            canonical_hw=self.canonical_hw,
            canonicalize=False,
        )
        return key_hash[:8]

    def make_cache_key(self, *, split: str, video_id: str, lag_ms: float) -> Tuple[str, FrameTargetMeta]:
        """Compute the cache key/meta tuple for a clip."""

        return make_target_cache_key(
            split=split,
            video_id=video_id,
            lag_ms=lag_ms,
            fps=self.fps,
            frames=self.frames,
            tolerance=self.tolerance,
            dilation=self.dilation,
            canonical_hw=self.canonical_hw,
        )

    def summary(self, *, lag_source: str = "guardrail") -> str:
        tol_str = f"{self.tolerance:.3f}".rstrip("0").rstrip(".")
        if tol_str:
            tol_display = f"{tol_str}s"
        else:
            tol_display = f"{self.tolerance:.3f}s"
        return (
            "targets_conf: "
            f"T={self.frames}, "
            f"tol={tol_display}, "
            f"dilate={self.dilation}, "
            f"lag_source={lag_source}, "
            f"cache_key={self.cache_key_prefix}"
        )


@dataclass(frozen=True)
class SoftTargetConfig:
    """Normalised configuration for train-only soft target smoothing."""

    enabled: bool
    apply_onset: bool
    apply_pitch: bool
    apply_offset: bool
    onset_kernel: Tuple[float, ...]
    frame_kernel: Tuple[float, ...]


_DEFAULT_ONSET_KERNEL: Tuple[float, ...] = (0.5, 1.0, 0.5)
_DEFAULT_FRAME_KERNEL: Tuple[float, ...] = (0.5, 1.0, 0.5)
_SOFT_TARGET_LOGGED = False


def _coerce_kernel(
    values: Optional[Sequence[Any]],
    fallback: Tuple[float, ...],
    *,
    require_odd: bool = False,
    min_len: int = 0,
) -> Tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return fallback
    try:
        kernel = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return fallback
    if not kernel:
        return fallback
    if require_odd and len(kernel) % 2 == 0:
        return fallback
    if min_len and len(kernel) < min_len:
        return fallback
    return kernel


def resolve_soft_target_config(cfg: Optional[Mapping[str, Any]]) -> Optional[SoftTargetConfig]:
    """Normalise soft target settings from configuration."""

    if not isinstance(cfg, Mapping):
        return None

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None

    apply_cfg = cfg.get("apply_to", {}) or {}
    apply_onset = bool(apply_cfg.get("onset", True))
    apply_pitch = bool(apply_cfg.get("pitch", True))
    apply_offset = bool(apply_cfg.get("offset", False))
    onset_kernel = _coerce_kernel(
        cfg.get("onset_kernel"),
        _DEFAULT_ONSET_KERNEL,
        require_odd=True,
        min_len=3,
    )
    frame_kernel = _coerce_kernel(
        cfg.get("frame_kernel"),
        _DEFAULT_FRAME_KERNEL,
        min_len=3,
    )

    return SoftTargetConfig(
        enabled=True,
        apply_onset=apply_onset,
        apply_pitch=apply_pitch,
        apply_offset=apply_offset,
        onset_kernel=onset_kernel,
        frame_kernel=frame_kernel,
    )


@dataclass
class FrameTargetResult:
    """Payload returned when preparing frame targets for a clip."""

    payload: Optional[Dict[str, torch.Tensor]]
    status: str
    cache_key: Optional[str]
    cache_meta: Optional[FrameTargetMeta]
    lag_ms: Optional[int]
    lag_source: str
    lag_frames: Optional[int] = None


class _CacheKeyKwargs(TypedDict):
    split: str
    lag_ms: float
    fps: float
    frames: int
    tolerance: float
    dilation: int
    canonical_hw: Sequence[int]


def resolve_frame_target_spec(
    frame_cfg: Optional[Mapping[str, Any]],
    *,
    frames: int,
    stride: int,
    fps: float,
    canonical_hw: Sequence[int],
) -> Optional[FrameTargetSpec]:
    """Normalise configuration mapping into a :class:`FrameTargetSpec`."""

    if not frame_cfg or not bool(frame_cfg.get("enable", False)):
        return None

    if len(canonical_hw) < 2:
        raise ValueError("canonical_hw must provide at least two entries (H, W)")

    note_min = int(frame_cfg.get("note_min", 21))
    note_max = int(frame_cfg.get("note_max", 108))
    clef_cfg = frame_cfg.get("clef_thresholds", [60, 64])
    clef_tuple = _normalise_clef_thresholds(clef_cfg)

    return FrameTargetSpec(
        frames=int(frames),
        stride=int(stride),
        fps=float(fps),
        canonical_hw=(int(canonical_hw[0]), int(canonical_hw[1])),
        tolerance=float(frame_cfg.get("tolerance", 0.025)),
        dilation=int(frame_cfg.get("dilate_active_frames", 0)),
        note_min=note_min,
        note_max=note_max,
        fill_mode=str(frame_cfg.get("fill_mode", "overlap")),
        hand_from_pitch=bool(frame_cfg.get("hand_from_pitch", True)),
        clef_thresholds=clef_tuple,
        targets_sparse=bool(frame_cfg.get("targets_sparse", False)),
    )


def resolve_lag_ms(lag_result: Optional[AVLagResult]) -> Tuple[float, str]:
    """Derive the final lag (ms) and descriptive source string."""

    if lag_result is None or not getattr(lag_result, "success", False):
        return 0.0, "guardrail"

    lag_ms = float(lag_result.lag_ms)
    source_bits = ["guardrail"]
    if getattr(lag_result, "from_cache", False):
        source_bits.append("cache")
    if getattr(lag_result, "used_video_median", False):
        source_bits.append("median")
    if getattr(lag_result, "low_corr_zero", False):
        source_bits.append("zero")
    if getattr(lag_result, "hit_bound", False):
        source_bits.append("bound")
    if getattr(lag_result, "clamped", False):
        source_bits.append("clamped")
    if getattr(lag_result, "lag_timeout", False):
        source_bits.append("timeout")
    return lag_ms, "+".join(source_bits)


def build_dense_frame_targets(
    labels: Optional[torch.Tensor],
    *,
    T: int,
    stride: int,
    fps: float,
    note_min: int,
    note_max: int,
    tol: float,
    fill_mode: str,
    hand_from_pitch: bool,
    clef_thresholds: Tuple[int, int],
    dilate_active_frames: int,
    targets_sparse: bool,
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Build per-frame dense targets aligned to sampled frames."""

    hop_seconds = stride / max(1.0, float(fps))
    duration_sec = hop_seconds * max(T - 1, 0)

    T = int(T)
    P = int(note_max - note_min + 1)
    pitch_roll = torch.zeros((T, P), dtype=torch.float32)
    onset_roll = torch.zeros((T, P), dtype=torch.float32)
    offset_roll = torch.zeros((T, P), dtype=torch.float32)

    if labels is None or labels.numel() == 0:
        hand_frame = torch.zeros((T,), dtype=torch.long)
        clef_frame = torch.full((T,), 2, dtype=torch.long)
        return {
            "pitch_roll": pitch_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "hand_frame": hand_frame,
            "clef_frame": clef_frame,
        }

    on = labels[:, 0]
    off = labels[:, 1]
    pit = labels[:, 2].to(torch.int64)

    mask_pitch = (pit >= int(note_min)) & (pit <= int(note_max))
    on, off, pit = on[mask_pitch], off[mask_pitch], pit[mask_pitch] - int(note_min)

    window_start = 0.0
    window_end = duration_sec
    events_seen_total = int(len(on))
    in_window_mask = (on >= window_start) & (on < window_end)
    min_onset_used = float(on[in_window_mask].min().item()) if in_window_mask.any() else None
    max_onset_used = float(on[in_window_mask].max().item()) if in_window_mask.any() else None
    min_onset_seen = float(on.min().item()) if on.numel() > 0 else None
    max_onset_seen = float(on.max().item()) if on.numel() > 0 else None
    on = on[in_window_mask]
    off = off[in_window_mask]
    pit = pit[in_window_mask]
    events_in_window_total = int(len(on))

    on_frames = torch.as_tensor(sec_to_frame(on, hop_seconds), dtype=torch.int64)
    off_frames = torch.as_tensor(sec_to_frame(off, hop_seconds), dtype=torch.int64)
    on_frames = torch.clamp(on_frames, 0, T - 1)
    off_frames = torch.clamp(off_frames, 0, T)

    if trace is not None:
        trace.update(
            {
                "timebase": "shifted_to_clip",
                "clip_start_sec_used_for_counts": 0.0,
                "duration_sec_used_for_counts": duration_sec,
                "clip_end_sec_used_for_counts": duration_sec,
                "window_start": 0.0,
                "window_end": duration_sec,
                "hop_seconds": hop_seconds,
                "events_seen_total": events_seen_total,
                "events_in_window_total": events_in_window_total,
                "events_painted_total": 0,
                "min_onset_seen": min_onset_seen,
                "max_onset_seen": max_onset_seen,
                "min_onset_used": min_onset_used,
                "max_onset_used": max_onset_used,
                "unique_onset_center_frames_in_window": sorted({int(f.item()) for f in on_frames}),
            }
        )

    frames_touched: Set[int] = set()
    events_painted_total = 0
    for s, e, p in zip(on_frames, off_frames, pit):
        s = int(s)
        e = int(e)
        if e <= s:
            e = min(s + 1, T)
        pitch_roll[s:e, p] = 1.0
        onset_roll[s, p] = 1.0
        off_idx = min(e, T - 1)
        offset_roll[off_idx, p] = 1.0
        frames_touched.add(s)
        events_painted_total += 1

    if dilate_active_frames and dilate_active_frames > 0:
        import torch.nn.functional as F

        ksz = 2 * dilate_active_frames + 1
        ker = torch.ones((1, 1, ksz), dtype=torch.float32)
        for roll in (pitch_roll, onset_roll, offset_roll):
            x = roll.transpose(0, 1).unsqueeze(1)
            x = F.conv1d(x, ker, padding=dilate_active_frames)
            roll.copy_((x.squeeze(1) > 0).transpose(0, 1).float())
            if roll is onset_roll:
                active_frames = (roll.sum(dim=1) > 0).nonzero(as_tuple=False).view(-1).tolist()
                frames_touched.update(int(f) for f in active_frames)

    # Hand head consumes 2-way per-frame labels (0=left, 1=right) and the
    # current implementation feeds it pseudo-supervision inferred from MIDI.
    # With no hand annotations available, we average active note pitches per
    # frame and threshold them (clef thresholds below) to decide left/right;
    # the resulting heuristic hand_frame drives both the hand loss/metric.
    if hand_from_pitch:
        active = pitch_roll > 0
        pitch_ids = torch.arange(P, dtype=torch.float32) + float(note_min)
        sums = (pitch_roll * pitch_ids[None, :]).sum(dim=1)
        counts = active.sum(dim=1).clamp(min=1)
        avg_pitch = torch.where(active.any(dim=1), sums / counts, torch.full((T,), 60.0))

        lh_thr, rh_thr = int(clef_thresholds[0]), int(clef_thresholds[1])
        hand_frame = (avg_pitch >= lh_thr).long()
        clef_frame = torch.where(
            avg_pitch < lh_thr,
            torch.zeros_like(hand_frame),
            torch.where(
                avg_pitch > rh_thr,
                torch.ones_like(hand_frame),
                torch.full_like(hand_frame, 2),
            ),
        )
    else:
        hand_frame = torch.zeros((T,), dtype=torch.long)
        clef_frame = torch.full((T,), 2, dtype=torch.long)

    if targets_sparse:
        pass

    out = {
        "pitch_roll": pitch_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "hand_frame": hand_frame,
        "clef_frame": clef_frame,
    }
    if trace is not None:
        trace["onset_frames_touched_by_painting"] = sorted(frames_touched)
        trace["target_onset_cells_actual"] = int(torch.count_nonzero(onset_roll).item())
        trace["target_onset_frames_any_actual"] = int(torch.count_nonzero(onset_roll.sum(dim=1)).item())
        trace["painted_pairs_unique"] = int(torch.count_nonzero(onset_roll).item())
        trace["onset_frames_touched_count"] = len(frames_touched)
        trace["events_painted_total"] = events_painted_total

    # Offset trace
    if trace is not None:
        offset_frames_touched: Set[int] = set()
        offset_events_painted = 0
        offset_center_frames = set()
        for e, p in zip(off_frames, pit):
            off_idx = int(torch.clamp(e, 0, T - 1).item())
            offset_center_frames.add(off_idx)
            offset_frames_touched.add(off_idx)
            offset_events_painted += 1
        trace["offset_timebase"] = "shifted_to_clip"
        trace["offset_window_start"] = window_start
        trace["offset_window_end"] = window_end
        trace["offset_events_seen_total"] = events_seen_total
        trace["offset_events_in_window_total"] = events_in_window_total
        trace["offset_events_painted_total"] = offset_events_painted
        trace["offset_min_used"] = float(on.min().item()) if on.numel() > 0 else None
        trace["offset_max_used"] = float(on.max().item()) if on.numel() > 0 else None
        trace["offset_painted_pairs_unique"] = int(torch.count_nonzero(offset_roll).item())
        trace["offset_target_cells_actual"] = int(torch.count_nonzero(offset_roll).item())
        trace["offset_target_frames_any_actual"] = int(torch.count_nonzero(offset_roll.sum(dim=1)).item())
        trace["offset_frames_touched_by_painting"] = sorted(offset_frames_touched)
        trace["offset_frames_touched_count"] = len(offset_frames_touched)
        trace["offset_unique_center_frames_in_window"] = sorted(offset_center_frames)

    # Pitch trace
    if trace is not None:
        pitch_painted_pairs = int(torch.count_nonzero(pitch_roll).item())
        pitch_frames_any_actual = int(torch.count_nonzero(pitch_roll.sum(dim=1)).item())
        trace["pitch_events_seen_total"] = events_seen_total
        trace["pitch_events_in_window_total"] = events_in_window_total
        trace["pitch_events_painted_total"] = events_painted_total
        trace["pitch_painted_pairs_unique"] = pitch_painted_pairs
        trace["pitch_target_cells_actual"] = pitch_painted_pairs
        trace["pitch_frames_any_actual"] = pitch_frames_any_actual
        trace["pitch_frames_any_painted_count"] = pitch_frames_any_actual
        trace["pitch_frames_any_painted"] = (
            sorted([int(i) for i in (pitch_roll.sum(dim=1) > 0).nonzero(as_tuple=False).view(-1).tolist()])
            if pitch_roll.numel() > 0
            else []
        )
        trace["pitch_min_used"] = int((pit + note_min).min().item()) if pit.numel() > 0 else None
        trace["pitch_max_used"] = int((pit + note_min).max().item()) if pit.numel() > 0 else None
        trace["pitch_invalid_pitch_events_dropped"] = int(events_seen_total - len(pit))
    return out


def _shift_bool_mask(mask: torch.Tensor, delta: int) -> torch.Tensor:
    if mask.dim() != 2:
        return torch.zeros_like(mask)
    T = mask.shape[0]
    if delta == 0:
        return mask.clone()
    if abs(delta) >= T:
        return torch.zeros_like(mask)
    shifted = torch.zeros_like(mask)
    if delta > 0:
        shifted[delta:, :] = mask[: T - delta, :]
    else:
        shifted[: T + delta, :] = mask[-delta:, :]
    return shifted


def _apply_onset_soft_targets(onset_roll: torch.Tensor, kernel: Sequence[float]) -> None:
    if onset_roll.numel() == 0:
        return
    if not kernel:
        return
    center = len(kernel) // 2
    base_mask = onset_roll >= 0.5
    for idx, weight in enumerate(kernel):
        weight_val = max(0.0, min(1.0, float(weight)))
        if weight_val <= 0.0:
            continue
        delta = idx - center
        shifted = _shift_bool_mask(base_mask, delta)
        candidate = shifted.to(dtype=onset_roll.dtype) * weight_val
        torch.maximum(onset_roll, candidate, out=onset_roll)


def _apply_pitch_soft_targets(pitch_roll: torch.Tensor, kernel: Sequence[float]) -> None:
    if pitch_roll.numel() == 0 or len(kernel) < 3:
        return
    pre = max(0.0, min(1.0, float(kernel[0])))
    interior = max(0.0, min(1.0, float(kernel[1])))
    post = max(0.0, min(1.0, float(kernel[-1])))
    base_active = pitch_roll >= 0.5
    if interior > 0.0:
        torch.maximum(
            pitch_roll,
            base_active.to(dtype=pitch_roll.dtype) * interior,
            out=pitch_roll,
        )
    if pre > 0.0:
        prev_active = torch.zeros_like(base_active)
        prev_active[1:, :] = base_active[:-1, :]
        start_mask = base_active & (~prev_active)
        if start_mask.any():
            frames, pitches = start_mask.nonzero(as_tuple=True)
            valid = frames > 0
            if valid.any():
                frames = frames[valid] - 1
                pitches = pitches[valid]
                current = pitch_roll[frames, pitches]
                pitch_roll[frames, pitches] = torch.maximum(
                    current,
                    current.new_full(current.shape, pre),
                )
    if post > 0.0:
        next_active = torch.zeros_like(base_active)
        next_active[:-1, :] = base_active[1:, :]
        end_mask = base_active & (~next_active)
        if end_mask.any():
            frames, pitches = end_mask.nonzero(as_tuple=True)
            valid = frames < (pitch_roll.shape[0] - 1)
            if valid.any():
                frames = frames[valid] + 1
                pitches = pitches[valid]
                current = pitch_roll[frames, pitches]
                pitch_roll[frames, pitches] = torch.maximum(
                    current,
                    current.new_full(current.shape, post),
                )


def _log_soft_target_summary(payload: Mapping[str, torch.Tensor], split: str) -> None:
    global _SOFT_TARGET_LOGGED
    if _SOFT_TARGET_LOGGED:
        return
    onset = payload.get("onset_roll")
    pitch = payload.get("pitch_roll")
    if not torch.is_tensor(onset) or not torch.is_tensor(pitch):
        return
    def _stats(t: torch.Tensor) -> Tuple[float, float, float]:
        return (
            float(t.min().item()),
            float(t.mean().item()),
            float(t.max().item()),
        )
    onset_stats = _stats(onset)
    pitch_stats = _stats(pitch)
    LOGGER.info(
        "[targets:soft] split=%s onset[min=%.3f mean=%.3f max=%.3f] pitch[min=%.3f mean=%.3f max=%.3f]",
        split,
        onset_stats[0],
        onset_stats[1],
        onset_stats[2],
        pitch_stats[0],
        pitch_stats[1],
        pitch_stats[2],
    )
    _SOFT_TARGET_LOGGED = True


def _maybe_apply_soft_targets(
    payload: Optional[Dict[str, torch.Tensor]],
    soft_cfg: Optional[SoftTargetConfig],
    *,
    split: str,
) -> Optional[Dict[str, torch.Tensor]]:
    if payload is None or soft_cfg is None or not soft_cfg.enabled:
        return payload
    if str(split).lower() != "train":
        return payload

    applied = False
    onset_roll = payload.get("onset_roll")
    if soft_cfg.apply_onset and torch.is_tensor(onset_roll):
        _apply_onset_soft_targets(onset_roll, soft_cfg.onset_kernel)
        applied = True

    pitch_roll = payload.get("pitch_roll")
    if soft_cfg.apply_pitch and torch.is_tensor(pitch_roll):
        _apply_pitch_soft_targets(pitch_roll, soft_cfg.frame_kernel)
        applied = True

    # NOTE: Offsets intentionally remain binary spikes; apply_to.offset exists
    # as a hook for future experiments mirroring the onset smoothing pattern.

    if applied:
        _log_soft_target_summary(payload, split)
    return payload


def prepare_frame_targets(
    *,
    labels: Optional[torch.Tensor],
    lag_result: Optional[AVLagResult],
    spec: FrameTargetSpec,
    cache: FrameTargetCache,
    split: str,
    video_id: str,
    clip_start: float,
    soft_targets: Optional[SoftTargetConfig] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> FrameTargetResult:
    """Load or construct frame targets for a clip using a shared pipeline."""

    lag_ms, lag_source = resolve_lag_ms(lag_result)
    canon_video = canonical_video_id(video_id)
    aliases = id_aliases(canon_video)
    key_kwargs: _CacheKeyKwargs = {
        "split": split,
        "lag_ms": lag_ms,
        "fps": spec.fps,
        "frames": spec.frames,
        "tolerance": spec.tolerance,
        "dilation": spec.dilation,
        "canonical_hw": spec.canonical_hw,
    }

    primary_key_hash, primary_key_meta = make_target_cache_key(
        video_id=canon_video,
        canonicalize=True,
        scheme="frame",
        **key_kwargs,
    )
    primary_lag_ms_val = primary_key_meta.get("lag_ms")
    primary_lag_frames_val = primary_key_meta.get("lag_frames")
    primary_lag_ms_int = int(round(float(primary_lag_ms_val))) if primary_lag_ms_val is not None else None
    primary_lag_frames_int = int(primary_lag_frames_val) if primary_lag_frames_val is not None else None

    candidates = []
    seen_hashes = set()
    for scheme in ("frame", "legacy_ms"):
        for alias in aliases:
            canonicalize_alias = alias == canon_video
            key_hash, key_meta = make_target_cache_key(
                video_id=alias,
                canonicalize=canonicalize_alias,
                scheme=scheme,
                **key_kwargs,
            )
            if key_hash in seen_hashes:
                continue
            seen_hashes.add(key_hash)
            candidates.append((key_hash, key_meta, alias))

    for key_hash, key_meta, alias in candidates:
        cached_targets, _ = cache.load(key_hash)
        if cached_targets is None:
            continue
        if not all(k in cached_targets for k in FRAME_TARGET_KEYS):
            continue
        if alias != canon_video:
            log_legacy_id_hit(alias, canon_video)
        lag_ms_meta = key_meta.get("lag_ms")
        lag_frames_meta = key_meta.get("lag_frames", None)
        lag_ms_int = int(round(float(lag_ms_meta))) if lag_ms_meta is not None else None
        lag_frames_int = int(lag_frames_meta) if lag_frames_meta is not None else None
        payload = _maybe_apply_soft_targets(cached_targets, soft_targets, split=split)
        return FrameTargetResult(
            payload,
            "reused",
            key_hash,
            key_meta,
            lag_ms_int,
            lag_source,
            lag_frames=lag_frames_int,
        )

    labels_local: Optional[torch.Tensor]
    if labels is None or labels.numel() == 0:
        labels_local = None
    else:
        labels_local = labels.clone()
        if labels_local.numel() > 0:
            labels_local[:, 0:2] -= float(clip_start)

    build_start = time.perf_counter()
    ft = build_dense_frame_targets(
        labels_local,
        T=spec.frames,
        stride=spec.stride,
        fps=spec.fps,
        note_min=spec.note_min,
        note_max=spec.note_max,
        tol=spec.tolerance,
        fill_mode=spec.fill_mode,
        hand_from_pitch=spec.hand_from_pitch,
        clef_thresholds=spec.clef_thresholds,
        dilate_active_frames=spec.dilation,
        targets_sparse=spec.targets_sparse,
        trace=trace,
    )
    build_duration = time.perf_counter() - build_start
    if build_duration > 3.0:
        return FrameTargetResult(
            None,
            "failed",
            primary_key_hash,
            primary_key_meta,
            primary_lag_ms_int,
            lag_source,
            lag_frames=primary_lag_frames_int,
        )

    cache_payload = {key: ft[key] for key in FRAME_TARGET_KEYS}
    payload = _maybe_apply_soft_targets(ft, soft_targets, split=split)
    cache.save(primary_key_hash, primary_key_meta, cache_payload)
    lag_ms_int = primary_lag_ms_int
    lag_frames_int = primary_lag_frames_int
    status = "built"
    return FrameTargetResult(
        payload,
        status,
        primary_key_hash,
        primary_key_meta,
        lag_ms_int,
        lag_source,
        lag_frames=lag_frames_int,
    )


# Backwards-compatible wrappers retained for callers using the legacy names.
def build_frame_target_spec(frame_cfg: Mapping[str, Any], *, frames: int, stride: int, fps: float, canonical_hw: Sequence[int]) -> Optional[FrameTargetSpec]:
    """Alias for resolve_frame_target_spec."""

    return resolve_frame_target_spec(frame_cfg, frames=frames, stride=stride, fps=fps, canonical_hw=canonical_hw)


def build_soft_target_cfg(frame_cfg: Mapping[str, Any]) -> Optional[SoftTargetConfig]:
    """Alias for resolve_soft_target_config."""

    return resolve_soft_target_config(frame_cfg)


def build_frame_targets(
    *,
    labels: Optional[torch.Tensor],
    lag_result: Optional[AVLagResult],
    spec: FrameTargetSpec,
    cache: FrameTargetCache,
    split: str,
    video_id: str,
    clip_start: float,
    soft_targets: Optional[SoftTargetConfig] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> FrameTargetResult:
    """Alias for prepare_frame_targets."""

    return prepare_frame_targets(
        labels=labels,
        lag_result=lag_result,
        spec=spec,
        cache=cache,
        split=split,
        video_id=video_id,
        clip_start=clip_start,
        soft_targets=soft_targets,
        trace=trace,
    )


__all__ = [
    "FRAME_TARGET_KEYS",
    "FrameTargetResult",
    "FrameTargetSpec",
    "SoftTargetConfig",
    "build_dense_frame_targets",
    "build_frame_target_spec",
    "build_soft_target_cfg",
    "build_frame_targets",
    "prepare_frame_targets",
    "resolve_frame_target_spec",
    "resolve_soft_target_config",
    "resolve_lag_ms",
]
