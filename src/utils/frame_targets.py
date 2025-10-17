"""Utilities for consistent frame-target configuration and caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import time

import torch

from .av_sync import AVLagResult
from .frame_target_cache import FrameTargetCache, FrameTargetMeta, make_target_cache_key
from .identifiers import canonical_video_id, id_aliases, log_legacy_id_hit
from .time_grid import sec_to_frame

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
) -> Dict[str, torch.Tensor]:
    """Build per-frame dense targets aligned to sampled frames."""

    hop_seconds = stride / max(1.0, float(fps))

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

    on_frames = torch.as_tensor(sec_to_frame(on, hop_seconds), dtype=torch.int64)
    off_frames = torch.as_tensor(sec_to_frame(off, hop_seconds), dtype=torch.int64)
    on_frames = torch.clamp(on_frames, 0, T - 1)
    off_frames = torch.clamp(off_frames, 0, T)

    for s, e, p in zip(on_frames, off_frames, pit):
        s = int(s)
        e = int(e)
        if e <= s:
            e = min(s + 1, T)
        pitch_roll[s:e, p] = 1.0
        onset_roll[s, p] = 1.0
        off_idx = min(e, T - 1)
        offset_roll[off_idx, p] = 1.0

    if dilate_active_frames and dilate_active_frames > 0:
        import torch.nn.functional as F

        ksz = 2 * dilate_active_frames + 1
        ker = torch.ones((1, 1, ksz), dtype=torch.float32)
        for roll in (pitch_roll, onset_roll, offset_roll):
            x = roll.transpose(0, 1).unsqueeze(1)
            x = F.conv1d(x, ker, padding=dilate_active_frames)
            roll.copy_((x.squeeze(1) > 0).transpose(0, 1).float())

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

    return {
        "pitch_roll": pitch_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "hand_frame": hand_frame,
        "clef_frame": clef_frame,
    }


def prepare_frame_targets(
    *,
    labels: Optional[torch.Tensor],
    lag_result: Optional[AVLagResult],
    spec: FrameTargetSpec,
    cache: FrameTargetCache,
    split: str,
    video_id: str,
    clip_start: float,
) -> FrameTargetResult:
    """Load or construct frame targets for a clip using a shared pipeline."""

    lag_ms, lag_source = resolve_lag_ms(lag_result)
    canon_video = canonical_video_id(video_id)
    aliases = id_aliases(canon_video)
    key_kwargs = dict(
        split=split,
        lag_ms=lag_ms,
        fps=spec.fps,
        frames=spec.frames,
        tolerance=spec.tolerance,
        dilation=spec.dilation,
        canonical_hw=spec.canonical_hw,
    )

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
        return FrameTargetResult(
            cached_targets,
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
    cache.save(primary_key_hash, primary_key_meta, cache_payload)
    lag_ms_int = primary_lag_ms_int
    lag_frames_int = primary_lag_frames_int
    status = "built"
    return FrameTargetResult(
        cache_payload,
        status,
        primary_key_hash,
        primary_key_meta,
        lag_ms_int,
        lag_source,
        lag_frames=lag_frames_int,
    )


__all__ = [
    "FRAME_TARGET_KEYS",
    "FrameTargetResult",
    "FrameTargetSpec",
    "build_dense_frame_targets",
    "prepare_frame_targets",
    "resolve_frame_target_spec",
    "resolve_lag_ms",
]
