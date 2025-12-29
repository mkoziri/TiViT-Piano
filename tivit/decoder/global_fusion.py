"""Global fusion helpers for per-tile TiViT logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from collections import Counter

import numpy as np
import torch

from .tile_keymap import TileMaskResult, build_tile_key_mask
from .registration_geometry import build_canonical_registration_metadata
from .tile_support_cache import CacheScope, TileSupportCache, make_tile_cache_key
from tivit.data.targets.identifiers import canonical_video_id
from tivit.data.roi.keyboard_roi import RegistrationRefiner

_DEFAULT_APPLY_TO = ("onset", "offset", "pitch")
_REG_LOOKUP_LOGGED: Set[str] = set()
_CACHE_PROBE_LOGGED = False


def _metadata_width_hint(payload: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not payload:
        return None
    for key in ("rectified_width", "frame_width", "width"):
        val = payload.get(key)
        if isinstance(val, (int, float)) and float(val) > 0:
            return float(val)
    target_hw = payload.get("target_hw") or payload.get("canonical_hw")
    if isinstance(target_hw, Sequence) and len(target_hw) >= 2:
        try:
            width = float(target_hw[1])
        except (TypeError, ValueError):
            width = None
        else:
            if width > 0:
                return width
    return None


def _registration_cache_key(video_uid: Optional[str]) -> Optional[str]:
    if video_uid is None:
        return None
    key = canonical_video_id(video_uid)
    key = key.strip()
    if not key:
        return None
    if video_uid not in _REG_LOOKUP_LOGGED:
        print(f"[fusion] registration lookup dataset_id={video_uid} cache_key={key}", flush=True)
        _REG_LOOKUP_LOGGED.add(video_uid)
    return key


def _log_cache_probe_once(refiner: RegistrationRefiner) -> None:
    global _CACHE_PROBE_LOGGED
    if _CACHE_PROBE_LOGGED:
        return
    keys = refiner.peek_cache_keys(1)
    if not keys:
        return
    key = keys[0]
    payload = refiner.get_cache_entry_payload(key) or {}
    status = payload.get("status", "missing")
    target_hw = payload.get("target_hw")
    print(
        f"[fusion] registration cache probe example_key={key} status={status} target_hw={target_hw}",
        flush=True,
    )
    _CACHE_PROBE_LOGGED = True


@dataclass(frozen=True)
class GlobalFusionConfig:
    """Resolved configuration for decode-time global fusion."""

    enabled: bool = False
    mode: str = "masked_mean"
    cushion_keys: int = 2
    apply_to: Tuple[str, ...] = _DEFAULT_APPLY_TO
    weighting: str = "uniform"
    consistency_check: bool = False
    consistency_batches: int = 0

    @property
    def needs_per_tile(self) -> bool:
        return self.enabled or self.consistency_check


@dataclass
class ComparisonAccumulator:
    baseline_f1_sum: float = 0.0
    fused_f1_sum: float = 0.0
    baseline_pred_rate_sum: float = 0.0
    fused_pred_rate_sum: float = 0.0
    pos_rate_sum: float = 0.0
    count: int = 0
    max_abs_logit_diff: float = 0.0


class FusionDebugState:
    """Collects coverage/shape/comparison summaries for debug logging."""

    def __init__(self, num_tiles: int, n_keys: int = 88) -> None:
        self.num_tiles = max(int(num_tiles), 0)
        self.n_keys = max(int(n_keys), 0)
        self.coverage_sum: List[float] = [0.0 for _ in range(self.num_tiles)]
        self.coverage_min: List[float] = [float("inf") for _ in range(self.num_tiles)]
        self.coverage_max: List[float] = [0.0 for _ in range(self.num_tiles)]
        self.coverage_clips: int = 0
        self.shape_lines: List[str] = []
        self.comparison: dict[str, ComparisonAccumulator] = {}
        self.boundary_hist: Counter[int] = Counter()
        self.key_index_min: List[int] = [self.n_keys for _ in range(self.num_tiles)]
        self.key_index_max: List[int] = [-1 for _ in range(self.num_tiles)]
        self.registration_clips: int = 0
        self.fallback_clips: int = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.clip_reports: List[str] = []
        self._clip_report_limit = 5

    def record_shape(self, head: str, tile_shape: Sequence[int], fused_shape: Sequence[int]) -> None:
        head_key = str(head)
        if any(head_key in line for line in self.shape_lines):
            return
        self.shape_lines.append(
            f"shapes {head_key}: tiles={tuple(int(v) for v in tile_shape)} fused={tuple(int(v) for v in fused_shape)}"
        )

    def record_mask(self, mask: np.ndarray, *, boundary_count: Optional[int] = None) -> None:
        if self.num_tiles <= 0:
            return
        if mask.ndim != 2 or mask.shape[0] != self.num_tiles:
            return
        counts = mask.sum(axis=1).astype(np.float32)
        self.coverage_clips += 1
        for idx, count in enumerate(counts.tolist()):
            self.coverage_sum[idx] += float(count)
            self.coverage_min[idx] = min(self.coverage_min[idx], float(count))
            self.coverage_max[idx] = max(self.coverage_max[idx], float(count))
            covered = np.flatnonzero(mask[idx])
            if covered.size > 0:
                self.key_index_min[idx] = min(self.key_index_min[idx], int(covered[0]))
                self.key_index_max[idx] = max(self.key_index_max[idx], int(covered[-1]))
        if boundary_count is None:
            overlap = mask.sum(axis=0)
            boundary_count = int(np.count_nonzero(overlap > 1))
        self.boundary_hist[int(boundary_count)] += 1

    def record_mask_result(self, record: TileMaskResult, clip_id: Optional[str] = None) -> None:
        self.record_mask(record.mask, boundary_count=record.boundary_keys)
        if record.registration_based:
            self.registration_clips += 1
        else:
            self.fallback_clips += 1
            reason = record.fallback_reason or "unknown"
            self.fallback_reasons[reason] += 1
        if len(self.clip_reports) >= self._clip_report_limit:
            return
        parts = []
        for idx, (rng, count) in enumerate(zip(record.tile_key_ranges, record.tile_key_counts)):
            parts.append(f"tile{idx}:keys={count} range=[{rng[0]},{rng[1]}]")
        mode = "registration" if record.registration_based else f"fallback({record.fallback_reason or 'unknown'})"
        clip_label = clip_id or "?"
        self.clip_reports.append(f"clip {clip_label} {mode} " + ", ".join(parts))

    def record_comparison(
        self,
        head: str,
        *,
        baseline_logits: torch.Tensor,
        fused_logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        prob_threshold: float,
        f1_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> bool:
        if targets is None:
            return False
        if not (torch.is_tensor(baseline_logits) and torch.is_tensor(fused_logits)):
            return False
        if baseline_logits.shape != fused_logits.shape:
            return False
        if targets.shape != fused_logits.shape:
            return False
        baseline = baseline_logits.detach()
        fused = fused_logits.detach()
        target = targets.detach()
        baseline_probs = torch.sigmoid(baseline)
        fused_probs = torch.sigmoid(fused)
        thr = float(prob_threshold)
        baseline_preds = (baseline_probs >= thr).float()
        fused_preds = (fused_probs >= thr).float()
        stats = self.comparison.setdefault(str(head), ComparisonAccumulator())
        stats.count += 1
        stats.pos_rate_sum += float(target.mean().item())
        stats.baseline_pred_rate_sum += float(baseline_preds.mean().item())
        stats.fused_pred_rate_sum += float(fused_preds.mean().item())
        stats.baseline_f1_sum += float(f1_fn(baseline_preds.reshape(-1), target.reshape(-1)) or 0.0)
        stats.fused_f1_sum += float(f1_fn(fused_preds.reshape(-1), target.reshape(-1)) or 0.0)
        diff = torch.max(torch.abs(fused - baseline)).item()
        stats.max_abs_logit_diff = max(stats.max_abs_logit_diff, float(diff))
        return True

    def summary_lines(self, prefix: str = "[fusion]") -> List[str]:
        lines: List[str] = []
        for line in self.shape_lines:
            lines.append(f"{prefix} {line}")
        if self.coverage_clips > 0:
            parts: List[str] = []
            for idx in range(self.num_tiles):
                if self.coverage_min[idx] == float("inf"):
                    continue
                avg = self.coverage_sum[idx] / max(1, self.coverage_clips)
                min_val = int(round(self.coverage_min[idx]))
                max_val = int(round(self.coverage_max[idx]))
                range_lo = self.key_index_min[idx]
                range_hi = self.key_index_max[idx]
                if range_lo > range_hi:
                    range_desc = "range=[-, -]"
                else:
                    range_desc = f"range=[{range_lo},{range_hi}]"
                parts.append(f"tile{idx}:avg={avg:.1f}[{min_val},{max_val}] {range_desc}")
            if parts:
                lines.append(f"{prefix} tile coverage (clips={self.coverage_clips}): " + ", ".join(parts))
        if self.registration_clips or self.fallback_clips:
            lines.append(
                f"{prefix} coverage sources registration={self.registration_clips} fallback={self.fallback_clips}"
            )
        if self.fallback_reasons:
            reason_desc = ", ".join(f"{k}:{v}" for k, v in sorted(self.fallback_reasons.items()))
            lines.append(f"{prefix} fallback reasons {{{reason_desc}}}")
        if self.boundary_hist:
            hist_desc = ", ".join(f"{k}:{v}" for k, v in sorted(self.boundary_hist.items()))
            lines.append(f"{prefix} boundary-keys {{{hist_desc}}}")
        for report in self.clip_reports:
            lines.append(f"{prefix} {report}")
        for head, stats in self.comparison.items():
            if stats.count <= 0:
                continue
            base_f1 = stats.baseline_f1_sum / stats.count
            fused_f1 = stats.fused_f1_sum / stats.count
            base_rate = stats.baseline_pred_rate_sum / stats.count
            fused_rate = stats.fused_pred_rate_sum / stats.count
            pos_rate = stats.pos_rate_sum / stats.count
            lines.append(
                f"{prefix} {head} baseline_f1={base_f1:.3f} fused_f1={fused_f1:.3f} "
                f"pred_rate={base_rate:.4f}->{fused_rate:.4f} pos_rate≈{pos_rate:.4f} "
                f"max|Δlogit|={stats.max_abs_logit_diff:.3e}"
            )
        return lines


def resolve_global_fusion_config(decoder_cfg: Optional[Mapping[str, Any]]) -> GlobalFusionConfig:
    cfg = decoder_cfg or {}
    raw = cfg.get("global_fusion") if isinstance(cfg, Mapping) else None
    block = raw if isinstance(raw, Mapping) else {}

    enabled = bool(block.get("enabled"))
    mode = str(block.get("mode", "masked_mean"))
    cushion_keys = int(block.get("cushion_keys", 2) or 0)
    apply_to_raw = block.get("apply_to", _DEFAULT_APPLY_TO)
    if isinstance(apply_to_raw, str):
        apply_candidates = [apply_to_raw]
    else:
        apply_candidates = list(apply_to_raw) if isinstance(apply_to_raw, Sequence) else []
    apply_to = tuple(
        sorted(
            {
                str(head).strip().lower()
                for head in apply_candidates
                if head is not None and str(head).strip()
            }
        )
    ) or _DEFAULT_APPLY_TO
    weighting = str(block.get("weighting", "uniform"))

    consistency_block = block.get("consistency_check")
    if not isinstance(consistency_block, Mapping):
        consistency_block = {}
    consistency_enabled = bool(consistency_block.get("enabled"))
    consistency_batches = int(consistency_block.get("batches", 2) or 0)

    return GlobalFusionConfig(
        enabled=enabled,
        mode=mode,
        cushion_keys=cushion_keys,
        apply_to=apply_to,
        weighting=weighting,
        consistency_check=consistency_enabled and enabled,
        consistency_batches=consistency_batches if consistency_enabled and enabled else 0,
    )


def resolve_tile_key_mask(
    video_uid: Optional[str],
    *,
    cache: TileSupportCache,
    cache_scope: CacheScope,
    reg_meta_cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    reg_refiner: Optional[RegistrationRefiner] = None,
    num_tiles: int,
    cushion_keys: int,
    n_keys: int,
    canonical_hw: Optional[Tuple[int, int]] = None,
) -> TileMaskResult:
    if cache is None:
        raise ValueError("TileSupportCache instance is required for resolve_tile_key_mask")
    reg_meta_cache = reg_meta_cache or {}
    lookup_key = _registration_cache_key(video_uid) if video_uid else None
    key_hw = canonical_hw or getattr(reg_refiner, "canonical_hw", None)
    cache_key = make_tile_cache_key(
        lookup_key,
        num_tiles=num_tiles,
        cushion_keys=cushion_keys,
        n_keys=n_keys,
        canonical_hw=key_hw,
    )
    cached = cache.get(cache_scope, cache_key)
    if cached is not None:
        return cached
    reg_meta_key = lookup_key or video_uid
    reg_meta = reg_meta_cache.get(reg_meta_key) if reg_meta_key else None
    if reg_meta is None and reg_meta_key and reg_refiner is not None:
        reg_meta = reg_refiner.get_geometry_metadata(reg_meta_key)
        if reg_meta is not None:
            reg_meta_cache[reg_meta_key] = reg_meta
    lookup_attempted = reg_meta_key is not None
    result = build_tile_key_mask(
        reg_meta,
        num_tiles=num_tiles,
        cushion_keys=cushion_keys,
        n_keys=n_keys,
    )
    synthetic_used = False
    synthetic_attempted = False
    if (not result.registration_based) and video_uid and reg_refiner is not None:
        reason = result.fallback_reason or "unknown"
        if reason in {"missing_key_geometry", "missing_tile_bounds"}:
            synthetic_attempted = True
            width_hint = _metadata_width_hint(reg_meta) or float(reg_refiner.canonical_hw[1])
            synth_meta = build_canonical_registration_metadata(width_hint, num_tiles, n_keys=n_keys)
            synth_meta["target_hw"] = list(reg_refiner.canonical_hw)
            reg_meta = synth_meta
            if reg_meta_key:
                reg_meta_cache[reg_meta_key] = synth_meta
            result = build_tile_key_mask(
                reg_meta,
                num_tiles=num_tiles,
                cushion_keys=cushion_keys,
                n_keys=n_keys,
            )
            synthetic_used = result.registration_based
    if (not result.registration_based) and video_uid:
        reason = result.fallback_reason or "unknown"
        if not lookup_attempted:
            cache_state = "skipped"
        elif reg_meta is None:
            cache_state = "no_entry"
        elif synthetic_attempted:
            cache_state = "synthetic_failed"
        else:
            cache_state = "entry_unusable"
        query = reg_meta_key or "<none>"
        refiner_entries = reg_refiner.cache_size if reg_refiner is not None else len(reg_meta_cache)
        print(
            "[fusion] fallback coverage clip={} query_key={} cache_state={} reason={} geometry_source={} tile_source={} refiner_entries={}".format(
                video_uid,
                query,
                cache_state,
                reason,
                result.geometry_source,
                result.tile_source,
                refiner_entries,
            ),
            flush=True,
        )
        if reg_refiner is not None and refiner_entries > 0:
            _log_cache_probe_once(reg_refiner)
    elif synthetic_used and video_uid:
        print(f"[fusion] synthesized registration metadata clip={video_uid} modality=canonical", flush=True)
    cache.put(cache_scope, cache_key, result)
    return result


@dataclass(frozen=True)
class TileMaskBatch:
    tensor: torch.Tensor
    records: List[TileMaskResult]


def build_batch_tile_mask(
    video_uids: Sequence[Optional[str]],
    *,
    cache: TileSupportCache,
    cache_scope: CacheScope = "train",
    include_records: bool = False,
    reg_meta_cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    reg_refiner: Optional[RegistrationRefiner] = None,
    num_tiles: int,
    cushion_keys: int,
    n_keys: int,
    canonical_hw: Optional[Tuple[int, int]] = None,
) -> TileMaskBatch:
    if not video_uids:
        empty = torch.zeros((0, num_tiles, n_keys), dtype=torch.float32)
        return TileMaskBatch(tensor=empty, records=[])
    masks: List[np.ndarray] = []
    records: List[TileMaskResult] = [] if include_records else []
    for video_uid in video_uids:
        record = resolve_tile_key_mask(
            video_uid,
            cache=cache,
            cache_scope=cache_scope,
            reg_meta_cache=reg_meta_cache,
            reg_refiner=reg_refiner,
            num_tiles=num_tiles,
            cushion_keys=cushion_keys,
            n_keys=n_keys,
            canonical_hw=canonical_hw,
        )
        if include_records:
            records.append(record)
        masks.append(record.mask.astype(np.float32))
    stacked = np.stack(masks, axis=0)
    return TileMaskBatch(tensor=torch.from_numpy(stacked), records=records)


_FUSE_SHAPE_LOGGED = False


def fuse_tile_logits(tile_logits: torch.Tensor, tile_mask: torch.Tensor, *, mode: str = "masked_mean") -> torch.Tensor:
    if mode != "masked_mean":
        raise ValueError(f"Unsupported fusion mode '{mode}'")
    if tile_logits.ndim != 4:
        raise ValueError(f"Expected 4D tensor for tile logits, got shape {tuple(tile_logits.shape)}")
    batch, time_steps, tiles, key_dim = tile_logits.shape
    if batch == 0 or tiles == 0:
        return tile_logits.mean(dim=2)
    mask = tile_mask
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[1] != tiles or mask.shape[2] != key_dim:
        raise ValueError(
            f"Tile mask shape {tuple(mask.shape)} incompatible with per-tile logits {tuple(tile_logits.shape)} "
            "(expected canonical (B, T, tiles, K) layout)."
        )
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1)
    elif mask.shape[0] != batch:
        raise ValueError(f"Tile mask batch {mask.shape[0]} != logits batch {batch}")
    mask = mask.to(tile_logits.device, dtype=tile_logits.dtype)
    global _FUSE_SHAPE_LOGGED
    if not _FUSE_SHAPE_LOGGED:
        print(
            "[fusion] per-tile multiply: logits(B,T,tiles,K)={} mask(B,tiles,K)={} tile_axis=2".format(
                tuple(tile_logits.shape),
                tuple(mask.shape),
            ),
            flush=True,
        )
        _FUSE_SHAPE_LOGGED = True
    weights = mask.unsqueeze(1)  # (B, 1, tiles, keys)
    weighted_sum = (tile_logits * weights).sum(dim=2)  # (B, T, keys)
    counts = weights.sum(dim=2)  # (B, 1, keys)
    fused = weighted_sum / counts.clamp_min(1e-6)
    fallback = tile_logits.mean(dim=2)
    no_cover = counts <= 1e-6
    if no_cover.any():
        fused = torch.where(no_cover.expand(-1, time_steps, -1), fallback, fused)
    return fused


__all__ = [
    "GlobalFusionConfig",
    "FusionDebugState",
    "resolve_global_fusion_config",
    "build_batch_tile_mask",
    "fuse_tile_logits",
    "resolve_tile_key_mask",
    "TileMaskBatch",
]
