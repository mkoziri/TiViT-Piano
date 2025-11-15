"""Global fusion helpers for per-tile TiViT logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from collections import Counter

import numpy as np
import torch

from .tile_keymap import TileMaskResult, build_tile_key_mask

_DEFAULT_APPLY_TO = ("onset", "offset", "pitch")
_FALLBACK_CACHE_KEY = "__fallback__"


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
    clip_id: Optional[str],
    *,
    reg_meta_cache: Mapping[str, Mapping[str, Any]],
    mask_cache: MutableMapping[str, TileMaskResult],
    num_tiles: int,
    cushion_keys: int,
    n_keys: int,
) -> TileMaskResult:
    cache_key = clip_id or f"{_FALLBACK_CACHE_KEY}_{n_keys}"
    if cache_key in mask_cache:
        return mask_cache[cache_key]
    reg_meta = reg_meta_cache.get(clip_id) if clip_id else None
    result = build_tile_key_mask(
        reg_meta,
        num_tiles=num_tiles,
        cushion_keys=cushion_keys,
        n_keys=n_keys,
    )
    if (not result.registration_based) and clip_id:
        reason = result.fallback_reason or "unknown"
        print(f"[fusion] fallback coverage clip={clip_id} reason={reason}", flush=True)
    mask_cache[cache_key] = result
    return result


@dataclass(frozen=True)
class TileMaskBatch:
    tensor: torch.Tensor
    records: List[TileMaskResult]


def build_batch_tile_mask(
    clip_ids: Sequence[Optional[str]],
    *,
    reg_meta_cache: Mapping[str, Mapping[str, Any]],
    mask_cache: MutableMapping[str, TileMaskResult],
    num_tiles: int,
    cushion_keys: int,
    n_keys: int,
) -> TileMaskBatch:
    if not clip_ids:
        empty = torch.zeros((0, num_tiles, n_keys), dtype=torch.float32)
        return TileMaskBatch(tensor=empty, records=[])
    masks: List[np.ndarray] = []
    records: List[TileMaskResult] = []
    for clip_id in clip_ids:
        record = resolve_tile_key_mask(
            clip_id,
            reg_meta_cache=reg_meta_cache,
            mask_cache=mask_cache,
            num_tiles=num_tiles,
            cushion_keys=cushion_keys,
            n_keys=n_keys,
        )
        records.append(record)
        masks.append(record.mask.astype(np.float32))
    stacked = np.stack(masks, axis=0)
    return TileMaskBatch(tensor=torch.from_numpy(stacked), records=records)


def fuse_tile_logits(tile_logits: torch.Tensor, tile_mask: torch.Tensor, *, mode: str = "masked_mean") -> torch.Tensor:
    if mode != "masked_mean":
        raise ValueError(f"Unsupported fusion mode '{mode}'")
    if tile_logits.ndim != 4:
        raise ValueError(f"Expected 4D tensor for tile logits, got shape {tuple(tile_logits.shape)}")
    batch, tiles, time_steps, key_dim = tile_logits.shape
    if batch == 0 or tiles == 0:
        return tile_logits.mean(dim=1)
    mask = tile_mask
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[1] != tiles or mask.shape[2] != key_dim:
        raise ValueError(f"Tile mask shape {tuple(mask.shape)} incompatible with logits {tuple(tile_logits.shape)}")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1)
    elif mask.shape[0] != batch:
        raise ValueError(f"Tile mask batch {mask.shape[0]} != logits batch {batch}")
    mask = mask.to(tile_logits.device, dtype=tile_logits.dtype)
    weights = mask.unsqueeze(2)  # (B, tiles, 1, keys)
    weighted_sum = (tile_logits * weights).sum(dim=1)  # (B, T, keys)
    counts = weights.sum(dim=1)  # (B, 1, keys)
    fused = weighted_sum / counts.clamp_min(1e-6)
    fallback = tile_logits.mean(dim=1)
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
