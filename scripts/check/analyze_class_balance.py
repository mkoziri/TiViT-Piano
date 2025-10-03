#!/usr/bin/env python3
"""Purpose:
    Analyze class distributions across pitch, onset, and offset targets to inform
    loss weighting and model design decisions. This is crucial for handling
    extreme class imbalance in piano transcription.

Key Functions:
    - compute_class_stats(): Calculate positive rates and imbalance ratios
    - analyze_frame_level_targets(): Detailed frame-wise statistics
    - analyze_clip_level_targets(): Clip-aggregated statistics
    - generate_imbalance_report(): Summary with recommendations

CLI:
    python scripts/check/analyze_class_balance.py --config configs/config.yaml --split train --num-batches 50
with default values for arguments if not present the ones given here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# repo imports (without relying on ``pip install -e``)
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config  # type: ignore  # local module
from data.loader import make_dataloader  # type: ignore  # local module


@dataclass
class ClassDistribution:
    """Lightweight container storing aggregated class counts."""

    positive: np.ndarray
    total: np.ndarray

    @property
    def positive_rate(self) -> np.ndarray:
        total = np.clip(self.total.astype(np.float64), a_min=1.0, a_max=None)
        return self.positive.astype(np.float64) / total

    @property
    def imbalance_ratio(self) -> np.ndarray:
        pos_rate = self.positive_rate
        neg_rate = np.clip(1.0 - pos_rate, a_min=0.0, a_max=None)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pos_rate > 0.0, neg_rate / np.maximum(pos_rate, 1e-12), np.inf)
        return ratio

    @property
    def overall_positive_rate(self) -> float:
        total = float(np.clip(self.total.sum(), 1.0, None))
        return float(self.positive.sum() / total)

    @property
    def suggested_pos_weight(self) -> float:
        """Heuristic BCE ``pos_weight`` based on imbalance."""

        rate = self.overall_positive_rate
        if rate <= 0.0:
            return float("inf")
        return float((1.0 - rate) / max(rate, 1e-12))

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "positive": self.positive,
            "total": self.total,
            "positive_rate": self.positive_rate,
            "imbalance_ratio": self.imbalance_ratio,
        }


def compute_class_stats(positive: np.ndarray, total: np.ndarray) -> ClassDistribution:
    """Calculate positive/total statistics for a binary target array."""

    pos = np.asarray(positive, dtype=np.float64)
    tot = np.asarray(total, dtype=np.float64)
    if pos.shape != tot.shape:
        raise ValueError(f"Shape mismatch: positive {pos.shape} vs total {tot.shape}")
    return ClassDistribution(positive=pos, total=tot)


def _ensure_numpy(arr: torch.Tensor) -> np.ndarray:
    return arr.detach().to(dtype=torch.float64).cpu().numpy()


def _sum_over_batch_time(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Sum tensor across all dims except the last (pitch dimension)."""

    if tensor.dim() == 0:
        raise ValueError("Expected at least 1D tensor with pitch dimension")
    if tensor.dim() == 1:
        # already pitch dimension only
        summed = tensor
        total_factors = 1
    else:
        reduce_dims = tuple(range(tensor.dim() - 1))
        summed = tensor.sum(dim=reduce_dims)
        total_factors = int(np.prod(tensor.shape[:-1]))
    summed_np = _ensure_numpy(summed)
    totals_np = np.full_like(summed_np, fill_value=float(total_factors), dtype=np.float64)
    return summed_np, totals_np


def analyze_frame_level_targets(
    loader, num_batches: Optional[int] = None
) -> Mapping[str, ClassDistribution]:
    """Aggregate frame-wise class statistics from the dataloader."""

    accum_pos: MutableMapping[str, np.ndarray] = {}
    accum_tot: MutableMapping[str, np.ndarray] = {}

    keys = ("pitch_roll", "onset_roll", "offset_roll")

    for idx, batch in enumerate(loader):
        if num_batches is not None and idx >= num_batches:
            break
        if not isinstance(batch, Mapping):
            continue
        for key in keys:
            tensor = batch.get(key)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            tensor = tensor.float()
            pos_np, tot_np = _sum_over_batch_time(tensor)
            if key not in accum_pos:
                accum_pos[key] = np.zeros_like(pos_np)
                accum_tot[key] = np.zeros_like(tot_np)
            # shapes should match; broadcast defensively
            accum_pos[key] = accum_pos[key] + pos_np
            accum_tot[key] = accum_tot[key] + tot_np

    stats: Dict[str, ClassDistribution] = {}
    for key in accum_pos:
        stats[key] = compute_class_stats(accum_pos[key], accum_tot[key])
    return stats


def analyze_clip_level_targets(
    loader, num_batches: Optional[int] = None
) -> Mapping[str, ClassDistribution]:
    """Aggregate clip-level class statistics from the dataloader."""

    accum_pos: MutableMapping[str, np.ndarray] = {}
    accum_tot: MutableMapping[str, np.ndarray] = {}

    keys = ("pitch", "onset", "offset")

    for idx, batch in enumerate(loader):
        if num_batches is not None and idx >= num_batches:
            break
        if not isinstance(batch, Mapping):
            continue
        for key in keys:
            tensor = batch.get(key)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            tensor = tensor.float()
            pos_np, tot_np = _sum_over_batch_time(tensor)
            if key not in accum_pos:
                accum_pos[key] = np.zeros_like(pos_np)
                accum_tot[key] = np.zeros_like(tot_np)
            accum_pos[key] = accum_pos[key] + pos_np
            accum_tot[key] = accum_tot[key] + tot_np

    stats: Dict[str, ClassDistribution] = {}
    for key in accum_pos:
        stats[key] = compute_class_stats(accum_pos[key], accum_tot[key])
    return stats


def _format_topk_rates(
    stats: ClassDistribution, midi_offset: int, k: int = 5, *, label: str
) -> str:
    pos_rate = stats.positive_rate
    order = np.argsort(pos_rate)
    lowest = order[:k]
    highest = order[-k:][::-1]
    parts = [f"  • Lowest {label} positive rates:"]
    for idx in lowest:
        midi = midi_offset + int(idx)
        rate = pos_rate[idx]
        imb = stats.imbalance_ratio[idx]
        parts.append(f"    - MIDI {midi:3d}: rate={rate:.6f}, neg/pos≈{imb:.1f}")
    parts.append(f"  • Highest {label} positive rates:")
    for idx in highest:
        midi = midi_offset + int(idx)
        rate = pos_rate[idx]
        imb = stats.imbalance_ratio[idx]
        parts.append(f"    - MIDI {midi:3d}: rate={rate:.6f}, neg/pos≈{imb:.1f}")
    return "\n".join(parts)


def generate_imbalance_report(
    frame_stats: Mapping[str, ClassDistribution],
    clip_stats: Mapping[str, ClassDistribution],
    *,
    frame_note_min: int,
    clip_note_min: int,
) -> str:
    """Generate a human-readable summary with imbalance recommendations."""

    lines = []
    if frame_stats:
        lines.append("Frame-level targets:")
        for key, stats in frame_stats.items():
            overall = stats.overall_positive_rate
            pos_weight = stats.suggested_pos_weight
            lines.append(
                f"- {key}: overall positive rate={overall:.6f} (pos_weight≈{pos_weight:.1f})"
            )
            if stats.positive.size:
                lines.append(
                    _format_topk_rates(stats, frame_note_min, label=f"{key} frame")
                )
    else:
        lines.append("Frame-level targets: [not available]")

    if clip_stats:
        lines.append("\nClip-level targets:")
        for key, stats in clip_stats.items():
            overall = stats.overall_positive_rate
            pos_weight = stats.suggested_pos_weight
            lines.append(
                f"- {key}: overall positive rate={overall:.6f} (pos_weight≈{pos_weight:.1f})"
            )
            if stats.positive.size:
                lines.append(
                    _format_topk_rates(stats, clip_note_min, label=f"{key} clip")
                )
    else:
        lines.append("\nClip-level targets: [not available]")

    lines.append(
        "\nRecommendation: consider scaling BCE/Focal pos_weight using the above "
        "neg:pos ratios and rebalancing heads with very low activity."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute class balance statistics for piano transcription datasets"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-batches", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    frame_cfg = dataset_cfg.get("frame_targets", {}) if isinstance(dataset_cfg, Mapping) else {}
    frame_note_min = int(frame_cfg.get("note_min", 21))
    clip_note_min = 21  # dataset uses piano MIDI range by default

    loader = make_dataloader(cfg, split=args.split, drop_last=False)

    num_batches = None if args.num_batches is None or args.num_batches <= 0 else args.num_batches

    frame_stats = analyze_frame_level_targets(loader, num_batches=num_batches)
    clip_stats = analyze_clip_level_targets(loader, num_batches=num_batches)

    report = generate_imbalance_report(
        frame_stats,
        clip_stats,
        frame_note_min=frame_note_min,
        clip_note_min=clip_note_min,
    )

    print(report)


if __name__ == "__main__":
    main()
