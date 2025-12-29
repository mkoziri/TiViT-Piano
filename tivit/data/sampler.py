"""
Purpose:
    Build onset-balanced samplers for the new data layout, reusing legacy
    weighting logic and metadata handling.

Key Functions/Classes:
    - SamplerGroups: Holds onset/nearmiss/background indices.
    - build_onset_balanced_sampler(): Returns a WeightedRandomSampler when enabled.

CLI Arguments:
    (none)

Usage:
    sampler = build_onset_balanced_sampler(dataset, sampler_cfg, base_seed=seed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import WeightedRandomSampler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplerGroups:
    """Container for the sampler's index groups."""

    size: int
    onset: Tuple[int, ...]
    nearmiss: Tuple[int, ...]
    background: Tuple[int, ...]


def build_onset_balanced_sampler(dataset, sampler_cfg: Optional[Mapping[str, Any]], *, base_seed: int) -> Optional[WeightedRandomSampler]:
    """Return a WeightedRandomSampler when onset balancing is enabled."""

    if not sampler_cfg:
        return None
    mode = str(sampler_cfg.get("mode", "")).strip().lower()
    if mode != "onset_balanced":
        return None

    dataset_len = len(dataset)
    if dataset_len <= 0:
        LOGGER.warning("Onset-balanced sampler requested but dataset is empty.")
        return None

    fractions = _normalize_fractions(
        sampler_cfg.get("onset_frac"),
        sampler_cfg.get("nearmiss_frac"),
        sampler_cfg.get("bg_frac"),
    )
    if fractions is None:
        LOGGER.warning("Onset-balanced sampler disabled; invalid fraction configuration: %s", sampler_cfg)
        return None

    groups = _collect_sampler_groups(dataset, sampler_cfg, dataset_len)
    if groups is None:
        LOGGER.info("Onset-balanced sampler falling back to uniform sampling (metadata unavailable).")
        return None

    weights_tensor = _build_group_weights(groups, fractions)
    if weights_tensor is None:
        LOGGER.info("Onset-balanced sampler skipped; missing valid group assignments.")
        return None
    weight_values = weights_tensor.tolist()

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(base_seed) + 131)

    LOGGER.info(
        "Onset-balanced sampler enabled | size=%d onset=%d nearmiss=%d bg=%d fractions=%s radius=%s",
        groups.size,
        len(groups.onset),
        len(groups.nearmiss),
        len(groups.background),
        fractions,
        sampler_cfg.get("nearmiss_radius"),
    )

    return WeightedRandomSampler(
        weights=weight_values,
        num_samples=groups.size,
        replacement=True,
        generator=sampler_generator,
    )


def _normalize_fractions(onset, nearmiss, background) -> Optional[Dict[str, float]]:
    vals = [
        _coerce_non_negative_float(onset, default=0.5),
        _coerce_non_negative_float(nearmiss, default=0.25),
        _coerce_non_negative_float(background, default=0.25),
    ]
    total = sum(vals)
    if total <= 0:
        return None
    onset_frac, near_frac, bg_frac = (v / total for v in vals)
    return {"onset": onset_frac, "nearmiss": near_frac, "background": bg_frac}


def _coerce_non_negative_float(value, default: float) -> float:
    if value is None:
        return max(0.0, float(default))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(0.0, float(default))
    return max(0.0, parsed)


def _collect_sampler_groups(dataset, sampler_cfg: Mapping[str, Any], dataset_len: int) -> Optional[SamplerGroups]:
    radius = int(_coerce_non_negative_float(sampler_cfg.get("nearmiss_radius"), 0.0))
    metadata = _resolve_sampler_metadata(dataset, radius)
    if not metadata:
        return None

    onset = _collect_indices(metadata, dataset_len, ["onset", "onset_indices", "positive"])
    nearmiss = _collect_indices(metadata, dataset_len, ["nearmiss", "near_miss", "nearmiss_indices"])

    start_map = _collect_start_frames(metadata, dataset_len)
    if not nearmiss and radius > 0 and start_map:
        nearmiss = _derive_nearmiss_from_radius(onset, start_map, radius)

    background = _collect_indices(metadata, dataset_len, ["background", "background_indices"])
    if not background:
        claimed = set(onset) | set(nearmiss)
        background = tuple(idx for idx in range(dataset_len) if idx not in claimed)

    if not (onset or nearmiss or background):
        return None

    return SamplerGroups(size=dataset_len, onset=tuple(onset), nearmiss=tuple(nearmiss), background=tuple(background))


def _resolve_sampler_metadata(dataset, radius: int) -> Optional[Mapping[str, Any]]:
    builder = getattr(dataset, "build_onset_sampler_metadata", None)
    if callable(builder):
        try:
            meta = builder(nearmiss_radius=radius)
        except TypeError:
            try:
                meta = builder()
            except TypeError:
                meta = builder(radius)
        if isinstance(meta, Mapping):
            return dict(meta)

    attr = getattr(dataset, "onset_sampler_metadata", None)
    if isinstance(attr, Mapping):
        return dict(attr)

    attr = getattr(dataset, "sampler_metadata", None)
    if isinstance(attr, Mapping):
        payload = attr.get("onset_balanced", attr)
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def _collect_indices(metadata: Mapping[str, Any], dataset_len: int, keys: Sequence[str]) -> Tuple[int, ...]:
    for key in keys:
        values = metadata.get(key)
        if isinstance(values, Sequence):
            indices = [int(v) for v in values if 0 <= int(v) < dataset_len]
            return tuple(indices)
    return tuple()


def _collect_start_frames(metadata: Mapping[str, Any], dataset_len: int) -> Dict[int, int]:
    result: Dict[int, int] = {}
    starts = metadata.get("start_frames")
    if isinstance(starts, Mapping):
        for key, value in starts.items():
            try:
                idx = int(key)
                val = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < dataset_len:
                result[idx] = val
    return result


def _derive_nearmiss_from_radius(onset: Sequence[int], start_map: Mapping[int, int], radius: int) -> Tuple[int, ...]:
    nearmiss: set[int] = set()
    onset_set = set(onset)
    for idx, start in start_map.items():
        if idx in onset_set:
            continue
        for onset_idx in onset:
            start_on = start_map.get(onset_idx, None)
            if start_on is None:
                continue
            if abs(start - start_on) <= radius:
                nearmiss.add(idx)
                break
    return tuple(sorted(nearmiss))


def _build_group_weights(groups: SamplerGroups, fractions: Mapping[str, float]) -> Optional[torch.Tensor]:
    size = groups.size
    if size <= 0:
        return None
    weights = torch.zeros(size, dtype=torch.float32)
    for key, indices in (("onset", groups.onset), ("nearmiss", groups.nearmiss), ("background", groups.background)):
        if not indices:
            continue
        weight = float(fractions.get(key, 0.0))
        if weight <= 0:
            continue
        weights[list(indices)] = weight
    if weights.sum() <= 0:
        return None
    return weights


__all__ = ["build_onset_balanced_sampler", "SamplerGroups"]
