"""Helper utilities for optional dataset samplers.

This module centralizes logic for building sampling strategies that bias
towards onset-heavy clips while remaining opt-in.  Datasets can expose custom
metadata (index lists, cached start positions, etc.) to inform the sampler,
but the helper gracefully falls back to standard uniform sampling whenever
that metadata is missing.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Dict, Any

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


def build_onset_balanced_sampler(
    dataset,
    sampler_cfg: Optional[Mapping[str, Any]],
    *,
    base_seed: int,
) -> Optional[WeightedRandomSampler]:
    """Return a ``WeightedRandomSampler`` when onset balancing is enabled.

    Parameters
    ----------
    dataset:
        The PyTorch dataset instance attached to the dataloader.
    sampler_cfg:
        The ``dataset.sampler`` configuration mapping.
    base_seed:
        Seed used to initialize the sampler's RNG for reproducibility.
    """

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
        LOGGER.warning(
            "Onset-balanced sampler disabled; invalid fraction configuration: %s",
            sampler_cfg,
        )
        return None

    groups = _collect_sampler_groups(dataset, sampler_cfg, dataset_len)
    if groups is None:
        LOGGER.info(
            "Onset-balanced sampler falling back to uniform sampling "
            "(metadata unavailable).",
        )
        return None

    weights_tensor = _build_group_weights(groups, fractions)
    if weights_tensor is None:
        LOGGER.info(
            "Onset-balanced sampler skipped; missing valid group assignments.",
        )
        return None
    weight_values = weights_tensor.tolist()

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(base_seed) + 131)

    LOGGER.info(
        "Onset-balanced sampler enabled | size=%d onset=%d nearmiss=%d bg=%d "
        "fractions=%s radius=%s",
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


def _collect_sampler_groups(
    dataset,
    sampler_cfg: Mapping[str, Any],
    dataset_len: int,
) -> Optional[SamplerGroups]:
    radius = int(_coerce_non_negative_float(sampler_cfg.get("nearmiss_radius"), 0.0))
    metadata = _resolve_sampler_metadata(dataset, radius)
    if not metadata:
        return None

    onset = _collect_indices(metadata, dataset_len, ["onset", "onset_indices", "positive"])
    nearmiss = _collect_indices(
        metadata,
        dataset_len,
        ["nearmiss", "near_miss", "nearmiss_indices"],
    )

    start_map = _collect_start_frames(metadata, dataset_len)
    if not nearmiss and radius > 0 and start_map:
        nearmiss = _derive_nearmiss_from_radius(onset, start_map, radius)

    background = _collect_indices(
        metadata,
        dataset_len,
        ["background", "background_indices"],
    )
    if not background:
        claimed = set(onset) | set(nearmiss)
        background = tuple(idx for idx in range(dataset_len) if idx not in claimed)

    if not (onset or nearmiss or background):
        return None

    return SamplerGroups(
        size=dataset_len,
        onset=tuple(onset),
        nearmiss=tuple(nearmiss),
        background=tuple(background),
    )


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


def _collect_indices(
    metadata: Mapping[str, Any],
    dataset_len: int,
    keys: Sequence[str],
) -> Tuple[int, ...]:
    collected: set[int] = set()
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        collected.update(_normalize_index_container(raw, dataset_len))
    return tuple(sorted(collected))


def _normalize_index_container(raw: Any, dataset_len: int) -> set[int]:
    indices: set[int] = set()
    if isinstance(raw, Mapping):
        iterator = raw.items()
        for idx, flag in iterator:
            if not flag:
                continue
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < dataset_len:
                indices.add(idx_int)
        return indices

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        is_mask = len(raw) == dataset_len and all(
            isinstance(v, (bool, int, float)) for v in raw
        )
        if is_mask:
            for idx, flag in enumerate(raw):
                if bool(flag) and 0 <= idx < dataset_len:
                    indices.add(idx)
            return indices
        for val in raw:
            try:
                idx_int = int(val)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < dataset_len:
                indices.add(idx_int)
        return indices

    try:
        idx_int = int(raw)
    except (TypeError, ValueError):
        return indices
    if 0 <= idx_int < dataset_len:
        indices.add(idx_int)
    return indices


def _collect_start_frames(metadata: Mapping[str, Any], dataset_len: int) -> Dict[int, int]:
    source = (
        metadata.get("start_frames")
        or metadata.get("start_indices")
        or metadata.get("starts")
    )
    if source is None:
        return {}
    result: Dict[int, int] = {}
    if isinstance(source, Mapping):
        iterator = source.items()
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        iterator = enumerate(source)
    else:
        return {}
    for idx, value in iterator:
        if value is None:
            continue
        try:
            idx_int = int(idx)
            start_int = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_int < dataset_len:
            result[idx_int] = start_int
    return result


def _derive_nearmiss_from_radius(
    onset: Sequence[int],
    start_map: Mapping[int, int],
    radius: int,
) -> Tuple[int, ...]:
    if not onset or radius <= 0:
        return tuple()
    ordered = sorted((start, idx) for idx, start in start_map.items())
    starts = [start for start, _ in ordered]
    candidates = set()
    for idx in onset:
        start = start_map.get(idx)
        if start is None:
            continue
        left = bisect.bisect_left(starts, start - radius)
        right = bisect.bisect_right(starts, start + radius)
        for pos in range(left, right):
            neighbor_idx = ordered[pos][1]
            if neighbor_idx == idx:
                continue
            candidates.add(neighbor_idx)
    candidates.difference_update(onset)
    return tuple(sorted(candidates))


def _build_group_weights(
    groups: SamplerGroups,
    fractions: Mapping[str, float],
) -> Optional[torch.Tensor]:
    weights = torch.ones(groups.size, dtype=torch.float64)
    total_assigned = 0.0

    for key, indices in (
        ("onset", groups.onset),
        ("nearmiss", groups.nearmiss),
        ("background", groups.background),
    ):
        frac = fractions.get(key, 0.0)
        if frac <= 0 or not indices:
            continue
        per_weight = frac / max(len(indices), 1)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        weights[idx_tensor] = per_weight
        total_assigned += frac

    if total_assigned <= 0:
        return None
    return weights
