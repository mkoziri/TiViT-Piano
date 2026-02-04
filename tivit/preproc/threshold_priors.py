"""Dataset-only priors and threshold bounds generator.

Purpose:
    - Compute per-split label statistics without model scores.
    - Emit recommended threshold ranges and peak-picking constraints.
    - Keep the workflow independent from training/runtime pipelines.

Key Functions/Classes:
    - compute_split_stats(): Iterate dataset entries to build SplitStats.
    - recommend_bounds(): Derive threshold ranges and constraints.
    - main(): CLI entrypoint for writing YAML outputs.

CLI Arguments:
    - config: Dataset config YAML.
    - output: Output YAML path (stats + recommendations).
    - splits: Optional comma-separated splits to process.
    - max-clips: Optional cap on dataset entries per split.
    - disable-av-sync: Skip AV sync adjustments.
    - disable-coverage: Skip tile/key visibility coverage.
    - disable-targets: Skip frame-target roll stats.

Usage:
    python -m tivit.preproc.threshold_priors --config tivit/configs/dataset/pianoyt.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
import torch

from tivit.core.config import resolve_config_chain
from tivit.data.cache.frame_target_cache import FrameTargetCache
from tivit.data.sync import apply_sync, resolve_sync
from tivit.data.targets.frame_targets import FrameTargetSpec, prepare_frame_targets
from tivit.decoder.registration_geometry import build_canonical_registration_metadata
from tivit.decoder.tile_keymap import build_tile_key_mask
from tivit.preproc.threshold_stats import SplitStats, build_log_bins
from tivit.utils.imbalance import map_ratio_to_band, sanitize_ratio


class NullFrameTargetCache(FrameTargetCache):
    """Cache stub that never persists frame-target tensors."""

    def __init__(self) -> None:
        self.cache_dir = Path(".")
        self._lock_timeout_logged = False

    def load(self, _: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        return None, False

    def save(self, *_: Any, **__: Any) -> bool:
        return False


def _resolve_dataset_class(name: str):
    key = (name or "").strip().lower()
    if key == "omaps":
        from tivit.data.datasets.omaps_impl import OMAPSDataset

        return OMAPSDataset
    if key == "pianoyt":
        from tivit.data.datasets.pianoyt_impl import PianoYTDataset

        return PianoYTDataset
    if key == "pianovam":
        from tivit.data.datasets.pianovam_impl import PianoVAMDataset

        return PianoVAMDataset
    raise ValueError(f"Unsupported dataset name '{name}'. Expected omaps, pianoyt, or pianovam.")


def _collect_splits(dataset_cfg: Mapping[str, Any]) -> List[str]:
    splits: List[str] = []

    def _add(val: Any) -> None:
        if isinstance(val, str) and val and val not in splits:
            splits.append(val)

    for key in ("split_train", "split_val", "split_test"):
        _add(dataset_cfg.get(key))
    manifest = dataset_cfg.get("manifest")
    if isinstance(manifest, Mapping):
        for key in manifest.keys():
            _add(str(key))
    _add(dataset_cfg.get("split"))
    return list(splits) if splits else ["train"]




def _resolve_pos_weight_bands(cfg: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    heads = cfg.get("training", {}).get("loss", {}).get("heads", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(heads, Mapping):
        heads = {}

    def _band(head: str, default: Tuple[float, float]) -> Tuple[float, float]:
        head_cfg = heads.get(head, {}) if isinstance(heads.get(head), Mapping) else {}
        band = head_cfg.get("pos_weight_band")
        if isinstance(band, Sequence) and not isinstance(band, (str, bytes)) and len(band) == 2:
            return float(band[0]), float(band[1])
        return default

    return {
        "pitch": _band("pitch", (2.0, 4.0)),
        "onset": _band("onset", (3.0, 5.0)),
        "offset": _band("offset", (3.0, 5.0)),
    }


def _pos_weights_from_stats(stats: SplitStats, bands: Mapping[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    weights: Dict[str, List[float]] = {}
    for head in ("pitch", "onset", "offset"):
        pos_counts = np.asarray(stats.pos_counts_by_key.get(head, []), dtype=np.float64)
        if pos_counts.size == 0 or stats.n_keys <= 0:
            continue
        total_cells = float(stats.total_cells.get(head, 0))
        if total_cells <= 0:
            continue
        total_per_key = total_cells / float(stats.n_keys)
        neg = np.clip(total_per_key - pos_counts, 0.0, None)
        ratio = np.where(pos_counts > 0.0, neg / np.maximum(pos_counts, 1e-12), np.inf)
        ratio = sanitize_ratio(ratio)
        weights[head] = map_ratio_to_band(torch.as_tensor(ratio), bands[head]).tolist()
    return weights

def _resolve_threshold_baselines(cfg: Mapping[str, Any]) -> Dict[str, float]:
    metrics_cfg = cfg.get("training", {}).get("metrics", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(metrics_cfg, Mapping):
        metrics_cfg = {}
    base = float(metrics_cfg.get("prob_threshold", 0.2) or 0.2)
    return {
        "pitch": float(metrics_cfg.get("prob_threshold", base) or base),
        "onset": float(metrics_cfg.get("prob_threshold_onset", base) or base),
        "offset": float(metrics_cfg.get("prob_threshold_offset", base) or base),
    }


def _resolve_prior_means(cfg: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    heads = cfg.get("training", {}).get("loss", {}).get("heads", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(heads, Mapping):
        heads = {}
    onset = heads.get("onset", {}) if isinstance(heads.get("onset"), Mapping) else {}
    offset = heads.get("offset", {}) if isinstance(heads.get("offset"), Mapping) else {}
    return {
        "onset": _coerce_optional_float(onset.get("prior_mean")),
        "offset": _coerce_optional_float(offset.get("prior_mean")),
    }


def _resolve_cushion_keys(cfg: Mapping[str, Any]) -> int:
    per_tile = cfg.get("training", {}).get("loss", {}).get("per_tile", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(per_tile, Mapping):
        cushion = _coerce_optional_int(per_tile.get("mask_cushion_keys"))
        if cushion is not None:
            return max(cushion, 0)
    decoder_cfg = cfg.get("decoder", {}) if isinstance(cfg, Mapping) else {}
    global_fusion = decoder_cfg.get("global_fusion", {}) if isinstance(decoder_cfg, Mapping) else {}
    if isinstance(global_fusion, Mapping):
        cushion = _coerce_optional_int(global_fusion.get("cushion_keys"))
        if cushion is not None:
            return max(cushion, 0)
    return 0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(float(value), high))


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _coerce_float(value: Any, default: float) -> float:
    parsed = _coerce_optional_float(value)
    if parsed is None:
        return float(default)
    return parsed


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _threshold_range(
    *,
    baseline: float,
    pos_rate: float,
    ref_rate: float,
    min_width: float,
    max_width: float,
    digits: int,
) -> Dict[str, float | List[float]]:
    eps = 1e-6
    ratio = (pos_rate + eps) / (ref_rate + eps)
    log_ratio = _clamp(math.log10(ratio), -1.0, 1.0)
    shift = -0.08 * log_ratio
    width = min_width + (max_width - min_width) * min(1.0, abs(log_ratio))
    center = _clamp(baseline + shift, 0.05, 0.95)
    low = _clamp(center - width, 0.05, 0.95)
    high = _clamp(center + width, 0.05, 0.95)
    return {
        "range": [round(low, digits), round(high, digits)],
        "default": round(center, digits),
        "baseline": round(float(baseline), digits),
        "pos_rate": round(float(pos_rate), digits),
        "ref_rate": round(float(ref_rate), digits),
    }


def _threshold_bounds(
    thresholds: Mapping[str, Mapping[str, Any]],
    head: str,
    *,
    fallback: float,
    width: float,
) -> Tuple[float, float]:
    entry = thresholds.get(head, {}) if isinstance(thresholds, Mapping) else {}
    raw_range = entry.get("range") if isinstance(entry, Mapping) else None
    if isinstance(raw_range, Sequence) and not isinstance(raw_range, (str, bytes)) and len(raw_range) >= 2:
        low = _coerce_float(raw_range[0], fallback)
        high = _coerce_float(raw_range[1], fallback + width)
        if high < low:
            high = low
        return low, high
    center = _coerce_float(entry.get("default") if isinstance(entry, Mapping) else None, fallback)
    half = max(float(width) * 0.5, 0.0)
    return max(0.0, center - half), min(1.0, center + half)


def _pick_reference_split(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    if "train" in summaries:
        return "train"
    return next(iter(summaries.keys()))


def _derive_peak_picking(summary: Mapping[str, Any]) -> Dict[str, Any]:
    hop = float(summary.get("hop_seconds", 0.0) or 0.0)
    durations = summary.get("note_duration", {}).get("percentiles_sec", {})
    ioi = summary.get("inter_onset_interval", {}).get("percentiles_sec", {})

    ioi_p10 = float(ioi.get("p10", 0.0) or 0.0)
    dur_p10 = float(durations.get("p10", 0.0) or 0.0)

    min_sep_frames = 1
    if hop > 0.0 and ioi_p10 > 0.0:
        min_sep_frames = max(1, int(round(ioi_p10 / hop)))
    min_sep_sec = min_sep_frames * hop

    min_on = 1
    if hop > 0.0 and dur_p10 > 0.0:
        min_on = max(1, int(round(dur_p10 / hop)))
    min_off = min_on
    merge_gap = max(0, min_sep_frames - 1)
    median = max(1, min(7, 2 * min_sep_frames - 1))

    return {
        "min_separation_frames": int(min_sep_frames),
        "min_separation_sec": round(float(min_sep_sec), 6),
        "min_on": int(min_on),
        "min_off": int(min_off),
        "merge_gap": int(merge_gap),
        "median": int(median),
    }


def _derive_aggregation(summary: Mapping[str, Any]) -> Dict[str, Any]:
    poly = summary.get("polyphony", {})
    p50 = int(poly.get("p50", 0) or 0)
    p90 = int(poly.get("p90", 0) or 0)
    tiles = int(summary.get("tiles", 1) or 1)
    tiles = max(1, tiles)

    k_val = max(1, min(3, int(round(p50 / tiles)) or 1))
    top_k = max(3, min(8, int(round(p90)) or 3))

    return {
        "mode": "k_of_p",
        "k": {"onset": k_val, "offset": k_val},
        "top_k": top_k,
        "basis": {"polyphony_p50": p50, "polyphony_p90": p90, "tiles": tiles},
    }


def _derive_tile_priors(summary: Mapping[str, Any], *, digits: int) -> Optional[Dict[str, Any]]:
    per_tile = summary.get("event_density", {}).get("per_tile", {})
    onset_per_min = per_tile.get("onset_per_min")
    coverage_summary = summary.get("key_visibility", {}).get("coverage_summary")
    if onset_per_min is None or coverage_summary is None:
        return None

    density = [float(v) for v in onset_per_min]
    if not density:
        return None
    mean_density = sum(density) / max(1, len(density))
    coverage_means = coverage_summary.get("per_tile_mean") or []
    if not coverage_means:
        return None
    coverage = [float(v) for v in coverage_means]
    mean_coverage = sum(coverage) / max(1, len(coverage))

    deltas = []
    basis = []
    for idx, (d_val, c_val) in enumerate(zip(density, coverage)):
        density_ratio = (d_val / mean_density) if mean_density > 0 else 1.0
        coverage_ratio = (c_val / mean_coverage) if mean_coverage > 0 else 1.0
        ease = max(density_ratio * coverage_ratio, 1e-6)
        delta = _clamp(0.05 * math.log(ease), -0.05, 0.05)
        deltas.append(round(float(delta), digits))
        basis.append(
            {
                "tile": idx,
                "density_ratio": round(float(density_ratio), digits),
                "coverage_ratio": round(float(coverage_ratio), digits),
            }
        )

    return {"threshold_delta": deltas, "basis": basis}


def recommend_bounds(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    digits: int = 6,
) -> Mapping[str, Any]:
    ref_split = _pick_reference_split(summaries)
    ref_summary = summaries[ref_split]
    baselines = _resolve_threshold_baselines(cfg)
    priors = _resolve_prior_means(cfg)

    pos_rates = ref_summary.get("positive_rate", {})
    pos_pitch = _coerce_float(pos_rates.get("pitch"), 0.0)
    pos_onset = _coerce_float(pos_rates.get("onset"), 0.0)
    pos_offset = _coerce_float(pos_rates.get("offset"), 0.0)
    prior_onset = priors.get("onset")
    prior_offset = priors.get("offset")
    ref_rates = {
        "pitch": pos_pitch,
        "onset": prior_onset if prior_onset is not None else pos_onset,
        "offset": prior_offset if prior_offset is not None else pos_offset,
    }
    for head in ("pitch", "onset", "offset"):
        if ref_rates[head] <= 0.0:
            ref_rates[head] = _coerce_float(pos_rates.get(head), 0.0)

    thresholds = {
        head: _threshold_range(
            baseline=baselines.get(head, 0.2),
            pos_rate=_coerce_float(pos_rates.get(head), 0.0),
            ref_rate=float(ref_rates.get(head, 0.0) or 0.0),
            min_width=0.06,
            max_width=0.12,
            digits=digits,
        )
        for head in ("pitch", "onset", "offset")
    }
    onset_low, onset_high = _threshold_bounds(
        thresholds,
        "onset",
        fallback=baselines.get("onset", 0.2),
        width=0.12,
    )
    offset_low, offset_high = _threshold_bounds(
        thresholds,
        "offset",
        fallback=baselines.get("offset", 0.2),
        width=0.12,
    )

    peak_picking = _derive_peak_picking(ref_summary)
    aggregation = _derive_aggregation(ref_summary)

    hysteresis_ratio = {}
    for head in ("onset", "offset"):
        pos_rate = _coerce_float(pos_rates.get(head), 0.0)
        ref_rate = float(ref_rates.get(head, 0.0) or 0.0)
        ratio = (pos_rate + 1e-6) / (ref_rate + 1e-6) if ref_rate > 0 else 1.0
        log_ratio = _clamp(math.log10(ratio), -1.0, 1.0)
        low_ratio = _clamp(0.6 - 0.08 * log_ratio, 0.45, 0.75)
        hysteresis_ratio[head] = round(float(low_ratio), digits)

    hysteresis = {
        "low_ratio": hysteresis_ratio,
        "open_range": {
            "onset": [round(onset_low, digits), round(onset_high, digits)],
            "offset": [round(offset_low, digits), round(offset_high, digits)],
        },
        "hold_range": {
            "onset": [
                round(onset_low * hysteresis_ratio["onset"], digits),
                round(onset_high * hysteresis_ratio["onset"], digits),
            ],
            "offset": [
                round(offset_low * hysteresis_ratio["offset"], digits),
                round(offset_high * hysteresis_ratio["offset"], digits),
            ],
        },
        "min_on": peak_picking["min_on"],
        "min_off": peak_picking["min_off"],
        "merge_gap": peak_picking["merge_gap"],
        "median": peak_picking["median"],
    }

    tile_priors = _derive_tile_priors(ref_summary, digits=digits)

    return {
        "reference_split": ref_split,
        "thresholds": thresholds,
        "peak_picking": peak_picking,
        "hysteresis": hysteresis,
        "aggregation": aggregation,
        "per_tile_priors": tile_priors,
    }


def _normalize_events(raw: Any) -> List[Tuple[float, float, int]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    events: List[Tuple[float, float, int]] = []
    for evt in raw:
        if not isinstance(evt, (list, tuple)) or len(evt) < 3:
            continue
        try:
            onset = float(evt[0])
            offset = float(evt[1])
            pitch = int(evt[2])
        except (TypeError, ValueError):
            continue
        events.append((onset, offset, pitch))
    return events


def _resolve_tile_mask(
    *,
    reg_meta: Optional[Mapping[str, Any]],
    fallback_meta: Mapping[str, Any],
    tiles: int,
    cushion_keys: int,
    note_min: int,
    note_max: int,
) -> Tuple[Optional[Any], bool]:
    midi_low = 21
    midi_high = 108
    n_keys_full = midi_high - midi_low + 1
    if tiles <= 0:
        return None, False
    used_fallback = reg_meta is None
    meta = reg_meta if reg_meta is not None else fallback_meta
    result = build_tile_key_mask(meta, tiles, cushion_keys, n_keys=n_keys_full)
    mask = result.mask
    if note_min != midi_low or note_max != midi_high:
        start = max(0, note_min - midi_low)
        stop = min(n_keys_full, start + (note_max - note_min + 1))
        if stop <= start:
            return None, False
        mask = mask[:, start:stop]
    registration_based = bool(result.registration_based) and not used_fallback
    return mask, registration_based


def compute_split_stats(
    *,
    cfg: Mapping[str, Any],
    split: str,
    max_clips: Optional[int],
    disable_av_sync: bool,
    disable_coverage: bool,
    disable_targets: bool,
) -> Tuple[SplitStats, FrameTargetSpec | None]:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("Config must contain a 'dataset' mapping.")
    dataset_name = str(dataset_cfg.get("name", "omaps"))
    dataset_cls = _resolve_dataset_class(dataset_name)
    dataset = dataset_cls(cfg, split=split, full_cfg=cfg)

    spec = dataset.frame_target_spec
    if spec is None:
        disable_targets = True

    frame_cfg = dataset_cfg.get("frame_targets") if isinstance(dataset_cfg, Mapping) else None
    if not isinstance(frame_cfg, Mapping):
        frame_cfg = {}
    note_min = int(spec.note_min if spec is not None else frame_cfg.get("note_min", 21))
    note_max = int(spec.note_max if spec is not None else frame_cfg.get("note_max", 108))
    tiles = int(getattr(dataset, "tiles", dataset_cfg.get("tiles", 1)))

    duration_bins = build_log_bins(0.02, 8.0, 40)
    ioi_bins = build_log_bins(0.02, 6.0, 40)
    stats = SplitStats(
        split=split,
        frames=dataset.frames,
        stride=dataset.stride,
        fps=dataset.decode_fps,
        note_min=note_min,
        note_max=note_max,
        num_tiles=tiles,
        duration_bins=duration_bins,
        ioi_bins=ioi_bins,
    )

    entries = getattr(dataset, "entries", [])
    max_limit = int(max_clips) if max_clips is not None else None

    cache = NullFrameTargetCache()
    cushion_keys = _resolve_cushion_keys(cfg)
    fallback_meta = build_canonical_registration_metadata(
        width=float(getattr(dataset, "canonical_hw", (0, 0))[1] or 800.0),
        num_tiles=tiles,
        n_keys=88,
    )

    for idx, entry in enumerate(entries):
        if max_limit is not None and idx >= max_limit:
            break
        raw_labels = {}
        try:
            raw_labels = dataset._read_labels(entry)  # type: ignore[attr-defined]
        except Exception:
            raw_labels = {}
        events = _normalize_events(raw_labels.get("events", []) if isinstance(raw_labels, Mapping) else [])
        if events and not disable_av_sync:
            sync_cache = getattr(dataset, "_av_sync_cache", None)
            sync = resolve_sync(entry.video_id, entry.metadata, sync_cache)
            sample = {"events": events}
            apply_sync(sample, sync)
            events = sample.get("events", events)
        stats.record_events(events, clip_start=float(getattr(dataset, "skip_seconds", 0.0) or 0.0))

        if not disable_coverage:
            reg_meta = None
            refiner = getattr(dataset, "registration_refiner", None)
            if refiner is not None and hasattr(refiner, "get_geometry_metadata"):
                try:
                    reg_meta = refiner.get_geometry_metadata(entry.video_id)
                except Exception:
                    reg_meta = None
            mask, registration_based = _resolve_tile_mask(
                reg_meta=reg_meta,
                fallback_meta=fallback_meta,
                tiles=tiles,
                cushion_keys=cushion_keys,
                note_min=note_min,
                note_max=note_max,
            )
            stats.record_tile_mask(mask, registration_based=registration_based)

        if disable_targets:
            continue

        if not events:
            stats.record_empty_rolls()
            continue

        labels = [[evt[0], evt[1], evt[2]] for evt in events]
        labels_tensor = None
        try:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        except Exception:
            labels_tensor = None
        if labels_tensor is None or spec is None:
            stats.record_empty_rolls()
            continue

        with torch.no_grad():
            result = prepare_frame_targets(
                labels=labels_tensor,
                lag_result=None,
                spec=spec,
                cache=cache,
                split=split,
                video_id=str(entry.video_id),
                clip_start=float(getattr(dataset, "skip_seconds", 0.0) or 0.0),
                soft_targets=None,
                trace=None,
            )
        payload = result.payload or {}
        pitch_roll = payload.get("pitch_roll")
        onset_roll = payload.get("onset_roll")
        offset_roll = payload.get("offset_roll")
        if pitch_roll is None or onset_roll is None or offset_roll is None:
            stats.record_empty_rolls()
            continue
        stats.record_rolls(pitch_roll=pitch_roll, onset_roll=onset_roll, offset_roll=offset_roll)

    return stats, spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset-only priors and threshold bounds.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument(
        "--output",
        default="tivit/configs/priors/threshold_priors.yaml",
        help="Output YAML path.",
    )
    parser.add_argument("--splits", default=None, help="Comma-separated split list (overrides config).")
    parser.add_argument("--max-clips", type=int, default=None, help="Cap clips per split.")
    parser.add_argument("--disable-av-sync", action="store_true", help="Skip AV sync adjustments.")
    parser.add_argument("--disable-coverage", action="store_true", help="Skip tile/key coverage stats.")
    parser.add_argument("--disable-targets", action="store_true", help="Skip frame-target roll stats.")
    args = parser.parse_args()

    cfg = resolve_config_chain([Path(args.config)], default_base=None)
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("Config must contain a 'dataset' mapping.")

    splits = _collect_splits(dataset_cfg)
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    summaries: Dict[str, Mapping[str, Any]] = {}
    stats_by_split: Dict[str, SplitStats] = {}
    specs: Dict[str, Mapping[str, Any]] = {}
    for split in splits:
        stats, spec = compute_split_stats(
            cfg=cfg,
            split=split,
            max_clips=args.max_clips,
            disable_av_sync=bool(args.disable_av_sync),
            disable_coverage=bool(args.disable_coverage),
            disable_targets=bool(args.disable_targets),
        )
        summary = stats.summary()
        stats_by_split[split] = stats
        summaries[split] = summary
        if spec is not None:
            specs[split] = {
                "frames": int(spec.frames),
                "stride": int(spec.stride),
                "fps": float(spec.fps),
                "tolerance": float(spec.tolerance),
                "dilation": int(spec.dilation),
                "note_min": int(spec.note_min),
                "note_max": int(spec.note_max),
                "fill_mode": str(spec.fill_mode),
            }

    recommendations = recommend_bounds(summaries, cfg=cfg)

    output = {
        "meta": {
            "config": str(args.config),
            "splits": splits,
            "enabled": True,
        },
        "frame_target_spec": specs,
        "stats": summaries,
        "recommendations": recommendations,
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)

    print(f"[threshold_priors] wrote {out_path}", flush=True)
    ref_split = _pick_reference_split(summaries)
    ref_stats = stats_by_split.get(ref_split)
    if ref_stats is not None:
        bands = _resolve_pos_weight_bands(cfg)
        weights = _pos_weights_from_stats(ref_stats, bands)
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        weights_path = log_dir / "pos_weights.json"
        payload = {
            "meta": {
                "config": str(args.config),
                "split": ref_split,
            },
            "bands": {k: list(v) for k, v in bands.items()},
            "weights": weights,
        }
        weights_path.write_text(json.dumps(payload, indent=2))
        print(f"[threshold_priors] wrote pos_weight file {weights_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compute_split_stats", "recommend_bounds"]
