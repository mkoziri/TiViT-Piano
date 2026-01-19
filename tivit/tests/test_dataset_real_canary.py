#!/usr/bin/env python3
"""
Purpose:
    Unified deterministic real-data canary gate for any registered dataset.
    Selects a dataset config at runtime, applies testing overrides from
    ``dataset.testing``, and runs the shared invariants used by the legacy
    OMAPS/PianoVAM/PianoYT canary scripts.

Usage:
    python tivit/tests/test_dataset_real_canary.py --config tivit/configs/dataset/omaps.yaml
    python tivit/tests/test_dataset_real_canary.py --dataset pianovam --regen
    TIVIT_DATASET_CONFIG=tivit/configs/dataset/pianoyt.yaml TIVIT_CANARY_AUDIT_DIR=out/audit python tivit/tests/test_dataset_real_canary.py

Runtime overrides (precedence: CLI/env > dataset.testing > built-ins):
    --config / TIVIT_DATASET_CONFIG : dataset YAML (or --dataset shorthand)
    --regen / TIVIT_CANARY_REGEN    : regenerate canary list
    --count / TIVIT_CANARY_COUNT    : number of canaries to regenerate
    --audit-dir / TIVIT_CANARY_AUDIT_DIR : export audit artifacts
    --cache-equiv / TIVIT_CANARY_CACHE_EQUIV : compare cache on/off
    --cache-off / TIVIT_CANARY_CACHE_OFF : disable label caching
    --strict-sampling / TIVIT_CANARY_STRICT_SAMPLING : tighten sampler checks
    --strict-fallback / TIVIT_CANARY_STRICT_FALLBACK : fail on ROI/sync fallbacks
    --strict-labels / TIVIT_CANARY_STRICT_LABELS : force label expectations on
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw  # type: ignore
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore

try:  # optional visuals
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore

from tivit.core.config import load_yaml_file
from tivit.data.decode.video_reader import VideoReaderConfig, load_clip
from tivit.data.loaders import make_dataloader

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIGS = {
    "omaps": REPO_ROOT / "tivit" / "configs" / "dataset" / "omaps.yaml",
    "pianovam": REPO_ROOT / "tivit" / "configs" / "dataset" / "pianovam.yaml",
    "pianoyt": REPO_ROOT / "tivit" / "configs" / "dataset" / "pianoyt.yaml",
}

FRAME_TARGET_DEFAULTS: Mapping[str, Any] = {
    "enable": True,
    "tolerance": 0.03,
    "dilate_active_frames": 1,
    "fill_mode": "overlap",
    "hand_from_pitch": True,
    "clef_thresholds": [60, 64],
    "note_min": 21,
    "note_max": 108,
}
DEFAULT_CANARY_COUNT = 2


@dataclass(frozen=True)
class Canary:
    video_rel: str
    split: str

    @property
    def abs_path(self) -> Path:
        return (REPO_ROOT / self.video_rel).resolve()

    @property
    def video_id(self) -> str:
        return Path(self.video_rel).stem


@dataclass
class TestingOptions:
    regen: bool
    canary_count: int
    canary_json: Path
    cache_equiv: bool
    cache_off: bool
    strict_sampling: bool
    strict_labels: bool
    strict_fallback: bool
    enable_audit: bool
    audit_dir: Optional[Path]
    log_level: str
    export_report_json: bool
    report_json_path: Optional[Path]
    max_event_examples: int
    warn_fps_tolerance: float
    warn_out_of_window_ratio: float
    enable_debug_extras: bool
    warn_refine_worsened_delta: float
    max_event_spotchecks: int
    onset_unique_time_quantization: str
    warn_onset_cells_margin: float
    enable_spotcheck: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified real-data dataset canary gate.")
    parser.add_argument("--config", type=str, help="Path to dataset YAML (repo-relative ok).")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset shorthand (omaps, pianovam, pianoyt) to pick a built-in config.",
    )
    parser.add_argument("--regen", dest="regen", action="store_const", const=True, default=None, help="Regenerate canaries.")
    parser.add_argument("--count", type=int, default=None, help="Canary count for regeneration.")
    parser.add_argument("--audit-dir", type=str, default=None, help="Directory for audit exports (implies --enable-audit).")
    parser.add_argument("--cache-equiv", dest="cache_equiv", action="store_const", const=True, default=None, help="Check cache equivalence.")
    parser.add_argument("--cache-off", dest="cache_off", action="store_const", const=True, default=None, help="Disable label caching during run.")
    parser.add_argument("--strict-sampling", dest="strict_sampling", action="store_const", const=True, default=None, help="Enable strict sampling checks.")
    parser.add_argument("--strict-fallback", dest="strict_fallback", action="store_const", const=True, default=None, help="Enable strict ROI/sync fallback checks.")
    parser.add_argument("--strict-labels", dest="strict_labels", action="store_const", const=True, default=None, help="Force label presence checks.")
    return parser.parse_args()


def _env_flag(name: str) -> Optional[bool]:
    return True if os.environ.get(name) is not None else None


def _first_non_none(*vals: Any) -> Any:
    for val in vals:
        if val is not None:
            return val
    return None


def _to_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_config_path(args: argparse.Namespace) -> Path:
    cfg_from_env = os.environ.get("TIVIT_DATASET_CONFIG")
    dataset_key = (args.dataset or os.environ.get("TIVIT_DATASET") or "").strip().lower()
    if args.config:
        return _to_repo_path(args.config)
    if cfg_from_env:
        return _to_repo_path(cfg_from_env)
    if dataset_key:
        if dataset_key not in DATASET_CONFIGS:
            raise SystemExit(f"Unknown dataset shorthand '{dataset_key}'. Choose from {', '.join(sorted(DATASET_CONFIGS))}.")
        return DATASET_CONFIGS[dataset_key]
    return DATASET_CONFIGS["omaps"]


def _resolve_testing_options(testing_cfg: Mapping[str, Any], args: argparse.Namespace) -> TestingOptions:
    count_env = os.environ.get("TIVIT_CANARY_COUNT")
    audit_dir_env = os.environ.get("TIVIT_CANARY_AUDIT_DIR")

    canary_count = int(
        _first_non_none(
            args.count,
            int(count_env) if count_env else None,
            testing_cfg.get("default_canary_count"),
            DEFAULT_CANARY_COUNT,
        )
    )
    canary_json_raw = testing_cfg.get("canary_json")
    if not canary_json_raw:
        raise SystemExit("dataset.testing.canary_json is required for the canary gate.")
    audit_dir_raw = _first_non_none(args.audit_dir, audit_dir_env, testing_cfg.get("audit_dir"))
    audit_dir = _to_repo_path(audit_dir_raw) if audit_dir_raw else None
    enable_audit = bool(
        _first_non_none(
            True if args.audit_dir is not None else None,
            True if audit_dir_env is not None else None,
            testing_cfg.get("enable_audit_export"),
            False,
        )
    )
    regen = bool(_first_non_none(args.regen, _env_flag("TIVIT_CANARY_REGEN"), testing_cfg.get("regen_canaries"), False))
    cache_equiv = bool(_first_non_none(args.cache_equiv, _env_flag("TIVIT_CANARY_CACHE_EQUIV"), testing_cfg.get("enable_cache_equiv"), False))
    cache_off = bool(_first_non_none(args.cache_off, _env_flag("TIVIT_CANARY_CACHE_OFF"), testing_cfg.get("cache_off"), False))
    strict_sampling = bool(_first_non_none(args.strict_sampling, _env_flag("TIVIT_CANARY_STRICT_SAMPLING"), testing_cfg.get("strict_sampling"), False))
    strict_fallback = bool(_first_non_none(args.strict_fallback, _env_flag("TIVIT_CANARY_STRICT_FALLBACK"), testing_cfg.get("strict_fallback"), False))
    strict_labels = bool(_first_non_none(args.strict_labels, _env_flag("TIVIT_CANARY_STRICT_LABELS"), testing_cfg.get("strict_labels"), False))
    log_level = str(testing_cfg.get("log_level", "detailed")).lower()
    export_report_json = bool(testing_cfg.get("export_report_json", False))
    report_json_raw = testing_cfg.get("report_json_path")
    report_json_path = _to_repo_path(report_json_raw) if report_json_raw else None
    max_event_examples = int(testing_cfg.get("max_event_examples", 2))
    warn_fps_tolerance = float(testing_cfg.get("warn_fps_tolerance", 0.5))
    warn_out_of_window_ratio = float(testing_cfg.get("warn_out_of_window_ratio", 0.2))
    enable_debug_extras = True
    warn_refine_worsened_delta = float(testing_cfg.get("warn_refine_worsened_delta", 0.5))
    max_event_spotchecks = int(testing_cfg.get("max_event_spotchecks", 2))
    onset_unique_time_quantization = str(testing_cfg.get("onset_unique_time_quantization", "frame"))
    warn_onset_cells_margin = float(testing_cfg.get("warn_onset_cells_margin", 0.25))
    enable_spotcheck = bool(testing_cfg.get("enable_spotcheck", True))
    if export_report_json and audit_dir is not None and report_json_path is None:
        report_json_path = audit_dir / "model_readiness_report.jsonl"
    export_report_json = bool(export_report_json and enable_audit and report_json_path is not None)
    return TestingOptions(
        regen=regen,
        canary_count=canary_count,
        canary_json=_to_repo_path(canary_json_raw),
        cache_equiv=cache_equiv,
        cache_off=cache_off,
        strict_sampling=strict_sampling,
        strict_labels=strict_labels,
        strict_fallback=strict_fallback,
        enable_audit=enable_audit,
        audit_dir=audit_dir,
        log_level=log_level,
        export_report_json=export_report_json,
        report_json_path=report_json_path,
        max_event_examples=max_event_examples,
        warn_fps_tolerance=warn_fps_tolerance,
        warn_out_of_window_ratio=warn_out_of_window_ratio,
        enable_debug_extras=enable_debug_extras,
        warn_refine_worsened_delta=warn_refine_worsened_delta,
        max_event_spotchecks=max_event_spotchecks,
        onset_unique_time_quantization=onset_unique_time_quantization,
        warn_onset_cells_margin=warn_onset_cells_margin,
        enable_spotcheck=enable_spotcheck,
    )


def _prepare_dataset_cfg(cfg: Mapping[str, Any], testing_cfg: Mapping[str, Any], options: TestingOptions) -> Mapping[str, Any]:
    cfg = dict(cfg)
    dataset_cfg: Dict[str, Any] = dict(cfg.get("dataset", {}))
    name = str(dataset_cfg.get("name", "dataset")).lower()
    root_dir = dataset_cfg.get("root_dir")
    if not root_dir:
        raise RuntimeError("dataset.root_dir is required for the canary gate.")
    dataset_cfg["root_dir"] = str(_to_repo_path(root_dir))
    annotations_root = dataset_cfg.get("annotations_root") or dataset_cfg["root_dir"]
    dataset_cfg["annotations_root"] = str(_to_repo_path(annotations_root))

    frame_targets_cfg: Dict[str, Any] = dict(dataset_cfg.get("frame_targets", {}) or {})
    for key, val in FRAME_TARGET_DEFAULTS.items():
        frame_targets_cfg.setdefault(key, val)
    cache_prefix = f"tivit_canary_ft_off_{name}_" if options.cache_off else f"tivit_canary_ft_{name}_"
    frame_targets_cfg["cache_dir"] = tempfile.mkdtemp(prefix=cache_prefix)
    if options.cache_off:
        frame_targets_cfg["cache_labels"] = False
    dataset_cfg["frame_targets"] = frame_targets_cfg
    dataset_cfg["num_workers"] = 0
    dataset_cfg["prefetch_factor"] = None
    testing_cfg_local: Dict[str, Any] = dict(testing_cfg or {})
    testing_cfg_local.setdefault("log_level", options.log_level)
    testing_cfg_local["enable_debug_extras"] = bool(testing_cfg_local.get("enable_debug_extras", False) or options.enable_debug_extras)
    testing_cfg_local.setdefault("export_report_json", options.export_report_json)
    testing_cfg_local.setdefault("report_json_path", str(options.report_json_path) if options.report_json_path else None)
    testing_cfg_local.setdefault("max_event_examples", options.max_event_examples)
    testing_cfg_local.setdefault("warn_fps_tolerance", options.warn_fps_tolerance)
    testing_cfg_local.setdefault("warn_out_of_window_ratio", options.warn_out_of_window_ratio)
    testing_cfg_local.setdefault("warn_refine_worsened_delta", options.warn_refine_worsened_delta)
    testing_cfg_local.setdefault("max_event_spotchecks", options.max_event_spotchecks)
    testing_cfg_local.setdefault("onset_unique_time_quantization", options.onset_unique_time_quantization)
    testing_cfg_local.setdefault("warn_onset_cells_margin", options.warn_onset_cells_margin)
    testing_cfg_local.setdefault("enable_spotcheck", options.enable_spotcheck)
    dataset_cfg["testing"] = testing_cfg_local

    cfg["dataset"] = dataset_cfg
    return cfg


def _resolve_manifest_path(dataset_cfg: Mapping[str, Any], split: str) -> Optional[Path]:
    manifest_cfg = dataset_cfg.get("manifest")
    if isinstance(manifest_cfg, Mapping):
        candidate = manifest_cfg.get(split)
    elif isinstance(manifest_cfg, (str, Path)):
        candidate = manifest_cfg
    else:
        candidate = None
    if not candidate:
        return None
    return _to_repo_path(candidate)


def _resolve_split(dataset_cfg: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    candidates = [
        dataset_cfg.get("split_val"),
        dataset_cfg.get("split"),
        dataset_cfg.get("split_test"),
        dataset_cfg.get("split_train"),
        "val",
        "test",
        "train",
    ]
    root = Path(dataset_cfg["root_dir"])
    trace: Dict[str, Any] = {"candidates": [], "selected": None, "found_path": None}
    for split in candidates:
        if not split:
            continue
        manifest_path = _resolve_manifest_path(dataset_cfg, str(split))
        manifest_exists = bool(manifest_path is not None and manifest_path.exists())
        dir_exists = (root / str(split)).exists()
        trace["candidates"].append(
            {"name": str(split), "dir_exists": dir_exists, "manifest": str(manifest_path) if manifest_path else None}
        )
        if manifest_exists and manifest_path is not None:
            resolved_manifest = manifest_path.resolve()
            trace["selected"] = str(split)
            trace["found_path"] = str(resolved_manifest)
            return str(split), trace
        if dir_exists:
            trace["selected"] = str(split)
            trace["found_path"] = str((root / str(split)).resolve())
            return str(split), trace
    fallback = str(dataset_cfg.get("split", "val"))
    trace["selected"] = fallback
    trace["found_path"] = str((root / fallback).resolve())
    return fallback, trace


def _gather_candidates(split: str, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    root = Path(dataset_cfg["root_dir"])
    manifest_path = _resolve_manifest_path(dataset_cfg, split)
    if manifest_path and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        entries = payload.get(split, [])
        candidates: List[Path] = []
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, Mapping):
                    rel = item.get("video") or item.get("video_path")
                elif isinstance(item, str):
                    rel = item
                else:
                    continue
                if not rel:
                    continue
                path = Path(rel)
                if not path.is_absolute():
                    path = root / rel
                if path.exists():
                    candidates.append(path)
        return candidates
    split_dir = root / split
    return sorted(split_dir.rglob("*.mp4"))


def _select_decodable(candidates: List[Path], count: int, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    selected: List[Path] = []
    skip_seconds = float(dataset_cfg.get("skip_seconds", 0.0))
    start_frame = max(0, int(round(skip_seconds * float(dataset_cfg.get("decode_fps", 30.0)))))
    vr_cfg = VideoReaderConfig(
        frames=int(dataset_cfg.get("frames", 32)),
        stride=1,
        resize_hw=tuple(dataset_cfg.get("resize", [224, 224])),
        channels=int(dataset_cfg.get("channels", 3)),
        start_frame=start_frame,
    )
    for path in candidates:
        if len(selected) >= count:
            break
        try:
            load_clip(path, vr_cfg)
            selected.append(path)
        except Exception:
            continue
    return selected


def _save_canaries(split: str, candidates: List[Path], count: int, canary_path: Path, dataset_cfg: Mapping[str, Any]) -> List[Canary]:
    canary_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _select_decodable(candidates, count, dataset_cfg)
    if len(selected) < count:
        raise RuntimeError(f"[regen] only {len(selected)} decodable clips found; expected {count}")

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    payload = {
        "split": split,
        "count": len(selected),
        "canaries": [_rel(path) for path in selected],
    }
    canary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return [Canary(video_rel=rel, split=split) for rel in payload["canaries"]]


def _load_canaries(path: Path) -> List[Canary]:
    if not path.exists():
        raise FileNotFoundError(f"Canary list missing at {path}. Regenerate with TIVIT_CANARY_REGEN=1.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    canaries = payload.get("canaries", [])
    if not canaries:
        raise SystemExit(f"Canary list at {path} is empty; regenerate with TIVIT_CANARY_REGEN=1.")
    split = payload.get("split", "val")
    return [Canary(video_rel=rel, split=split) for rel in canaries]


def _assert_finite(tensor: torch.Tensor, name: str, canary: Canary) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} has non-finite values for canary {canary.video_rel}")


def _check_frames(sample: Mapping[str, Any], expected_T: int, canary: Canary, tiles: int, *, resize_hw: Iterable[int], stride: int) -> None:
    video = sample.get("video")
    if not torch.is_tensor(video):
        raise AssertionError(f"video tensor missing for canary {canary.video_rel}")
    if video.ndim != 5:
        raise AssertionError(f"video tensor rank {video.ndim} != 5 for canary {canary.video_rel}")
    T, C, K, H, W = video.shape
    if T != expected_T:
        raise AssertionError(f"T mismatch: expected {expected_T}, got {T} for {canary.video_rel}")
    if C != 3:
        raise AssertionError(f"channels {C} unexpected for {canary.video_rel}")
    if K != tiles:
        raise AssertionError(f"tiles {K} != expected {tiles} for {canary.video_rel}")
    if H <= 0 or W <= 0:
        raise AssertionError(f"tile dims degenerate ({H}, {W}) for {canary.video_rel}")
    if stride < 1:
        raise AssertionError(f"stride {stride} invalid for {canary.video_rel}")
    expected_h = int(list(resize_hw)[0]) if resize_hw else H
    expected_w = int(list(resize_hw)[1]) if resize_hw else W
    if H != expected_h:
        raise AssertionError(f"tile height {H} != resize height {expected_h} for {canary.video_rel}")
    if W < max(32, expected_w * 0.2):
        raise AssertionError(f"tile width {W} unexpectedly small vs resize width {expected_w} for {canary.video_rel}")
    total_width = W * tiles
    if expected_w > 0 and not (0.8 * expected_w <= total_width <= 1.3 * expected_w):
        raise AssertionError(f"tiling coverage {total_width} px out of bounds for expected ~{expected_w} on {canary.video_rel}")
    _assert_finite(video, "video", canary)
    vmin, vmax = float(video.min()), float(video.max())
    if not (-5.0 <= vmin <= 5.0 and -5.0 <= vmax <= 5.0):
        raise AssertionError(f"video values out of expected range [{vmin}, {vmax}] for {canary.video_rel}")
    tiles_view = video.permute(2, 0, 1, 3, 4)  # K,T,C,H,W
    for k in range(tiles):
        if float(tiles_view[k].var()) < 1e-8:
            raise AssertionError(f"tile {k} appears degenerate for {canary.video_rel}")


def _check_targets(sample: Mapping[str, Any], T: int, canary: Canary, *, require_labels: bool) -> None:
    expected_keys = ("pitch", "onset", "offset", "hand", "clef")
    for key in expected_keys:
        tensor = sample.get(key)
        if tensor is None:
            if require_labels:
                raise AssertionError(f"missing target '{key}' for {canary.video_rel}")
            continue
        if not torch.is_tensor(tensor):
            if require_labels:
                raise AssertionError(f"target '{key}' not a tensor for {canary.video_rel}")
            continue
        _assert_finite(tensor, key, canary)
        if tensor.ndim >= 2 and tensor.shape[0] != T:
            raise AssertionError(f"target '{key}' length {tensor.shape[0]} != T {T} for {canary.video_rel}")
        if tensor.dtype not in (torch.float32, torch.float64, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise AssertionError(f"target '{key}' dtype {tensor.dtype} unexpected for {canary.video_rel}")
        if tensor.is_floating_point():
            if not (tensor.min() >= -1e-3 and tensor.max() <= 1.0 + 1e-3):
                raise AssertionError(f"target '{key}' outside [0,1] for {canary.video_rel}")
            mean_val = float(tensor.mean())
            if mean_val <= 1e-6 or mean_val >= 0.5:
                raise AssertionError(f"target '{key}' mean {mean_val:.3f} suspicious for {canary.video_rel}")
            if require_labels and float(tensor.sum()) <= 0.0:
                raise AssertionError(f"target '{key}' has zero positives for {canary.video_rel}")
            pos_rate = mean_val
            min_band = 1e-4
            max_band = 0.35 if key in {"pitch", "onset", "offset"} else 1.0
            if pos_rate < min_band:
                raise AssertionError(f"target '{key}' too sparse (mean {pos_rate:.6f}) for {canary.video_rel}")
            if pos_rate > max_band:
                raise AssertionError(f"target '{key}' too dense (mean {pos_rate:.3f}) for {canary.video_rel}")
        else:
            if tensor.min() < 0:
                raise AssertionError(f"target '{key}' has negative values for {canary.video_rel}")
            if require_labels and int(tensor.sum().item()) <= 0:
                raise AssertionError(f"target '{key}' has zero positives for {canary.video_rel}")


def _check_determinism(ds: Any, idx: int, canary: Canary) -> None:
    first = ds[idx]
    second = ds[idx]
    for key in ("video", "pitch", "onset", "offset", "hand", "clef"):
        t1 = first.get(key)
        t2 = second.get(key)
        if torch.is_tensor(t1) and torch.is_tensor(t2):
            if not torch.allclose(t1, t2, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"determinism check failed for {key} in {canary.video_rel}")


def _check_roi_and_fallbacks(sample: Mapping[str, Any], canary: Canary, *, strict_fallback: bool) -> None:
    meta = sample.get("metadata", {}) or {}
    crop = meta.get("crop")
    if crop is not None:
        try:
            vals = list(crop)
            if len(vals) >= 4:
                def _as_xyxy(values: Sequence[Any]) -> Optional[Tuple[float, float, float, float]]:
                    try:
                        y0, y1, x0, x1 = map(float, values[:4])
                    except (TypeError, ValueError):
                        return None
                    if x1 > x0 and y1 > y0:
                        return x0, y0, x1, y1
                    try:
                        x0, y0, x1, y1 = map(float, values[:4])
                    except (TypeError, ValueError):
                        return None
                    if x1 > x0 and y1 > y0:
                        return x0, y0, x1, y1
                    return None

                resolved = _as_xyxy(vals)
                if resolved is None:
                    raise AssertionError(f"degenerate crop {crop} for {canary.video_rel}")
                x0, y0, x1, y1 = resolved
                if (x1 - x0) < 8 or (y1 - y0) < 8:
                    raise AssertionError(f"crop too small for {canary.video_rel}: {crop}")
        except Exception:
            raise AssertionError(f"invalid crop metadata for {canary.video_rel}: {crop}")
    sync = sample.get("sync", {}) or {}
    sync_src = str(sync.get("source", "")).lower()
    if strict_fallback and (sync_src.startswith("fallback") or sync_src == "default"):
        raise AssertionError(f"sync fallback detected for {canary.video_rel}: {sync_src}")
    meta_str = json.dumps(meta).lower()
    if strict_fallback and ("fallback" in meta_str or meta.get("status") == "fallback"):
        raise AssertionError(f"fallback detected in metadata for {canary.video_rel}")


def _check_sampling_rules(ds: Any, sample: Mapping[str, Any], canary: Canary, *, strict_sampling: bool) -> None:
    T = ds.frames
    hop = ds.stride / max(ds.decode_fps, 1e-6)
    duration = hop * max(T - 1, 0)
    if duration <= 0:
        raise AssertionError(f"invalid duration derived from stride/fps for {canary.video_rel}")
    sampler_meta = sample.get("sampler_meta", {}) or {}
    start_frame = sampler_meta.get("start_frame")
    if start_frame is not None:
        # start_frame is computed on full-video timeline; account for skip_seconds when checking range
        offset_frames = int(round(getattr(ds, "skip_seconds", 0.0) * ds.decode_fps))
        rel_start = start_frame - offset_frames
        if rel_start >= ds.frames and strict_sampling:
            raise AssertionError(f"start_frame {start_frame} outside clip for {canary.video_rel}")
    if ds.stride < 1:
        raise AssertionError(f"stride {ds.stride} invalid for {canary.video_rel}")


def _export_audit(sample: Mapping[str, Any], canary: Canary, audit_dir: Path, *, tiles: int, norm_mean: Iterable[float], norm_std: Iterable[float], resize_hw: Optional[Iterable[int]] = None) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    video = sample["video"]
    T = video.shape[0]
    frames_idx = [0, T // 2]

    def _denorm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(list(norm_mean), device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(list(norm_std), device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        return (x * std) + mean

    vid_denorm = _denorm(video).clamp(0.0, 1.0)
    for idx in frames_idx:
        idx = int(max(0, min(T - 1, idx)))
        tile_tensor = vid_denorm[idx]  # C, K, H, W
        tiles_np_list = [
            tile_tensor[:, k].permute(1, 2, 0).cpu().numpy() for k in range(tile_tensor.shape[1])
        ]  # H, W, C per tile
        tiles_np = tiles_np_list[0] if len(tiles_np_list) == 1 else np.concatenate(tiles_np_list, axis=1)
        target_h, target_w = (None, None)
        if resize_hw:
            resize_list = list(resize_hw)
            if len(resize_list) >= 2:
                target_h, target_w = int(resize_list[0]), int(resize_list[1])
        if target_h and target_w and (tiles_np.shape[0] != target_h or tiles_np.shape[1] != target_w):
            try:
                import cv2  # type: ignore
            except Exception:
                cv2 = None  # type: ignore
            if cv2 is not None:
                tiles_np = cv2.resize(tiles_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            elif Image is not None:
                img_tmp = Image.fromarray((tiles_np * 255).astype("uint8"))
                resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR", 2))
                img_tmp = img_tmp.resize((target_w, target_h), resample=resampling)
                tiles_np = np.array(img_tmp).astype("float32") / 255.0
        out_path = audit_dir / f"{canary.video_id}_frame{idx:03d}_tiles.png"
        saved = False
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            imageio = None  # type: ignore
        if imageio is not None:
            imageio.imwrite(out_path, (tiles_np * 255).astype("uint8"))
            saved = True
        elif Image is not None:
            img = Image.fromarray((tiles_np * 255).astype("uint8"))
            img.save(out_path)
            saved = True

        # Overlay showing tile grid (ROI proxy is full extent).
        overlay_name = audit_dir / f"{canary.video_id}_frame{idx:03d}_overlay.png"
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None  # type: ignore
        if cv2 is not None:
            overlay = (tiles_np * 255).astype("uint8").copy()
            h, w, _ = overlay.shape
            step = w // tiles if tiles > 0 else w
            for k in range(1, tiles):
                x = k * step
                cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)
            cv2.imwrite(str(overlay_name), overlay)
        elif Image is not None and ImageDraw is not None and saved:
            overlay_img = Image.open(out_path).copy()
            draw = ImageDraw.Draw(overlay_img)
            w, h = overlay_img.size
            step = w // tiles if tiles > 0 else w
            for k in range(1, tiles):
                x = k * step
                draw.line([(x, 0), (x, h - 1)], fill=(0, 255, 0), width=2)
            draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 0, 0), width=2)
            overlay_img.save(overlay_name)

    # Original frame with crop rectangle (first frame).
    crop_meta = sample.get("metadata", {}).get("crop") if isinstance(sample.get("metadata"), Mapping) else None
    if crop_meta is not None:
        try:
            cfg_resize: Optional[Tuple[int, int]] = None
            if resize_hw:
                resize_list = list(resize_hw)
                if len(resize_list) >= 2:
                    cfg_resize = (int(resize_list[0]), int(resize_list[1]))
            vr_cfg = VideoReaderConfig(frames=1, stride=1, resize_hw=cfg_resize or video.shape[-2:], channels=3)
            decoded = load_clip(canary.abs_path, vr_cfg)
            frame0 = decoded[0].permute(1, 2, 0).cpu().numpy()
            h, w, _ = frame0.shape
            vals = list(crop_meta)
            if len(vals) >= 4:
                x0, y0, x1, y1 = map(float, vals[:4])
                x0 = int(max(0, min(w - 1, x0)))
                x1 = int(max(0, min(w - 1, x1)))
                y0 = int(max(0, min(h - 1, y0)))
                y1 = int(max(0, min(h - 1, y1)))
                # Previously drew crop box on original frame; removed per request.
        except Exception:
            pass

    if plt is not None:
        onset = sample.get("onset")
        offset = sample.get("offset")
        if torch.is_tensor(onset) and torch.is_tensor(offset):
            fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
            ax[0].plot(onset.sum(dim=1).cpu().numpy())
            ax[0].set_title("Onset activity")
            ax[1].plot(offset.sum(dim=1).cpu().numpy())
            ax[1].set_title("Offset activity")
            fig.suptitle(f"{canary.video_id}")
            fig.tight_layout()
            fig.savefig(audit_dir / f"{canary.video_id}_targets.png")
            plt.close(fig)

    summary = {
        "video": canary.video_rel,
        "shape": list(video.shape),
        "dtype": str(video.dtype),
        "video_range": [float(video.min()), float(video.max())],
        "metadata": sample.get("metadata", {}),
        "targets": {k: list(sample[k].shape) if torch.is_tensor(sample.get(k)) else None for k in ("pitch", "onset", "offset", "hand", "clef")},
    }
    (audit_dir / f"{canary.video_id}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _build_dataset(cfg: Mapping[str, Any], split: str):
    loader = make_dataloader(cfg, split=split, drop_last=False)
    return loader.dataset


class ModelReadinessReporter:
    """Collect and emit per-run and per-canary readiness reports."""

    def __init__(
        self,
        *,
        dataset_name: str,
        config_path: Path,
        dataset_cfg: Mapping[str, Any],
        resolved_split: str,
        split_trace: Mapping[str, Any],
        options: TestingOptions,
        require_labels: bool,
    ) -> None:
        self.dataset_name = dataset_name
        self.config_path = config_path
        self.dataset_cfg = dataset_cfg
        self.resolved_split = resolved_split
        self.split_trace = dict(split_trace)
        self.require_labels = bool(require_labels)
        self.log_level = str(options.log_level or "detailed").lower()
        self.max_event_examples = max(0, int(options.max_event_examples))
        self.warn_fps_tolerance = float(options.warn_fps_tolerance)
        self.warn_out_of_window_ratio = float(options.warn_out_of_window_ratio)
        self.warn_refine_worsened_delta = float(options.warn_refine_worsened_delta)
        self.max_event_spotchecks = max(0, int(options.max_event_spotchecks))
        self.onset_unique_time_quantization = str(options.onset_unique_time_quantization or "frame")
        self.warn_onset_cells_margin = float(options.warn_onset_cells_margin)
        self.enable_spotcheck = bool(options.enable_spotcheck)
        self.export_json = bool(options.export_report_json and options.report_json_path)
        self.json_path = options.report_json_path if self.export_json else None
        self.json_records: List[Dict[str, Any]] = []
        self.run_header_emitted = False
        self.testing_flags = {
            "log_level": options.log_level,
            "export_report_json": options.export_report_json,
            "report_json_path": str(options.report_json_path) if options.report_json_path else None,
            "max_event_examples": self.max_event_examples,
            "warn_fps_tolerance": self.warn_fps_tolerance,
            "warn_out_of_window_ratio": self.warn_out_of_window_ratio,
            "warn_refine_worsened_delta": self.warn_refine_worsened_delta,
            "enable_debug_extras": options.enable_debug_extras,
            "strict_sampling": options.strict_sampling,
            "strict_labels": options.strict_labels,
            "strict_fallback": options.strict_fallback,
            "max_event_spotchecks": self.max_event_spotchecks,
            "onset_unique_time_quantization": self.onset_unique_time_quantization,
            "warn_onset_cells_margin": self.warn_onset_cells_margin,
            "enable_spotcheck": self.enable_spotcheck,
        }
        self.footer_counts = {
            "ok": 0,
            "fail": 0,
            "warnings": {},
            "refine_status": {},
        }

    def _resolve_seed(self, ds: Any) -> Optional[int]:
        seed_cfg = None
        try:
            if isinstance(getattr(ds, "full_cfg", None), Mapping):
                seed_cfg = ds.full_cfg.get("experiment", {}).get("seed")
        except Exception:
            seed_cfg = None
        try:
            if seed_cfg is None and hasattr(ds, "_rng") and hasattr(ds._rng, "initial_seed"):
                return int(ds._rng.initial_seed())
        except Exception:
            pass
        return int(seed_cfg) if seed_cfg is not None else None

    def _grayscale_mode(self, ds: Any) -> str:
        if getattr(ds, "grayscale", False):
            return "decode_gray" if getattr(ds, "channels", 3) == 1 else "rgb_to_gray"
        return "rgb"

    def emit_run_header(self, ds: Any) -> None:
        if self.run_header_emitted:
            return
        hop_s = ds.stride / max(ds.decode_fps, 1e-6)
        duration_s = hop_s * max(ds.frames - 1, 0)
        try:
            deterministic = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            deterministic = False
        requested_split = {
            key: self.dataset_cfg.get(key)
            for key in ("split", "split_train", "split_val", "split_test")
            if self.dataset_cfg.get(key) is not None
        }
        header_bits = [
            f"dataset={self.dataset_name}",
            f"config={self.config_path}",
            f"root={self.dataset_cfg.get('root_dir')}",
            f"split_req={requested_split}",
            f"split_candidates={self.split_trace.get('candidates')}",
            f"split_resolved={self.resolved_split}",
            f"split_found_path={self.split_trace.get('found_path')}",
            f"seed={self._resolve_seed(ds)}",
            f"deterministic={deterministic}",
            f"grayscale={self._grayscale_mode(ds)}",
        ]
        decode_bits = [
            f"decode_fps={ds.decode_fps}",
            f"stride={ds.stride}",
            f"frames={ds.frames}",
            f"hop_s={hop_s:.3f}",
            f"duration_s={duration_s:.3f}",
            f"resize={getattr(ds, 'resize_hw', None)}",
            f"canonical={getattr(ds, 'canonical_hw', None)}",
            f"crop={'on' if getattr(ds, 'apply_crop', False) else 'off'}",
            f"registration={'on' if getattr(ds, 'registration_enabled', False) else 'off'}",
            f"tiles={getattr(ds, 'tiles', None)}",
        ]
        print(f"[run] {' '.join(header_bits)}")
        print(f"[run.decode] {' '.join(decode_bits)}")
        if self.export_json and self.json_path is not None:
            run_record = {
                "record_type": "run_header",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dataset": self.dataset_name,
                "config_path": str(self.config_path),
                "root_dir": self.dataset_cfg.get("root_dir"),
                "split_request": requested_split,
                "split_trace": self.split_trace,
                "split_resolved": self.resolved_split,
                "seed": self._resolve_seed(ds),
                "decode_fps": ds.decode_fps,
                "stride": ds.stride,
                "frames": ds.frames,
                "hop_seconds": hop_s,
                "duration_sec": duration_s,
                "resize": getattr(ds, "resize_hw", None),
                "canonical_hw": getattr(ds, "canonical_hw", None),
                "crop_enabled": bool(getattr(ds, "apply_crop", False)),
                "registration_enabled": bool(getattr(ds, "registration_enabled", False)),
                "tiles": getattr(ds, "tiles", None),
                "grayscale": {
                    "enabled": bool(getattr(ds, "grayscale", False)),
                    "mode": self._grayscale_mode(ds),
                },
                "testing_flags": self.testing_flags,
                "deterministic": deterministic,
            }
            self.json_records.append(run_record)
        self.run_header_emitted = True

    def _tensor_shape(self, tensor: Any) -> Optional[List[int]]:
        if torch.is_tensor(tensor):
            return [int(v) for v in tensor.shape]
        return None

    def _summarize_target(self, name: str, tensor: Any, spec: Optional[Any]) -> Dict[str, Any]:
        if not torch.is_tensor(tensor):
            return {"name": name, "missing": True}
        shape = self._tensor_shape(tensor)
        summary: Dict[str, Any] = {"name": name, "shape": shape, "missing": False}
        if tensor.numel() == 0:
            summary.update({"pos_rate": 0.0, "count_pos": 0})
            return summary
        if tensor.ndim >= 2:
            pos_rate = float(tensor.float().mean().item())
            count = int(torch.count_nonzero(tensor).item())
            min_midi = None
            max_midi = None
            active = torch.nonzero(tensor, as_tuple=False)
            if active.numel() > 0:
                min_idx = int(active[:, -1].min().item())
                max_idx = int(active[:, -1].max().item())
                if spec is not None and getattr(spec, "note_min", None) is not None:
                    min_midi = int(spec.note_min + min_idx)
                    max_midi = int(spec.note_min + max_idx)
                else:
                    min_midi = min_idx
                    max_midi = max_idx
            summary.update(
                {
                    "pos_rate": pos_rate,
                    "count_pos": count,
                    "min_midi": min_midi,
                    "max_midi": max_midi,
                }
            )
        else:
            flat = tensor.reshape(-1)
            unique_vals = flat.unique(sorted=True)
            uniq_list = [self._jsonable(v.item()) for v in unique_vals[:8]]
            summary.update(
                {
                    "unique_values": uniq_list,
                    "mean_value": float(flat.float().mean().item()),
                }
            )
        return summary

    def _event_window_stats(
        self, events: List[Tuple[float, float, float]], clip_start: float, duration_s: float
    ) -> Tuple[int, int, float]:
        in_window = 0
        out_window = 0
        if duration_s <= 0.0:
            return in_window, len(events), 1.0 if events else 0.0
        clip_end = clip_start + duration_s
        for evt in events:
            try:
                onset = float(evt[0])
            except Exception:
                continue
            if clip_start <= onset <= clip_end:
                in_window += 1
            else:
                out_window += 1
        total = max(in_window + out_window, 1)
        return in_window, out_window, float(out_window) / float(total)

    def _event_examples(
        self,
        before: List[Tuple[float, float, float]],
        after: List[Tuple[float, float, float]],
        clip_start: float,
        duration_s: float,
        hop_s: float,
    ) -> List[Dict[str, Any]]:
        if not before or not after or self.max_event_examples <= 0:
            return []
        examples: List[Dict[str, Any]] = []
        paired = list(zip(before, after))
        clip_end = clip_start + duration_s
        in_window_example = None
        out_window_example = None
        for pre, post in paired:
            onset_post = float(post[0])
            if clip_start <= onset_post <= clip_end:
                in_window_example = (pre, post)
                break
        for pre, post in paired:
            onset_post = float(post[0])
            if onset_post < clip_start or onset_post > clip_end:
                out_window_example = (pre, post)
                break
        for pair in (in_window_example, out_window_example):
            if pair is None:
                continue
            pre, post = pair
            onset_pre = float(pre[0])
            onset_post = float(post[0])
            frame_idx = int(round((onset_post - clip_start) / max(hop_s, 1e-6)))
            examples.append({"onset": onset_pre, "onset_shifted": onset_post, "frame_idx": frame_idx})
            if len(examples) >= self.max_event_examples:
                break
        return examples

    def _spotcheck_onsets(
        self,
        events_in_window: List[Tuple[float, Optional[float], Any]],
        clip_start: float,
        hop_s: float,
        onset_target: Any,
        spec: Any,
    ) -> Dict[str, Any]:
        if not self.enable_spotcheck or not torch.is_tensor(onset_target) or onset_target.ndim < 2:
            return {"enabled": False, "results": [], "ok": 0, "fail": 0}
        if not events_in_window:
            return {"enabled": True, "results": [], "ok": 0, "fail": 0}
        T = onset_target.shape[0]
        K = onset_target.shape[1] if onset_target.ndim >= 2 else 0
        max_checks = min(self.max_event_spotchecks, len(events_in_window))
        results: List[Dict[str, Any]] = []
        ok = 0
        fail = 0
        note_min = getattr(spec, "note_min", None)
        for evt in events_in_window[:max_checks]:
            onset_time = float(evt[0])
            pitch_val = evt[2] if len(evt) >= 3 else None
            frame_idx = int(round((onset_time - clip_start) / max(hop_s, 1e-6)))
            key_idx = None
            if pitch_val is not None:
                key_idx = int(round(float(pitch_val)))
                if note_min is not None:
                    key_idx = key_idx - int(note_min)
            frame_idx_clamped = min(max(frame_idx, 0), max(T - 1, 0))
            key_idx_clamped = min(max(key_idx if key_idx is not None else 0, 0), max(K - 1, 0))
            val_at = None
            max_neighbor = None
            if 0 <= frame_idx_clamped < T and 0 <= key_idx_clamped < K:
                val_at = float(onset_target[frame_idx_clamped, key_idx_clamped].item())
                low = max(frame_idx_clamped - 1, 0)
                high = min(frame_idx_clamped + 1, T - 1)
                max_neighbor = float(onset_target[low : high + 1, key_idx_clamped].max().item())
            ok_flag = max_neighbor is not None and max_neighbor > 0.0
            if ok_flag:
                ok += 1
            else:
                fail += 1
            results.append(
                {
                    "onset": onset_time,
                    "frame_idx": frame_idx,
                    "key_idx": key_idx,
                    "target_val": val_at,
                    "target_max_neighbor": max_neighbor,
                    "ok": ok_flag,
                }
            )
        return {"enabled": True, "results": results, "ok": ok, "fail": fail}

    def collect(self, *, canary: Canary, sample: Mapping[str, Any], ds: Any, dataset_index: int) -> Dict[str, Any]:
        debug_meta = sample.get("_debug_extras", {}) if isinstance(sample, Mapping) else {}
        video = sample.get("video")
        tensor_shape = self._tensor_shape(video)
        hop_s = ds.stride / max(ds.decode_fps, 1e-6)
        duration_s = hop_s * max(ds.frames - 1, 0)
        clip_start_ds = float(getattr(ds, "skip_seconds", 0.0))
        clip_start = clip_start_ds

        decode_meta = debug_meta.get("decode", {}) if isinstance(debug_meta, Mapping) else {}
        tile_meta = debug_meta.get("tiling", {}) if isinstance(debug_meta, Mapping) else {}
        crop_meta = debug_meta.get("crop", {}) if isinstance(debug_meta, Mapping) else {}
        refine_meta = debug_meta.get("registration", {}) if isinstance(debug_meta, Mapping) else {}
        sync_meta = debug_meta.get("sync", {}) if isinstance(debug_meta, Mapping) else {}
        target_build = debug_meta.get("target_build", {}) if isinstance(debug_meta, Mapping) else {}
        if isinstance(target_build, Mapping):
            try:
                tb_clip_start = target_build.get("clip_start_sec")
                if tb_clip_start is not None:
                    clip_start = float(tb_clip_start)
            except Exception:
                clip_start = clip_start_ds
        events_before = list(sync_meta.get("events_before_sync") or [])
        events_after = list(sync_meta.get("events_after_sync") or [])
        actual_fps = decode_meta.get("actual_fps", decode_meta.get("decode_fps", ds.decode_fps))
        requested_crop = (crop_meta or {}).get("requested")
        clamped_crop = (crop_meta or {}).get("clamped")
        roi_xyxy = None
        if clamped_crop and len(clamped_crop) >= 4:
            roi_xyxy = {"x1": float(clamped_crop[2]), "y1": float(clamped_crop[0]), "x2": float(clamped_crop[3]), "y2": float(clamped_crop[1])}
        elif requested_crop and len(requested_crop) >= 4:
            roi_xyxy = {"x1": float(requested_crop[2]), "y1": float(requested_crop[0]), "x2": float(requested_crop[3]), "y2": float(requested_crop[1])}
        roi_issues = (crop_meta or {}).get("issues") or []
        roi_valid = bool(roi_xyxy and roi_xyxy["x2"] > roi_xyxy["x1"] and roi_xyxy["y2"] > roi_xyxy["y1"] and not roi_issues)

        lag_seconds = float(sample.get("sync", {}).get("lag_seconds", 0.0))
        lag_source = str(sample.get("sync", {}).get("source", target_build.get("lag_source", "unknown")))
        lag_ms = target_build.get("lag_ms")
        lag_frames = target_build.get("lag_frames")
        examples = self._event_examples(events_before, events_after, clip_start, duration_s, hop_s)
        reg_cache_path = None
        try:
            reg_cache_path = getattr(getattr(ds, "registration_refiner", None), "cache_path", None)
        except Exception:
            reg_cache_path = None

        targets = {
            name: self._summarize_target(name, sample.get(name), getattr(ds, "frame_target_spec", None))
            for name in ("pitch", "onset", "offset", "hand", "clef")
        }
        pitch_trace = {
            "painted_pairs_unique": target_build.get("pitch_painted_pairs_unique"),
            "frames_any_actual": target_build.get("pitch_frames_any_actual"),
            "events_in_window_total": target_build.get("pitch_events_in_window_total"),
            "events_seen_total": target_build.get("pitch_events_seen_total"),
            "min_used": target_build.get("pitch_min_used"),
            "max_used": target_build.get("pitch_max_used"),
        }
        offset_trace = {
            "painted_pairs_unique": target_build.get("offset_painted_pairs_unique"),
            "frames_any_actual": target_build.get("offset_target_frames_any_actual"),
            "events_in_window_total": target_build.get("offset_events_in_window_total"),
            "events_seen_total": target_build.get("offset_events_seen_total"),
            "min_used": target_build.get("offset_min_used"),
            "max_used": target_build.get("offset_max_used"),
        }
        hand_trace = {
            "enabled": sample.get("hand") is not None,
        }

        def _safe_int(val: Any) -> Optional[int]:
            try:
                return int(val)
            except Exception:
                return None

        # Prefer per-note counts from target_build (same source as targets), fall back to sync_meta.
        tb_note_total = _safe_int(target_build.get("events_seen_total"))
        tb_note_after_lag = tb_note_total
        tb_chord_in_total = _safe_int(target_build.get("events_in_window_total"))
        tb_painted_pairs = _safe_int(target_build.get("painted_pairs_unique"))
        tb_note_in = tb_painted_pairs
        tb_chord_total = None
        tb_chord_in = tb_chord_in_total
        unique_center_frames = target_build.get("unique_onset_center_frames_in_window") or []
        tb_unique_frames = len(unique_center_frames)
        tb_events_in_window_sample: List[Any] = []

        note_events_sync: List[Tuple[float, Optional[float], Any]] = []
        for evt in events_after:
            if not isinstance(evt, (list, tuple)) or len(evt) < 2:
                continue
            onset_t = float(evt[0])
            offset_t = float(evt[1]) if len(evt) >= 2 else None
            pitch_val = evt[2] if len(evt) >= 3 else None
            note_events_sync.append((onset_t, offset_t, pitch_val))

        note_events_sync_in_window = [
            evt for evt in note_events_sync if clip_start <= evt[0] < (clip_start + duration_s)
        ]
        note_onsets_total_fb = len(note_events_sync)
        note_onsets_in_window_fb = len(note_events_sync_in_window)
        note_offsets_in_window = sum(
            1
            for evt in note_events_sync
            if evt[1] is not None and clip_start <= float(evt[1]) < (clip_start + duration_s)
        )

        unique_onset_frames_fb: Set[int] = set()
        for evt in note_events_sync_in_window:
            if self.onset_unique_time_quantization.lower() == "frame":
                frame_idx = int(round((evt[0] - clip_start) / max(hop_s, 1e-6)))
                unique_onset_frames_fb.add(frame_idx)
            else:
                unique_onset_frames_fb.add(int(round(evt[0] * 1000.0)))

        note_onsets_total = tb_note_total if tb_note_total is not None else note_onsets_total_fb
        note_onsets_in_window = tb_note_in if tb_note_in is not None else note_onsets_in_window_fb
        unique_onsets = tb_unique_frames if tb_unique_frames is not None else len(unique_onset_frames_fb)
        note_onsets_outside = max(note_onsets_total - note_onsets_in_window, 0)
        note_events_in_window_for_spotcheck: List[Tuple[float, Optional[float], Any]] = []
        if tb_events_in_window_sample:
            for evt in tb_events_in_window_sample:
                if not isinstance(evt, (list, tuple)) or len(evt) < 1:
                    continue
                onset_t = float(evt[0])
                offset_t = float(evt[1]) if len(evt) >= 2 else None
                pitch_val = evt[2] if len(evt) >= 3 else None
                note_events_in_window_for_spotcheck.append((onset_t, offset_t, pitch_val))
        if not note_events_in_window_for_spotcheck:
            note_events_in_window_for_spotcheck = note_events_sync_in_window
        note_events_in_window_count = note_onsets_in_window
        avg_notes_per_onset_time = float(note_onsets_in_window) / max(unique_onsets, 1)
        in_window = note_onsets_in_window
        out_window = note_onsets_outside
        out_ratio = float(out_window) / float(max(note_onsets_total, 1))

        counts_source = (
            "target_build" if tb_note_total is not None and tb_note_in is not None and tb_unique_frames is not None else "sync_fallback"
        )
        sync_counts = {
            "total": note_onsets_total_fb,
            "in_window": note_onsets_in_window_fb,
            "unique_frames": len(unique_onset_frames_fb),
        }
        tb_counts = {
            "total": tb_note_total,
            "after_lag_total": tb_note_after_lag,
            "in_window": tb_note_in,
            "unique_frames": tb_unique_frames,
            "sample_len": len(tb_events_in_window_sample),
            "expected_onset_cells_from_events": _safe_int(target_build.get("expected_onset_cells_from_events")),
            "target_onset_cells_actual": _safe_int(target_build.get("target_onset_cells_actual")),
            "target_onset_frames_any_actual": _safe_int(target_build.get("target_onset_frames_any_actual")),
            "note_events_in_window_total": tb_note_in,
            "note_events_raw_total": tb_note_total,
            "note_events_after_lag_total": tb_note_after_lag,
            "chord_events_total": tb_chord_total,
            "chord_events_in_window": tb_chord_in,
            "timebase": target_build.get("timebase"),
            "window_start": target_build.get("window_start"),
            "window_end": target_build.get("window_end"),
            "min_onset_used": target_build.get("min_onset_used"),
            "max_onset_used": target_build.get("max_onset_used"),
            "onset_frames_touched_by_painting": target_build.get("onset_frames_touched_by_painting"),
            "painted_pairs_unique": target_build.get("painted_pairs_unique"),
            "unique_onset_center_frames_in_window": target_build.get("unique_onset_center_frames_in_window"),
            "events_in_window_total": target_build.get("events_in_window_total"),
            "events_seen_total": target_build.get("events_seen_total"),
            "events_painted_total": target_build.get("events_painted_total"),
            "frames_touched_count": target_build.get("onset_frames_touched_count"),
        }
        avg_notes_per_frame_tb = (
            float(tb_note_in) / max(float(tb_unique_frames), 1.0) if tb_note_in is not None and tb_unique_frames else None
        )
        multi_pitch_frame = False
        if tb_events_in_window_sample:
            frame_pitch: Dict[int, Set[Any]] = {}
            for evt in tb_events_in_window_sample:
                try:
                    onset_t = float(evt[0])
                    pitch_val = evt[2] if len(evt) >= 3 else None
                except Exception:
                    continue
                frame_idx = int(round((onset_t - clip_start) / max(hop_s, 1e-6)))
                if pitch_val is not None:
                    frame_pitch.setdefault(frame_idx, set()).add(pitch_val)
            multi_pitch_frame = any(len(p) > 1 for p in frame_pitch.values())
        dedup_suspect = bool(
            avg_notes_per_frame_tb is not None
            and abs(avg_notes_per_frame_tb - 1.0) < 1e-6
            and not multi_pitch_frame
            and (tb_note_in or 0) > 0
        )

        spec = getattr(ds, "frame_target_spec", None)
        tolerance_s = float(getattr(spec, "tolerance", 0.0) or 0.0)
        dilation_frames = int(getattr(spec, "dilation", 0) or 0)
        targets_sparse = bool(getattr(spec, "targets_sparse", False))
        frames_per_note_low = 1
        if targets_sparse:
            frames_per_note_high = 1
        else:
            frames_per_note_high = int(math.ceil((2.0 * tolerance_s) / max(hop_s, 1e-6)) + 1 + dilation_frames)
        tb_expected_cells = _safe_int(target_build.get("painted_pairs_unique")) or _safe_int(
            target_build.get("expected_onset_cells_from_events")
        )
        if tb_expected_cells is not None:
            expected_onset_cells_low = tb_expected_cells
            expected_onset_cells_high = tb_expected_cells
        else:
            expected_onset_cells_low = note_onsets_in_window * frames_per_note_low
            expected_onset_cells_high = note_onsets_in_window * frames_per_note_high
        onset_cells_actual = targets.get("onset", {}).get("count_pos")

        spotcheck_summary = self._spotcheck_onsets(
            note_events_in_window_for_spotcheck, clip_start, hop_s, sample.get("onset"), spec
        )
        onset_target = sample.get("onset")
        target_cells = _safe_int(target_build.get("target_onset_cells_actual"))
        target_any_frames = _safe_int(target_build.get("target_onset_frames_any_actual"))
        if target_cells is None or target_any_frames is None:
            if torch.is_tensor(onset_target):
                target_cells = int(torch.count_nonzero(onset_target).item())
                target_any_frames = int(torch.count_nonzero(onset_target.sum(dim=1)).item()) if onset_target.ndim >= 2 else None
        margin = 1.0 + self.warn_onset_cells_margin
        target_build_mismatch = False
        if targets_sparse and target_cells is not None:
            compare_to = tb_expected_cells if tb_expected_cells is not None else tb_note_in
            if compare_to is not None:
                upper_tb = float(compare_to) * margin
                lower_tb = float(compare_to) / margin
                if float(target_cells) > upper_tb or float(target_cells) < lower_tb:
                    target_build_mismatch = True
        if target_any_frames is not None:
            compare_frames = tb_unique_frames
            if compare_frames is not None:
                upper_frames = float(compare_frames) * margin
                lower_frames = float(compare_frames) / margin if compare_frames > 0 else 0.0
                if float(target_any_frames) > upper_frames or float(target_any_frames) < lower_frames:
                    target_build_mismatch = True
        clip_start_mismatch = abs(float(clip_start) - float(clip_start_ds)) > 1e-6

        tile_bounds = tile_meta.get("tile_bounds_px")
        tile_xyxy = tile_meta.get("tile_xyxy")
        if self.log_level != "detailed":
            tile_bounds = None
            tile_xyxy = None
        pad_left_list = tile_meta.get("pad_w_left")
        pad_right_list = tile_meta.get("pad_w_right")
        pad_left_total = max(pad_left_list or [0]) if isinstance(pad_left_list, list) else pad_left_list
        pad_right_total = max(pad_right_list or [0]) if isinstance(pad_right_list, list) else pad_right_list
        pad_total: List[int] = []
        pad_total_max: Optional[int] = None
        if isinstance(pad_left_list, list) or isinstance(pad_right_list, list):
            left = pad_left_list if isinstance(pad_left_list, list) else []
            right = pad_right_list if isinstance(pad_right_list, list) else []
            max_len = max(len(left), len(right), 0)
            if len(left) < max_len:
                left = left + [0] * (max_len - len(left))
            if len(right) < max_len:
                right = right + [0] * (max_len - len(right))
            pad_total = [int(l) + int(r) for l, r in zip(left, right)]
            pad_total_max = max(pad_total) if pad_total else None

        reg_enabled = bool(getattr(ds, "registration_enabled", False))
        refine_status_raw = refine_meta.get("status", "disabled" if not reg_enabled else "unknown")
        status_simple = "disabled" if not reg_enabled else str(refine_status_raw)
        if reg_enabled:
            if str(refine_status_raw).startswith("fallback"):
                status_simple = "fallback_used"
            else:
                try:
                    err_before_raw = refine_meta.get("err_before")
                    err_after_raw = refine_meta.get("err_after")
                    if err_before_raw is not None and err_after_raw is not None:
                        err_before = float(err_before_raw)
                        err_after = float(err_after_raw)
                        status_simple = "worsened" if err_after - err_before > self.warn_refine_worsened_delta else "ok"
                except Exception:
                    status_simple = str(refine_status_raw)

        record = {
            "record_type": "canary_report",
            "canary": {
                "video_rel": canary.video_rel,
                "path": str(canary.abs_path),
                "split": canary.split,
                "dataset_index": dataset_index,
            },
            "decode": {
                "decode_fps": decode_meta.get("decode_fps", ds.decode_fps),
                "actual_fps": actual_fps,
                "stride": ds.stride,
                "frames": ds.frames,
                "hop_seconds": hop_s,
                "clip_start_sec": clip_start,
                "duration_sec": duration_s,
                "source_hw": decode_meta.get("source_hw"),
                "tensor_shape": tensor_shape,
                "tensor_layout": "T,C,K,H,W",
                "grayscale": {"enabled": bool(getattr(ds, "grayscale", False)), "mode": self._grayscale_mode(ds)},
                "tiles": tensor_shape[2] if tensor_shape and len(tensor_shape) >= 3 else getattr(ds, "tiles", None),
            },
            "roi": {
                "source": (crop_meta or {}).get("crop_source") or ("metadata" if crop_meta else "none"),
                "applied": bool((crop_meta or {}).get("applied")),
                "valid": roi_valid,
                "xyxy": roi_xyxy,
                "output_hw": (crop_meta or {}).get("output_hw"),
                "issues": roi_issues,
            },
            "tiling": {
                "tiles_count": tile_meta.get("tiles", getattr(ds, "tiles", None)),
                "tile_bounds_px": tile_bounds,
                "tile_xyxy": tile_xyxy,
                "tile_hw": tile_meta.get("tile_hw"),
                "aligned_width": tile_meta.get("aligned_width"),
                "original_width": tile_meta.get("original_width"),
                "canonical_before_pad": tile_meta.get("canonical_before_pad"),
                "canonical_after_pad": tile_meta.get("canonical_after_pad"),
                "tokens_per_tile": tile_meta.get("tokens_per_tile"),
                "patch_w": tile_meta.get("patch_w"),
                "pad_w_left": tile_meta.get("pad_w_left"),
                "pad_w_right": tile_meta.get("pad_w_right"),
                "pad_w_left_max": pad_left_total,
                "pad_w_right_max": pad_right_total,
                "pad_w_total": pad_total,
                "pad_w_total_max": pad_total_max,
                "coord_system": tile_meta.get("coord_system"),
            },
            "refinement": {
                "enabled": bool(getattr(ds, "registration_enabled", False)),
                "status": refine_status_raw,
                "status_simple": status_simple,
                "err_before_px": refine_meta.get("err_before"),
                "err_after_px": refine_meta.get("err_after"),
                "err_white_px": refine_meta.get("err_white"),
                "err_black_px": refine_meta.get("err_black"),
                "frames": refine_meta.get("frames"),
                "source_hw": refine_meta.get("source_hw"),
                "target_hw": refine_meta.get("target_hw"),
                "geometry": refine_meta.get("cache_geometry") if self.log_level == "detailed" else None,
                "cache_path": str(reg_cache_path) if reg_cache_path is not None else None,
            },
            "lag": {
                "lag_seconds": lag_seconds,
                "lag_source": lag_source,
                "lag_ms": lag_ms,
                "lag_frames": lag_frames,
                "apply_to": "labels",
                "clip_start_used": clip_start,
                "clip_start_dataset": clip_start_ds,
                "counts_source": counts_source,
                "tb_keys": sorted(list(target_build.keys())) if isinstance(target_build, Mapping) else None,
                "sync_keys": sorted(list(sync_meta.keys())) if isinstance(sync_meta, Mapping) else None,
                "events_total": note_onsets_total,
                "events_in_window": in_window,
                "events_outside_window": out_window,
                "outside_ratio": out_ratio,
                "note_onsets_total": note_onsets_total,
                "note_onsets_in_window": note_onsets_in_window,
                "note_onsets_outside": note_onsets_outside,
                "note_offsets_in_window": note_offsets_in_window,
                "unique_onset_times_in_window": unique_onsets,
                "avg_notes_per_onset_time": avg_notes_per_onset_time,
                "quantization": self.onset_unique_time_quantization,
                "frames_per_note_low": frames_per_note_low,
                "frames_per_note_high": frames_per_note_high,
                "expected_onset_cells_low": expected_onset_cells_low,
                "expected_onset_cells_high": expected_onset_cells_high,
                "onset_cells_actual": onset_cells_actual,
                "target_onset_cells": target_cells,
                "target_onset_frames_any": target_any_frames,
                "note_events_in_window_count": note_events_in_window_count,
                "examples": examples,
                "tb_counts": tb_counts,
                "sync_counts": sync_counts,
                "avg_notes_per_frame_tb": avg_notes_per_frame_tb,
                "dedup_suspect": dedup_suspect,
                "multi_pitch_frame": multi_pitch_frame,
                "target_build_mismatch": target_build_mismatch,
                "clip_start_mismatch": clip_start_mismatch,
            },
            "targets": targets,
            "pitch_trace": pitch_trace,
            "offset_trace": offset_trace,
            "hand_trace": hand_trace,
            "spotcheck": spotcheck_summary,
            "frame_targets": {
                "tolerance_s": tolerance_s,
                "dilate_active_frames": dilation_frames,
                "targets_sparse": targets_sparse,
            },
            "warnings": [],
        }
        record["warnings"] = self._derive_warnings(record)
        return record

    def _derive_warnings(self, record: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        decode = record.get("decode", {})
        actual_fps = decode.get("actual_fps")
        if actual_fps is not None:
            try:
                if abs(float(actual_fps) - float(decode.get("decode_fps", 0.0))) > self.warn_fps_tolerance:
                    warnings.append("fps_mismatch")
            except Exception:
                pass
        roi_meta = record.get("roi", {}) or {}
        for issue in roi_meta.get("issues") or []:
            warnings.append(f"roi_{issue}")
        if roi_meta and not roi_meta.get("applied") and roi_meta.get("source") != "none":
            warnings.append("roi_not_applied")
        if roi_meta.get("xyxy") is not None and not roi_meta.get("valid"):
            warnings.append("roi_invalid")
        lag_meta = record.get("lag", {}) or {}
        refinement = record.get("refinement", {}) or {}
        if str(refinement.get("status", "")).startswith("fallback"):
            warnings.append("refine_fallback")
        try:
            err_before_raw = refinement.get("err_before_px")
            err_after_raw = refinement.get("err_after_px")
            if err_before_raw is not None and err_after_raw is not None:
                err_before = float(err_before_raw)
                err_after = float(err_after_raw)
                if err_after - err_before > self.warn_refine_worsened_delta:
                    warnings.append("refine_worsened")
        except Exception:
            pass
        targets = record.get("targets", {}) or {}
        onset_meta = targets.get("onset", {})
        onset_count = onset_meta.get("count_pos")
        note_in = lag_meta.get("note_onsets_in_window", 0)
        expected_high = lag_meta.get("expected_onset_cells_high")
        onset_cells_actual = lag_meta.get("onset_cells_actual")
        target_cells = lag_meta.get("target_onset_cells")
        target_any_frames = lag_meta.get("target_onset_frames_any")
        unique_onsets = lag_meta.get("unique_onset_times_in_window")
        if self.require_labels and note_in == 0:
            warnings.append("no_note_onsets_in_window")
        if onset_count is not None and onset_count > 0 and note_in == 0:
            warnings.append("targets_have_onsets_but_labels_none")
        if note_in > 0 and (onset_count is None or onset_count == 0):
            warnings.append("labels_have_onsets_but_targets_empty")
        try:
            if onset_cells_actual is not None and expected_high is not None:
                upper = float(expected_high) * (1.0 + self.warn_onset_cells_margin)
                lower = max(float(lag_meta.get("expected_onset_cells_low", 0.0)) / max(1.0, (1.0 + self.warn_onset_cells_margin)), 0.0)
                if float(onset_cells_actual) > upper or float(onset_cells_actual) < lower:
                    warnings.append("onset_cells_out_of_expected_range")
            if target_cells is not None and expected_high is not None:
                upper_t = float(expected_high) * (1.0 + self.warn_onset_cells_margin)
                if float(target_cells) > upper_t:
                    warnings.append("target_cells_out_of_expected_range")
            frames_touched = lag_meta.get("tb_counts", {}).get("frames_touched_count") if lag_meta.get("tb_counts") else None
            if frames_touched is None and target_any_frames is not None and unique_onsets is not None:
                if float(target_any_frames) > float(unique_onsets) * (1.0 + self.warn_onset_cells_margin):
                    warnings.append("target_any_frames_excess")
        except Exception:
            pass
        painted_pairs = lag_meta.get("tb_counts", {}).get("painted_pairs_unique") if lag_meta.get("tb_counts") else None
        frames_touched = lag_meta.get("tb_counts", {}).get("frames_touched_count") if lag_meta.get("tb_counts") else None
        if painted_pairs is not None and target_cells is not None and painted_pairs != target_cells:
            warnings.append("target_cells_mismatch_trace")
        if frames_touched is not None and target_any_frames is not None and frames_touched != target_any_frames:
            warnings.append("target_frames_mismatch_trace")
        if lag_meta.get("clip_start_mismatch"):
            warnings.append("clip_start_mismatch")
        spotcheck = record.get("spotcheck", {}) or {}
        if spotcheck.get("enabled") and spotcheck.get("fail", 0) > 0:
            warnings.append("spotcheck_failed")
        for head, meta in targets.items():
            if meta.get("missing"):
                warnings.append(f"missing_{head}")
        return warnings

    def _format_block(self, record: Dict[str, Any]) -> List[str]:
        lines = []
        decode = record.get("decode", {})
        roi = record.get("roi", {})
        tiling = record.get("tiling", {})
        refinement = record.get("refinement", {})
        lag_meta = record.get("lag", {})
        targets = record.get("targets", {})
        frame_targets = record.get("frame_targets", {}) or {}
        canary = record.get("canary", {})
        gray_info = decode.get("grayscale", {}) or {}
        gray_str = f"{'on' if gray_info.get('enabled') else 'off'}({gray_info.get('mode')})"
        lines.append(
            "[canary] id={video_rel} idx={idx} split={split} path={path} t0={t0:.3f}s dur={dur:.3f}s".format(
                video_rel=canary.get("video_rel"),
                idx=canary.get("dataset_index"),
                split=canary.get("split"),
                path=canary.get("path"),
                t0=decode.get("clip_start_sec", 0.0),
                dur=decode.get("duration_sec", 0.0),
            )
        )
        tensor_shape = decode.get("tensor_shape")
        lines.append(
            "[decode] req_fps={req} actual_fps={actual} stride={stride} frames={frames} hop_s={hop:.3f} "
            "source_hw={src} tensor_shape={shape} tiles={tiles} gray={gray}".format(
                req=decode.get("decode_fps"),
                actual=decode.get("actual_fps", "n/a"),
                stride=decode.get("stride"),
                frames=decode.get("frames"),
                hop=decode.get("hop_seconds", 0.0),
                src=decode.get("source_hw"),
                shape=tensor_shape,
                tiles=decode.get("tiles"),
                gray=gray_str,
            )
        )
        lines.append(
            "[roi] source={source} applied={applied} valid={valid} xyxy={coords} output_hw={out_hw} issues={issues}".format(
                source=roi.get("source"),
                applied=roi.get("applied"),
                valid=roi.get("valid"),
                coords=roi.get("xyxy"),
                out_hw=roi.get("output_hw"),
                issues=roi.get("issues"),
            )
        )
        lines.append(
            "[tiling] tiles={tiles} tile_hw={tile_hw} tile_xyxy={tile_xyxy} coord={coord} canon_before={canon_b} canon_after={canon_a} pad_left={pad_l} pad_left_max={pad_l_max} pad_right={pad_r} pad_right_max={pad_r_max} pad_total={pad_tot} pad_total_max={pad_tot_max}".format(
                tiles=tiling.get("tiles_count"),
                tile_hw=tiling.get("tile_hw"),
                tile_xyxy=tiling.get("tile_xyxy"),
                coord=tiling.get("coord_system"),
                canon_b=tiling.get("canonical_before_pad"),
                canon_a=tiling.get("canonical_after_pad"),
                pad_l=tiling.get("pad_w_left"),
                pad_l_max=tiling.get("pad_w_left_max"),
                pad_r=tiling.get("pad_w_right"),
                pad_r_max=tiling.get("pad_w_right_max"),
                pad_tot=tiling.get("pad_w_total"),
                pad_tot_max=tiling.get("pad_w_total_max"),
            )
        )
        lines.append(
            "[refine] enabled={enabled} status={status} err_before={err_b} err_after={err_a} err_white={err_w} err_black={err_k} cache={cache}".format(
                enabled=refinement.get("enabled"),
                status=refinement.get("status"),
                err_b=refinement.get("err_before_px"),
                err_a=refinement.get("err_after_px"),
                err_w=refinement.get("err_white_px"),
                err_k=refinement.get("err_black_px"),
                cache=refinement.get("cache_path"),
            )
        )
        lines.append(
            "[lag.provenance] counts_source={src} tb_keys={tbk} sync_keys={sk} tb_counts={tb_counts} sync_counts={sc} "
            "clip_start_used={clip} clip_start_ds={clip_ds}".format(
                src=lag_meta.get("counts_source"),
                tbk=lag_meta.get("tb_keys"),
                sk=lag_meta.get("sync_keys"),
                tb_counts=lag_meta.get("tb_counts"),
                sc=lag_meta.get("sync_counts"),
                clip=lag_meta.get("clip_start_used"),
                clip_ds=lag_meta.get("clip_start_dataset"),
            )
        )
        lines.append(
            "[lag.tb_counts] tb_note_total={tb_total} tb_note_in={tb_in} tb_unique_frames={tb_uniq} avg_notes_per_frame_tb={avg_tb} "
            "tb_after_lag={tb_after_lag} tb_in_window_total={tb_in_win} tb_chord_total={tb_chord_total} tb_chord_in={tb_chord_in} "
            "sync_total={sync_total} sync_in={sync_in} sync_unique={sync_uniq} "
            "sparse={sparse} mismatch={mismatch} dedup_suspect={dedup} target_cells={target_cells} target_any_frames={target_any} "
            "expected_from_events={exp_cells} painted_pairs={painted_pairs}".format(
                tb_total=lag_meta.get("tb_counts", {}).get("total") if lag_meta.get("tb_counts") else None,
                tb_in=lag_meta.get("tb_counts", {}).get("in_window") if lag_meta.get("tb_counts") else None,
                tb_uniq=lag_meta.get("tb_counts", {}).get("unique_frames") if lag_meta.get("tb_counts") else None,
                avg_tb=lag_meta.get("avg_notes_per_frame_tb"),
                tb_after_lag=lag_meta.get("tb_counts", {}).get("after_lag_total") if lag_meta.get("tb_counts") else None,
                tb_in_win=lag_meta.get("tb_counts", {}).get("note_events_in_window_total") if lag_meta.get("tb_counts") else None,
                tb_chord_total=lag_meta.get("tb_counts", {}).get("chord_events_total") if lag_meta.get("tb_counts") else None,
                tb_chord_in=lag_meta.get("tb_counts", {}).get("chord_events_in_window") if lag_meta.get("tb_counts") else None,
                sync_total=lag_meta.get("sync_counts", {}).get("total") if lag_meta.get("sync_counts") else None,
                sync_in=lag_meta.get("sync_counts", {}).get("in_window") if lag_meta.get("sync_counts") else None,
                sync_uniq=lag_meta.get("sync_counts", {}).get("unique_frames") if lag_meta.get("sync_counts") else None,
                sparse=frame_targets.get("targets_sparse"),
                mismatch=lag_meta.get("target_build_mismatch"),
                dedup=lag_meta.get("dedup_suspect"),
                target_cells=lag_meta.get("target_onset_cells"),
                target_any=lag_meta.get("target_onset_frames_any"),
                exp_cells=lag_meta.get("tb_counts", {}).get("expected_onset_cells_from_events") if lag_meta.get("tb_counts") else None,
                painted_pairs=lag_meta.get("tb_counts", {}).get("painted_pairs_unique") if lag_meta.get("tb_counts") else None,
            )
        )
        lines.append(
            "[lag.trace] timebase={timebase} window=[{ws},{we}] min_onset_used={min_on} max_onset_used={max_on} "
            "frames_touched_count={ftc} frames_touched={ft} centers={centers}".format(
                timebase=lag_meta.get("tb_counts", {}).get("timebase") if lag_meta.get("tb_counts") else None,
                ws=lag_meta.get("tb_counts", {}).get("window_start") if lag_meta.get("tb_counts") else None,
                we=lag_meta.get("tb_counts", {}).get("window_end") if lag_meta.get("tb_counts") else None,
                min_on=lag_meta.get("tb_counts", {}).get("min_onset_used") if lag_meta.get("tb_counts") else None,
                max_on=lag_meta.get("tb_counts", {}).get("max_onset_used") if lag_meta.get("tb_counts") else None,
                ftc=len(lag_meta.get("tb_counts", {}).get("onset_frames_touched_by_painting") or []),
                ft=lag_meta.get("tb_counts", {}).get("onset_frames_touched_by_painting"),
                centers=lag_meta.get("tb_counts", {}).get("unique_onset_center_frames_in_window") if lag_meta.get("tb_counts") else None,
            )
        )
        lines.append(
            "[lag] seconds={lag} source={source} frames={frames} chords_in={chords} note_onsets_cells={cells_used} centers={uniq} frames_touched={ft} "
            "avg_notes_per_frame={avg:.2f} onset_cells={cells} expected=[{low},{high}] target_any_frames={any_frames} q={quant} examples={examples}".format(
                lag=lag_meta.get("lag_seconds"),
                source=lag_meta.get("lag_source"),
                frames=lag_meta.get("lag_frames"),
                chords=lag_meta.get("tb_counts", {}).get("events_in_window_total") if lag_meta.get("tb_counts") else lag_meta.get("events_in_window"),
                cells_used=lag_meta.get("tb_counts", {}).get("painted_pairs_unique") if lag_meta.get("tb_counts") else lag_meta.get("note_onsets_in_window"),
                uniq=lag_meta.get("unique_onset_times_in_window"),
                ft=lag_meta.get("tb_counts", {}).get("frames_touched_count") if lag_meta.get("tb_counts") else None,
                avg=float(lag_meta.get("avg_notes_per_onset_time", 0.0)),
                cells=lag_meta.get("onset_cells_actual"),
                low=lag_meta.get("expected_onset_cells_low"),
                high=lag_meta.get("expected_onset_cells_high"),
                any_frames=lag_meta.get("target_onset_frames_any"),
                quant=lag_meta.get("quantization"),
                examples=lag_meta.get("examples"),
            )
        )
        spotcheck = record.get("spotcheck", {}) or {}
        if spotcheck.get("enabled"):
            lines.append(
                "[spotcheck.onset] n={n} ok={ok} fail={fail}".format(
                    n=len(spotcheck.get("results") or []),
                    ok=spotcheck.get("ok", 0),
                    fail=spotcheck.get("fail", 0),
                )
            )
        target_bits = []
        for key in ("pitch", "onset", "offset", "hand", "clef"):
            meta = targets.get(key, {})
            target_bits.append(
                f"{key}:shape={meta.get('shape')} pos={meta.get('pos_rate')} count={meta.get('count_pos')} min_midi={meta.get('min_midi')} max_midi={meta.get('max_midi')} unique={meta.get('unique_values')} missing={meta.get('missing')}"
            )
        lines.append(f"[targets] {' | '.join(target_bits)}")
        warnings = record.get("warnings", [])
        if warnings:
            lines.append(f"[warnings] {warnings}")
        return lines

    def _format_scorecard(self, record: Dict[str, Any], status: str, error: Optional[str]) -> str:
        roi = record.get("roi", {}) or {}
        tiling = record.get("tiling", {}) or {}
        decode = record.get("decode", {}) or {}
        lag_meta = record.get("lag", {}) or {}
        targets = record.get("targets", {}) or {}
        warnings = list(record.get("warnings", []))
        if error:
            warnings.append(str(error))
        gray_info = decode.get("grayscale", {}) or {}
        gray_str = f"{'on' if gray_info.get('enabled') else 'off'}({gray_info.get('mode')})"
        tile_hw = tiling.get("tile_hw") or []
        tile_summary = "?"
        if tile_hw:
            h, w = tile_hw[0]
            tile_summary = f"{len(tile_hw)}x({h}x{w})"
        onset_count = targets.get("onset", {}).get("count_pos")
        offset_count = targets.get("offset", {}).get("count_pos")
        pitch_rate = targets.get("pitch", {}).get("pos_rate")
        notes_in = lag_meta.get("note_onsets_in_window")
        uniq_onsets = lag_meta.get("unique_onset_times_in_window")
        onset_cells_actual = lag_meta.get("onset_cells_actual")
        onset_cells_high = lag_meta.get("expected_onset_cells_high")
        target_cells = lag_meta.get("target_onset_cells")
        warning_str = ";".join(warnings)
        return (
            "[score] {status} video={video} split={split} roi={roi_src}(valid={valid}) tiles={tiles} "
            "canon_after={canon_after} gray={gray} lag={lag:.3f}s note_in={notes_in}/{notes_total} chord_frames={uniq_on} "
            "onset_cells={cells} exp_high={cells_high} target_cells={target_cells} on={on} off={off} pitch_pos={pitch} warnings=[{warn}]"
        ).format(
            status=status,
            video=record.get("canary", {}).get("video_rel"),
            split=record.get("canary", {}).get("split"),
            roi_src=roi.get("source"),
            valid="Y" if roi.get("applied") and roi.get("valid") else "N",
            tiles=tile_summary,
            canon_after=tiling.get("canonical_after_pad"),
            gray=gray_str,
            lag=float(lag_meta.get("lag_seconds", 0.0)),
            notes_in=notes_in,
            notes_total=lag_meta.get("note_onsets_total"),
            uniq_on=uniq_onsets,
            cells=onset_cells_actual,
            cells_high=onset_cells_high,
            target_cells=target_cells,
            on=onset_count,
            off=offset_count,
            pitch=f"{pitch_rate:.4f}" if pitch_rate is not None else "n/a",
            warn=warning_str,
        )

    def emit(self, record: Optional[Dict[str, Any]], *, status: str, error: Optional[str] = None) -> None:
        record = record or {}
        record.setdefault("record_type", "canary_report")
        record["status"] = status
        if record.get("warnings") is None:
            record["warnings"] = []
        if error:
            record.setdefault("warnings", []).append(str(error))
        # footer counters
        if status.upper() == "OK":
            self.footer_counts["ok"] += 1
        else:
            self.footer_counts["fail"] += 1
        for w in record.get("warnings", []):
            self.footer_counts["warnings"][w] = self.footer_counts["warnings"].get(w, 0) + 1
        refine_key = (record.get("refinement") or {}).get("status_simple") or (record.get("refinement") or {}).get("status", "unknown")
        self.footer_counts["refine_status"][refine_key] = self.footer_counts["refine_status"].get(refine_key, 0) + 1
        if self.log_level == "detailed":
            for line in self._format_block(record):
                print(line)
        if self.log_level in {"detailed", "summary"}:
            print(self._format_scorecard(record, status=status, error=error))
        if self.export_json and self.json_path is not None:
            payload = dict(record)
            if error:
                payload["error"] = str(error)
            self.json_records.append(payload)

    def flush(self) -> None:
        if not self.export_json or not self.json_records or self.json_path is None:
            return
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        serializable: List[Dict[str, Any]] = []
        for rec in self.json_records:
            serializable.append(self._jsonable(rec))
        self.json_path.write_text(
            "\n".join(json.dumps(rec, sort_keys=True) for rec in serializable), encoding="utf-8"
        )

    def emit_footer(self) -> None:
        total = self.footer_counts["ok"] + self.footer_counts["fail"]
        print(
            "[run.summary] total={total} ok={ok} fail={fail}".format(
                total=total,
                ok=self.footer_counts["ok"],
                fail=self.footer_counts["fail"],
            )
        )
        warn_items = sorted(self.footer_counts["warnings"].items(), key=lambda x: x[0])
        if warn_items:
            print("[run.summary.warnings] " + " ".join(f"{k}={v}" for k, v in warn_items))
        refine_items = sorted(self.footer_counts["refine_status"].items(), key=lambda x: x[0])
        if refine_items:
            print("[run.summary.refine] " + " ".join(f"{k}={v}" for k, v in refine_items))

    def _jsonable(self, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        try:
            if isinstance(obj, np.generic):  # type: ignore[attr-defined]
                return obj.item()
        except Exception:
            pass
        if torch.is_tensor(obj):
            return obj.tolist()
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        try:
            return json.loads(json.dumps(obj))
        except Exception:
            return str(obj)

def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args)
    cfg = dict(load_yaml_file(config_path))
    dataset_cfg = dict(cfg.get("dataset", {}))
    testing_cfg = dict(dataset_cfg.get("testing", {}))
    options = _resolve_testing_options(testing_cfg, args)
    cfg = _prepare_dataset_cfg(cfg, testing_cfg, options)
    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", config_path.stem)).lower()

    split, split_trace = _resolve_split(dataset_cfg)
    candidates = _gather_candidates(split, dataset_cfg)

    if options.regen:
        if not candidates:
            raise SystemExit(f"[regen] no videos found under {dataset_cfg['root_dir']}/{split}")
        canaries = _save_canaries(split, candidates, options.canary_count, options.canary_json, dataset_cfg)
        print(f"[regen] wrote {len(canaries)} canaries to {options.canary_json}")
        return

    try:
        canaries = _load_canaries(options.canary_json)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    if any(c.split != split for c in canaries):
        raise SystemExit(f"Canary split mismatch: config resolved split '{split}' but canary list expects '{canaries[0].split}'. Regenerate canaries.")

    ds = _build_dataset(cfg, split=split)
    entry_map = {Path(e.video_path).resolve(): idx for idx, e in enumerate(ds.entries)}
    alt_ds = None
    alt_entry_map: Dict[Path, int] = {}
    if options.cache_equiv:
        alt_cfg = json.loads(json.dumps(cfg))
        alt_dataset_cfg = alt_cfg.get("dataset", {}) or {}
        ft_cfg = dict(alt_dataset_cfg.get("frame_targets", {}) or {})
        ft_cfg["cache_dir"] = tempfile.mkdtemp(prefix=f"tivit_canary_ft_alt_{dataset_name}_")
        if options.cache_off:
            ft_cfg["cache_labels"] = False
        alt_dataset_cfg["frame_targets"] = ft_cfg
        alt_dataset_cfg["num_workers"] = 0
        alt_cfg["dataset"] = alt_dataset_cfg
        alt_ds = _build_dataset(alt_cfg, split=split)
        alt_entry_map = {Path(e.video_path).resolve(): idx for idx, e in enumerate(alt_ds.entries)}

    require_labels_default = bool(getattr(ds, "require_labels", dataset_cfg.get("require_labels", False)))
    require_labels = bool(require_labels_default or options.strict_labels)

    global_stats: Dict[str, Any] = {"videos": [], "ranges": [], "targets": {}}
    reporter = ModelReadinessReporter(
        dataset_name=dataset_name,
        config_path=config_path,
        dataset_cfg=dataset_cfg,
        resolved_split=split,
        split_trace=split_trace,
        options=options,
        require_labels=require_labels,
    )
    reporter.emit_run_header(ds)

    for canary in canaries:
        if not canary.abs_path.exists():
            raise FileNotFoundError(f"Canary video missing: {canary.abs_path}")
        if canary.abs_path not in entry_map:
            raise AssertionError(f"Canary video not present in dataset entries: {canary.abs_path}")
        idx = entry_map[canary.abs_path]
        record: Optional[Dict[str, Any]] = None
        collect_error: Optional[BaseException] = None
        try:
            sample = ds[idx]
        except FileNotFoundError as exc:
            reporter.emit(
                record,
                status="FAIL",
                error=(
                    SystemExit(
                        f"Missing label for canary {canary.video_rel} (split={split}). "
                        "Ensure annotations/manifest are wired up or disable require_labels for this dataset."
                    ).__str__()
                    if require_labels
                    else str(exc)
                ),
            )
            if require_labels:
                raise SystemExit(
                    f"Missing label for canary {canary.video_rel} (split={split}). "
                    "Ensure annotations/manifest are wired up or disable require_labels for this dataset."
                ) from exc
            raise
        except Exception as exc:  # pragma: no cover - fail loudly
            reporter.emit(record, status="FAIL", error=str(exc))
            raise RuntimeError(f"Failed to load canary {canary.video_rel}: {exc}. Regenerate canaries or fix the source clip.") from exc

        try:
            try:
                record = reporter.collect(canary=canary, sample=sample, ds=ds, dataset_index=idx)
            except BaseException as exc:
                collect_error = exc

            T = ds.frames
            _check_frames(sample, T, canary, ds.tiles, resize_hw=ds.resize_hw, stride=ds.stride)
            _check_targets(sample, T, canary, require_labels=require_labels)
            _check_determinism(ds, idx, canary)
            _check_roi_and_fallbacks(sample, canary, strict_fallback=options.strict_fallback)
            _check_sampling_rules(ds, sample, canary, strict_sampling=options.strict_sampling)
            global_stats["videos"].append(canary.video_rel)
            video = sample["video"]
            global_stats["ranges"].append((float(video.min()), float(video.max())))
            for key in ("pitch", "onset", "offset", "hand", "clef"):
                tensor = sample.get(key)
                if torch.is_tensor(tensor):
                    stats = global_stats["targets"].setdefault(key, [])
                    stats.append(float(tensor.float().mean()))

            if alt_ds is not None:
                if canary.abs_path not in alt_entry_map:
                    raise AssertionError(f"cache equiv: canary not present in alt dataset {canary.abs_path}")
                sample_alt = alt_ds[alt_entry_map[canary.abs_path]]
                for key in ("video", "pitch", "onset", "offset", "hand", "clef"):
                    t1 = sample.get(key)
                    t2 = sample_alt.get(key)
                    if torch.is_tensor(t1) and torch.is_tensor(t2):
                        if not torch.allclose(t1, t2, atol=1e-5, rtol=1e-5):
                            raise AssertionError(f"cache on/off mismatch for {key} in {canary.video_rel}")

            if options.enable_audit and options.audit_dir:
                _export_audit(sample, canary, options.audit_dir, tiles=ds.tiles, norm_mean=ds.norm_mean, norm_std=ds.norm_std, resize_hw=ds.resize_hw)

            reporter.emit(record, status="OK", error=str(collect_error) if collect_error else None)
        except SystemExit as exc:
            reporter.emit(record, status="FAIL", error=str(exc))
            raise
        except Exception as exc:
            reporter.emit(record, status="FAIL", error=str(exc))
            raise

    if options.enable_audit and options.audit_dir:
        summary = {
            "videos": global_stats["videos"],
            "video_range": {
                "min": float(min(r[0] for r in global_stats["ranges"])),
                "max": float(max(r[1] for r in global_stats["ranges"])),
            },
            "target_means": {k: sum(v) / max(len(v), 1) for k, v in global_stats["targets"].items()},
        }
        (options.audit_dir / "global_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    reporter.emit_footer()
    reporter.flush()


if __name__ == "__main__":
    main()
