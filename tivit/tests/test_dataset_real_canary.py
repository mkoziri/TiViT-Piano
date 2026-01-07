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
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

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

    cfg["dataset"] = dataset_cfg
    return cfg


def _resolve_split(dataset_cfg: Mapping[str, Any]) -> str:
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
    for split in candidates:
        if split and (root / str(split)).exists():
            return str(split)
    return str(dataset_cfg.get("split", "val"))


def _gather_candidates(split: str, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    split_dir = Path(dataset_cfg["root_dir"]) / split
    return sorted(split_dir.rglob("*.mp4"))


def _select_decodable(candidates: List[Path], count: int, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    selected: List[Path] = []
    vr_cfg = VideoReaderConfig(
        frames=int(dataset_cfg.get("frames", 32)),
        stride=1,
        resize_hw=tuple(dataset_cfg.get("resize", [224, 224])),
        channels=int(dataset_cfg.get("channels", 3)),
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
                x0, y0, x1, y1 = map(float, vals[:4])
                if x1 <= x0 or y1 <= y0:
                    raise AssertionError(f"degenerate crop {crop} for {canary.video_rel}")
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
    if start_frame is not None and start_frame >= ds.frames and strict_sampling:
        raise AssertionError(f"start_frame {start_frame} outside clip for {canary.video_rel}")
    if ds.stride < 1:
        raise AssertionError(f"stride {ds.stride} invalid for {canary.video_rel}")


def _export_audit(sample: Mapping[str, Any], canary: Canary, audit_dir: Path, *, tiles: int, norm_mean: Iterable[float], norm_std: Iterable[float]) -> None:
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
        tile_tensor = vid_denorm[idx]
        tiles_np = tile_tensor.permute(1, 0, 2, 3).reshape(3, tiles * tile_tensor.shape[-2], tile_tensor.shape[-1])
        tiles_np = tiles_np.permute(1, 2, 0).cpu().numpy()
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

    split = _resolve_split(dataset_cfg)
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

    for canary in canaries:
        if not canary.abs_path.exists():
            raise FileNotFoundError(f"Canary video missing: {canary.abs_path}")
        if canary.abs_path not in entry_map:
            raise AssertionError(f"Canary video not present in dataset entries: {canary.abs_path}")
        idx = entry_map[canary.abs_path]
        try:
            sample = ds[idx]
        except FileNotFoundError as exc:
            if require_labels:
                raise SystemExit(
                    f"Missing label for canary {canary.video_rel} (split={split}). "
                    "Ensure annotations/manifest are wired up or disable require_labels for this dataset."
                ) from exc
            raise
        except Exception as exc:  # pragma: no cover - fail loudly
            raise RuntimeError(f"Failed to load canary {canary.video_rel}: {exc}. Regenerate canaries or fix the source clip.") from exc

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
            _export_audit(sample, canary, options.audit_dir, tiles=ds.tiles, norm_mean=ds.norm_mean, norm_std=ds.norm_std)
        print(f"[ok] {canary.video_rel} split={split} idx={idx}")

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


if __name__ == "__main__":
    main()
