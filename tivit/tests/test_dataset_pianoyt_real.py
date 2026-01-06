#!/usr/bin/env python3
"""
Purpose:
    Deterministic real-data canary gate for PianoYT using the validation split.
    Fails loudly on missing canaries or invariants; supports regenerating the
    canary list and optional audit exports.

Usage:
    python tivit/tests/test_dataset_pianoyt_real.py           # run gate
    TIVIT_CANARY_REGEN=1 python tivit/tests/test_dataset_pianoyt_real.py  # regenerate list
    TIVIT_CANARY_AUDIT_DIR=out/audit python tivit/tests/test_dataset_pianoyt_real.py  # export visuals
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

try:  # optional visuals
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore

from tivit.core.config import load_yaml_file
from tivit.data.datasets.pianoyt_impl import PianoYTDataset
from tivit.data.decode.video_reader import VideoReaderConfig, load_clip
from tivit.data.targets.frame_targets import prepare_frame_targets, resolve_soft_target_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tivit" / "configs" / "dataset" / "pianoyt.yaml"
CANARY_PATH = REPO_ROOT / "tivit" / "tests" / "resources" / "pianoyt_canaries.json"
DEFAULT_CANARY_COUNT = 2
STRICT_LABELS = os.environ.get("TIVIT_CANARY_STRICT_LABELS") is not None
STRICT_FALLBACK = os.environ.get("TIVIT_CANARY_STRICT_FALLBACK") is not None
CACHE_EQUIV = os.environ.get("TIVIT_CANARY_CACHE_EQUIV") is not None
CACHE_EQUIV = os.environ.get("TIVIT_CANARY_CACHE_EQUIV") is not None
CACHE_OFF = os.environ.get("TIVIT_CANARY_CACHE_OFF") is not None
STRICT_SAMPLING = os.environ.get("TIVIT_CANARY_STRICT_SAMPLING") is not None


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


def _load_config() -> dict:
    cfg = dict(load_yaml_file(CONFIG_PATH))
    dataset_cfg = dict(cfg.get("dataset", {}))
    root_dir = dataset_cfg.get("root_dir", "data/PianoYT")
    root_path = (REPO_ROOT / root_dir).resolve()
    dataset_cfg["root_dir"] = str(root_path)
    dataset_cfg["annotations_root"] = str(dataset_cfg.get("annotations_root", root_path))
    # Force cache isolation for this test
    frame_targets_cfg = dict(dataset_cfg.get("frame_targets", {}))
    frame_targets_cfg["cache_dir"] = tempfile.mkdtemp(prefix="tivit_canary_ft_off_" if CACHE_OFF else "tivit_canary_ft_")
    if CACHE_OFF:
        frame_targets_cfg["cache_labels"] = False
    dataset_cfg["frame_targets"] = frame_targets_cfg
    dataset_cfg["num_workers"] = 0
    dataset_cfg["require_labels"] = False  # default relaxed; strict via env
    cfg["dataset"] = dataset_cfg
    return cfg


def _gather_candidates(split: str, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    split_dir = Path(dataset_cfg["root_dir"]) / split
    return sorted(split_dir.rglob("*.mp4"))


def _select_decodable(candidates: List[Path], count: int, dataset_cfg: Mapping[str, Any]) -> List[Path]:
    """Pick decodable videos deterministically."""

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


def _save_canaries(split: str, candidates: List[Path], count: int, dataset_cfg: Mapping[str, Any]) -> List[Canary]:
    CANARY_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    CANARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return [Canary(video_rel=rel, split=split) for rel in payload["canaries"]]


def _load_canaries() -> List[Canary]:
    if not CANARY_PATH.exists():
        raise FileNotFoundError(f"canary list missing at {CANARY_PATH}. Regenerate with TIVIT_CANARY_REGEN=1.")
    payload = json.loads(CANARY_PATH.read_text(encoding="utf-8"))
    canaries = payload.get("canaries", [])
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
    if H != expected_h:
        raise AssertionError(f"tile height {H} != resize height {expected_h} for {canary.video_rel}")
    expected_w = int(list(resize_hw)[1]) if resize_hw else W
    if W < max(32, expected_w * 0.2):  # allow for crops; only fail on degenerate widths
        raise AssertionError(f"tile width {W} unexpectedly small vs resize width {expected_w} for {canary.video_rel}")
    total_width = W * tiles
    if expected_w > 0 and not (0.8 * expected_w <= total_width <= 1.3 * expected_w):
        raise AssertionError(
            f"tiling coverage {total_width} px out of bounds for expected ~{expected_w} on {canary.video_rel}"
        )
    _assert_finite(video, "video", canary)
    vmin, vmax = float(video.min()), float(video.max())
    if not (-5.0 <= vmin <= 5.0 and -5.0 <= vmax <= 5.0):
        raise AssertionError(f"video values out of expected range [{vmin}, {vmax}] for {canary.video_rel}")
    # Basic tile coverage: no zero-variance tiles
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
            else:
                continue
        if not torch.is_tensor(tensor):
            if require_labels:
                raise AssertionError(f"target '{key}' not a tensor for {canary.video_rel}")
            else:
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


def _check_determinism(ds: PianoYTDataset, idx: int, canary: Canary) -> None:
    first = ds[idx]
    second = ds[idx]
    for key in ("video", "pitch", "onset", "offset", "hand", "clef"):
        t1 = first.get(key)
        t2 = second.get(key)
        if torch.is_tensor(t1) and torch.is_tensor(t2):
            if not torch.allclose(t1, t2, atol=1e-5, rtol=1e-5):
                raise AssertionError(f"determinism check failed for {key} in {canary.video_rel}")


def _check_roi_and_fallbacks(sample: Mapping[str, Any], canary: Canary) -> None:
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
    if STRICT_FALLBACK and (sync_src.startswith("fallback") or sync_src == "default"):
        raise AssertionError(f"sync fallback detected for {canary.video_rel}: {sync_src}")
    meta_str = json.dumps(meta).lower()
    if STRICT_FALLBACK and ("fallback" in meta_str or meta.get("status") == "fallback"):
        raise AssertionError(f"fallback detected in metadata for {canary.video_rel}")


def _check_sampling_rules(ds: PianoYTDataset, sample: Mapping[str, Any], canary: Canary) -> None:
    T = ds.frames
    hop = ds.stride / max(ds.decode_fps, 1e-6)
    duration = hop * max(T - 1, 0)
    if duration <= 0:
        raise AssertionError(f"invalid duration derived from stride/fps for {canary.video_rel}")
    sampler_meta = sample.get("sampler_meta", {}) or {}
    start_frame = sampler_meta.get("start_frame")
    if start_frame is not None and start_frame >= ds.frames and STRICT_SAMPLING:
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
        tiles_np = vid_denorm[idx].permute(1, 0, 2, 3).reshape(3, tiles * vid_denorm.shape[-2], vid_denorm.shape[-1])
        tiles_np = tiles_np.permute(1, 2, 0).cpu().numpy()
        out_path = audit_dir / f"{canary.video_id}_frame{idx:03d}_tiles.png"
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            continue
        imageio.imwrite(out_path, (tiles_np * 255).astype("uint8"))

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

    # JSON summary
    summary = {
        "video": canary.video_rel,
        "shape": list(video.shape),
        "dtype": str(video.dtype),
        "video_range": [float(video.min()), float(video.max())],
        "metadata": sample.get("metadata", {}),
        "targets": {k: list(sample[k].shape) if torch.is_tensor(sample.get(k)) else None for k in ("pitch", "onset", "offset", "hand", "clef")},
    }
    (audit_dir / f"{canary.video_id}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    regen = os.environ.get("TIVIT_CANARY_REGEN") is not None
    audit_dir_env = os.environ.get("TIVIT_CANARY_AUDIT_DIR")
    audit_dir = Path(audit_dir_env) if audit_dir_env else None

    cfg = _load_config()
    dataset_cfg = cfg["dataset"]
    split = dataset_cfg.get("split_val") or dataset_cfg.get("split") or "val"
    candidates = _gather_candidates(split, dataset_cfg)

    if regen:
        if not candidates:
            raise SystemExit(f"[regen] no videos found under {dataset_cfg['root_dir']}/{split}")
        count = int(os.environ.get("TIVIT_CANARY_COUNT", DEFAULT_CANARY_COUNT))
        canaries = _save_canaries(split, candidates, count, dataset_cfg)
        print(f"[regen] wrote {len(canaries)} canaries to {CANARY_PATH}")
        return

    try:
        canaries = _load_canaries()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    if not canaries:
        raise SystemExit("Canary list is empty; regenerate with TIVIT_CANARY_REGEN=1.")

    ds = PianoYTDataset(cfg, split=split, full_cfg=cfg)
    entry_map = {Path(e.video_path).resolve(): idx for idx, e in enumerate(ds.entries)}

    for canary in canaries:
        if not canary.abs_path.exists():
            raise FileNotFoundError(f"Canary video missing: {canary.abs_path}")
        if canary.abs_path not in entry_map:
            raise AssertionError(f"Canary video not present in dataset entries: {canary.abs_path}")
        idx = entry_map[canary.abs_path]
        try:
            sample = ds[idx]
        except Exception as exc:
            raise RuntimeError(f"Failed to load canary {canary.video_rel}: {exc}. "
                               "Regenerate canaries or fix the source clip.") from exc
        T = ds.frames
        _check_frames(sample, T, canary, ds.tiles, resize_hw=ds.resize_hw, stride=ds.stride)
        _check_targets(sample, T, canary, require_labels=STRICT_LABELS or bool(dataset_cfg.get("require_labels", False)))
        _check_determinism(ds, idx, canary)
        _check_roi_and_fallbacks(sample, canary)
        _check_sampling_rules(ds, sample, canary)
        if CACHE_EQUIV:
            alt_cfg = json.loads(json.dumps(cfg))
            alt_cfg["dataset"]["frame_targets"]["cache_dir"] = tempfile.mkdtemp(prefix="tivit_canary_ft_alt_")
            ds_alt = PianoYTDataset(alt_cfg, split=split, full_cfg=alt_cfg)
            entry_map_alt = {Path(e.video_path).resolve(): j for j, e in enumerate(ds_alt.entries)}
            if canary.abs_path not in entry_map_alt:
                raise AssertionError(f"cache equiv: canary not present in alt dataset {canary.abs_path}")
            sample_alt = ds_alt[entry_map_alt[canary.abs_path]]
            for key in ("video", "pitch", "onset", "offset", "hand", "clef"):
                t1 = sample.get(key)
                t2 = sample_alt.get(key)
                if torch.is_tensor(t1) and torch.is_tensor(t2):
                    if not torch.allclose(t1, t2, atol=1e-5, rtol=1e-5):
                        raise AssertionError(f"cache on/off mismatch for {key} in {canary.video_rel}")
        if audit_dir:
            _export_audit(sample, canary, audit_dir, tiles=ds.tiles, norm_mean=ds.norm_mean, norm_std=ds.norm_std)
        print(f"[ok] {canary.video_rel} split={split} idx={idx}")


if __name__ == "__main__":
    main()
