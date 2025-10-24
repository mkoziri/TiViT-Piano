#!/usr/bin/env python3
"""Rebuild the PianoYT per-video audio/video lag cache.

Purpose:
    Iterate PianoYT splits, sample representative clip windows, and estimate a
    per-video lag (milliseconds) that is persisted to ``av_lags.json`` for reuse
    by datasets, calibration, and training scripts.

Key Functions/Classes:
    - probe_video: Gather fps, frame count, and dimensions for a video path.
    - resolve_keyboard_bbox: Determine the keyboard ROI or fall back to full frame.
    - main: CLI entry point orchestrating PianoYT iteration and cache updates.

CLI Arguments:
    --split {train,val,test} (required)
        Dataset split to process; no default provided.
    --root PATH (default: auto resolved by PianoYT helpers)
        Override PianoYT root directory resolution.
    --max-videos INT (default: 500)
        Limit the number of videos processed during the run.
    --tries-per-video INT (default: 4)
        Candidate windows evaluated per video to avoid quiet spans.
    --frames INT (default: 96)
        Frames decoded per window (stride implicitly 1/fps).
    --fps FLOAT (default: native fps or 30.0 fallback)
        Override decode fps; when omitted, derive from video metadata.
    --search-ms INT (default: 500)
        Cross-correlation half-window in milliseconds for lag search.
    --keyboard-bbox {reg,full} (default: reg)
        Keyboard ROI mode: registration-aware boxes or full frame.
    --debug-bbox INT (default: 0)
        Print the resolved keyboard ROI for the first N videos when > 0.
    --refresh (default: False)
        Recompute and overwrite existing cache entries when set.
    --min-onsets INT (default: 1)
        Minimum note onsets required within a sampled window.
    --min-corr FLOAT (default: 0.25)
        Minimum correlation required before accepting the estimated lag.
    --max-runtime-s FLOAT (default: 3.0)
        Maximum runtime in seconds allocated to lag estimation per video.

Usage:
    python scripts/rebuild_av_lags.py --split train --refresh --min-corr 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.omaps_dataset import apply_registration_crop, _load_clip_with_random_start  # noqa: E402
from data.pianoyt_dataset import (  # noqa: E402
    _expand_root,
    _load_metadata,
    _read_midi_events,
    _read_split_ids,
    _resolve_media_paths,
)
from utils.av_sync import AVLagCache, AVLagResult, compute_av_lag  # noqa: E402
from utils.identifiers import canonical_video_id  # noqa: E402
from utils.registration_refinement import RegistrationRefiner  # noqa: E402
from utils.time_grid import frame_to_sec  # noqa: E402


try:  # Optional fast-path
    import decord  # type: ignore

    _HAVE_DECORD = True
except Exception:  # pragma: no cover - optional dependency
    decord = None  # type: ignore
    _HAVE_DECORD = False


DEFAULT_CANONICAL_HW = (145, 800)


@dataclass
class VideoInfo:
    fps: float
    num_frames: int
    duration: float
    height: int
    width: int


def _safe_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    return cv2


def probe_video(path: Path) -> VideoInfo:
    """Return (fps, num_frames, duration, height, width) for ``path``."""

    if _HAVE_DECORD:
        try:
            assert decord is not None
            vr = decord.VideoReader(str(path))
        except Exception:
            vr = None
        else:
            fps = float(vr.get_avg_fps()) or 0.0
            length = int(len(vr))
            duration = length / fps if fps > 0 else 0.0
            height = width = 0
            if length > 0:
                sample = vr[0]
                if hasattr(sample, "asnumpy"):
                    sample_arr = sample.asnumpy()
                elif isinstance(sample, torch.Tensor):
                    sample_arr = sample.detach().cpu().numpy()
                else:
                    sample_arr = np.asarray(sample)
                height, width = int(sample_arr.shape[0]), int(sample_arr.shape[1])
            return VideoInfo(fps=fps, num_frames=length, duration=duration, height=height, width=width)

    cv2 = _safe_import_cv2()
    if cv2 is None:
        raise RuntimeError("Neither decord nor OpenCV is available for video probing.")

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frames / fps if fps > 0 else 0.0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    finally:
        cap.release()
    return VideoInfo(fps=fps, num_frames=frames, duration=duration, height=height, width=width)


def _select_window_start(
    onsets: np.ndarray,
    duration: float,
    *,
    tries: int,
    frames: int,
    hop_seconds: float,
    min_onsets: int,
) -> float:
    if frames <= 1 or hop_seconds <= 0:
        return 0.0
    window_extent = frames * hop_seconds
    max_start = max(duration - window_extent, 0.0)
    if not math.isfinite(max_start):
        max_start = 0.0
    tries = max(1, int(tries))
    if max_start <= 0:
        candidates = np.zeros(1, dtype=np.float32)
    else:
        candidates = np.linspace(0.0, max_start, num=tries, dtype=np.float32)
    best_start = float(candidates[0])
    best_count = -1
    fallback_start = float(candidates[0])
    fallback_count = -1
    for cand in candidates:
        start = float(cand)
        end = start + window_extent
        if onsets.size > 0:
            count = int(((onsets >= start) & (onsets < end)).sum())
        else:
            count = 0
        if fallback_count < count:
            fallback_count = count
            fallback_start = start
        if count >= min_onsets:
            best_start = start
            best_count = count
            break
    if best_count >= min_onsets:
        return best_start
    return fallback_start


def _infer_canonical_hw(reg_cache_path: Path) -> Tuple[int, int]:
    if not reg_cache_path.exists():
        return DEFAULT_CANONICAL_HW
    try:
        with reg_cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return DEFAULT_CANONICAL_HW
    if not isinstance(data, dict):
        return DEFAULT_CANONICAL_HW
    for payload in data.values():
        target_hw = None
        if isinstance(payload, dict):
            target_hw = payload.get("target_hw")
        if isinstance(target_hw, Sequence) and len(target_hw) >= 2:
            try:
                h = int(target_hw[0])
                w = int(target_hw[1])
            except (TypeError, ValueError):
                continue
            if h > 0 and w > 0:
                return (h, w)
    return DEFAULT_CANONICAL_HW

def _load_registration_metadata(reg_cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not reg_cache_path.exists():
        return {}
    try:
        with reg_cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, payload in data.items():
        if isinstance(payload, dict):
            result[canonical_video_id(str(key))] = payload
    return result


def resolve_keyboard_bbox(
    video_id: str,
    frame_w: int,
    frame_h: int,
    reg_dict: Optional[Any],
) -> Tuple[int, int, int, int]:
    """Resolve a keyboard ROI for ``video_id`` within a rectified frame."""

    width = int(max(frame_w, 0))
    height = int(max(frame_h, 0))
    default_bbox = (0, 0, width, height) if width > 0 and height > 0 else (0, 0, 0, 0)

    if width <= 0 or height <= 0:
        return default_bbox
    if reg_dict is None:
        return default_bbox

    candidate: Optional[Sequence[Any]] = None

    if isinstance(reg_dict, dict):
        bbox_value = reg_dict.get("bbox")
        if isinstance(bbox_value, Sequence) and not isinstance(bbox_value, (str, bytes)):
            candidate = bbox_value
        else:
            key_sets = [
                ("x0", "y0", "x1", "y1"),
                ("min_x", "min_y", "max_x", "max_y"),
                ("minX", "minY", "maxX", "maxY"),
                ("left", "top", "right", "bottom"),
            ]
            for keys in key_sets:
                if all(k in reg_dict for k in keys):
                    candidate = [reg_dict[k] for k in keys]
                    break
    elif isinstance(reg_dict, Sequence) and not isinstance(reg_dict, (str, bytes)):
        if len(reg_dict) >= 4:
            candidate = reg_dict

    if candidate is None:
        return default_bbox

    try:
        values = [float(candidate[i]) for i in range(4)]
    except (TypeError, ValueError, IndexError):
        return default_bbox

    normalized = all(0.0 <= v <= 1.0 for v in values)
    if normalized:
        x0 = values[0] * width
        y0 = values[1] * height
        x1 = values[2] * width
        y1 = values[3] * height
    else:
        x0, y0, x1, y1 = values

    x0 = int(math.floor(x0))
    y0 = int(math.floor(y0))
    x1 = int(math.ceil(x1))
    y1 = int(math.ceil(y1))

    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        return default_bbox
    return (x0, y0, x1, y1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild PianoYT per-video A/V lag cache.")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"), help="Dataset split to process.")
    parser.add_argument("--root", default=None, help="Override PianoYT root (defaults to dataset helper resolution).")
    parser.add_argument("--max-videos", type=int, default=500, help="Maximum videos to process (default: 500).")
    parser.add_argument("--tries-per-video", type=int, default=4, help="Candidate windows per video (default: 4).")
    parser.add_argument("--frames", type=int, default=96, help="Frames per window (default: 96).")
    parser.add_argument("--fps", type=float, default=None, help="Override decode fps (default: auto from metadata).")
    parser.add_argument("--search-ms", type=int, default=500, help="Cross-correlation half-window in ms (default: 500).")
    parser.add_argument(
        "--keyboard-bbox",
        choices=("reg", "full"),
        default="reg",
        help="Keyboard ROI mode: 'reg' uses reg_refined.json when available (default), 'full' disables ROI.",
    )
    parser.add_argument(
        "--debug-bbox",
        type=int,
        default=0,
        help="Print the resolved keyboard ROI for the first N videos when > 0 (default: 0).",
    )
    parser.add_argument("--refresh", action="store_true", help="Overwrite existing cache entries.")
    parser.add_argument("--min-onsets", type=int, default=1, help="Minimum onsets per window (default: 1).")
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.25,
        help="Minimum correlation required before preserving a lag estimate (default: 0.25).",
    )
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=3.0,
        help="Per-video runtime budget for lag estimator (default: 3.0).",
    )
    return parser.parse_args()


def _load_cache_values(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        canonical_video_id(str(key)): float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def main() -> None:
    args = _parse_args()

    split = args.split
    max_videos = max(1, int(args.max_videos)) if args.max_videos is not None else None
    frames = max(2, int(args.frames))
    min_onsets = max(0, int(args.min_onsets))
    tries = max(1, int(args.tries_per_video))
    search_ms = max(1, int(args.search_ms))
    max_runtime_s = float(args.max_runtime_s) if args.max_runtime_s is not None else 3.0

    cache_path = Path("av_lags.json")
    reg_cache_path = Path("reg_refined.json")

    print(f"[setup] split={split} cache={cache_path} refresh={args.refresh}")

    root = _expand_root(args.root)
    root = root.resolve()
    print(f"[setup] dataset_root={root}")

    ids = _read_split_ids(root, split)
    if not ids:
        raise SystemExit(f"No ids found for split '{split}'.")

    metadata = _load_metadata(root)
    reg_metadata_map = _load_registration_metadata(reg_cache_path) if args.keyboard_bbox == "reg" else {}
    canonical_hw = _infer_canonical_hw(reg_cache_path)
    registration_refiner = RegistrationRefiner(
        canonical_hw,
        cache_path=reg_cache_path,
        sample_frames=max(frames, 32),
    )

    cache = AVLagCache(cache_path=cache_path)

    processed = 0
    skipped = 0
    failures = 0
    total_candidates = 0
    debug_bbox_limit = max(0, int(getattr(args, "debug_bbox", 0) or 0))
    debug_bbox_count = 0

    for video_id in ids:
        canon_id = canonical_video_id(video_id)
        if max_videos is not None and processed >= max_videos:
            break
        total_candidates += 1

        existing = cache.get(canon_id)
        if existing is not None and not args.refresh:
            skipped += 1
            continue

        video_path, midi_path = _resolve_media_paths(root, split, canon_id)
        if video_path is None or not video_path.exists():
            print(f"[warn] video missing: {canon_id}")
            failures += 1
            continue
        if midi_path is None or not midi_path.exists():
            print(f"[warn] midi missing: {canon_id}")
            failures += 1
            continue

        try:
            info = probe_video(video_path)
        except Exception as exc:
            print(f"[warn] probe failed for {canon_id}: {exc}")
            failures += 1
            continue

        fps_eff = float(args.fps) if args.fps and args.fps > 0 else float(info.fps) if info.fps > 0 else 30.0
        if fps_eff <= 0:
            fps_eff = 30.0
        stride = 1
        hop_seconds = stride / fps_eff
        clip_span_frames = (frames - 1) * stride + 1
        clip_duration = clip_span_frames / fps_eff

        labels = _read_midi_events(midi_path)
        onset_arr = labels[:, 0].detach().cpu().numpy() if labels.numel() > 0 else np.zeros(0, dtype=np.float32)
        start_sec = _select_window_start(
            onset_arr,
            info.duration,
            tries=tries,
            frames=frames,
            hop_seconds=hop_seconds,
            min_onsets=min_onsets,
        )
        start_sec = max(0.0, min(start_sec, max(0.0, info.duration - clip_duration)))
        clip_start = start_sec
        clip_end = clip_start + clip_duration
        if info.duration > 0:
            clip_end = min(info.duration, clip_end)

        preferred_start_idx = int(round(start_sec * fps_eff))
        try:
            clip, start_idx = _load_clip_with_random_start(
                path=video_path,
                frames=frames,
                stride=stride,
                channels=3,
                training=False,
                decode_fps=fps_eff,
                preferred_start_idx=preferred_start_idx,
            )
        except Exception as exc:
            print(f"[warn] decode failed for {canon_id}: {exc}")
            failures += 1
            cache.set(canon_id, 0.0)
            continue
        clip = clip.to(torch.float32)

        meta = metadata.get(canon_id)
        if meta is not None:
            clip = apply_registration_crop(clip, meta, cfg=None)

        try:
            clip = registration_refiner.transform_clip(
                clip,
                video_id=canon_id,
                video_path=video_path,
                crop_meta=meta,
                interp="bilinear",
            )
        except Exception as exc:
            print(f"[warn] registration failed for {canon_id}: {exc}")

        clip = clip.contiguous()

        start_idx = int(start_idx)
        clip_start = float(frame_to_sec(start_idx, 1.0 / fps_eff))
        clip_end = float(frame_to_sec(start_idx + clip_span_frames, 1.0 / fps_eff))
        if info.duration > 0:
            clip_end = min(info.duration, clip_end)

        bbox_arg: Optional[Sequence[int]] = None
        if args.keyboard_bbox == "reg":
            frame_h = int(clip.shape[-2])
            frame_w = int(clip.shape[-1])
            reg_entry = reg_metadata_map.get(canon_id)
            resolved_bbox = resolve_keyboard_bbox(canon_id, frame_w, frame_h, reg_entry)
            bbox_arg: Optional[Sequence[int]] = tuple(resolved_bbox)
            if debug_bbox_limit > 0 and debug_bbox_count < debug_bbox_limit:
                x0, y0, x1, y1 = resolved_bbox
                valid = 0 <= x0 < x1 <= frame_w and 0 <= y0 < y1 <= frame_h
                width_px = x1 - x0
                height_px = y1 - y0
                print(
                    f"[bbox] {canon_id} x0={x0} y0={y0} x1={x1} y1={y1} "
                    f"(w={width_px} h={height_px}) valid={valid}"
                )
                debug_bbox_count += 1

        try:
            result: Optional[AVLagResult] = compute_av_lag(
                video_id=canon_id,
                frames=clip,
                fps=fps_eff,
                events=labels,
                clip_start=clip_start,
                clip_end=clip_end,
                cache=cache,
                window_ms=float(search_ms),
                keyboard_bbox=bbox_arg,
                max_runtime_s=max_runtime_s,
                min_corr=float(args.min_corr),
            )
        except Exception as exc:
            print(f"[warn] estimator raised for {canon_id}: {exc}")
            cache.set(canon_id, 0.0)
            failures += 1
            continue

        if result is None or not result.success or not math.isfinite(result.lag_ms):
            cache.set(canon_id, 0.0)
            lag_ms = 0.0
            corr = float("nan")
            flags = set()
            failures += 1
        else:
            lag_ms = float(result.lag_ms)
            corr = float(result.corr)
            flags = set(result.flags or set())
            cache.set(canon_id, lag_ms)

        processed += 1
        flag_text = ",".join(sorted(flags)) if flags else "-"
        corr_text = f"{corr:.3f}" if math.isfinite(corr) else "nan"
        if bbox_arg is not None:
            x0, y0, x1, y1 = (int(bbox_arg[0]), int(bbox_arg[1]), int(bbox_arg[2]), int(bbox_arg[3]))
            bbox_text = f"x0={x0} y0={y0} x1={x1} y1={y1}"
        else:
            bbox_text = "full"
        print(
            f"[lag] {canon_id:<20} lag_ms={lag_ms:+6.1f} corr={corr_text} "
            f"bbox={bbox_text} flags={flag_text}"
        )

    summary = (
        f"[summary] processed={processed} skipped={skipped} failures={failures} "
        f"total_candidates={total_candidates} cache={cache_path}"
    )
    print(summary)

    cache_values = _load_cache_values(cache_path)
    if not cache_values:
        print(f"[stats] cache empty: {cache_path}")
        return

    values = np.asarray([abs(val) for val in cache_values.values()], dtype=np.float32)
    count = int(values.size)
    nonzero_frac = float(np.count_nonzero(values > 1e-3) / count) if count > 0 else 0.0
    abs_p50 = float(np.percentile(values, 50)) if count > 0 else 0.0
    abs_p95 = float(np.percentile(values, 95)) if count > 0 else 0.0
    pass_target = nonzero_frac >= 0.7 and abs_p95 <= 180.0
    print(
        "[stats] count={:d} nonzero_frac={:.3f} abs_p50={:.1f}ms abs_p95={:.1f}ms target={}".format(
            count, nonzero_frac, abs_p50, abs_p95, "PASS" if pass_target else "FAIL"
        )
    )


if __name__ == "__main__":
    main()
