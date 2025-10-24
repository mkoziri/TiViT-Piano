#!/usr/bin/env python3
"""Rebuild the PianoYT per-video audio/video lag cache.

Purpose:
    Iterate PianoYT splits, sample representative clip windows, and estimate a
    per-video lag (milliseconds) that is persisted to ``av_lags.json`` for reuse
    by datasets, calibration, and training scripts.

Key Functions/Classes:
    - probe_video: Gather fps, frame count, and dimensions for a video path.
    - decode_window: Decode a fixed-length frame window via decord/OpenCV.
    - main: CLI entry point orchestrating PianoYT iteration and cache updates.

CLI:
    --split {train,val,test} (required)
        Dataset split to process; no default.
    --root PATH (default: auto resolved by PianoYT helpers)
        Override PianoYT root directory.
    --max-videos INT (default: 500)
        Limit the number of videos processed for the run.
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
    --refresh (default: False)
        Recompute and overwrite existing cache entries.
    --min-onsets INT (default: 1)
        Minimum note onsets required within a sampled window.
    --max-runtime-s FLOAT (default: 3.0)
        Maximum runtime in seconds allocated to lag estimation per video.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.omaps_dataset import apply_registration_crop  # noqa: E402
from data.pianoyt_dataset import (  # noqa: E402
    _expand_root,
    _load_metadata,
    _read_midi_events,
    _read_split_ids,
    _resolve_media_paths,
)
from utils.av_sync import AVLagCache, AVLagResult, compute_av_lag  # noqa: E402
from utils.identifiers import canonical_video_id  # noqa: E402
from utils.registration_refinement import RegistrationRefiner, RegistrationResult  # noqa: E402


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
                sample = vr[0].asnumpy()
                height, width = int(sample.shape[0]), int(sample.shape[1])
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


def _decode_with_decord(
    path: Path,
    start_sec: float,
    frames: int,
    hop_seconds: float,
) -> torch.Tensor:
    assert decord is not None
    vr = decord.VideoReader(str(path))
    native_fps = float(vr.get_avg_fps()) or 0.0
    max_idx = max(len(vr) - 1, 0)
    times = [start_sec + k * hop_seconds for k in range(frames)]
    if native_fps <= 0:
        native_fps = 30.0
    idxs = [
        max(0, min(max_idx, int(round(t * native_fps))))
        for t in times
    ]
    batch = vr.get_batch(idxs)  # T,H,W,C
    tensor = batch.to(torch.float32) / 255.0  # type: ignore[operator]
    if tensor.shape[0] < frames:
        pad = tensor[-1:].repeat(frames - tensor.shape[0], 1, 1, 1)
        tensor = torch.cat([tensor, pad], dim=0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # T,C,H,W
    return tensor


def _decode_with_cv2(
    path: Path,
    start_sec: float,
    frames: int,
    hop_seconds: float,
    info: VideoInfo,
) -> torch.Tensor:
    cv2 = _safe_import_cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV not available for decoding and decord failed.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    native_fps = info.fps if info.fps > 0 else float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    max_idx = max(info.num_frames - 1, 0)
    times = [start_sec + k * hop_seconds for k in range(frames)]
    idxs = [
        max(0, min(max_idx, int(round(t * native_fps))))
        for t in times
    ]
    imgs: List[np.ndarray] = []
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
    finally:
        cap.release()
    if not imgs:
        raise RuntimeError(f"Failed to decode frames from {path}")
    while len(imgs) < frames:
        imgs.append(imgs[-1])
    batch = np.stack(imgs, axis=0)  # T,H,W,C
    tensor = torch.from_numpy(batch).to(torch.float32) / 255.0
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def decode_window(
    path: Path,
    info: VideoInfo,
    start_sec: float,
    frames: int,
    hop_seconds: float,
) -> torch.Tensor:
    if _HAVE_DECORD:
        try:
            return _decode_with_decord(path, start_sec, frames, hop_seconds)
        except Exception:
            pass
    return _decode_with_cv2(path, start_sec, frames, hop_seconds, info)


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


def _bbox_from_registration(result: Optional[RegistrationResult]) -> Optional[Tuple[int, int, int, int]]:
    if result is None:
        return None
    target_h, target_w = result.target_hw
    if target_h <= 0 or target_w <= 0:
        return None
    keyboard_height = float(result.keyboard_height)
    slope = float(result.baseline_slope)
    intercept = float(result.baseline_intercept)
    if not (math.isfinite(keyboard_height) and keyboard_height > 0):
        return None
    xs = np.asarray([0.0, float(target_w - 1)], dtype=np.float32)
    ys = slope * xs + intercept
    y_bottom = float(np.max(ys))
    y_top = float(np.min(ys) - keyboard_height)
    y0 = max(0, int(math.floor(y_top)))
    y1 = min(target_h, int(math.ceil(y_bottom)))
    if y1 <= y0:
        y0 = max(0, min(target_h - 1, int(math.floor(y_bottom - keyboard_height))))
        y1 = min(target_h, max(y0 + 1, int(math.ceil(y_bottom))))
    x0 = 0
    x1 = target_w
    return (y0, y1, x0, x1)


def _load_reg_bboxes(reg_cache_path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    if not reg_cache_path.exists():
        return {}
    try:
        with reg_cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Tuple[int, int, int, int]] = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        target_hw = payload.get("target_hw")
        keyboard_height = payload.get("keyboard_height")
        slope = payload.get("baseline_slope")
        intercept = payload.get("baseline_intercept")
        if not (
            isinstance(target_hw, Sequence)
            and len(target_hw) >= 2
            and isinstance(keyboard_height, (int, float))
            and isinstance(slope, (int, float))
            and isinstance(intercept, (int, float))
        ):
            continue
        try:
            target_h = int(target_hw[0])
            target_w = int(target_hw[1])
        except (TypeError, ValueError):
            continue
        dummy = RegistrationResult(
            homography=np.eye(3, dtype=np.float32),
            source_hw=(target_h, target_w),
            target_hw=(target_h, target_w),
            err_before=0.0,
            err_after=0.0,
            err_white_edges=0.0,
            err_black_gaps=0.0,
            frames=0,
            status="cache",
            baseline_slope=float(slope),
            baseline_intercept=float(intercept),
            keyboard_height=float(keyboard_height),
            timestamp=time.time(),
            x_warp_ctrl=None,
            grid=None,
        )
        bbox = _bbox_from_registration(dummy)
        if bbox is None:
            continue
        result[canonical_video_id(str(key))] = bbox
    return result


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
    parser.add_argument("--refresh", action="store_true", help="Overwrite existing cache entries.")
    parser.add_argument("--min-onsets", type=int, default=1, help="Minimum onsets per window (default: 1).")
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
    reg_bbox_map = _load_reg_bboxes(reg_cache_path) if args.keyboard_bbox == "reg" else {}
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
        hop_seconds = 1.0 / fps_eff if fps_eff > 0 else 1.0 / 30.0
        clip_duration = frames * hop_seconds

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

        try:
            clip = decode_window(video_path, info, clip_start, frames, hop_seconds)
        except Exception as exc:
            print(f"[warn] decode failed for {canon_id}: {exc}")
            failures += 1
            cache.set(canon_id, 0.0)
            continue

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
            reg_result = registration_refiner.refine(
                video_id=canon_id,
                video_path=video_path,
                crop_meta=meta,
            )
        except Exception as exc:
            print(f"[warn] registration failed for {canon_id}: {exc}")
            reg_result = None

        keyboard_bbox = None
        if args.keyboard_bbox == "reg":
            keyboard_bbox = reg_bbox_map.get(canon_id)
            if keyboard_bbox is None and reg_result is not None:
                keyboard_bbox = _bbox_from_registration(reg_result)
        if keyboard_bbox is not None and len(keyboard_bbox) == 4:
            bbox_arg: Optional[Sequence[int]] = tuple(int(v) for v in keyboard_bbox)
        else:
            bbox_arg = None

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
        print(
            f"[lag] {canon_id:<20} lag_ms={lag_ms:+6.1f} corr={corr_text} "
            f"bbox={'y0y1x0x1' if bbox_arg else 'full'} flags={flag_text}"
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
