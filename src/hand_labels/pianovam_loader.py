"""PianoVAM Handskeleton loader.

This module only parses PianoVAM Handskeleton JSON files and aligns left/right
hand landmarks to a TiViT clip's frame grid.  It does not apply canonical
keyboard transforms or build supervision targets; callers can feed the returned
landmarks into downstream transforms or heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

_HandArray = torch.Tensor  # (21, 3) -> x, y, confidence
_HandMask = torch.Tensor   # bool (21,)


@dataclass
class AlignedHandLandmarks:
    """
    Container for clip-aligned hand landmarks.

    Attributes:
        landmarks: Tensor of shape (T, 2, 21, 3) with ``(x, y, confidence)`` per
            point. Order: time, hand (0=left, 1=right), keypoint, coord.
        mask: Bool tensor of shape (T, 2, 21) indicating which points are valid
            after alignment and confidence filtering.
        frame_times: Tensor of shape (T,) giving the clip frame times in
            seconds relative to the source video.
        source_fps: Native Handskeleton frame rate if available; otherwise the
            caller-provided fallback.
        clip_start_sec: Clip start time in seconds (decode grid).
        metadata: Diagnostic fields (counts, tolerances) to aid logging.
    """

    landmarks: torch.Tensor
    mask: torch.Tensor
    frame_times: torch.Tensor
    source_fps: float
    clip_start_sec: float
    metadata: Dict[str, Any]


def _coerce_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor(out)):
        return None
    return out


def _infer_source_fps(payload: Any, fallback: float) -> float:
    if isinstance(payload, Mapping):
        for key in ("fps", "frame_rate", "frameRate", "source_fps"):
            fps_val = _coerce_float(payload.get(key))
            if fps_val is not None and fps_val > 0:
                return fps_val
    return float(fallback)


def _extract_frames_container(raw: Any) -> Optional[Iterable[Any]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        for key in ("frames", "data", "hands", "results"):
            maybe = raw.get(key)
            if isinstance(maybe, Iterable):
                return maybe
        # Some dumps use a dict of {frame_idx: {...}}
        values = list(raw.values())
        if values and all(isinstance(v, Mapping) for v in values):
            return values
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return raw
    return None


def _point_from_mapping(entry: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], float]:
    x = entry.get("x", entry.get("X"))
    y = entry.get("y", entry.get("Y"))
    conf = (
        entry.get("conf")
        or entry.get("confidence")
        or entry.get("score")
        or entry.get("visibility")
        or entry.get("prob")
        or entry.get("probability")
    )
    x_f = _coerce_float(x)
    y_f = _coerce_float(y)
    conf_f = _coerce_float(conf)
    if conf_f is None:
        conf_f = 1.0
    return x_f, y_f, float(conf_f)


def _parse_hand_points(obj: Any) -> Tuple[_HandArray, _HandMask]:
    """
    Normalise a single hand's keypoints into (21,3) with a validity mask.
    Accepts:
      - list of dicts/tuples
      - dict with x/y arrays
      - flat numeric list
    """
    coords = torch.zeros((21, 3), dtype=torch.float32)
    mask = torch.zeros((21,), dtype=torch.bool)

    if obj is None:
        return coords, mask

    # dict-of-arrays form: {"x": [...], "y": [...], "confidence": [...]}
    if isinstance(obj, Mapping):
        xs = obj.get("x") or obj.get("X") or obj.get("xs")
        ys = obj.get("y") or obj.get("Y") or obj.get("ys")
        cs = obj.get("confidence") or obj.get("conf") or obj.get("scores") or obj.get("visibility")
        if isinstance(xs, Sequence) and isinstance(ys, Sequence):
            for idx in range(min(len(xs), len(ys), 21)):
                x_f = _coerce_float(xs[idx])
                y_f = _coerce_float(ys[idx])
                c_f = _coerce_float(cs[idx]) if isinstance(cs, Sequence) and idx < len(cs) else None
                if x_f is None or y_f is None:
                    continue
                coords[idx, 0] = x_f
                coords[idx, 1] = y_f
                coords[idx, 2] = float(c_f) if c_f is not None else 1.0
                mask[idx] = True
            return coords, mask

    # list/tuple of dicts or tuples
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if obj and all(isinstance(v, Mapping) for v in obj):
            for idx, point in enumerate(obj[:21]):
                x_f, y_f, c_f = _point_from_mapping(point)
                if x_f is None or y_f is None:
                    continue
                coords[idx, 0] = x_f
                coords[idx, 1] = y_f
                coords[idx, 2] = c_f
                mask[idx] = True
            return coords, mask
        if obj and all(not isinstance(v, Mapping) for v in obj):
            # Flat numeric sequence
            flat_vals = [v for v in obj if _coerce_float(v) is not None]
            if len(flat_vals) >= 42:  # 21 * (x,y)
                for idx in range(21):
                    base = idx * 2
                    x_f = _coerce_float(flat_vals[base])
                    y_f = _coerce_float(flat_vals[base + 1])
                    if x_f is None or y_f is None:
                        continue
                    coords[idx, 0] = x_f
                    coords[idx, 1] = y_f
                    coords[idx, 2] = 1.0
                    mask[idx] = True
                return coords, mask

    return coords, mask


def _parse_time_sec(entry: Mapping[str, Any], source_fps: float) -> Optional[float]:
    for key in ("timestamp_sec", "time_sec", "ts_sec", "time", "timestamp"):
        t_val = _coerce_float(entry.get(key))
        if t_val is not None:
            return t_val
    for key in ("timestamp_ms", "time_ms", "ts_ms"):
        t_val = _coerce_float(entry.get(key))
        if t_val is not None:
            return t_val / 1000.0
    for key in ("frame", "frame_index", "index", "idx"):
        idx_val = _coerce_float(entry.get(key))
        if idx_val is not None and source_fps > 0:
            return idx_val / source_fps
    return None


def _parse_frame_entry(entry: Any, source_fps: float) -> Optional[Tuple[float, _HandArray, _HandMask, _HandArray, _HandMask]]:
    if not isinstance(entry, Mapping):
        return None
    time_sec = _parse_time_sec(entry, source_fps)
    if time_sec is None:
        return None

    left_data = None
    right_data = None

    for key in ("left_hand", "hand_left", "left", "L", "lhand"):
        if key in entry:
            left_data = entry[key]
            break
    for key in ("right_hand", "hand_right", "right", "R", "rhand"):
        if key in entry:
            right_data = entry[key]
            break

    left_pts, left_mask = _parse_hand_points(left_data)
    right_pts, right_mask = _parse_hand_points(right_data)
    return float(time_sec), left_pts, left_mask, right_pts, right_mask


def _empty_response(
    *,
    frames: int,
    clip_start_sec: float,
    stride: int,
    fps: float,
    source_fps: float,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AlignedHandLandmarks:
    hop_seconds = float(stride) / max(float(fps), 1e-6)
    frame_times = torch.arange(frames, dtype=torch.float32) * hop_seconds + float(clip_start_sec)
    meta = {
        "reason": reason,
        "source_frames": 0,
        "filled_frames": 0,
    }
    if metadata:
        meta.update(metadata)
    return AlignedHandLandmarks(
        landmarks=torch.zeros((frames, 2, 21, 3), dtype=torch.float32),
        mask=torch.zeros((frames, 2, 21), dtype=torch.bool),
        frame_times=frame_times,
        source_fps=float(source_fps),
        clip_start_sec=float(clip_start_sec),
        metadata=meta,
    )


def _choose_nearest_index(source_times: torch.Tensor, target_time: float) -> Tuple[Optional[int], float]:
    if source_times.numel() == 0:
        return None, float("inf")
    diffs = (source_times - target_time).abs()
    idx = int(diffs.argmin().item())
    gap = float(diffs[idx].item())
    return idx, gap


def load_pianovam_hand_landmarks(
    json_path: Union[str, Path],
    *,
    clip_start_sec: float,
    frames: int,
    stride: int,
    decode_fps: float,
    min_confidence: float = 0.0,
    time_tolerance: Optional[float] = None,
) -> AlignedHandLandmarks:
    """
    Load and time-align PianoVAM Handskeleton landmarks to a clip grid.

    Args:
        json_path: Path to ``Handskeleton/{record_id}.json``.
        clip_start_sec: Clip start time (seconds) on the decode_fps grid.
        frames: Number of frames in the clip (T).
        stride: Temporal stride used when sampling the clip.
        decode_fps: Decode frame rate used for the clip grid.
        min_confidence: Minimum per-point confidence to treat a landmark as
            valid; points below are masked out but coordinates are preserved.
        time_tolerance: Maximum allowed |source_time - clip_time| in seconds for
            nearest-neighbour alignment. Defaults to one hop (stride/fps).

    Returns:
        AlignedHandLandmarks with shapes (T, 2, 21, 3) and validity mask.
    """
    path = Path(json_path)
    hop_seconds = float(stride) / max(float(decode_fps), 1e-6)
    tol = hop_seconds if time_tolerance is None else float(time_tolerance)
    tol = max(tol, 0.0)

    if not path.exists():
        return _empty_response(
            frames=frames,
            clip_start_sec=clip_start_sec,
            stride=stride,
            fps=decode_fps,
            source_fps=decode_fps,
            reason="missing_file",
            metadata={"path": str(path)},
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return _empty_response(
            frames=frames,
            clip_start_sec=clip_start_sec,
            stride=stride,
            fps=decode_fps,
            source_fps=decode_fps,
            reason="json_parse_error",
            metadata={"path": str(path)},
        )

    source_fps = _infer_source_fps(raw, decode_fps)
    container = _extract_frames_container(raw)
    if container is None:
        return _empty_response(
            frames=frames,
            clip_start_sec=clip_start_sec,
            stride=stride,
            fps=decode_fps,
            source_fps=source_fps,
            reason="no_frames_found",
            metadata={"path": str(path)},
        )

    parsed_frames: List[Tuple[float, _HandArray, _HandMask, _HandArray, _HandMask]] = []
    for entry in container:
        parsed = _parse_frame_entry(entry, source_fps)
        if parsed is not None:
            parsed_frames.append(parsed)

    if not parsed_frames:
        return _empty_response(
            frames=frames,
            clip_start_sec=clip_start_sec,
            stride=stride,
            fps=decode_fps,
            source_fps=source_fps,
            reason="no_valid_frames",
            metadata={"path": str(path)},
        )

    parsed_frames.sort(key=lambda item: item[0])
    source_times = torch.tensor([p[0] for p in parsed_frames], dtype=torch.float32)

    out_landmarks = torch.zeros((frames, 2, 21, 3), dtype=torch.float32)
    out_mask = torch.zeros((frames, 2, 21), dtype=torch.bool)
    frame_times = torch.arange(frames, dtype=torch.float32) * hop_seconds + float(clip_start_sec)

    filled_frames = 0
    for t_idx, target_time in enumerate(frame_times.tolist()):
        src_idx, gap = _choose_nearest_index(source_times, target_time)
        if src_idx is None or gap > tol:
            continue
        _, left_pts, left_mask, right_pts, right_mask = parsed_frames[src_idx]

        for hand_idx, (pts, mask) in enumerate(((left_pts, left_mask), (right_pts, right_mask))):
            if mask is None or pts is None:
                continue
            valid = mask.clone()
            if min_confidence > 0.0:
                valid &= pts[:, 2] >= float(min_confidence)
            if not valid.any():
                continue
            out_landmarks[t_idx, hand_idx] = pts
            out_mask[t_idx, hand_idx] = valid
        if out_mask[t_idx].any():
            filled_frames += 1

    metadata = {
        "path": str(path),
        "source_frames": len(parsed_frames),
        "filled_frames": filled_frames,
        "time_tolerance": tol,
        "min_confidence": float(min_confidence),
    }
    return AlignedHandLandmarks(
        landmarks=out_landmarks,
        mask=out_mask,
        frame_times=frame_times,
        source_fps=float(source_fps),
        clip_start_sec=float(clip_start_sec),
        metadata=metadata,
    )


__all__ = ["AlignedHandLandmarks", "load_pianovam_hand_landmarks"]
