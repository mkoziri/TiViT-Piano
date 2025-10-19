"""
Registration refinement utilities for per-video keyboard rectification.

This module estimates a refined homography that maps cropped piano keyboard
frames to the canonical resolution (e.g. 145x800).  The refinement is driven by
edge detections aggregated across a handful of frames sampled throughout each
video.  Results are cached on disk so repeated runs avoid recomputation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None  # type: ignore

try:
    import decord  # type: ignore

    _HAVE_DECORD = True
except Exception:  # pragma: no cover - decord optional
    decord = None  # type: ignore
    _HAVE_DECORD = False


def _extract_crop_values(
    meta: Optional[Sequence[float] | Dict[str, Any]],
) -> Optional[Tuple[float, float, float, float]]:
    if meta is None:
        return None
    if isinstance(meta, dict):
        keys = ("min_y", "max_y", "min_x", "max_x")
        vals: Optional[List[float]] = None
        for candidate in (keys, ("top", "bottom", "left", "right"), ("y0", "y1", "x0", "x1")):
            if all(k in meta for k in candidate):
                vals = [meta[k] for k in candidate]  # type: ignore[index]
                break
        if vals is None and "crop" in meta:
            crop_val = meta["crop"]
            if isinstance(crop_val, (list, tuple)) and len(crop_val) >= 4:
                vals = list(crop_val[:4])
        if vals is None:
            return None
    elif isinstance(meta, Sequence):
        if len(meta) < 4:
            return None
        vals = list(meta[:4])
    else:
        return None

    try:
        min_y, max_y, min_x, max_x = vals[:4]
        return (
            float(min_y),
            float(max_y),
            float(min_x),
            float(max_x),
        )
    except (TypeError, ValueError):
        return None


def _apply_crop_np(frame: np.ndarray, meta: Optional[Sequence[float] | Dict[str, Any]]) -> np.ndarray:
    coords = _extract_crop_values(meta)
    if coords is None:
        return frame

    min_y, max_y, min_x, max_x = coords
    h, w = frame.shape[:2]
    is_normalised = all(0.0 <= v <= 1.0 for v in (min_y, max_y, min_x, max_x))
    if is_normalised:
        min_y *= h
        max_y *= h
        min_x *= w
        max_x *= w

    y0 = int(math.floor(min_y))
    y1 = int(math.ceil(max_y))
    x0 = int(math.floor(min_x))
    x1 = int(math.ceil(max_x))

    y0 = max(0, min(y0, h - 1))
    x0 = max(0, min(x0, w - 1))
    y1 = max(y0 + 1, min(y1, h))
    x1 = max(x0 + 1, min(x1, w))
    return frame[y0:y1, x0:x1]


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if not arr:
        return None
    arr.sort()
    mid = len(arr) // 2
    if len(arr) % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def _detect_baseline(gray: np.ndarray) -> Optional[Tuple[float, float]]:
    if cv2 is None:
        return None
    edges = cv2.Canny(gray, 80, 160, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1.0, np.pi / 180.0, threshold=max(80, int(gray.shape[1] * 0.4)))
    if lines is None:
        return None

    best: Optional[Tuple[float, float]] = None
    best_votes = -1
    for candidate in lines[:, 0]:
        rho, theta = float(candidate[0]), float(candidate[1])
        # Horizontal line corresponds to theta approx pi/2
        if abs(theta - (math.pi / 2.0)) > math.radians(25.0):
            continue
        votes = 1
        if best is None or votes > best_votes:
            best = (rho, theta)
            best_votes = votes

    if best is None:
        return None

    rho, theta = best
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    if abs(sin_t) < 1e-6:
        return None

    slope = -cos_t / sin_t
    intercept = rho / sin_t
    return slope, intercept


def _gaussian_smooth_1d(data: np.ndarray, kernel: int = 9) -> np.ndarray:
    if cv2 is None or data.size == 0:
        return data.astype(np.float32)
    kernel = max(3, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = cv2.GaussianBlur(data.reshape(1, -1), (kernel, 1), 0, borderType=cv2.BORDER_REFLECT)
    return smoothed.reshape(-1).astype(np.float32)


def _find_peak(profile: np.ndarray, center: float, window: int) -> float:
    """Return the index of the strongest response around ``center``."""
    length = profile.shape[0]
    if length == 0:
        return float(center)
    left = int(max(0, math.floor(center - window)))
    right = int(min(length, math.ceil(center + window)))
    if right <= left:
        return float(center)
    segment = profile[left:right]
    if segment.size == 0:
        return float(center)
    idx = int(np.argmax(segment))
    return float(left + idx)


def _snap_edges_ransac(edges: np.ndarray, expected_spacing: float) -> np.ndarray:
    """Project detected edges onto a regular grid while limiting local drift."""
    if edges.size < 4:
        return edges
    indices = np.arange(edges.size, dtype=np.float32)
    spacing = max(float(expected_spacing), 1.0)
    inlier_thresh = max(spacing * 0.35, 1.5)
    rng = np.random.default_rng(12345)

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_error = float("inf")

    for _ in range(120):
        sample = rng.choice(edges.size, size=2, replace=False)
        i0, i1 = int(sample[0]), int(sample[1])
        denom = float(indices[i1] - indices[i0])
        if abs(denom) < 1e-6:
            continue
        slope = float((edges[i1] - edges[i0]) / denom)
        intercept = float(edges[i0] - slope * indices[i0])
        fitted = slope * indices + intercept
        residuals = np.abs(fitted - edges)
        inliers = residuals <= inlier_thresh
        count = int(np.count_nonzero(inliers))
        if count < 4:
            continue
        error = float(residuals[inliers].mean()) if count > 0 else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_inliers = inliers
            best_count = count
            best_error = error

    if best_inliers is None:
        slope, intercept = np.polyfit(indices, edges, deg=1)
    else:
        slope, intercept = np.polyfit(indices[best_inliers], edges[best_inliers], deg=1)

    fitted = slope * indices + intercept
    residuals = edges - fitted
    clip_val = max(spacing * 0.6, 3.0)
    residuals = np.clip(residuals, -clip_val, clip_val)
    refined = fitted + residuals
    refined = np.maximum.accumulate(refined)
    return refined.astype(np.float32)


def _compute_keyboard_edges(
    grad_profile: np.ndarray,
    width: int,
    num_white_keys: int = 52,
) -> Tuple[np.ndarray, float, float]:
    """Estimate white-key edge locations across the keyboard span."""
    if grad_profile.size == 0:
        indices = np.linspace(0.0, float(width - 1), num_white_keys + 1, dtype=np.float32)
        return indices, float(indices[0]), float(indices[-1])

    profile = _gaussian_smooth_1d(grad_profile.astype(np.float32), kernel=9)
    if profile.max(initial=0.0) <= 1e-6:
        indices = np.linspace(0.0, float(width - 1), num_white_keys + 1, dtype=np.float32)
        return indices, float(indices[0]), float(indices[-1])

    threshold = float(profile.max() * 0.2)
    active = np.where(profile >= threshold)[0]
    if active.size == 0:
        x_left = 0.0
        x_right = float(width - 1)
    else:
        x_left = float(active[0])
        x_right = float(active[-1])
        if (x_right - x_left) < width * 0.6:
            x_left = 0.0
            x_right = float(width - 1)

    span = max(x_right - x_left, 1.0)
    step = span / float(num_white_keys)
    window = max(5.0, step * 0.35)
    edges = []
    for idx in range(num_white_keys + 1):
        center = x_left + step * idx
        peak = _find_peak(profile, center, int(round(window)))
        edges.append(peak)
    edge_arr = np.array(edges, dtype=np.float32)
    edge_arr = np.clip(edge_arr, 0.0, float(width - 1))
    edge_arr = _snap_edges_ransac(edge_arr, step)
    edge_arr = np.clip(edge_arr, 0.0, float(width - 1))
    return edge_arr, float(edge_arr[0]), float(edge_arr[-1])


def _aggregate_gradients(frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = None
    grad_y = None
    for frame in frames:
        gray = frame
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY) if cv2 is not None else np.mean(gray, axis=-1)
        if cv2 is not None:
            gray = cv2.medianBlur(gray.astype(np.uint8), 5)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        else:
            grayf = gray.astype(np.float32)
            gx = np.abs(np.gradient(grayf, axis=1))
            gy = np.abs(np.gradient(grayf, axis=0))
        gx = np.abs(gx).sum(axis=0)
        gy = np.abs(gy).sum(axis=1)
        grad_x = gx if grad_x is None else grad_x + gx
        grad_y = gy if grad_y is None else grad_y + gy
    if grad_x is None:
        grad_x = np.zeros(frames[0].shape[1], dtype=np.float32)
    if grad_y is None:
        grad_y = np.zeros(frames[0].shape[0], dtype=np.float32)
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def _estimate_keyboard_height(
    grad_y_profile: np.ndarray,
    baseline_row: int,
    height: int,
) -> int:
    if baseline_row <= 0 or grad_y_profile.size == 0:
        return max(int(height * 0.65), 1)
    smoothed = _gaussian_smooth_1d(grad_y_profile.astype(np.float32), kernel=9)
    cutoff = max(0, baseline_row - 3)
    if cutoff <= 0:
        return max(int(height * 0.65), 1)
    upper = smoothed[:cutoff]
    if upper.size == 0:
        return max(int(height * 0.65), 1)
    top_idx = int(np.argmax(upper))
    est = baseline_row - top_idx
    low = max(int(height * 0.35), 1)
    high = max(int(height * 0.9), low + 1)
    return int(min(max(est, low), high))


def _homography_to_grid(
    homography: np.ndarray,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    h_dst, w_dst = target_hw
    h_src, w_src = source_hw
    ys = np.linspace(0.0, float(h_dst - 1), h_dst, dtype=np.float32)
    xs = np.linspace(0.0, float(w_dst - 1), w_dst, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    ones = np.ones_like(xv)
    coords = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T
    mapped = homography @ coords
    denom = mapped[2] + 1e-6
    x_src = (mapped[0] / denom).reshape(h_dst, w_dst)
    y_src = (mapped[1] / denom).reshape(h_dst, w_dst)
    w_src_f = max(float(w_src), 1.0)
    h_src_f = max(float(h_src), 1.0)
    x_norm = ((x_src + 0.5) / w_src_f) * 2.0 - 1.0
    y_norm = ((y_src + 0.5) / h_src_f) * 2.0 - 1.0
    grid = np.stack(
        [
            np.clip(x_norm, -2.0, 2.0),
            np.clip(y_norm, -2.0, 2.0),
        ],
        axis=-1,
    )
    return torch.from_numpy(grid.astype(np.float32))


def _homography_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Flatten a homography into an 8D vector with bottom-right entry normalised."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"expected homography shape (3, 3); got {arr.shape}")
    scale = float(arr[2, 2]) if abs(float(arr[2, 2])) > 1e-6 else 1.0
    arr = arr / scale
    return np.array(
        [
            float(arr[0, 0]),
            float(arr[0, 1]),
            float(arr[0, 2]),
            float(arr[1, 0]),
            float(arr[1, 1]),
            float(arr[1, 2]),
            float(arr[2, 0]),
            float(arr[2, 1]),
        ],
        dtype=np.float32,
    )


def _solve_regularized_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    base_h: np.ndarray,
    *,
    reg_lambda: float = 0.05,
) -> Optional[np.ndarray]:
    """Solve a least-squares homography with Tikhonov regularisation."""
    if src_pts.shape[0] < 4:
        return None

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    n = src.shape[0]
    A = np.zeros((2 * n, 8), dtype=np.float32)
    b = np.zeros((2 * n,), dtype=np.float32)

    for i in range(n):
        x, y = float(src[i, 0]), float(src[i, 1])
        u, v = float(dst[i, 0]), float(dst[i, 1])
        row = 2 * i
        A[row, 0:3] = [x, y, 1.0]
        A[row, 6:8] = [-u * x, -u * y]
        b[row] = u
        A[row + 1, 3:6] = [x, y, 1.0]
        A[row + 1, 6:8] = [-v * x, -v * y]
        b[row + 1] = v

    if reg_lambda > 0.0:
        lam = float(reg_lambda)
        sqrt_lam = math.sqrt(lam)
        A_reg = sqrt_lam * np.eye(8, dtype=np.float32)
        b_reg = sqrt_lam * _homography_to_vector(base_h)
        A = np.vstack([A, A_reg])
        b = np.concatenate([b, b_reg])

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    H = np.array(
        [
            [sol[0], sol[1], sol[2]],
            [sol[3], sol[4], sol[5]],
            [sol[6], sol[7], 1.0],
        ],
        dtype=np.float32,
    )
    return H


def _invert_homography(matrix: Any) -> np.ndarray:
    """Return the inverse of a 3x3 homography as float32 array."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"expected homography shape (3, 3); got {arr.shape}")
    return np.linalg.inv(arr)


@dataclass
class RegistrationResult:
    homography: np.ndarray
    source_hw: Tuple[int, int]
    target_hw: Tuple[int, int]
    err_before: float
    err_after: float
    frames: int
    status: str
    baseline_slope: float
    baseline_intercept: float
    keyboard_height: float
    timestamp: float
    grid: Optional[torch.Tensor] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "homography": [float(x) for x in self.homography.reshape(-1)],
            "source_hw": list(self.source_hw),
            "target_hw": list(self.target_hw),
            "err_before": float(self.err_before),
            "err_after": float(self.err_after),
            "frames": int(self.frames),
            "status": str(self.status),
            "baseline_slope": float(self.baseline_slope),
            "baseline_intercept": float(self.baseline_intercept),
            "keyboard_height": float(self.keyboard_height),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "RegistrationResult":
        homography = np.asarray(payload.get("homography", []), dtype=np.float32).reshape(3, 3)
        source_hw = tuple(int(v) for v in payload.get("source_hw", (0, 0)))  # type: ignore[arg-type]
        target_hw = tuple(int(v) for v in payload.get("target_hw", (0, 0)))  # type: ignore[arg-type]
        return cls(
            homography=homography,
            source_hw=source_hw,  # type: ignore[arg-type]
            target_hw=target_hw,  # type: ignore[arg-type]
            err_before=float(payload.get("err_before", 0.0)),
            err_after=float(payload.get("err_after", 0.0)),
            frames=int(payload.get("frames", 0)),
            status=str(payload.get("status", "unknown")),
            baseline_slope=float(payload.get("baseline_slope", 0.0)),
            baseline_intercept=float(payload.get("baseline_intercept", 0.0)),
            keyboard_height=float(payload.get("keyboard_height", 0.0)),
            timestamp=float(payload.get("timestamp", time.time())),
            grid=None,
        )


@contextmanager
def _file_lock(path: Path):
    lock_handle = None
    try:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(lock_path, "w")
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        if lock_handle:
            try:
                import fcntl  # type: ignore

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_handle.close()


class RegistrationRefiner:
    """Refines per-video keyboard registration using homography estimation."""

    def __init__(
        self,
        canonical_hw: Sequence[int],
        *,
        cache_path: Optional[Path] = None,
        sample_frames: int = 30,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if len(canonical_hw) < 2:
            raise ValueError("canonical_hw must provide (H, W)")
        self.canonical_hw: Tuple[int, int] = (int(canonical_hw[0]), int(canonical_hw[1]))
        self.sample_frames = int(max(sample_frames, 8))
        self.cache_path = cache_path or Path("reg_refined.json")
        self.logger = logger or LOGGER
        self._cache: Dict[str, RegistrationResult] = {}
        self._load_cache()

    # ------------------------------------------------------------------ I/O --

    def _load_cache(self) -> None:
        path = self.cache_path
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.logger.warning("reg_refined: unable to read cache %s (%s)", path, exc)
            return

        if not isinstance(data, dict):
            return
        for key, payload in data.items():
            if not isinstance(payload, dict):
                continue
            try:
                result = RegistrationResult.from_json(payload)
            except Exception:
                continue
            if tuple(result.target_hw) != self.canonical_hw:
                continue
            self._cache[str(key)] = result

    def _persist_cache(self) -> None:
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: result.to_json() for key, result in self._cache.items()}
        tmp_path = path.with_suffix(".tmp")
        with _file_lock(path):
            try:
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                os.replace(tmp_path, path)
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                self.logger.warning("reg_refined: failed to persist cache %s (%s)", path, exc)

    # --------------------------------------------------------------- Sampling --

    def _sample_video_frames(
        self,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
        *,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        target_frames = int(max_frames or self.sample_frames)
        if target_frames <= 0:
            return []

        frames: List[np.ndarray] = []

        if _HAVE_DECORD:
            assert decord is not None
            try:
                vr = decord.VideoReader(str(video_path))
                total = len(vr)
                if total <= 0:
                    return []
                step = max(total // target_frames, 1)
                indices = list(range(0, total, step))
                if len(indices) > target_frames:
                    indices = indices[:target_frames]
                for idx in indices:
                    img = vr[idx].asnumpy()  # type: ignore[attr-defined]
                    if img is None:
                        continue
                    frame = _apply_crop_np(img, crop_meta)
                    frames.append(frame)
                if frames:
                    return frames
            except Exception as exc:  # pragma: no cover - fallback to cv2
                self.logger.debug("reg_refined: decord failed for %s (%s)", video_path, exc)

        if cv2 is None:
            return frames

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return frames
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0 or not math.isfinite(total):
                total = target_frames * 4
            indices = np.linspace(0, max(total - 1, 0), num=target_frames, dtype=int)
            seen = set()
            for idx in indices:
                idx = int(idx)
                if idx in seen:
                    continue
                seen.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(_apply_crop_np(frame_rgb, crop_meta))
                if len(frames) >= target_frames:
                    break
        finally:
            cap.release()
        return frames

    # ------------------------------------------------------------ Computation --

    def _compute_refinement(
        self,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
    ) -> RegistrationResult:
        frames = self._sample_video_frames(video_path, crop_meta)
        canonical = self.canonical_hw
        if not frames:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=canonical,
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                frames=0,
                status="fallback_no_frames",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
            )

        heights = [frame.shape[0] for frame in frames]
        widths = [frame.shape[1] for frame in frames]
        height = int(np.median(heights))
        width = int(np.median(widths))
        if height <= 0 or width <= 0:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=canonical,
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                frames=len(frames),
                status="fallback_bad_dims",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
            )

        grayscale_frames: List[np.ndarray] = []
        slopes: List[float] = []
        intercepts: List[float] = []

        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if cv2 is not None else frame
            if frame.ndim == 3:
                if cv2 is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = np.mean(np.asarray(frame, dtype=np.float32), axis=-1)
            else:
                gray = frame
            gray = gray.astype(np.uint8)
            if cv2 is not None:
                gray = cv2.medianBlur(gray, 5)
            baseline = _detect_baseline(gray)
            if baseline:
                slopes.append(baseline[0])
                intercepts.append(baseline[1])
            grayscale_frames.append(gray)

        slope = _median_or_none(slopes)
        intercept = _median_or_none(intercepts)
        if slope is None or intercept is None:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=(height, width),
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                frames=len(frames),
                status="fallback_no_baseline",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
            )

        grad_x_profile, grad_y_profile = _aggregate_gradients(grayscale_frames)

        edges, x_left, x_right = _compute_keyboard_edges(grad_x_profile, width)
        canon_edges = np.linspace(0.0, float(canonical[1] - 1), edges.shape[0], dtype=np.float32)

        scale_x = (canonical[1] - 1) / max(width - 1, 1)
        scale_y = (canonical[0] - 1) / max(height - 1, 1)
        x_before = edges * scale_x
        err_before = float(np.mean(np.abs(x_before - canon_edges)))

        xs = edges
        ys = np.clip(slope * xs + intercept, 0.0, height - 1.0)
        baseline_row = int(np.clip(np.median(ys), 0.0, height - 1.0))
        keyboard_height = _estimate_keyboard_height(grad_y_profile, baseline_row, height)
        keyboard_height = max(float(keyboard_height), 1.0)
        target_baseline_y = baseline_row * scale_y
        canonical_keyboard_height = keyboard_height * scale_y
        target_top_y = max(0.0, target_baseline_y - canonical_keyboard_height)
        source_top = np.clip(ys - keyboard_height, 0.0, height - 1.0)

        src_pts: List[List[float]] = []
        dst_pts: List[List[float]] = []
        for x_val, y_base, y_top, canon_x in zip(xs, ys, source_top, canon_edges):
            src_pts.append([float(x_val), float(y_top)])
            dst_pts.append([float(canon_x), float(target_top_y)])
            src_pts.append([float(x_val), float(y_base)])
            dst_pts.append([float(canon_x), float(target_baseline_y)])

        src_pts.extend(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [0.0, float(height - 1)],
                [float(width - 1), float(height - 1)],
            ]
        )
        dst_pts.extend(
            [
                [0.0, 0.0],
                [float(canonical[1] - 1), 0.0],
                [0.0, float(canonical[0] - 1)],
                [float(canonical[1] - 1), float(canonical[0] - 1)],
            ]
        )

        src_arr = np.asarray(src_pts, dtype=np.float32)
        dst_arr = np.asarray(dst_pts, dtype=np.float32)
        base_h = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        status = "ok"
        err_after = err_before
        H: Optional[np.ndarray]

        if src_arr.shape[0] < 4:
            H = base_h
            status = "fallback_points"
        else:
            inlier_mask: Optional[np.ndarray] = None
            if cv2 is not None:
                _, mask = cv2.findHomography(
                    src_arr,
                    dst_arr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2500,
                    confidence=0.997,
                )
                if mask is not None and mask.size == src_arr.shape[0]:
                    mask = mask.ravel().astype(bool)
                    if int(mask.sum()) >= 4:
                        inlier_mask = mask

            src_fit = src_arr[inlier_mask] if inlier_mask is not None else src_arr
            dst_fit = dst_arr[inlier_mask] if inlier_mask is not None else dst_arr

            H_candidate = _solve_regularized_homography(src_fit, dst_fit, base_h, reg_lambda=0.05)
            if H_candidate is None or not np.all(np.isfinite(H_candidate)):
                H = base_h
                status = "fallback_homography"
            else:
                base_norm = float(np.linalg.norm(base_h))
                delta_norm = float(np.linalg.norm(H_candidate - base_h))
                if base_norm > 0.0 and delta_norm > 0.75 * base_norm:
                    blend = min(1.0, (0.75 * base_norm) / (delta_norm + 1e-6))
                    H_candidate = base_h + blend * (H_candidate - base_h)
                pts_h = np.stack([xs, ys, np.ones_like(xs)], axis=0)
                proj = H_candidate @ pts_h
                proj /= np.maximum(proj[2], 1e-6)
                x_proj = proj[0]
                err_after = float(np.mean(np.abs(x_proj - canon_edges)))
                if not math.isfinite(err_after):
                    H = base_h
                    status = "fallback_invalid"
                    err_after = err_before
                else:
                    H = H_candidate

        if status != "ok":
            err_after = err_before
            H = base_h if H is None else H
        else:
            err_after = float(err_after)

        H_arr = np.asarray(H, dtype=np.float32)
        if self.logger.isEnabledFor(logging.DEBUG):
            delta_norm = float(np.linalg.norm(H_arr - base_h))
            err_delta = float(err_before - err_after)
            self.logger.debug(
                "reg_refined.debug: %s status=%s err_before=%.2fpx err_after=%.2fpx Δerr=%.2fpx ||ΔH||=%.3f frames=%d",
                video_id,
                status,
                err_before,
                err_after,
                err_delta,
                delta_norm,
                len(frames),
            )
        H_inv = _invert_homography(H_arr)
        grid = _homography_to_grid(H_inv, (height, width), canonical)
        return RegistrationResult(
            homography=H_arr,
            source_hw=(height, width),
            target_hw=canonical,
            err_before=err_before,
            err_after=err_after,
            frames=len(frames),
            status=status,
            baseline_slope=float(slope),
            baseline_intercept=float(intercept),
            keyboard_height=float(keyboard_height),
            timestamp=time.time(),
            grid=grid,
        )

    # --------------------------------------------------------------- Public API --

    def refine(
        self,
        *,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
    ) -> RegistrationResult:
        canon_id = canonical_video_id(video_id)
        cached = self._cache.get(canon_id)
        if cached is not None:
            if cached.grid is None:
                try:
                    H_inv = _invert_homography(cached.homography)
                    cached.grid = _homography_to_grid(H_inv, cached.source_hw, cached.target_hw)
                except Exception:
                    cached.grid = None
            return cached

        result = self._compute_refinement(canon_id, video_path, crop_meta)
        if result.grid is None:
            try:
                H_inv = _invert_homography(result.homography)
                result.grid = _homography_to_grid(H_inv, result.source_hw, result.target_hw)
            except Exception:
                result.grid = None
        self._cache[canon_id] = result
        if result.status.startswith("fallback"):
            self.logger.warning(
                "reg_refined: %s status=%s err_before=%.2fpx err_after=%.2fpx frames=%d",
                canon_id,
                result.status,
                result.err_before,
                result.err_after,
                result.frames,
            )
        else:
            self.logger.info(
                "reg_refined: %s err_before=%.2fpx err_after=%.2fpx frames=%d",
                canon_id,
                result.err_before,
                result.err_after,
                result.frames,
            )
        self._persist_cache()
        return result

    def transform_clip(
        self,
        clip: torch.Tensor,
        *,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
        interp: str = "bilinear",
    ) -> torch.Tensor:
        result = self.refine(video_id=video_id, video_path=video_path, crop_meta=crop_meta)
        target_h, target_w = self.canonical_hw
        alignable = {"linear", "bilinear", "bicubic", "trilinear"}
        if result.grid is None or clip.ndim != 4:
            if interp in alignable:
                return F.interpolate(clip, size=self.canonical_hw, mode=interp, align_corners=False)
            return F.interpolate(clip, size=self.canonical_hw, mode=interp)

        grid = result.grid.to(clip.device, clip.dtype)
        clip_in = clip.contiguous()
        T = clip.shape[0]
        grid_batched = grid.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
        mode = interp if interp in {"bilinear", "nearest"} else "bilinear"
        warped = F.grid_sample(
            clip_in,
            grid_batched,
            mode=mode,
            align_corners=False,
            padding_mode="border",
        )
        if warped.shape[-2:] != (target_h, target_w):
            if interp in alignable:
                warped = F.interpolate(warped, size=self.canonical_hw, mode=interp, align_corners=False)
            else:
                warped = F.interpolate(warped, size=self.canonical_hw, mode=interp)
        return warped


__all__ = ["RegistrationRefiner", "RegistrationResult"]
