"""Purpose:
    Estimate and cache audio/video alignment offsets used across TiViT
    datasets.  Provides guarded correlation search plus disk-backed lag cache.

Key Functions/Classes:
    - AVLagResult: Dataclass capturing lag, correlation, and guardrail flags.
    - AVLagCache: Thread-safe JSON cache storing per-video lag estimates.
    - compute_av_lag(): Main entry point that measures lag with guardrails.

CLI:
    Not a standalone CLI; imported by dataset loaders and calibration scripts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

from statistics import median

import numpy as np
import torch
import torch.nn.functional as F

from .identifiers import canonical_video_id, id_aliases, log_legacy_id_hit

LOGGER = logging.getLogger(__name__)


@dataclass
class AVLagResult:
    """Container for audio/visual lag estimation outputs."""

    lag_frames: int
    lag_ms: float
    corr: float
    from_cache: bool
    success: bool
    runtime_s: float = 0.0
    flags: Set[str] = field(default_factory=set)

    @property
    def used_video_median(self) -> bool:
        return "used_video_median" in self.flags

    @property
    def low_corr_zero(self) -> bool:
        return "low_corr_zero" in self.flags

    @property
    def hit_bound(self) -> bool:
        return "hit_bound" in self.flags

    @property
    def clamped(self) -> bool:
        return "clamped" in self.flags

    @property
    def lag_timeout(self) -> bool:
        return "lag_timeout" in self.flags


_CACHE_LOCK = threading.Lock()
_CACHE_TMP_SUFFIX = ".tmp"


class AVLagCache:
    """Cache of per-video lag estimates persisted to JSON."""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        default_path = project_root / "av_lags.json"
        self.cache_path = Path(cache_path) if cache_path else default_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict[str, float]] = None
        self._loaded = False
        self._lock_timeout_logged = False
        self._preload_slow_logged = False

    def _lock_path(self) -> Path:
        return self.cache_path.with_suffix(self.cache_path.suffix + ".lock")

    def _acquire_file_lock(self, timeout: float) -> bool:
        lock_path = self._lock_path()
        if timeout is None or timeout < 0:
            timeout = 0.0
        deadline = time.perf_counter() + timeout if timeout > 0 else None
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                if deadline is None or time.perf_counter() >= deadline:
                    return False
                time.sleep(0.05)

    def _release_file_lock(self) -> None:
        lock_path = self._lock_path()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _load(self, *, lock_timeout: float = 1.0, warn_on_timeout: bool = False, locked: bool = False) -> bool:
        if self._loaded:
            return True
        if not locked:
            acquired = self._acquire_file_lock(lock_timeout)
            if not acquired:
                if warn_on_timeout and not self._lock_timeout_logged:
                    LOGGER.debug("lag_cache preload timeout; continuing without preload")
                    self._lock_timeout_logged = True
                return False
        try:
            with _CACHE_LOCK:
                if self._loaded:
                    return True
                data: Dict[str, float] = {}
                if self.cache_path.exists():
                    try:
                        with self.cache_path.open("r", encoding="utf-8") as handle:
                            raw = json.load(handle)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.warning("Failed to load A/V lag cache from %s (%s)", self.cache_path, exc)
                        raw = None
                    if isinstance(raw, dict):
                        for key, value in raw.items():
                            try:
                                data[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue
                self._cache = data
                self._loaded = True
                return True
        finally:
            if not locked:
                self._release_file_lock()

    def preload(self) -> None:
        """Eagerly load the cache once during dataset initialisation."""

        if self._loaded:
            return
        start = time.perf_counter()
        loaded = self._load(lock_timeout=1.0, warn_on_timeout=True, locked=False)
        elapsed = time.perf_counter() - start
        if not loaded:
            return
        if elapsed > 2.0:
            if not self._preload_slow_logged:
                LOGGER.debug("lag_cache preload exceeded threshold; skipping preload this run")
                self._preload_slow_logged = True
            with _CACHE_LOCK:
                self._cache = None
                self._loaded = False

    def get(self, video_id: str) -> Optional[float]:
        self._load()
        if self._cache is None:
            return None
        canon_id = canonical_video_id(video_id)
        for alias in id_aliases(canon_id):
            value = self._cache.get(alias)
            if value is not None:
                if alias != canon_id:
                    log_legacy_id_hit(alias, canon_id, logger=LOGGER)
                return float(value)
        return None

    def set(self, video_id: str, lag_ms: float) -> None:
        if not self._acquire_file_lock(1.0):
            LOGGER.warning("Skipping A/V lag cache write for %s due to lock timeout", video_id)
            return
        try:
            self._load(lock_timeout=0.0, warn_on_timeout=False, locked=True)
            if self._cache is None:
                self._cache = {}
            key = canonical_video_id(video_id)
            new_value = float(lag_ms)
            tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + _CACHE_TMP_SUFFIX)
            with _CACHE_LOCK:
                current = self._cache.get(key)
                if current is not None and math.isclose(current, new_value, abs_tol=1e-3):
                    return
                self._cache[key] = new_value
                legacy_key = f"{key}.0"
                if legacy_key in self._cache and legacy_key != key:
                    del self._cache[legacy_key]
                try:
                    with tmp_path.open("w", encoding="utf-8") as handle:
                        json.dump(self._cache, handle, indent=2, sort_keys=True)
                    os.replace(tmp_path, self.cache_path)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to write A/V lag cache to %s (%s)", self.cache_path, exc)
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
        finally:
            self._release_file_lock()


def _normalize_series(values: np.ndarray) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return None
    norm = (values - vmin) / (vmax - vmin)
    return norm.astype(np.float32, copy=False)


def _sanitize_bbox(
    bbox: Optional[Sequence[int]],
    height: int,
    width: int,
) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    try:
        min_y, max_y, min_x, max_x = (int(b) for b in bbox)
    except (TypeError, ValueError):
        return None
    min_y = max(0, min(min_y, height - 1))
    max_y = max(0, min(max_y, height))
    min_x = max(0, min(min_x, width - 1))
    max_x = max(0, min(max_x, width))
    if max_y <= min_y or max_x <= min_x:
        return None
    return min_y, max_y, min_x, max_x


def _motion_envelope(
    frames: torch.Tensor,
    keyboard_bbox: Optional[Sequence[int]] = None,
    downsample: int = 2,
) -> Optional[np.ndarray]:
    if frames is None or frames.ndim != 4 or frames.shape[0] < 2:
        return None
    x = frames.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.to(torch.float32)
    if x.shape[1] == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    elif x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    if downsample > 1 and x.shape[-1] >= downsample and x.shape[-2] >= downsample:
        x = F.avg_pool2d(x, kernel_size=downsample, stride=downsample)
    diff = torch.abs(x[1:] - x[:-1])
    if keyboard_bbox is not None:
        height = diff.shape[-2]
        width = diff.shape[-1]
        region = _sanitize_bbox(keyboard_bbox, height, width)
        if region is not None:
            min_y, max_y, min_x, max_x = region
            diff = diff[..., min_y:max_y, min_x:max_x]
    env = diff.mean(dim=(1, 2, 3)).cpu().numpy()
    env = np.concatenate([np.zeros(1, dtype=env.dtype), env], axis=0)
    return _normalize_series(env)


def _label_envelope(labels: Optional[torch.Tensor],
                    clip_start: float,
                    clip_end: float,
                    hop_seconds: float,
                    num_frames: int,
                    sigma_frames: float = 2.5) -> Optional[np.ndarray]:
    if labels is None or labels.numel() == 0 or num_frames <= 1 or hop_seconds <= 0:
        return None
    arr = labels.detach().cpu().numpy()
    onset = arr[:, 0]
    offset = arr[:, 1]
    mask = (offset > clip_start) & (onset < clip_end)
    if not np.any(mask):
        return None
    onset = np.clip(onset[mask], clip_start, clip_end)
    offset = np.clip(offset[mask], clip_start, clip_end)
    env = np.zeros(int(num_frames), dtype=np.float32)
    frame_on = np.rint((onset - clip_start) / hop_seconds).astype(int)
    frame_off = np.rint((offset - clip_start) / hop_seconds).astype(int)
    frame_on = np.clip(frame_on, 0, num_frames - 1)
    frame_off = np.clip(frame_off, 0, num_frames - 1)
    for idx in frame_on:
        env[idx] += 1.0
    for idx in frame_off:
        env[idx] += 0.5
    if not np.any(env):
        return None
    radius = max(1, int(math.ceil(3.0 * sigma_frames)))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (grid / float(sigma_frames)) ** 2)
    kernel /= kernel.sum()
    env = np.convolve(env, kernel, mode="same")
    return _normalize_series(env.astype(np.float32, copy=False))


def _resample_to_length(values: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    if target_len <= 0:
        return values
    if values.size == target_len:
        return values.astype(np.float32, copy=False)
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if values.size == 1:
        return np.full(target_len, float(values[0]), dtype=np.float32)
    src_idx = np.linspace(0.0, values.size - 1, num=values.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, values.size - 1, num=target_len, dtype=np.float32)
    resampled = np.interp(dst_idx, src_idx, values.astype(np.float32, copy=False))
    return resampled.astype(np.float32, copy=False)


def _zscore_series(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    if values.size == 0:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-6:
        return None
    normalized = (values - mean) / std
    return normalized.astype(np.float32, copy=False)


def _corr_at_lag(video_env: np.ndarray, audio_env: np.ndarray, lag: int) -> float:
    if video_env is None or audio_env is None:
        return float("nan")
    if lag > 0:
        v = video_env[lag:]
        a = audio_env[:-lag] if lag < audio_env.size else np.empty(0, dtype=np.float32)
    elif lag < 0:
        lag_abs = abs(lag)
        v = video_env[:-lag_abs] if lag_abs < video_env.size else np.empty(0, dtype=np.float32)
        a = audio_env[lag_abs:]
    else:
        v = video_env
        a = audio_env
    length = min(v.size, a.size)
    if length <= 1:
        return float("nan")
    v = v[:length]
    a = a[:length]
    v_center = v - v.mean()
    a_center = a - a.mean()
    denom = np.linalg.norm(v_center) * np.linalg.norm(a_center)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(v_center, a_center) / denom)


_VIDEO_LAG_HISTORY: Dict[str, list[float]] = {}
_VIDEO_HISTORY_LIMIT = 32
_BOUND_TRACKER = {
    "hits": 0,
    "total": 0,
    "expanded": False,
}
_BOUND_THRESHOLD = 0.10
_DEFAULT_WINDOW_MS = 500.0
_EXPANDED_WINDOW_MS = 800.0
_CLAMP_LIMIT_MS = 400.0
_LOW_CORR_THRESHOLD = 0.15
_BORDERLINE_CORR_THRESHOLD = 0.25
_HIGH_CONFIDENCE_CORR = 0.35
_MAX_RUNTIME_S = 2.0


def _get_video_median(video_id: str) -> Optional[float]:
    key = canonical_video_id(video_id)
    history = _VIDEO_LAG_HISTORY.get(key)
    if not history:
        return None
    return float(median(history))


def _record_video_lag(video_id: str, lag_ms: float, corr: float, flags: Set[str]) -> None:
    if not math.isfinite(lag_ms) or not math.isfinite(corr):
        return
    if corr < _BORDERLINE_CORR_THRESHOLD:
        return
    if "low_corr_zero" in flags or "lag_timeout" in flags:
        return
    key = canonical_video_id(video_id)
    history = _VIDEO_LAG_HISTORY.setdefault(key, [])
    history.append(float(lag_ms))
    if len(history) > _VIDEO_HISTORY_LIMIT:
        del history[0 : len(history) - _VIDEO_HISTORY_LIMIT]
        return


def _maybe_expand_window() -> None:
    total = _BOUND_TRACKER["total"]
    hits = _BOUND_TRACKER["hits"]
    if total == 0 or _BOUND_TRACKER["expanded"]:
        return
    if hits / float(total) >= _BOUND_THRESHOLD:
        _BOUND_TRACKER["expanded"] = True


def _select_window_ms(requested_ms: float) -> float:
    limit = requested_ms if requested_ms > 0 else 0.0
    if _BOUND_TRACKER["expanded"]:
        limit = max(limit, _EXPANDED_WINDOW_MS)
    return limit


def _apply_guardrails(
    video_id: str,
    *,
    hop_seconds: float,
    raw_lag_ms: float,
    corr: float,
    hit_bound: bool,
    min_corr: float = _LOW_CORR_THRESHOLD,
    use_abs_corr: bool = True,
) -> Tuple[int, float, Set[str]]:
    video_id = canonical_video_id(video_id)
    selected_ms = float(raw_lag_ms)
    video_median_ms = _get_video_median(video_id)
    flags: Set[str] = set()

    corr_threshold = float(min_corr)
    if not math.isfinite(corr_threshold):
        corr_threshold = _LOW_CORR_THRESHOLD
    corr_threshold = min(max(corr_threshold, 0.0), 1.0)
    borderline_threshold = max(corr_threshold, _BORDERLINE_CORR_THRESHOLD)
    corr_eval = abs(corr) if use_abs_corr else corr

    if hit_bound:
        flags.add("hit_bound")
        if video_median_ms is not None:
            selected_ms = video_median_ms
            flags.add("used_video_median")
        else:
            selected_ms = 0.0

    if math.isfinite(corr):
        if corr_eval < corr_threshold:
            selected_ms = 0.0
            flags.add("low_corr_zero")
        elif corr_eval < borderline_threshold and "used_video_median" not in flags:
            if video_median_ms is not None:
                selected_ms = video_median_ms
                flags.add("used_video_median")
            else:
                selected_ms = 0.0
                flags.add("low_corr_zero")

    if (
        math.isfinite(corr)
        and abs(selected_ms) > _CLAMP_LIMIT_MS
        and corr_eval < _HIGH_CONFIDENCE_CORR
    ):
        selected_ms = math.copysign(_CLAMP_LIMIT_MS, selected_ms)
        flags.add("clamped")

    if hop_seconds > 0:
        lag_frames = int(round((selected_ms / 1000.0) / hop_seconds))
        lag_ms = lag_frames * hop_seconds * 1000.0
    else:
        lag_frames = 0
        lag_ms = 0.0

    return lag_frames, float(lag_ms), flags


def compute_av_lag(
    video_id: str,
    frames: torch.Tensor,
    *,
    fps: Optional[float] = None,
    hop_seconds: Optional[float] = None,
    events: Optional[torch.Tensor] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    cache: Optional[AVLagCache] = None,
    window_ms: float = _DEFAULT_WINDOW_MS,
    keyboard_bbox: Optional[Sequence[int]] = None,
    max_runtime_s: float = _MAX_RUNTIME_S,
    min_corr: Optional[float] = None,
    visual_curve: Optional[np.ndarray] = None,
    use_abs_corr: bool = False,
) -> AVLagResult:
    canon_id = canonical_video_id(video_id)
    start_time = time.perf_counter()
    deadline = start_time + float(max_runtime_s) if max_runtime_s and max_runtime_s > 0 else None
    flags: Set[str] = set()
    lag_frames = 0
    lag_ms = 0.0
    corr = float("nan")
    from_cache = False
    success = False
    pending_cache_value: Optional[float] = None
    timed_out = False

    if hop_seconds is None or hop_seconds <= 0:
        if fps is None or fps <= 0:
            raise ValueError("compute_av_lag requires fps>0 or hop_seconds>0")
        hop_seconds = 1.0 / float(fps)
    else:
        hop_seconds = float(hop_seconds)
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    if fps is not None and fps > 0:
        fps_est = float(fps)
    else:
        fps_est = (1.0 / hop_seconds) if hop_seconds > 0 else 0.0
    if not math.isfinite(fps_est) or fps_est <= 0:
        fps_est = 0.0
    window_frames_cap = int(round(fps_est * 0.5)) if fps_est > 0 else 0

    T = int(frames.shape[0]) if frames is not None else 0
    if clip_end is None:
        clip_end = clip_start + max(T - 1, 0) * hop_seconds

    min_corr_value = _LOW_CORR_THRESHOLD if min_corr is None else float(min_corr)
    if not math.isfinite(min_corr_value):
        min_corr_value = _LOW_CORR_THRESHOLD
    min_corr_value = min(max(min_corr_value, 0.0), 1.0)

    if visual_curve is not None:
        if isinstance(visual_curve, np.ndarray):
            video_env = visual_curve.astype(np.float32, copy=False)
        else:
            video_env = np.asarray(visual_curve, dtype=np.float32)
        video_env = video_env.reshape(-1)
    else:
        video_env = _motion_envelope(frames, keyboard_bbox)

    audio_env = _label_envelope(events, clip_start, clip_end, hop_seconds, T)

    video_env = _resample_to_length(video_env, T)
    if visual_curve is not None:
        video_env = _zscore_series(video_env)

    audio_env = _resample_to_length(audio_env, T)
    if video_env is None or audio_env is None:
        runtime_s = time.perf_counter() - start_time
        return AVLagResult(lag_frames, lag_ms, float("nan"), False, False, runtime_s, flags)

    video_var = float(np.var(video_env)) if video_env.size > 0 else 0.0
    audio_var = float(np.var(audio_env)) if audio_env.size > 0 else 0.0
    if video_var <= 1e-6 or audio_var <= 1e-6:
        flags.add("low_corr_zero")
        runtime_s = time.perf_counter() - start_time
        return AVLagResult(0, 0.0, 0.0, False, True, runtime_s, flags)

    cached_ms = cache.get(canon_id) if cache is not None else None
    if cached_ms is not None:
        from_cache = True
        raw_lag_ms = float(cached_ms)
        lag_frames_guess = int(round((raw_lag_ms / 1000.0) / hop_seconds)) if hop_seconds > 0 else 0
        if window_frames_cap > 0 and abs(lag_frames_guess) > window_frames_cap:
            lag_frames_guess = int(math.copysign(window_frames_cap, lag_frames_guess))
        raw_lag_ms = lag_frames_guess * hop_seconds * 1000.0
        corr = _corr_at_lag(video_env, audio_env, lag_frames_guess)
        if math.isfinite(corr):
            lag_frames, lag_ms, guard_flags = _apply_guardrails(
                canon_id,
                hop_seconds=hop_seconds,
                raw_lag_ms=raw_lag_ms,
                corr=corr,
                hit_bound=False,
                min_corr=min_corr_value,
                use_abs_corr=use_abs_corr,
            )
            if window_frames_cap > 0 and abs(lag_frames) > window_frames_cap:
                lag_frames = int(math.copysign(window_frames_cap, lag_frames))
                lag_ms = lag_frames * hop_seconds * 1000.0
            flags.update(guard_flags)
            success = True
            pending_cache_value = lag_ms
        else:
            lag_frames = lag_frames_guess
            lag_ms = raw_lag_ms
            success = False
    else:
        search_window_ms = _select_window_ms(window_ms if window_ms > 0 else _DEFAULT_WINDOW_MS)
        max_lag_frames = int(round((search_window_ms / 1000.0) / hop_seconds)) if hop_seconds > 0 else 0
        max_lag_frames = max(0, max_lag_frames)
        if window_frames_cap > 0:
            if max_lag_frames == 0:
                max_lag_frames = window_frames_cap
            else:
                max_lag_frames = min(max_lag_frames, window_frames_cap)
        hit_bound = False
        if max_lag_frames == 0:
            if deadline and time.perf_counter() > deadline:
                timed_out = True
            if timed_out:
                corr = float("nan")
                lag_frames_raw = 0
            else:
                corr_candidate = _corr_at_lag(video_env, audio_env, 0)
                if math.isfinite(corr_candidate):
                    corr = corr_candidate
                    lag_frames_raw = 0
                else:
                    corr = float("nan")
                    lag_frames_raw = 0
        else:
            best_corr = float("nan")
            best_score = float("-inf")
            best_lag = 0
            _BOUND_TRACKER["total"] += 1
            for lag in range(-max_lag_frames, max_lag_frames + 1):
                if deadline and time.perf_counter() > deadline:
                    timed_out = True
                    break
                candidate_corr = _corr_at_lag(video_env, audio_env, lag)
                if not math.isfinite(candidate_corr):
                    continue
                candidate_score = abs(candidate_corr) if use_abs_corr else candidate_corr
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_corr = candidate_corr
                    best_lag = lag
            if timed_out:
                corr = float("nan")
                lag_frames_raw = 0
            elif not math.isfinite(best_corr) or best_score == float("-inf"):
                corr = float("nan")
                lag_frames_raw = 0
            else:
                corr = best_corr
                lag_frames_raw = best_lag
                hit_bound = max_lag_frames > 0 and abs(best_lag) == max_lag_frames
                if hit_bound:
                    _BOUND_TRACKER["hits"] += 1
            _maybe_expand_window()
        if math.isfinite(corr):
            raw_lag_ms = lag_frames_raw * hop_seconds * 1000.0
            lag_frames, lag_ms, guard_flags = _apply_guardrails(
                canon_id,
                hop_seconds=hop_seconds,
                raw_lag_ms=raw_lag_ms,
                corr=corr,
                hit_bound=hit_bound,
                min_corr=min_corr_value,
                use_abs_corr=use_abs_corr,
            )
            if window_frames_cap > 0 and abs(lag_frames) > window_frames_cap:
                lag_frames = int(math.copysign(window_frames_cap, lag_frames))
                lag_ms = lag_frames * hop_seconds * 1000.0
            flags.update(guard_flags)
            success = True
            pending_cache_value = lag_ms
        else:
            lag_frames = 0
            lag_ms = 0.0
            success = False

    runtime_s = time.perf_counter() - start_time
    if timed_out or (max_runtime_s > 0 and runtime_s > max_runtime_s):
        flags = set(flags)
        flags.add("lag_timeout")
        success = False
        corr = 0.0
        lag_frames = 0
        lag_ms = 0.0
        pending_cache_value = None

    if success and math.isfinite(corr):
        _record_video_lag(canon_id, lag_ms, corr, flags)
        if cache is not None and pending_cache_value is not None:
            cache.set(canon_id, pending_cache_value)

    return AVLagResult(
        lag_frames=lag_frames,
        lag_ms=float(lag_ms),
        corr=float(corr),
        from_cache=from_cache,
        success=success,
        runtime_s=runtime_s,
        flags=set(flags),
    )


def estimate_av_lag(
    video_id: str,
    frames: torch.Tensor,
    labels: torch.Tensor,
    *,
    clip_start: float,
    clip_end: float,
    hop_seconds: float,
    cache: Optional[AVLagCache] = None,
    max_lag_ms: float = 500.0,
) -> AVLagResult:
    return compute_av_lag(
        video_id,
        frames,
        hop_seconds=hop_seconds,
        events=labels,
        clip_start=clip_start,
        clip_end=clip_end,
        cache=cache,
        window_ms=max_lag_ms,
    )


def round_lag_ms_for_cache(lag_ms: float, fps: float) -> int:
    """Round ``lag_ms`` to the nearest frame duration for cache keys."""

    try:
        lag_val = float(lag_ms)
    except (TypeError, ValueError):
        lag_val = 0.0
    try:
        fps_val = float(fps)
    except (TypeError, ValueError):
        fps_val = 0.0

    if not math.isfinite(fps_val) or fps_val <= 0:
        return int(round(lag_val))

    frame_ms = 1000.0 / fps_val
    frames = int(round(lag_val / frame_ms))
    rounded_ms = frames * frame_ms
    return int(round(rounded_ms))


def shift_label_events(labels: torch.Tensor,
                       lag_seconds: float,
                       *,
                       clip_start: float,
                       clip_end: float,
                       eps: float = 1e-4) -> torch.Tensor:
    if labels is None or labels.numel() == 0:
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        dtype = labels.dtype if isinstance(labels, torch.Tensor) else torch.float32
        return torch.zeros((0, 3), dtype=dtype, device=device)
    out = labels.clone()
    if abs(lag_seconds) > 1e-9:
        out[:, 0:2] = out[:, 0:2] + float(lag_seconds)
    mask = (out[:, 1] > clip_start) & (out[:, 0] < clip_end)
    out = out[mask]
    if out.numel() == 0:
        return out.reshape(0, 3)
    out[:, 0] = out[:, 0].clamp(min=clip_start, max=clip_end)
    out[:, 1] = out[:, 1].clamp(min=clip_start, max=clip_end)
    out[:, 1] = torch.maximum(out[:, 1], out[:, 0] + float(eps))
    return out
