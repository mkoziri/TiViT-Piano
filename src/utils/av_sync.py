from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


@dataclass
class AVLagResult:
    """Container for audio/visual lag estimation outputs."""

    lag_frames: int
    lag_ms: float
    corr: float
    from_cache: bool
    success: bool


class AVLagCache:
    """Cache of per-video lag estimates persisted to JSON."""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_path = project_root / "runs" / "av_lags.json"
        self.cache_path = Path(cache_path) if cache_path else default_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[dict] = None

    def _load(self) -> None:
        if self._cache is not None:
            return
        self._cache = {}
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load A/V lag cache from %s (%s)", self.cache_path, exc)
            return
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    self._cache[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

    def get(self, video_id: str) -> Optional[float]:
        self._load()
        assert self._cache is not None
        value = self._cache.get(video_id)
        return float(value) if value is not None else None

    def set(self, video_id: str, lag_ms: float) -> None:
        self._load()
        assert self._cache is not None
        self._cache[video_id] = float(lag_ms)
        try:
            with self.cache_path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to write A/V lag cache to %s (%s)", self.cache_path, exc)


def _normalize_series(values: np.ndarray) -> Optional[np.ndarray]:
    if values.size == 0:
        return None
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return None
    norm = (values - vmin) / (vmax - vmin)
    return norm.astype(np.float32, copy=False)


def _motion_envelope(frames: torch.Tensor, downsample: int = 2) -> Optional[np.ndarray]:
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
    env = diff.mean(dim=(1, 2, 3)).cpu().numpy()
    env = np.concatenate([np.zeros(1, dtype=env.dtype), env], axis=0)
    return _normalize_series(env)


def _label_envelope(labels: torch.Tensor,
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


def estimate_av_lag(video_id: str,
                    frames: torch.Tensor,
                    labels: torch.Tensor,
                    *,
                    clip_start: float,
                    clip_end: float,
                    hop_seconds: float,
                    cache: Optional[AVLagCache] = None,
                    max_lag_ms: float = 500.0) -> AVLagResult:
    T = int(frames.shape[0]) if frames is not None else 0
    video_env = _motion_envelope(frames)
    audio_env = _label_envelope(labels, clip_start, clip_end, hop_seconds, T)
    if video_env is None or audio_env is None:
        return AVLagResult(0, 0.0, float("nan"), False, False)

    cached_ms = cache.get(video_id) if cache is not None else None
    if cached_ms is not None:
        lag_frames = int(round((cached_ms / 1000.0) / hop_seconds)) if hop_seconds > 0 else 0
        corr = _corr_at_lag(video_env, audio_env, lag_frames)
        return AVLagResult(lag_frames, float(cached_ms), corr, True, True)

    max_lag_frames = int(round((max_lag_ms / 1000.0) / hop_seconds)) if hop_seconds > 0 else 0
    max_lag_frames = max(0, max_lag_frames)
    if max_lag_frames == 0:
        corr = _corr_at_lag(video_env, audio_env, 0)
        if not math.isfinite(corr):
            return AVLagResult(0, 0.0, float("nan"), False, False)
        if cache is not None:
            cache.set(video_id, 0.0)
        return AVLagResult(0, 0.0, corr, False, True)

    best_corr = float("-inf")
    best_lag = 0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        corr = _corr_at_lag(video_env, audio_env, lag)
        if not math.isfinite(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if not math.isfinite(best_corr) or best_corr == float("-inf"):
        return AVLagResult(0, 0.0, float("nan"), False, False)

    lag_ms = best_lag * hop_seconds * 1000.0
    if cache is not None:
        cache.set(video_id, lag_ms)
    return AVLagResult(best_lag, lag_ms, best_corr, False, True)


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
