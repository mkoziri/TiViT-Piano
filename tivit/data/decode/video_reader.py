"""
Purpose:
    Provide a lightweight video reader abstraction that loads clips using
    decord when available and falls back to OpenCV. Keeps tensors memory
    efficient and aligned to the expected shape (T,C,H,W).

Key Functions/Classes:
    - VideoReaderConfig: Captures clip loading parameters (frames/stride/resize/channels).
    - load_clip(): Decodes a clip to float32 in [0,1], applying stride/pad and resize.
    - _try_decord(): Attempt decode via decord.
    - _try_cv2(): Fallback decode via OpenCV.

CLI Arguments:
    (none)

Usage:
    cfg = VideoReaderConfig(frames=16, stride=2, resize_hw=(180,1536), channels=3)
    clip = load_clip(\"video.mp4\", cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


@dataclass
class VideoReaderConfig:
    """Lightweight config for clip loading."""

    frames: int
    stride: int
    resize_hw: Tuple[int, int] | None
    channels: int = 3
    start_frame: int = 0


def _try_decord(path: Path, cfg: VideoReaderConfig) -> torch.Tensor:
    """Attempt to decode using decord; returns T,H,W,C float32 in [0,1]."""
    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(path))
    total = len(vr)
    start = min(max(int(cfg.start_frame), 0), max(total - 1, 0))
    idxs = list(range(start, total, cfg.stride))[: cfg.frames]
    if not idxs:
        idxs = [0]
    if len(idxs) < cfg.frames:
        idxs += [idxs[-1]] * (cfg.frames - len(idxs))
    batch = vr.get_batch(idxs)  # T,H,W,C uint8
    x = batch.to(torch.float32) / 255.0
    return x


def _try_cv2(path: Path, cfg: VideoReaderConfig) -> torch.Tensor:
    """Fallback decode using OpenCV; returns T,H,W,C float32 in [0,1]."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    images = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < cfg.start_frame:
            i += 1
            continue
        rel_idx = i - cfg.start_frame
        if rel_idx % cfg.stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg.resize_hw and cfg.resize_hw[0] > 0 and cfg.resize_hw[1] > 0:
                frame = cv2.resize(frame, (cfg.resize_hw[1], cfg.resize_hw[0]), interpolation=cv2.INTER_AREA)
            images.append(frame)
            if len(images) == cfg.frames:
                break
        i += 1
    cap.release()
    if not images:
        raise RuntimeError(f"Failed to read frames from {path}")
    while len(images) < cfg.frames:
        images.append(images[-1])
    x = torch.from_numpy(np.stack(images, axis=0)).to(torch.float32) / 255.0
    return x


def load_clip(path: str | Path, cfg: VideoReaderConfig) -> torch.Tensor:
    """
    Load a clip as (T,C,H,W) float32 in [0,1].

    Prefers decord, falls back to cv2 without adding extra copies.
    """

    path = Path(path)
    try:
        x = _try_decord(path, cfg)
    except Exception:
        x = _try_cv2(path, cfg)

    if cfg.channels == 1 and x.shape[-1] == 3:
        x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)
    x = x.permute(0, 3, 1, 2)  # T,C,H,W
    if cfg.resize_hw and cfg.resize_hw[0] > 0 and cfg.resize_hw[1] > 0:
        h, w = x.shape[-2:]
        if (h, w) != cfg.resize_hw:
            x = torch.nn.functional.interpolate(x, size=cfg.resize_hw, mode="area")
    return x


__all__ = ["VideoReaderConfig", "load_clip"]
