#!/usr/bin/env python3
"""Purpose:
    Diagnose temporal alignment between model predictions and ground-truth
    onset/offset labels by sweeping integer frame shifts and evaluating
    correlation and F1 scores.

Key Functions/Classes:
    - _align(): Crops tensors after applying a temporal shift so correlations
      can be computed on equal-length sequences.
    - _load_window(): Fetches a temporal window from disk using the dataset's
      decoding parameters and tiling scheme.
    - main(): CLI entry point that loads configuration, runs the model on a
      selected clip, and sweeps lag offsets.

CLI:
    Run ``python scripts/lag_sweep.py --split val --clip <name> --seconds 5 --ckpt``
    to analyze a specific window.  Optional flags include ``--head`` to choose
    onset or offset logits and ``--max_shift`` for the sweep range.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Mapping
import torch.nn.functional as F

import numpy as np
import torch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src"))

from utils import load_config  # noqa: E402
from models import build_model  # noqa: E402
from data import make_dataloader 

def _align(a: torch.Tensor, b: torch.Tensor, delta: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if delta > 0:
        a = a[delta:]
        b = b[:-delta]
    elif delta < 0:
        a = a[:delta]
        b = b[-delta:]

    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized cross-correlation that is safe for zero-variance inputs."""

    a = a - a.mean()
    b = b - b.mean()
    na = a.norm()
    nb = b.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return float("nan")
    return float((a * b).sum().item() / (na.item() * nb.item()))


def _extract_dataset(loader_obj, split: str):
    if isinstance(loader_obj, dict):
        loader_obj = loader_obj.get(split) or next(iter(loader_obj.values()))
    if isinstance(loader_obj, (list, tuple)):
        loader_obj = loader_obj[0]
    dataset = getattr(loader_obj, "dataset", loader_obj)
    return loader_obj, dataset


def _select_clip_index(dataset, key: Optional[str], explicit_idx: Optional[int]) -> int:
    if explicit_idx is not None:
        if explicit_idx < 0 or explicit_idx >= len(dataset):
            raise IndexError(f"clip_idx {explicit_idx} out of range (len={len(dataset)})")
        return explicit_idx
    if key is None:
        return 0
    videos = getattr(dataset, "videos", None)
    if videos:
        for idx, path in enumerate(videos):
            p = Path(path)
            if key in p.stem or key in p.name or key in str(p):
                return idx
    for idx in range(len(dataset)):
        sample = dataset[idx]
        path = Path(sample.get("path", ""))
        if key in path.stem or key in path.name or key in str(path):
            return idx
    raise SystemExit(f"Clip '{key}' not found.")


def _compute_frames_needed(seconds: float, start_sec: float, hop_seconds: float, base_frames: int) -> Tuple[int, int, int]:
    start_frame = max(0, int(round(start_sec / hop_seconds)))
    window_frames = max(1, int(math.ceil(seconds / hop_seconds)))
    frames_needed = max(base_frames, start_frame + window_frames + 1)
    return frames_needed, start_frame, window_frames


def _clone_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def _ensure_frame_targets(cfg: dict) -> None:
    ft = cfg.setdefault("dataset", {}).setdefault("frame_targets", {})
    ft.setdefault("enable", True)
    ft.setdefault("tolerance", 0.10)
    ft.setdefault("dilate_active_frames", 1)


def _build_rolls_from_labels(
    labels: torch.Tensor,
    start_sec: float,
    seconds: float,
    hop_seconds: float,
    note_min: int,
    note_max: int,
    dilate: int,
) -> Dict[str, torch.Tensor]:
    frames = max(1, int(math.ceil(seconds / hop_seconds)))
    pitches = note_max - note_min + 1
    onset = torch.zeros((frames, pitches), dtype=torch.float32)
    offset = torch.zeros((frames, pitches), dtype=torch.float32)
    window_start = start_sec
    window_end = start_sec + seconds
    if labels is None or labels.numel() == 0:
        return {"onset_roll": onset, "offset_roll": offset}
    for onset_sec, offset_sec, pitch in labels.tolist():
        if offset_sec <= window_start or onset_sec >= window_end:
            continue
        pitch_idx = int(pitch - note_min)
        if pitch_idx < 0 or pitch_idx >= pitches:
            continue
        if window_start <= onset_sec < window_end:
            frame = int(round((onset_sec - window_start) / hop_seconds))
            frame = max(0, min(frames - 1, frame))
            onset[frame, pitch_idx] = 1.0
        if window_start < offset_sec <= window_end:
            frame = int(round((offset_sec - window_start) / hop_seconds))
            frame = max(0, min(frames - 1, frame))
            offset[frame, pitch_idx] = 1.0
    if dilate > 0:
        kernel = 2 * dilate + 1
        onset = F.max_pool1d(onset.permute(1, 0).unsqueeze(0), kernel_size=kernel, stride=1, padding=dilate)
        offset = F.max_pool1d(offset.permute(1, 0).unsqueeze(0), kernel_size=kernel, stride=1, padding=dilate)
        onset = onset.squeeze(0).permute(1, 0)
        offset = offset.squeeze(0).permute(1, 0)
    return {"onset_roll": onset, "offset_roll": offset}


def main():  # noqa: C901
    ap = argparse.ArgumentParser(description="Lag sweep between predictions and ground truth")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--clip", help="Clip id/substring (legacy alias for --clip_key)")
    ap.add_argument("--clip_key", help="Clip id/substring to search for")
    ap.add_argument("--clip_idx", type=int, help="Explicit dataset index")
    ap.add_argument("--seconds", type=float, required=True, help="Window length in seconds")
    ap.add_argument("--start_sec", type=float, default=0.0, help="Window start in seconds")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--head", choices=["onset", "offset"], required=True)
    ap.add_argument("--use_logits", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prob_threshold", type=float, default=0.2)
    ap.add_argument("--max_shift", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    time_cfg = cfg.get("time", cfg.get("dataset", {}))
    decode_fps = float(time_cfg.get("decode_fps", cfg.get("dataset", {}).get("decode_fps", 30.0)))
    hop_frames = time_cfg.get("hop_frames")
    if hop_frames is not None:
        hop_seconds = float(hop_frames) / decode_fps
    else:
        hop_seconds = float(time_cfg.get("hop_seconds", cfg.get("dataset", {}).get("hop_seconds", 1.0 / decode_fps)))
        hop_frames = int(round(hop_seconds * decode_fps))
    print(
        f"[GRID] decode_fps={decode_fps} hop_frames={hop_frames} hop_seconds={hop_seconds:.4f}",
        flush=True,
    )

    frames_base = int(cfg.get("dataset", {}).get("frames", 32))
    frames_needed, start_frame, window_frames = _compute_frames_needed(args.seconds, args.start_sec, hop_seconds, frames_base)

    cfg_local = _clone_cfg(cfg)
    cfg_local.setdefault("dataset", {})["frames"] = frames_needed
    _ensure_frame_targets(cfg_local)
    loader_obj = make_dataloader(cfg_local, split=args.split)
    loader, dataset = _extract_dataset(loader_obj, args.split)

    clip_key = args.clip_key or args.clip
    clip_idx = _select_clip_index(dataset, clip_key, args.clip_idx)
    sample = dataset[clip_idx]
    clip_path = Path(sample.get("path", f"idx_{clip_idx}"))

    video = sample["video"]
    if video.ndim == 5:
        # (tiles, T, C, H, W) or (T, tiles, C, H, W) -> flatten tiles dimension
        if video.shape[0] < video.shape[1]:
            video = video.permute(1, 0, 2, 3, 4)
        video = video.reshape(video.shape[0], -1, video.shape[-2], video.shape[-1])
    T = video.shape[0]
    if T < start_frame + window_frames:
        raise SystemExit(f"Clip too short ({T} frames) for requested window (need {start_frame + window_frames})")
    clip = video[start_frame : start_frame + window_frames].contiguous()

    ft_cfg = cfg_local.get("dataset", {}).get("frame_targets", {})
    note_min = int(ft_cfg.get("note_min", 21))
    note_max = int(ft_cfg.get("note_max", 108))
    dilate = int(ft_cfg.get("dilate_active_frames", 0))

    onset_roll = sample.get("onset_roll")
    offset_roll = sample.get("offset_roll")
    if onset_roll is None or offset_roll is None:
        labels = sample.get("labels")
        labels_tensor = labels if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        rolls = _build_rolls_from_labels(labels_tensor, args.start_sec, args.seconds, hop_seconds, note_min, note_max, dilate)
        onset_roll = rolls["onset_roll"]
        offset_roll = rolls["offset_roll"]
    else:
        onset_roll = onset_roll[start_frame : start_frame + window_frames]
        offset_roll = offset_roll[start_frame : start_frame + window_frames]

    roll = onset_roll if args.head == "onset" else offset_roll
    y_gt = roll.sum(dim=1)
    sum_gt = y_gt.sum().item()
    if sum_gt <= 0:
        raise SystemExit("No GT events in this crop; pick another clip/window.")
    nz_gt = int((y_gt > 0).sum().item())
    std_gt = float(y_gt.float().std().item())

    model = build_model(cfg_local).eval()
    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    ckpt_state = ckpt.get("model", ckpt)
    model.load_state_dict(ckpt_state, strict=False)
    with torch.no_grad():
        inp = clip.unsqueeze(0)
        out = model(inp)
    logits = out[f"{args.head}_logits"][0]
    if logits.ndim == 3:
        logits = logits.mean(dim=1)
    if args.use_logits:
        y_pred = logits.mean(dim=1)
    else:
        y_pred = torch.sigmoid(logits).mean(dim=1)

    min_p = float(y_pred.min().item())
    max_p = float(y_pred.max().item())
    std_p = float(y_pred.std().item())
    if std_p == 0.0 or torch.isnan(y_pred).any():
        raise SystemExit("Predictions are constant/NaN; check model/ckpt/head wiring.")

    print(f"[CLIP] path={clip_path} idx={clip_idx} frames={window_frames} start_frame={start_frame}")
    print(f"[STATS] y_gt: sum={sum_gt:.4f} nonzero_frames={nz_gt} std={std_gt:.6f}")
    print(f"[STATS] y_pred: min={min_p:.6f} max={max_p:.6f} std={std_p:.6f}")

    if args.use_logits:
        thr = float(torch.quantile(y_pred, 0.95).item())
    else:
        thr = float(args.prob_threshold)

    rows = []
    for d in range(-args.max_shift, args.max_shift + 1):
        pred_al, gt_al = _align(y_pred, y_gt, d)
        if pred_al.numel() == 0 or gt_al.numel() == 0:
            corr = float("nan")
            f1 = float("nan")
        else:
            corr = _safe_corr(pred_al, gt_al)
            pred_bin = pred_al >= thr
            gt_bin = gt_al > 0
            tp = (pred_bin & gt_bin).sum().item()
            fp = (pred_bin & ~gt_bin).sum().item()
            fn = (~pred_bin & gt_bin).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append({"delta": d, "corr": corr, "f1": f1})

    best = max(rows, key=lambda r: (r["f1"], r["corr"]))
    print(f"Best shift by F1: Δ={best['delta']} f1={best['f1']:.4f} corr={best['corr']:.4f}")
    for r in rows:
        print(f"Δ={r['delta']:>2d}  corr={r['corr']:.4f}  f1={r['f1']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())