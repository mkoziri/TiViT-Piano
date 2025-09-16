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
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config  # noqa: E402
from models import build_model  # noqa: E402
from data.omaps_dataset import (  # noqa: E402
    OMAPSDataset,
    _load_clip_with_random_start,
    _tile_vertical,
    _read_txt_events,
    _build_frame_targets,
)


def _align(a: torch.Tensor, b: torch.Tensor, delta: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align two 1-D tensors according to shift ``delta``.

    After shifting, tensors are cropped to their minimum common length so they
    can be compared element-wise.
    """

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


def _find_clip(dataset: OMAPSDataset, key: str) -> Path:
    """Return the path to the first clip whose name contains ``key``."""

    for p in dataset.videos:
        if key in p.name or key in p.stem or key in str(p):
            return p
    sys.exit(f"Clip '{key}' not found in split '{dataset.split}'.")


def _load_window(
    path: Path,
    decode_fps: float,
    hop_seconds: float,
    start_sec: float,
    seconds: float,
    resize: Tuple[int, int],
    tiles: int,
    channels: int,
) -> torch.Tensor:
    """Load a window ``[start_sec, start_sec+seconds)`` from ``path``.

    Returns ``T,tiles,C,H,W`` tensor on CPU with values in [0,1].
    """

    stride = int(round(hop_seconds * decode_fps))
    frames_needed = int(np.ceil(seconds / hop_seconds))
    frames_total = int(np.ceil((start_sec + seconds) / hop_seconds)) + 1

    clip_full, _ = _load_clip_with_random_start(
        path=path,
        frames=frames_total,
        stride=stride,
        resize_hw=resize,
        channels=channels,
        training=False,
        decode_fps=decode_fps,
    )

    clip_full = _tile_vertical(clip_full, tiles)  # T,tiles,C,H,Wt

    start_frame = int(np.floor(start_sec / hop_seconds))
    clip = clip_full[start_frame : start_frame + frames_needed]
    return clip
    
    
def main():  # noqa: C901 - command-line tool
    ap = argparse.ArgumentParser(description="Lag sweep between predictions and ground truth")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True, help="Dataset split")
    ap.add_argument("--clip", required=True, help="Clip id or substring")
    ap.add_argument("--seconds", type=float, required=True, help="Window length in seconds")
    ap.add_argument("--start_sec", type=float, default=0.0, help="Start time of window in seconds")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--head", choices=["onset", "offset"], required=True, help="Model head to use")
    ap.add_argument(
        "--use_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Operate on logits (default) or probabilities",
    )
    ap.add_argument("--prob_threshold", type=float, default=0.2, help="Threshold when using probabilities")
    ap.add_argument("--max_shift", type=int, default=5, help="Maximum +/- frame shift for sweep")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    
    time_cfg = cfg.get("time", cfg.get("dataset", {}))
    decode_fps = float(time_cfg.get("decode_fps"))
    hop_frames = time_cfg.get("hop_frames")
    if hop_frames is not None:
        hop_seconds = float(hop_frames) / decode_fps
    else:
        hop_seconds = float(time_cfg.get("hop_seconds"))
        hop_frames = int(round(hop_seconds * decode_fps))
    const_off = float(time_cfg.get("constant_frame_offset", 0))
    print(
        f"[GRID] decode_fps={decode_fps} "
        f"{'hop_frames=' + str(hop_frames) if hop_frames is not None else 'hop_seconds=' + str(hop_seconds)} "
        f"constant_frame_offset={const_off}",
        flush=True,
    )

    # instantiate dataset (to honour requirement and reuse label config)
    dcfg = cfg["dataset"]
    dataset = OMAPSDataset(
        root_dir=dcfg.get("root_dir"),
        split=args.split,
        frames=1,
        stride=hop_frames,
        resize=tuple(dcfg.get("resize", [224, 224])),
        tiles=int(dcfg.get("tiles", 3)),
        channels=int(dcfg.get("channels", 3)),
        normalize=bool(dcfg.get("normalize", True)),
        decode_fps=decode_fps,
    )
    dataset.annotations_root = dcfg.get("annotations_root")
    dataset.label_format = dcfg.get("label_format", "txt")
    dataset.label_targets = dcfg.get("label_targets", ["pitch", "onset", "offset", "hand", "clef"])
    dataset.require_labels = bool(dcfg.get("require_labels", False))
    dataset.frame_targets_cfg = dcfg.get("frame_targets", {})
    dataset.max_clips = dcfg.get("max_clips", None)

    clip_path = _find_clip(dataset, args.clip)

    # load video window
    clip = _load_window(
        path=clip_path,
        decode_fps=decode_fps,
        hop_seconds=hop_seconds,
        start_sec=args.start_sec,
        seconds=args.seconds,
        resize=tuple(dcfg.get("resize", [224, 224])),
        tiles=int(dcfg.get("tiles", 3)),
        channels=int(dcfg.get("channels", 3)),
    )
    T = clip.shape[0]
    x = clip.unsqueeze(0)  # B, T, tiles, C, H, W

    # load labels and build frame targets
    ann_path = None
    if dataset.annotations_root:
        cand = Path(dataset.annotations_root).expanduser() / f"{clip_path.stem}.txt"
        if cand.exists():
            ann_path = cand
    if ann_path is None:
        cand = clip_path.with_suffix(".txt")
        if cand.exists():
            ann_path = cand
    if ann_path is None:
        sys.exit(f"No GT annotation found for {clip_path}")
    labels = _read_txt_events(ann_path)  # (N,3) or (0,3)
    labels = labels.clone()
    labels[:, 0:2] -= args.start_sec

    ft_cfg = dataset.frame_targets_cfg or {}
    ft = _build_frame_targets(
        labels=labels,
        T=T,
        stride=hop_frames,
        fps=decode_fps,
        note_min=int(ft_cfg.get("note_min", 21)),
        note_max=int(ft_cfg.get("note_max", 108)),
        tol=float(ft_cfg.get("tolerance", 0.025)),
        fill_mode=str(ft_cfg.get("fill_mode", "overlap")),
        hand_from_pitch=bool(ft_cfg.get("hand_from_pitch", True)),
        clef_thresholds=tuple(ft_cfg.get("clef_thresholds", [60, 64])),
        dilate_active_frames=int(ft_cfg.get("dilate_active_frames", 0)),
        targets_sparse=bool(ft_cfg.get("targets_sparse", False)),
    )

    roll = ft["onset_roll" if args.head == "onset" else "offset_roll"]
    y_gt = roll.sum(dim=1)
    sum_gt = y_gt.sum().item()
    if sum_gt <= 0:
        sys.exit("No GT events in this crop; pick another clip/window.")
    nz_gt = int((y_gt > 0).sum().item())
    std_gt = float(y_gt.float().std().item())

    # model and predictions
    model = build_model(cfg).eval()
    state = torch.load(Path(args.ckpt), map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        out = model(x)
    logits = out[f"{args.head}_logits"][0]  # (T,P)
    if args.use_logits:
        y_pred = logits.mean(dim=1)
    else:
        y_pred = torch.sigmoid(logits).mean(dim=1)

    min_p = float(y_pred.min().item())
    max_p = float(y_pred.max().item())
    std_p = float(y_pred.std().item())
    if std_p == 0.0 or torch.isnan(y_pred).any():
        sys.exit("Predictions are constant/NaN; check model/ckpt/head wiring.")

    print(
        f"[STATS] y_gt: sum={sum_gt:.4f} nonzero_frames={nz_gt} std={std_gt:.6f}"
    )
    print(
        f"[STATS] y_pred: min={min_p:.6f} max={max_p:.6f} std={std_p:.6f}"
    )

    # thresholds for F1
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
    print(
        f"Best shift by F1: Δ={best['delta']} f1={best['f1']:.4f} corr={best['corr']:.4f}"
    )
    for r in rows:
        print(f"Δ={r['delta']:>2d}  corr={r['corr']:.4f}  f1={r['f1']:.4f}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
