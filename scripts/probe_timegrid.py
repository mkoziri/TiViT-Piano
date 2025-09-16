#!/usr/bin/env python3
"""Purpose:
    Diagnose the dataset time grid by examining the relationship between frame
    indices, seconds, and raw label timestamps for a single clip.

Key Functions/Classes:
    - main(): Loads configuration, fetches one batch from the test split, and
      prints time-grid samples along with label mappings.

CLI:
    Execute ``python scripts/probe_timegrid.py``.  The script has no arguments
    and prints diagnostic information to stdout.
"""

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config
from utils.time_grid import frame_to_sec, sec_to_frame
from data import make_dataloader
import torch

def main():
    cfg = load_config("configs/config.yaml")
    # Use exactly one clip for deterministic diagnostics
    cfg["dataset"]["max_clips"] = 1
    cfg["dataset"]["shuffle"] = False

    loader = make_dataloader(cfg, split="test")
    batch = next(iter(loader))

    T = batch["video"].shape[1]
    decode_fps = float(cfg["dataset"].get("decode_fps", 30.0))
    hop_seconds = float(cfg["dataset"].get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))
    start_idx = 0  # test split uses deterministic start

    t_vals = [frame_to_sec(start_idx + k * stride, 1.0 / decode_fps) for k in range(T)]
    print("First 10 t_k values:")
    print([round(t, 6) for t in t_vals[:10]])

    labels = batch.get("labels", [torch.zeros(0, 3)])
    labels = labels[0]
    if labels.numel() > 0:
        print("\nRaw label onset times and mapped k:")
        onset_times = labels[:5, 0]
        ks = sec_to_frame(onset_times, hop_seconds)
        for t, k in zip(onset_times.tolist(), ks.tolist()):
            print(f"  onset={t:.3f}s -> k={int(k)}")
    else:
        print("\nNo raw labels available")

    onset_roll = batch.get("onset_roll")
    if onset_roll is not None:
        onset_roll = onset_roll[0]  # (T,P)
        print("\nTable (first 5 frames): k, t_k, has_onset_any_pitch")
        for k in range(min(5, T)):
            has_on = bool(onset_roll[k].any())
            print(f"{k:2d}  {t_vals[k]:6.3f}  {int(has_on)}")
    else:
        print("\nNo onset_roll in batch")

if __name__ == "__main__":
    main()
