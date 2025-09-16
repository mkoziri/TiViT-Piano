#!/usr/bin/env python3
"""Purpose:
    Analyze onset and offset head behavior across a few batches by logging
    probability statistics, exporting CSV/plots, and optionally loading
    checkpoints on CPU or GPU.

Key Functions/Classes:
    - pick_input_tensor(): Finds the video tensor within heterogeneous batch
      structures produced by custom collates.
    - find_logits(): Extracts onset/offset logits from potentially nested model
      outputs.
    - summarize(): Computes descriptive statistics used for console and CSV
      reporting.

CLI:
    Example usage::

        python scripts/diagnose_onset_offsets.py --config configs/config.yaml \
            --ckpt checkpoints/tivit_best.pt --split val --batches 5

    Additional options include ``--device`` for CPU/GPU selection.
"""

import argparse, os, sys, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- CLI ---
ap = argparse.ArgumentParser("Diagnose onset/offset heads on a few batches")
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
ap.add_argument("--split", default="train", choices=["train","val","test"])
ap.add_argument("--batches", type=int, default=5)
ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
args = ap.parse_args()

# --- Repo roots / imports (per your structure) ---
# tivit/
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.config import load_config                           # src/utils/config.py
from models.factory import build_model                         # src/models/factory.py
from data.omaps_dataset import make_dataloader                 # src/data/omaps_dataset.py

# --- Config + device ---
cfg = load_config(args.config)

if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Build dataloader on requested split ---
# We’ll use the loader the same way your eval does (split argument is honored by make_dataloader)
loader = make_dataloader(cfg, split=args.split, drop_last=False)

# --- Build model + load checkpoint (robust to formats) ---
model = build_model(cfg).to(device).eval()

ckpt_path = Path(args.ckpt)
if not ckpt_path.exists():
    print(f"[ERROR] checkpoint not found: {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(str(ckpt_path), map_location=device)
state = ckpt.get("model", ckpt)  # accept either {"model": sd} or raw sd
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:   print("[load_state_dict] missing keys:", missing)
if unexpected:print("[load_state_dict] unexpected keys:", unexpected)
if "epoch" in ckpt:
    print(f"[ckpt] epoch: {ckpt['epoch']}")

# --- Helpers ---
def pick_input_tensor(batch):
    """Try common keys your collate might use."""
    if isinstance(batch, dict):
        for k in ("video","frames","images","x"):
            if k in batch and torch.is_tensor(batch[k]):
                return batch[k]
        # fallback: first tensor value
        for v in batch.values():
            if torch.is_tensor(v):
                return v
    # If batch is a tuple/list, try first tensor element
    if isinstance(batch, (list,tuple)):
        for v in batch:
            if torch.is_tensor(v):
                return v
    return None

def find_logits(output):
    """
    Return (onset_logits, offset_logits) from model output dict.
    Tries common key patterns.
    """
    onset = None; offset = None
    if isinstance(output, dict):
        for k, v in output.items():
            lk = k.lower()
            if onset is None and "onset" in lk and "logit" in lk:
                onset = v
            if offset is None and "offset" in lk and "logit" in lk:
                offset = v
        # some models nest heads
        if (onset is None or offset is None) and "heads" in output and isinstance(output["heads"], dict):
            for k, v in output["heads"].items():
                lk = k.lower()
                if onset is None and "onset" in lk and "logit" in lk:
                    onset = v
                if offset is None and "offset" in lk and "logit" in lk:
                    offset = v
    return onset, offset

def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    q = np.percentile
    return dict(
        mean=float(arr.mean()), std=float(arr.std()),
        min=float(arr.min()), p05=float(q(arr,5)), p25=float(q(arr,25)),
        p50=float(q(arr,50)), p75=float(q(arr,75)), p95=float(q(arr,95)),
        max=float(arr.max())
    )

# --- Iterate a few batches ---
reports_dir = ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
csv_path = reports_dir / "diagnostics_onset_offset.csv"

rows = []
all_on, all_off = [], []

model.eval()
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= args.batches: break
        x = pick_input_tensor(batch)
        if x is None:
            print(f"[WARN] batch {i}: could not find input tensor; batch keys={list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
            continue
        x = x.to(device, non_blocking=True)

        out = model(x)
        onset_logits, offset_logits = find_logits(out)
        if onset_logits is None:
            print("[ERROR] onset logits not found in model output keys:", list(out.keys()) if isinstance(out, dict) else type(out))
            sys.exit(1)

        on_prob = torch.sigmoid(onset_logits).detach().cpu().numpy()
        on_stats = summarize(on_prob)
        print(f"[batch {i}] onset mean={on_stats['mean']:.3f} p25={on_stats['p25']:.3f} p50={on_stats['p50']:.3f} p75={on_stats['p75']:.3f}")
        all_on.append(on_prob)

        off_stats = {}
        if offset_logits is not None:
            off_prob = torch.sigmoid(offset_logits).detach().cpu().numpy()
            off_stats = summarize(off_prob)
            all_off.append(off_prob)
            print(f"          offset mean={off_stats['mean']:.3f} p25={off_stats['p25']:.3f} p50={off_stats['p50']:.3f} p75={off_stats['p75']:.3f}")

        rows.append({"batch": i, **{f"on_{k}": v for k,v in on_stats.items()}, **{f"off_{k}": v for k,v in off_stats.items()}})

# --- Save CSV ---
if rows:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys(), key=lambda x: (x!="batch", x)))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] batch stats → {csv_path}")

# --- Histograms (optional) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_hist(probs_list, title, out_name):
        if not probs_list: return
        arr = np.concatenate([a.reshape(-1) for a in probs_list], axis=0)
        plt.figure()
        plt.hist(arr, bins=50)
        plt.xlabel("probability"); plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(reports_dir / out_name, dpi=160)
        plt.close()

    plot_hist(all_on,  "Onset sigmoid probabilities",  "diagnostics_onset_prob.png")
    plot_hist(all_off, "Offset sigmoid probabilities", "diagnostics_offset_prob.png")
    print(f"[OK] histograms → {reports_dir/'diagnostics_onset_prob.png'}"
          + (f", {reports_dir/'diagnostics_offset_prob.png'}" if all_off else ""))
except Exception as e:
    print("[WARN] could not save histograms (matplotlib missing?)", e)

# --- Overall printout ---
if all_on:
    arr = np.concatenate([a.reshape(-1) for a in all_on], axis=0)
    s = summarize(arr)
    print("\n[OVERALL onset probs] mean={mean:.3f} std={std:.3f} min={min:.3f} "
          "p05={p05:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p95={p95:.3f} max={max:.3f}".format(**s))
if all_off:
    arr = np.concatenate([a.reshape(-1) for a in all_off], axis=0)
    s = summarize(arr)
    print("[OVERALL offset probs] mean={mean:.3f} std={std:.3f} min={min:.3f} "
          "p05={p05:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p95={p95:.3f} max={max:.3f}".format(**s))

