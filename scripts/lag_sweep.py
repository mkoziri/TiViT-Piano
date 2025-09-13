#!/usr/bin/env python3
"""Evaluate temporal alignment between predicted and true onsets.

Aggregates onsets over pitch to obtain 1-D sequences ``y_gt[k]`` and
``y_pred[k]``.  Sweeps over integer shifts ``Δ`` in ``[-5, 5]`` and computes
correlation and F1 for each shift.  Prints the best shift for each metric and
optionally saves a CSV table.
"""
import argparse
import csv
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

import numpy as np
import torch
from utils import load_config
from data import make_dataloader
from models import build_model


def _align(a: torch.Tensor, b: torch.Tensor, delta: int):
     """Align two 1-D tensors according to shift ``delta``.

    ``a`` and ``b`` may have different lengths.  After applying the shift, this
    function crops both tensors to the minimum common length so that they can be
    compared element-wise.
    """
    if delta > 0:
        a = a[delta:]
        b = b[:-delta]
    elif delta < 0:
        a = a[:delta]
        b = b[-delta:]

    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]



def main():
    ap = argparse.ArgumentParser(description="Lag sweep between predictions and ground truth")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to save CSV table")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    cfg["dataset"]["max_clips"] = 1
    cfg["dataset"]["shuffle"] = False

    model = build_model(cfg).eval()
    loader = make_dataloader(cfg, split="test")
    batch = next(iter(loader))
    x = batch["video"]
    onset_roll = batch["onset_roll"][0]  # (T,P)

    with torch.no_grad():
        out = model(x)
    onset_logits = out["onset_logits"][0]  # (T,P)

    y_gt = onset_roll.sum(dim=1)  # (T,)
    y_pred = onset_logits.sigmoid().sum(dim=1)  # (T,)

    rows = []
    for d in range(-5, 6):
        pred, gt = _align(y_pred, y_gt, d)
        if pred.numel() < 2:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(pred.cpu().numpy(), gt.cpu().numpy())[0, 1])
        pred_bin = (pred >= 0.5)
        gt_bin = (gt > 0)
        tp = (pred_bin & gt_bin).sum().item()
        fp = (pred_bin & ~gt_bin).sum().item()
        fn = (~pred_bin & gt_bin).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append({"delta": d, "corr": corr, "f1": f1})

    best_corr = max(rows, key=lambda r: r["corr"] if not np.isnan(r["corr"]) else -np.inf)
    best_f1 = max(rows, key=lambda r: r["f1"])

    print(f"Best correlation: Δ={best_corr['delta']} corr={best_corr['corr']:.4f}")
    print(f"Best F1:          Δ={best_f1['delta']} f1={best_f1['f1']:.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["delta", "corr", "f1"])
            writer.writeheader()
            writer.writerows(rows)

    for r in rows:
        print(f"Δ={r['delta']:>2d}  corr={r['corr']:.4f}  f1={r['f1']:.4f}")

if __name__ == "__main__":
    main()
