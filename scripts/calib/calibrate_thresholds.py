#!/usr/bin/env python3
"""Purpose:
    Evaluate onset and offset predictions to determine calibrated logit or
    probability thresholds and produce reliability diagnostics.

Key Functions/Classes:
    - _collect(): Runs the model across a dataloader to gather logits,
      probabilities, and aligned targets.
    - _compute_metrics(): Sweeps thresholds to compute F1 scores, prediction
      rates, expected calibration error, and Brier scores.
    - main(): Command-line entry point that loads checkpoints, runs evaluation,
      and writes calibration summaries.

CLI:
    Invoke ``python scripts/calibrate_thresholds.py --ckpt <path> --split val``
    with optional ``--max-clips`` and ``--frames`` overrides to adjust dataset
    size.
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Repo setup so we can import from src/
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils import load_config, align_pitch_dim
from data import make_dataloader
from models import build_model


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling."""
    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)

def _binary_f1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Binary F1 score for tensors in {0,1}."""
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)

def _reliability_curve(probs: np.ndarray, targets: np.ndarray, n_bins: int, name: str):
    """Compute reliability data and save a diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    bin_sums = np.bincount(bin_ids, weights=probs, minlength=n_bins)
    bin_true = np.bincount(bin_ids, weights=targets, minlength=n_bins)
    bin_cnts = np.bincount(bin_ids, minlength=n_bins)

    nonzero = bin_cnts > 0
    prob_mean = np.zeros(n_bins)
    true_mean = np.zeros(n_bins)
    prob_mean[nonzero] = bin_sums[nonzero] / bin_cnts[nonzero]
    true_mean[nonzero] = bin_true[nonzero] / bin_cnts[nonzero]

    ece = np.sum(np.abs(true_mean - prob_mean) * bin_cnts / probs.size)
    brier = np.mean((probs - targets) ** 2)

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_mean[nonzero], true_mean[nonzero], marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"{name.capitalize()} reliability (ECE {ece:.3f}, Brier {brier:.3f})")
    plt.tight_layout()
    plt.savefig(f"calib_reliability_{name}.png")
    plt.close()
    return ece, brier

def _collect(model, loader):
    onset_logits, offset_logits = [], []
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["video"]
            out = model(x)

            on_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            off_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")

            onset_logits.append(on_logits.cpu())
            offset_logits.append(off_logits.cpu())
            onset_probs.append(torch.sigmoid(on_logits).cpu())
            offset_probs.append(torch.sigmoid(off_logits).cpu())
            onset_tgts.append(batch["onset_roll"].float().cpu())
            offset_tgts.append(batch["offset_roll"].float().cpu())

    onset_logits = torch.cat(onset_logits, dim=0)
    offset_logits = torch.cat(offset_logits, dim=0)
    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    T_logits, P_logits = onset_logits.shape[1], onset_logits.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = _pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = _pool_roll_BT(offset_tgts, T_logits)
    onset_tgts = align_pitch_dim(onset_logits, onset_tgts, "onset")
    offset_tgts = align_pitch_dim(offset_logits, offset_tgts, "offset")

    onset_tgts = (onset_tgts > 0).float()
    offset_tgts = (offset_tgts > 0).float()

    return onset_logits, offset_logits, onset_probs, offset_probs, onset_tgts, offset_tgts

def _compute_metrics(logits: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor, name: str):
    logits_flat = logits.reshape(-1)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)

    logit_grid = torch.arange(-4.0, 2.0 + 1e-9, 0.05)
    best_logit, best_f1_logit, pred_rate_logit = -4.0, -1.0, 0.0
    for thr in logit_grid:
        pred = (logits_flat >= thr).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > best_f1_logit:
            best_f1_logit = f1
            best_logit = thr.item()
            pred_rate_logit = pred.mean().item()

    prob_grid = torch.arange(0.01, 0.99 + 1e-9, 0.01)
    best_prob, best_f1_prob, pred_rate_prob = 0.5, -1.0, 0.0
    for thr in prob_grid:
        pred = (probs_flat >= thr).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > best_f1_prob:
            best_f1_prob = f1
            best_prob = thr.item()
            pred_rate_prob = pred.mean().item()

    pos_rate = targets_flat.mean().item()
    ece, brier = _reliability_curve(probs_flat.numpy(), targets_flat.numpy(), 10, name)

    return {
        "best_logit": best_logit,
        "best_prob": best_prob,
        "f1_logit": best_f1_logit,
        "f1_prob": best_f1_prob,
        "pred_rate_logit": pred_rate_logit,
        "pred_rate_prob": pred_rate_prob,
        "pos_rate": pos_rate,
        "ece": ece,
        "brier": brier,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    if args.max_clips is not None:
        cfg["dataset"]["max_clips"] = args.max_clips
    if args.frames is not None:
        cfg["dataset"]["frames"] = args.frames
    decode_fps = float(cfg["dataset"].get("decode_fps", 1.0))
    hop_seconds = float(cfg["dataset"].get("hop_seconds", 1.0 / decode_fps))
    split = args.split or cfg["dataset"].get("split_val", "val")
    loader = make_dataloader(cfg, split=split)
    if isinstance(loader, dict):
        loader = loader.get(split) or next(iter(loader.values()))
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    onset_logits, offset_logits, onset_probs, offset_probs, onset_tgts, offset_tgts = _collect(model, loader)

    onset_stats = _compute_metrics(onset_logits, onset_probs, onset_tgts, "onset")
    offset_stats = _compute_metrics(offset_logits, offset_probs, offset_tgts, "offset")

    with open("calibration.json", "w") as f:
        json.dump({
            "onset": {"best_logit": onset_stats["best_logit"], "best_prob": onset_stats["best_prob"]},
            "offset": {"best_logit": offset_stats["best_logit"], "best_prob": offset_stats["best_prob"]},
        }, f, indent=2)

    for name, stats in [("Onset", onset_stats), ("Offset", offset_stats)]:
        print(
            f"{name}: pos_rate={stats['pos_rate']:.4f} | "
            f"best_logit={stats['best_logit']:.2f} (pred_rate={stats['pred_rate_logit']:.4f}, F1={stats['f1_logit']:.3f}) | "
            f"best_prob={stats['best_prob']:.2f} (pred_rate={stats['pred_rate_prob']:.4f}, F1={stats['f1_prob']:.3f})"
        )

if __name__ == "__main__":
    main()

