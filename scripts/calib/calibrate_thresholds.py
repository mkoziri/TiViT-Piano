#!/usr/bin/env python3
"""Purpose:
    Evaluate onset and offset predictions to determine calibrated logit or
    probability thresholds, emit reliability diagnostics, and continuously write
    partial results to ``calibration.json`` during long sweeps.

Key Functions/Classes:
    - _collect(): Runs the model across a dataloader to gather logits,
      probabilities, and aligned targets while checkpointing partial metrics.
    - _compute_metrics(): Sweeps thresholds to compute F1 scores, prediction
      rates, expected calibration error, and Brier scores.
    - main(): Command-line entry point that loads checkpoints, runs evaluation,
      and writes calibration summaries.

CLI:
    Invoke ``python scripts/calib/calibrate_thresholds.py --ckpt <path> --split val``
    with optional ``--max-clips``/``--frames`` overrides or ``--timeout-mins`` to
    stop early while preserving partial statistics.
"""

import sys, json, argparse, time
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

PARTIAL_WRITE_INTERVAL = 32


def _print_progress(processed: int, total: int) -> None:
    if total > 0:
        pct = 100.0 * processed / total
        msg = f"[calib] processed {processed}/{total} clips ({pct:5.1f}%)"
    else:
        msg = f"[calib] processed {processed} clips"
    print(msg, flush=True)


def _write_partial_calibration(
    onset_logits_list,
    offset_logits_list,
    onset_probs_list,
    offset_probs_list,
    onset_tgts_list,
    offset_tgts_list,
) -> None:
    if not onset_logits_list:
        return
    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs_list, dim=0)
    offset_probs = torch.cat(offset_probs_list, dim=0)
    onset_tgts = torch.cat(onset_tgts_list, dim=0)
    offset_tgts = torch.cat(offset_tgts_list, dim=0)
    onset_stats = _compute_metrics(onset_logits, onset_probs, onset_tgts, "onset")
    offset_stats = _compute_metrics(offset_logits, offset_probs, offset_tgts, "offset")
    with open("calibration.json", "w") as f:
        json.dump(
            {
                "onset": {"best_logit": onset_stats["best_logit"], "best_prob": onset_stats["best_prob"]},
                "offset": {"best_logit": offset_stats["best_logit"], "best_prob": offset_stats["best_prob"]},
            },
            f,
            indent=2,
        )


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

def _collect(model, loader, max_clips: int, timeout_secs: float):
    onset_logits_list, offset_logits_list = [], []
    onset_probs_list, offset_probs_list = [], []
    onset_tgts_list, offset_tgts_list = [], []

    processed = 0
    timeout_hit = False
    target_total = max(0, max_clips)
    start_time = time.monotonic()

    with torch.no_grad():
        for batch in loader:
            remaining = None if target_total == 0 else target_total - processed
            if remaining is not None and remaining <= 0:
                break

            x = batch["video"]
            batch_size = x.shape[0]
            if remaining is not None and batch_size > remaining:
                idx = slice(0, remaining)
            else:
                idx = slice(None)

            x = x[idx]
            out = model(x)

            on_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            off_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")

            onset_probs = torch.sigmoid(on_logits)
            offset_probs = torch.sigmoid(off_logits)

            onset_roll = batch["onset_roll"][idx].float()
            offset_roll = batch["offset_roll"][idx].float()

            T_logits = on_logits.shape[1]
            if onset_roll.shape[1] != T_logits:
                onset_roll = _pool_roll_BT(onset_roll, T_logits)
                offset_roll = _pool_roll_BT(offset_roll, T_logits)

            onset_roll = align_pitch_dim(on_logits, onset_roll, "onset")
            offset_roll = align_pitch_dim(off_logits, offset_roll, "offset")

            onset_roll = (onset_roll > 0).float()
            offset_roll = (offset_roll > 0).float()

            onset_logits_list.append(on_logits.cpu())
            offset_logits_list.append(off_logits.cpu())
            onset_probs_list.append(onset_probs.cpu())
            offset_probs_list.append(offset_probs.cpu())
            onset_tgts_list.append(onset_roll.cpu())
            offset_tgts_list.append(offset_roll.cpu())

            processed += x.shape[0]
            _print_progress(processed, target_total)

            if processed % PARTIAL_WRITE_INTERVAL == 0 or (target_total and processed >= target_total):
                _write_partial_calibration(
                    onset_logits_list,
                    offset_logits_list,
                    onset_probs_list,
                    offset_probs_list,
                    onset_tgts_list,
                    offset_tgts_list,
                )

            if timeout_secs and time.monotonic() - start_time >= timeout_secs:
                timeout_hit = True
                _write_partial_calibration(
                    onset_logits_list,
                    offset_logits_list,
                    onset_probs_list,
                    offset_probs_list,
                    onset_tgts_list,
                    offset_tgts_list,
                )
                break

    if not onset_logits_list:
        raise RuntimeError("No clips processed during calibration")

    _write_partial_calibration(
        onset_logits_list,
        offset_logits_list,
        onset_probs_list,
        offset_probs_list,
        onset_tgts_list,
        offset_tgts_list,
    )

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs_list, dim=0)
    offset_probs = torch.cat(offset_probs_list, dim=0)
    onset_tgts = torch.cat(onset_tgts_list, dim=0)
    offset_tgts = torch.cat(offset_tgts_list, dim=0)

    return (
        onset_logits,
        offset_logits,
        onset_probs,
        offset_probs,
        onset_tgts,
        offset_tgts,
        processed,
        timeout_hit,
    )

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
    ap.add_argument("--max-clips", type=int, default=400, help="Limit number of clips evaluated (default: 400)")
    ap.add_argument("--frames", type=int, default=64, help="Frames per clip during calibration (default: 64)")
    ap.add_argument(
        "--timeout-mins",
        type=float,
        help="Optional timeout in minutes; stops early while keeping partial stats",
    )
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    if args.max_clips:
        cfg["dataset"]["max_clips"] = int(args.max_clips)
    if args.frames:
        cfg["dataset"]["frames"] = int(args.frames)
    decode_fps = float(cfg["dataset"].get("decode_fps", 1.0))
    hop_seconds = float(cfg["dataset"].get("hop_seconds", 1.0 / decode_fps))
    split = args.split or cfg["dataset"].get("split_val") or cfg["dataset"].get("split") or "val"
    loader = make_dataloader(cfg, split=split)
    if isinstance(loader, dict):
        loader = loader.get(split) or next(iter(loader.values()))
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    timeout_secs = float(args.timeout_mins * 60.0) if args.timeout_mins else 0.0
    (
        onset_logits,
        offset_logits,
        onset_probs,
        offset_probs,
        onset_tgts,
        offset_tgts,
        processed_clips,
        timeout_hit,
    ) = _collect(model, loader, int(args.max_clips or 0), timeout_secs)

    if timeout_hit:
        print(f"[calib] timeout reached after {processed_clips} clips", flush=True)

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

