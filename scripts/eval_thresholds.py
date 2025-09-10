#!/usr/bin/env python3
import sys, torch
import torch.nn.functional as F
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config
from data import make_dataloader
from models import build_model


# Default grid of thresholds.  We parse this argument manually so that callers
# can provide values in a comma-separated form (e.g., ``-3,-2.5,-2``) without
# needing to escape leading minus signs.  Argparse would otherwise interpret
# such tokens as new options.
DEFAULT_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]


def _parse_thresholds(argv):
    """Extract ``--thresholds`` from ``argv`` allowing comma/space separation."""
    for i, arg in enumerate(list(argv)):
        if arg.startswith("--thresholds"):
            if arg == "--thresholds":
                j = i + 1
                vals = []
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                if not vals:
                    raise ValueError("--thresholds expects at least one value")
                del argv[i:j]
                arg_str = " ".join(vals)
            else:  # handle --thresholds=... form
                arg_str = arg.split("=", 1)[1]
                del argv[i]
            arg_str = arg_str.replace(",", " ")
            return [float(v) for v in arg_str.split() if v]
    return DEFAULT_THRESHOLDS.copy()


def _binary_f1(pred, target, eps=1e-8):
    """Binary F1 score for tensors in {0,1}.

    Returns None if both pred and target are all zeros."""
    if target.sum().item() == 0 and pred.sum().item() == 0:
        return None
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)
   
   
def _pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling.

    This mirrors the alignment logic used during training so that frame-level
    targets match the model's temporal resolution ``Tprime``.
    """
    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)
    
def main():
    import argparse

    argv = sys.argv[1:]
    try:
        thresholds = _parse_thresholds(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument(
        "--thresholds",
        metavar="T",
        nargs="*",
        help="Threshold values (comma or space-separated)",
    )
    # Optional temperature and bias parameters for logit calibration
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Scale logits by this temperature before sigmoid; >1 softens predictions",
    )
    ap.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Additive bias applied to logits before sigmoid",
    )
    args = ap.parse_args(argv)
    args.thresholds = thresholds

    cfg = load_config("configs/config.yaml")
    # lock the SAME positive window you trained on
    cfg["dataset"]["split_val"] = "train"
    cfg["dataset"]["split_train"] = "train"
    cfg["dataset"]["max_clips"] = 1
    cfg["dataset"]["frames"] = 96
    cfg["dataset"]["stride"] = 1
    cfg["dataset"]["shuffle"] = False
    # keep frames=96 etc as you used in Stage-A

    # build val loader
    val_loader = make_dataloader(cfg, split="val")
    if isinstance(val_loader, dict): val_loader = val_loader.get("val", next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)): val_loader = val_loader[0]

    # load model + ckpt
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    # run model once to collect probabilities and targets
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []
@@ -111,47 +172,47 @@ def main():
    T_logits, P_logits = onset_probs.shape[1], onset_probs.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = _pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = _pool_roll_BT(offset_tgts, T_logits)
    if onset_tgts.shape[2] != P_logits:
        if onset_tgts.shape[2] == 128 and P_logits == 88:
            start = 21  # map MIDI 0-127 to piano range 21-108
            onset_tgts = onset_tgts[..., start : start + P_logits]
            offset_tgts = offset_tgts[..., start : start + P_logits]
        else:
            raise ValueError(
                f"Target pitch dim {onset_tgts.shape[2]} does not match model {P_logits}"
            )
    # diagnostic prints
    print(f"[OVERALL onset probs] mean={onset_probs.mean():.3f} min={onset_probs.min():.3f} max={onset_probs.max():.3f}")
    print(f"[OVERALL offset probs] mean={offset_probs.mean():.3f} min={offset_probs.min():.3f} max={offset_probs.max():.3f}")

    # Use all key/time positions rather than collapsing with ``any``.
    # Collapsing across the note dimension causes the predicted rate to be
    # either 0 or 1 for a clip, which in turn makes F1-threshold sweeps
    # uninformative.  Instead we compute metrics over the full pianoroll so
    # that the positive rate varies smoothly with the threshold.
    onset_true_bin = (onset_tgts > 0).float()
    offset_true_bin = (offset_tgts > 0).float()
    
    thrs = list(args.thresholds)
    print("Threshold\tonset_f1\toffset_f1\tonset_pred_rate\tonset_pos_rate\ttotal")
    for t in thrs:
        # Threshold probabilities at the given level and evaluate at the
        # pianoroll level (B, T, 88).  This yields more nuanced predicted
        # positive rates and F1 scores, which are useful for calibration.
        onset_pred_bin = (onset_probs >= t).float()
        offset_pred_bin = (offset_probs >= t).float()

        f1_on = _binary_f1(onset_pred_bin.reshape(-1), onset_true_bin.reshape(-1))
        f1_off = _binary_f1(offset_pred_bin.reshape(-1), offset_true_bin.reshape(-1))
        onset_pred_rate = onset_pred_bin.mean().item()
        onset_pos_rate = onset_true_bin.mean().item()

        f1_on = 0.0 if f1_on is None else f1_on
        f1_off = 0.0 if f1_off is None else f1_off

        print(f"{t:.2f}\t{f1_on:0.3f}\t{f1_off:0.3f}\t{onset_pred_rate:0.3f}\t{onset_pos_rate:0.3f}\t0.000")

if __name__ == "__main__":
    main()


