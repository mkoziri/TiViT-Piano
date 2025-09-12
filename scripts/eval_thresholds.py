#!/usr/bin/env python3
import sys, json, torch
import torch.nn.functional as F
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config, align_pitch_dim
from data import make_dataloader
from models import build_model


# Default probability grid used when sweeping thresholds without an explicit
# list.  We parse lists manually so callers can provide comma-separated values
# without escaping leading minus signs.
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


def _parse_list(argv, name):
    """Extract ``--<name>`` from ``argv`` allowing comma/space separation."""
    flag = f"--{name}"
    for i, arg in enumerate(list(argv)):
        if arg.startswith(flag):
            if arg == flag:
                j = i + 1
                vals = []
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                if not vals:
                    raise ValueError(f"{flag} expects at least one value")
                del argv[i:j]
                arg_str = " ".join(vals)
            else:  # handle --flag=... form
                arg_str = arg.split("=", 1)[1]
                del argv[i]
            arg_str = arg_str.replace(",", " ")
            return [float(v) for v in arg_str.split() if v]
    return None


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
        logit_thrs = _parse_list(argv, "thresholds")
        prob_thrs = _parse_list(argv, "prob_thresholds")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--thresholds", metavar="T", nargs="*", help="Logit threshold values")
    ap.add_argument(
        "--prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values",
    )
    ap.add_argument("--calibration", help="JSON file with calibrated thresholds")
    ap.add_argument("--head", choices=["onset", "offset"], help="Sweep thresholds for only one head")
    # Explicit thresholds for the non-swept head when no calibration is provided
    ap.add_argument("--fixed_offset_prob", type=float)
    ap.add_argument("--fixed_offset_logit", type=float)
    ap.add_argument("--fixed_onset_prob", type=float)
    ap.add_argument("--fixed_onset_logit", type=float)
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
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--stride", type=int)
    args = ap.parse_args(argv)
    args.thresholds = logit_thrs
    args.prob_thresholds = prob_thrs

    if args.thresholds is not None and args.prob_thresholds is not None:
        print("error: --thresholds and --prob_thresholds are mutually exclusive", file=sys.stderr)
        return

    # Unless a calibration file is provided and no head is specified, default to
    # sweeping over probability thresholds when none were specified explicitly.
    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()


    cfg = load_config("configs/config.yaml")
    if args.max_clips is not None:
        cfg["dataset"]["max_clips"] = args.max_clips
    if args.frames is not None:
        cfg["dataset"]["frames"] = args.frames
    if args.stride is not None:
        cfg["dataset"]["stride"] = args.stride
    split = args.split or cfg["dataset"].get("split_val", "val")

    # build loader
    val_loader = make_dataloader(cfg, split=split)
    if isinstance(val_loader, dict):
        val_loader = val_loader.get(split, next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    # load model + ckpt
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    # run model once to collect logits/probabilities and targets
    onset_logits_list, offset_logits_list = [], []
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["video"]
            out = model(x)

            # prefer *_logits if present; fallback to old naming
            onset_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            offset_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")

            # Apply temperature scaling and bias for calibration
            onset_logits = onset_logits / args.temperature + args.bias
            offset_logits = offset_logits / args.temperature + args.bias
            onset_prob = torch.sigmoid(onset_logits)
            offset_prob = torch.sigmoid(offset_logits)

            onset_logits_list.append(onset_logits.detach().cpu())
            offset_logits_list.append(offset_logits.detach().cpu())
            onset_probs.append(onset_prob.detach().cpu())
            offset_probs.append(offset_prob.detach().cpu())

            onset_tgts.append(batch["onset_roll"].float().cpu())
            offset_tgts.append(batch["offset_roll"].float().cpu())

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    T_logits, P_logits = onset_probs.shape[1], onset_probs.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = _pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = _pool_roll_BT(offset_tgts, T_logits)
    onset_tgts = align_pitch_dim(onset_probs, onset_tgts, "onset")
    offset_tgts = align_pitch_dim(offset_probs, offset_tgts, "offset")
    
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
    
    def _eval_pair(on_thr, off_thr, use_logits):
        if use_logits:
            onset_pred_bin = (onset_logits >= on_thr).float()
            offset_pred_bin = (offset_logits >= off_thr).float()
        else:
            onset_pred_bin = (onset_probs >= on_thr).float()
            offset_pred_bin = (offset_probs >= off_thr).float()

        f1_on = _binary_f1(onset_pred_bin.reshape(-1), onset_true_bin.reshape(-1))
        f1_off = _binary_f1(offset_pred_bin.reshape(-1), offset_true_bin.reshape(-1))
        onset_pred_rate = onset_pred_bin.mean().item()
        onset_pos_rate = onset_true_bin.mean().item()

        f1_on = 0.0 if f1_on is None else f1_on
        f1_off = 0.0 if f1_off is None else f1_off

        print(f"{on_thr:.2f}\t{off_thr:.2f}\t{f1_on:0.3f}\t{f1_off:0.3f}\t{onset_pred_rate:0.3f}\t{onset_pos_rate:0.3f}\t0.000")

    printed_header = False

    def _header():
        nonlocal printed_header
        if not printed_header:
            print("onset_thr\toffset_thr\tonset_f1\toffset_f1\tonset_pred_rate\tonset_pos_rate\ttotal")
            printed_header = True

    if args.head is None:
        # Evaluate at calibrated thresholds if provided.
        if args.calibration:
            with open(args.calibration) as f:
                calib = json.load(f)
            on_cal = calib.get("onset", {})
            off_cal = calib.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                _header()
                _eval_pair(on_cal["best_logit"], off_cal["best_logit"], use_logits=True)
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                _header()
                _eval_pair(on_cal["best_prob"], off_cal["best_prob"], use_logits=False)
            else:
                print("Calibration file missing best_logit/best_prob keys", file=sys.stderr)

        # Sweep over provided threshold grids.
        if args.thresholds:
            _header()
            for t in args.thresholds:
                _eval_pair(t, t, use_logits=True)
        if args.prob_thresholds:
            _header()
            for t in args.prob_thresholds:
                _eval_pair(t, t, use_logits=False)
    else:
        # Per-head sweep
        if args.thresholds is not None:
            sweep_vals = args.thresholds
            use_logits = True
            mode = "logit"
        else:
            sweep_vals = args.prob_thresholds
            use_logits = False
            mode = "prob"

        other_head = "offset" if args.head == "onset" else "onset"
        fixed_thr = None
        source = None

        if args.calibration:
            with open(args.calibration) as f:
                calib = json.load(f)
            other_cal = calib.get(other_head, {})
            if use_logits:
                if "best_logit" in other_cal:
                    fixed_thr = other_cal["best_logit"]
                elif "best_prob" in other_cal:
                    fixed_thr = torch.logit(torch.tensor(other_cal["best_prob"])).item()
            else:
                if "best_prob" in other_cal:
                    fixed_thr = other_cal["best_prob"]
                elif "best_logit" in other_cal:
                    fixed_thr = torch.sigmoid(torch.tensor(other_cal["best_logit"])).item()
            if fixed_thr is None:
                print("Calibration file missing threshold for", other_head, file=sys.stderr)
                return
            source = "calibration"
        else:
            if args.head == "onset":
                fixed_thr = args.fixed_offset_logit if use_logits else args.fixed_offset_prob
                flag_name = "--fixed_offset_logit" if use_logits else "--fixed_offset_prob"
            else:
                fixed_thr = args.fixed_onset_logit if use_logits else args.fixed_onset_prob
                flag_name = "--fixed_onset_logit" if use_logits else "--fixed_onset_prob"
            if fixed_thr is None:
                print(
                    f"error: specify {flag_name} or --calibration to fix {other_head} threshold",
                    file=sys.stderr,
                )
                return
            source = "flag"

        print(
            f"Per-head sweep: head={args.head}, mode={mode}, fixed_{other_head}={fixed_thr:.3f} (source={source})"
        )
        _header()
        for t in sweep_vals:
            if args.head == "onset":
                on_thr, off_thr = t, fixed_thr
            else:
                on_thr, off_thr = fixed_thr, t
            _eval_pair(on_thr, off_thr, use_logits)

if __name__ == "__main__":
    main()

