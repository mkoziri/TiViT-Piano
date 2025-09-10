#!/usr/bin/env python3
import sys, torch
from pathlib import Path
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config
from data import make_dataloader
from models import build_model


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
    
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    args = ap.parse_args()

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
    with torch.no_grad():
        for batch in val_loader:
            x = batch["video"]
            out = model(x)

            # prefer *_logits if present; fallback to old naming
            onset_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            offset_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")

            onset_prob = torch.sigmoid(onset_logits)
            offset_prob = torch.sigmoid(offset_logits)

            onset_probs.append(onset_prob.detach().cpu())
            offset_probs.append(offset_prob.detach().cpu())

            onset_tgts.append(batch["onset_roll"].float().cpu())
            offset_tgts.append(batch["offset_roll"].float().cpu())

    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    # diagnostic prints
    print(f"[OVERALL onset probs] mean={onset_probs.mean():.3f} min={onset_probs.min():.3f} max={onset_probs.max():.3f}")
    print(f"[OVERALL offset probs] mean={offset_probs.mean():.3f} min={offset_probs.min():.3f} max={offset_probs.max():.3f}")

    onset_true_any = (onset_tgts > 0).any(dim=-1).float()
    offset_true_any = (offset_tgts > 0).any(dim=-1).float()
    
    thrs = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    print("Threshold\tonset_f1\toffset_f1\tonset_pred_rate\tonset_pos_rate\ttotal")
    for t in thrs:
        onset_pred_any = (onset_probs >= t).any(dim=-1).float()
        offset_pred_any = (offset_probs >= t).any(dim=-1).float()

        f1_on = _binary_f1(onset_pred_any.reshape(-1), onset_true_any.reshape(-1))
        f1_off = _binary_f1(offset_pred_any.reshape(-1), offset_true_any.reshape(-1))
        onset_pred_rate = onset_pred_any.mean().item()
        onset_pos_rate = onset_true_any.mean().item()

        f1_on = 0.0 if f1_on is None else f1_on
        f1_off = 0.0 if f1_off is None else f1_off

        print(f"{t:.2f}\t{f1_on:0.3f}\t{f1_off:0.3f}\t{onset_pred_rate:0.3f}\t{onset_pos_rate:0.3f}\t0.000")

if __name__ == "__main__":
    main()

