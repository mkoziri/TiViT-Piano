#!/usr/bin/env python3
import sys, torch
from pathlib import Path
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config
from data import make_dataloader
from models import build_model
from train import evaluate_one_epoch  # uses cfg["training"]["metrics"]["prob_threshold"]

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

    # sweep thresholds
    thrs = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    print("Threshold\tonset_f1\toffset_f1\tonset_pred_rate\tonset_pos_rate\ttotal")
    for t in thrs:
        cfg.setdefault("training", {}).setdefault("metrics", {})["prob_threshold"] = t
        metrics = evaluate_one_epoch(model, val_loader, cfg)
        print(f"{t:.2f}\t{metrics.get('onset_f1'):0.3f}\t{metrics.get('offset_f1'):0.3f}\t"
              f"{metrics.get('onset_pred_rate'):0.3f}\t{metrics.get('onset_pos_rate'):0.3f}\t"
              f"{metrics.get('total'):0.3f}")

if __name__ == "__main__":
    main()

