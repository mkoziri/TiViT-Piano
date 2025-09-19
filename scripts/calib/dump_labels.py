#!/usr/bin/env python3
"""Purpose:
    Inspect dataloader outputs for debugging by printing label tensors,
    metadata, and summary statistics for a single batch.

Key Functions/Classes:
    - pick_train_loader(): Normalizes different return types from
      ``make_dataloader`` to a training loader instance.
    - describe_tensor(): Logs tensor shapes and positive rates for frame-level
      targets.
    - main(): Loads configuration, fetches one batch from the training split,
      and dumps relevant diagnostics.

CLI:
    Execute ``python scripts/dump_labels.py``.  The script has no arguments and
    prints results directly to stdout.
"""

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config
from data import make_dataloader
import torch

def pick_train_loader(res):
    """
    Accepts any of:
      - DataLoader (iterable)
      - dict with 'train' key
      - tuple/list of loaders (use the first)
    """
    # DataLoader is iterable and has __len__/__iter__, but so do dicts;
    # check dict/sequence first.
    if isinstance(res, dict):
        if "train" in res:
            return res["train"]
        # fall back: first value
        return next(iter(res.values()))
    if isinstance(res, (list, tuple)):
        if len(res) == 0:
            raise ValueError("make_dataloader returned an empty list/tuple")
        return res[0]
    # assume it's already a DataLoader-like object
    return res

def describe_tensor(name, x):
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, sum={float(x.sum())}")
    # global positive rate across all dims
    pos_rate = x.float().mean().item()
    print(f"  mean positive rate: {pos_rate:.6f}")
    # per-frame rate for first sample (if 3D: B,T,128)
    if x.dim() == 3 and x.size(0) > 0:
        per_t = x[0].float().mean(dim=1)  # mean over pitch/classes
        show = min(10, per_t.numel())
        print(f"  first {show} frames pos-rate (sample 0): {[round(v,6) for v in per_t[:show].tolist()]}")

def main():
    cfg = load_config("configs/config.yaml")

    # Force same conditions as the overfit check:
    cfg["dataset"]["split_train"] = "train"
    cfg["dataset"]["split_val"]   = "train"
    cfg["dataset"]["split"]       = "train"
    cfg["dataset"]["max_clips"]   = 1
    cfg["dataset"]["shuffle"]     = False

    res = make_dataloader(cfg, split="train")
    train_loader = pick_train_loader(res)

    batch = next(iter(train_loader))
    print("Batch keys:", sorted(batch.keys()))
    if "frames" in batch:
        B, T = batch["frames"].shape[:2]
        print(f"B={B}, T={T}")

    # Add near the end of dump_labels.py, after building 'batch'
    print("\nMeta:")
    print('path:', batch.get('path'))
    evs = batch["labels"][0]  # list-like of (onset, offset, pitch), seconds + midi
    print("\nFirst 5 events:")
    for e in evs[:5]:
        try:
            print(e)
        except Exception:
            pass
    # Some repos pack raw events under these keys; adapt if your keys differ:
    for k in ['labels', 'onset', 'offset', 'pitch']:
        if k in batch:
            v = batch[k]
            try:
                print(f"{k}: len={len(v)}  sample0_len={len(v[0]) if hasattr(v, '__getitem__') else 'n/a'}")
            except Exception:
                print(f"{k}: type={type(v)}")
            
    # Expect these keys in frame mode (names may differ slightly in your repo):
    for k in ["pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"]:
        if k not in batch:
            print(f"{k}: MISSING")
        else:
            describe_tensor(k, batch[k])

if __name__ == "__main__":
    main()

