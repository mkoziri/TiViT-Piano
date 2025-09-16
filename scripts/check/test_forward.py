"""Purpose:
    Validate a full dataloader-to-model pipeline by running a forward pass on a
    real batch and printing timing plus output tensor statistics.

Key Functions/Classes:
    - main(): Loads the configuration, builds the dataloader and model, executes
      a forward pass, and reports shapes and value ranges.

CLI:
    Execute ``python scripts/test_forward.py``.  The script requires no
    arguments and is useful for sanity checks.
"""

from utils import load_config
from models import build_model
from data import make_dataloader
import torch
from time import perf_counter

def main():
    cfg = load_config("configs/config.yaml")

    # Build model (eval, CPU)
    model = build_model(cfg).eval()

    # Build loader for the configured split (usually "test" right now)
    split = cfg["dataset"].get("split", "test")
    loader = make_dataloader(cfg, split=split)

    print(f"Using split: {split} | batch_size={loader.batch_size}")

    # Fetch exactly one batch
    batch = next(iter(loader))
    x = batch["video"]  # (B, T, tiles, C, H, W)
    paths = batch["path"]

    print("Input batch shape:", tuple(x.shape))
    print("First sample path:", paths[0])

    # Run forward pass (no gradients)
    with torch.no_grad():
        t0 = perf_counter()
        out = model(x)
        dt = perf_counter() - t0

    # Print shapes per head
    print(f"Forward time: {dt:.3f}s (CPU)")
    for k, v in out.items():
        print(f"{k:15s} -> {tuple(v.shape)}")

    # Optional: quick sanity on logits ranges
    for k, v in out.items():
        v = v.detach().cpu()
        print(f"{k:15s} min={v.min().item():.3f} max={v.max().item():.3f}")

if __name__ == "__main__":
    main()

