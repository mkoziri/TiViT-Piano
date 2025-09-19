"""Purpose:
    Iterate over the configured dataloader to print batch shapes and sample
    paths, ensuring dataset settings are correct.

Key Functions/Classes:
    - main(): Loads ``configs/config.yaml``, constructs a dataloader for the
      selected split, and prints details for the first two batches (or the
      requested number of batches).

CLI:
    Execute ``python scripts/test_loader.py``. Optional arguments can override
    the configuration path, dataset split, and the number of batches to print.
"""

import argparse

from utils import load_config
from data import make_dataloader

def main():
    parser = argparse.ArgumentParser(description="Inspect dataset loader output.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to load. Overrides the split from the config if provided.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=2,
        dest="max_batches",
        help="Number of batches to iterate over before stopping.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = args.split if args.split is not None else cfg["dataset"].get("split", "test")
    loader = make_dataloader(cfg, split=split)
    dataset_name = cfg["dataset"].get("name", "unknown")

    print(f"Dataset split: {split}")
    print(f"Dataset name: {dataset_name}")
    for i, batch in enumerate(loader):
        x = batch["video"]           # B, T, tiles, C, H, W
        paths = batch["path"]
        B, T, M, C, H, W = x.shape
        print(f"[{i}] batch: {B} clips | shape={x.shape}")
        print(" sample:", paths[0])
        print(f"  B={B}, T={T}, tiles={M}, C={C}, H={H}, W={W}")
        if i + 1 >= args.max_batches:
            break

if __name__ == "__main__":
    main()

