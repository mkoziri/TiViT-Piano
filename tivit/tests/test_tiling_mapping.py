#!/usr/bin/env python3
"""
Purpose:
    Smoke test for tiling mappings (no pytest needed).

Key Functions/Classes:
    - main(): Tile a dummy tensor and print mapping info.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_tiling_mapping.py
"""

from __future__ import annotations

import torch

from tivit.data.roi.tiling import tile_vertical_token_aligned


def main() -> None:
    """Tile a dummy tensor and print mapping info."""
    x = torch.randn(1, 3, 145, 160)  # T=1,C=3,H=145,W=160
    tiles, tokens_per_tile, widths, bounds, aligned_w, original_w = tile_vertical_token_aligned(
        x, tiles=2, patch_w=16, tokens_split="auto", overlap_tokens=0
    )
    print("tokens_per_tile", tokens_per_tile)
    print("bounds", bounds)
    print("aligned_w", aligned_w, "original_w", original_w)
    print("tile shapes", [t.shape for t in tiles])


if __name__ == "__main__":
    main()
