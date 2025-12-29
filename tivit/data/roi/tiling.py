"""Utility functions for token-aligned tiling used by pipeline v2 (migrated)."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch


def auto_split_tokens(total_tokens: int, tiles: int) -> List[int]:
    """Split ``total_tokens`` into ``tiles`` groups differing by at most one."""

    if tiles <= 0:
        raise ValueError("tiles must be positive")
    if tiles == 1:
        return [int(total_tokens)]

    base = total_tokens // tiles
    remainder = total_tokens % tiles
    splits = [base for _ in range(tiles)]

    if remainder == 0:
        return splits

    if tiles % 2 == 1:
        center = tiles // 2
        for offset in range(1, (remainder // 2) + 1):
            left = center - offset
            right = center + offset
            if left < 0 or right >= tiles:
                break
            splits[left] += 1
            splits[right] += 1
        if remainder % 2 == 1:
            splits[center] += 1
    else:
        left_center = tiles // 2 - 1
        right_center = tiles // 2
        for offset in range(remainder // 2):
            left = left_center - offset
            right = right_center + offset
            if left < 0 or right >= tiles:
                break
            splits[left] += 1
            splits[right] += 1
        if remainder % 2 == 1:
            splits[left_center] += 1

    return splits


def tile_vertical_token_aligned(
    x: torch.Tensor,
    tiles: int,
    *,
    patch_w: int,
    tokens_split: Union[str, Sequence[int]],
    overlap_tokens: int = 0,
) -> Tuple[List[torch.Tensor], List[int], List[int], List[Tuple[int, int]], int, int]:
    """Tile ``x`` so that widths align with the ViT patch grid."""

    if patch_w <= 0:
        raise ValueError("patch_w must be positive")

    T, C, H, W = x.shape
    original_w = int(W)
    total_tokens = original_w // patch_w
    if total_tokens <= 0:
        raise ValueError(
            f"Width {original_w} is insufficient for patch width {patch_w}."
        )

    if isinstance(tokens_split, str):
        if tokens_split.lower() != "auto":
            raise ValueError(f"Unsupported tokens_split='{tokens_split}'")
        tokens_per_tile = auto_split_tokens(total_tokens, tiles)
    else:
        tokens_per_tile = [int(v) for v in tokens_split]
        if len(tokens_per_tile) != tiles:
            raise ValueError(
                f"tokens_split length {len(tokens_per_tile)} != tiles {tiles}"
            )
        if sum(tokens_per_tile) != total_tokens:
            raise ValueError(
                f"tokens_split sum {sum(tokens_per_tile)} != total_tokens {total_tokens}"
            )

    widths = [int(tok) * int(patch_w) for tok in tokens_per_tile]
    aligned_w = sum(widths)
    if aligned_w <= 0:
        raise ValueError("Aligned width must be positive")

    if aligned_w != original_w:
        x = x[..., :aligned_w]
        W = aligned_w
    else:
        W = original_w

    overlap_tokens = max(int(overlap_tokens), 0)
    overlap_px = overlap_tokens * int(patch_w)

    slices: List[torch.Tensor] = []
    bounds: List[Tuple[int, int]] = []
    start = 0
    for idx, width in enumerate(widths):
        end = start + width
        left = start - overlap_px if idx > 0 else start
        right = end + overlap_px if idx < tiles - 1 else end
        left = max(left, 0)
        right = min(right, W)
        slices.append(x[..., left:right])
        bounds.append((left, right))
        start = end

    return slices, [int(t) for t in tokens_per_tile], widths, bounds, W, original_w


__all__ = ["auto_split_tokens", "tile_vertical_token_aligned"]

