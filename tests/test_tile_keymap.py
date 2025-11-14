from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tivit.decoder.tile_keymap import build_tile_key_mask


def test_uniform_fallback_and_cushion_behavior() -> None:
    base_mask = build_tile_key_mask(reg_meta=None, num_tiles=3, cushion_keys=0, n_keys=88)
    assert base_mask.shape == (3, 88)
    counts = base_mask.sum(axis=1)
    # Uniform split: each tile should cover roughly 29–30 keys without cushions.
    assert counts.min() >= 29
    assert counts.max() <= 30
    assert np.all(np.any(base_mask, axis=1)), "each tile must cover at least one key"
    assert np.all(np.any(base_mask, axis=0)), "all keys should be covered"

    cushioned = build_tile_key_mask(reg_meta=None, num_tiles=3, cushion_keys=2, n_keys=88)
    assert cushioned.shape == (3, 88)
    # Cushions expand coverage but clip to the valid [0, 87] range.
    assert cushioned[0, 0]
    assert cushioned[-1, -1]
    assert np.all(cushioned.sum(axis=1) >= counts)
    overlap = cushioned.sum(axis=0)
    assert overlap.max() <= 3, "no key should be assigned to more than all tiles"
    boundary_keys = np.flatnonzero(overlap > 1)
    assert boundary_keys.size > 0, "cushioning should create boundary overlaps"
