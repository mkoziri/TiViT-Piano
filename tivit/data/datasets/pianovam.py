"""
Purpose:
    PianoVAM dataset adapter using the shared base implementation.

Key Functions/Classes:
    - PianoVAMAdapter: Builds DataLoader with sampler support.
    - build_dataset(): Registry hook returning the dataloader.

CLI Arguments:
    (none)

Usage:
    adapter = PianoVAMAdapter(cfg, split)
    loader = adapter.dataloader()
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import DatasetAdapter, safe_collate_fn
from .pianovam_impl import PianoVAMDataset


class PianoVAMAdapter(DatasetAdapter):
    """Adapter that builds a DataLoader for PianoVAM using the shared base."""

    def dataloader(self, drop_last: bool = False):
        import torch

        dcfg = self.cfg.get("dataset", {}) if isinstance(self.cfg, dict) else {}
        ds = PianoVAMDataset(self.cfg, self.split, full_cfg=self.cfg)
        sampler = getattr(ds, "_sampler", None)
        num_workers = int(dcfg.get("num_workers", 0))
        prefetch = dcfg.get("prefetch_factor")
        pin_memory = bool(dcfg.get("pin_memory", torch.cuda.is_available()))
        if num_workers < 1:
            prefetch = None
        return torch.utils.data.DataLoader(
            ds,
            batch_size=int(dcfg.get("batch_size", 1)),
            shuffle=False if sampler is not None else (bool(dcfg.get("shuffle", True)) if self.split == "train" else False),
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            prefetch_factor=prefetch,
            sampler=sampler,
            collate_fn=safe_collate_fn,
        )


def build_dataset(cfg: Mapping[str, Any], split: str, drop_last: bool = False, *, seed: Optional[int] = None):
    """Registry hook that returns the PianoVAM dataloader."""

    adapter = PianoVAMAdapter(cfg, split, seed=seed)
    return adapter.dataloader(drop_last=drop_last)


__all__ = ["PianoVAMAdapter", "build_dataset"]
