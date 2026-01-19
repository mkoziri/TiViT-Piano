"""
Purpose:
    PianoYT dataset entrypoint using the shared BasePianoDataset.

Key Functions/Classes:
    - PianoYTAdapter: Builds DataLoader with sampler support.
    - build_dataset(): Registry hook returning the dataloader.

CLI Arguments:
    (none)

Usage:
    adapter = PianoYTAdapter(cfg, split)
    loader = adapter.dataloader()
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import DatasetAdapter, safe_collate_fn
from .pianoyt_impl import PianoYTDataset


class PianoYTAdapter(DatasetAdapter):
    """Adapter that builds a DataLoader for PianoYT."""

    def dataloader(self, drop_last: bool = False):
        import torch

        ds = PianoYTDataset(self.cfg, self.split, full_cfg=self.cfg)
        dcfg = self.cfg.get("dataset", {}) if isinstance(self.cfg, dict) else {}
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
    """Registry hook that returns the PianoYT dataloader."""

    adapter = PianoYTAdapter(cfg, split, seed=seed)
    return adapter.dataloader(drop_last=drop_last)


__all__ = ["PianoYTAdapter", "build_dataset"]
