"""
OMAPS dataset loader.

Purpose:
    - Resolve OMAPS roots, manifests, and label parsing.
    - Reuse shared decode/tiling/target logic from BasePianoDataset.
Key Functions/Classes:
    - OMAPSDataset
CLI Arguments:
    - (none)
Usage:
    - ds = OMAPSDataset(cfg, split="train", full_cfg=cfg)
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional

import torch

from tivit.data.datasets.base import BasePianoDataset, DatasetEntry
from tivit.data.targets.identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)


class OMAPSDataset(BasePianoDataset):
    """OMAPS dataset using shared decoding/target logic."""

    def _resolve_root(self, root_dir: Optional[str]) -> Path:
        """Resolve dataset root with env fallbacks."""
        if root_dir:
            return Path(root_dir).expanduser()
        env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
        if env:
            return Path(env).expanduser() / "OMAPS"
        return Path("~/datasets/OMAPS").expanduser()

    def _resolve_manifest(self) -> Optional[Mapping[str, Any]]:
        """Load manifest mapping from file when provided."""
        manifest_cfg = self.dataset_cfg.get("manifest", {}) or {}
        path = manifest_cfg.get(self.split)
        if not path:
            return None
        ids = set()
        p = Path(path).expanduser()
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    ids.add(canonical_video_id(line))
        return {vid: {} for vid in ids}

    def _list_entries(self, root: Path, split: str, manifest: Optional[Mapping[str, Any]]) -> List[DatasetEntry]:
        """List video/label entries for the split."""
        split_dir = root.joinpath(split)
        pattern = str((split_dir if split_dir.exists() else root).joinpath("**/*.mp4"))
        vids = [Path(p) for p in glob.glob(pattern, recursive=True)]
        vids.sort()
        allow_ids = set(manifest.keys()) if isinstance(manifest, Mapping) else None
        entries: List[DatasetEntry] = []
        for video_path in vids:
            vid = canonical_video_id(video_path.stem)
            if allow_ids is not None and vid not in allow_ids:
                continue
            label_path = video_path.with_suffix(".txt")
            entries.append(DatasetEntry(video_path=video_path, label_path=label_path, video_id=vid, metadata={}))
        return entries

    def _read_labels(self, entry: DatasetEntry) -> Mapping[str, Any]:
        """Parse events (and optional hand/clef) from sidecar labels."""
        if entry.label_path is None or not entry.label_path.exists():
            if self.require_labels:
                raise FileNotFoundError(f"Missing label for {entry.video_id}")
            return {}
        events = []
        hands: list[int] = []
        clefs: list[int] = []
        try:
            with entry.label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
                    hand_val = int(parts[3]) if len(parts) >= 4 else None
                    clef_val = int(parts[4]) if len(parts) >= 5 else None
                    events.append((onset, offset, pitch, hand_val, clef_val))
                    if hand_val is not None:
                        hands.append(hand_val)
                    if clef_val is not None:
                        clefs.append(clef_val)
        except Exception as exc:
            if self.require_labels:
                raise
            LOGGER.warning("Failed to read labels for %s (%s)", entry.video_id, exc)
        payload = {"events": events}
        if hands:
            payload["hand_seq"] = hands
        if clefs:
            payload["clef_seq"] = clefs
        return payload


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False, *, seed: Optional[int] = None):
    dcfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    batch_size = int(dcfg.get("batch_size", 1))
    shuffle = bool(dcfg.get("shuffle", True)) if split == "train" else False
    num_workers = int(dcfg.get("num_workers", 0))
    prefetch_factor = dcfg.get("prefetch_factor")

    ds = OMAPSDataset(cfg, split, full_cfg=cfg)
    return torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
