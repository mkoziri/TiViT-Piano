"""
PianoVAM dataset loader for TiViT-Piano.

This version is a simple, video-only baseline:
- It does NOT rely on metadata_v2.json.
- It scans all video files under the Video/ directory.
- It creates train / val / test splits using a simple 80 / 10 / 10 partition.
- It loads video clips as tensors with shape (T, C, H, W).
- It integrates with data/loader.py via make_dataloader().

This is intended for the first PianoVAM experiment (P1):
- We verify that the dataloader, model and training loop work end-to-end.
- We do not yet follow the official PianoVAM splits or labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import decord

# Use the PyTorch bridge so decord returns torch tensors directly.
decord.bridge.set_bridge("torch")


def _resolve_root(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve the root directory of the PianoVAM dataset.

    Priority:
      1. dataset.root_dir from the config (if provided)
      2. $TIVIT_DATA_DIR/PianoVAM_v1.0  (or $DATASETS_HOME/PianoVAM_v1.0)
      3. ~/datasets/PianoVAM_v1.0       (fallback)
    """
    dataset_cfg = cfg.get("dataset", {})
    root_dir = dataset_cfg.get("root_dir")

    if root_dir:
        return Path(root_dir)

    env_base = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env_base:
        candidate = Path(env_base) / "PianoVAM_v1.0"
        if candidate.is_dir():
            return candidate

    home_candidate = Path.home() / "datasets" / "PianoVAM_v1.0"
    if home_candidate.is_dir():
        return home_candidate

    # Last resort: assume env_base exists and append PianoVAM_v1.0
    return Path(env_base) / "PianoVAM_v1.0" if env_base else home_candidate


class PianoVAMDataset(Dataset):
    """
    Simple video-only dataset for PianoVAM.

    Each sample is a dict with:
      {
        "video": tensor(T, C, H, W),
        "path":  full path to the video file,
        "id":    filename stem (without extension)
      }

    Splits (train / val / test) are derived by partitioning the
    list of available videos (80 / 10 / 10) rather than using
    the official PianoVAM split metadata.
    """

    def __init__(self, cfg: Mapping[str, Any], split: str = "train") -> None:
        super().__init__()

        self.cfg = cfg
        self.root = _resolve_root(cfg)
        dataset_cfg = cfg.get("dataset", {})

        # Frames per clip.
        self.frames = int(dataset_cfg.get("frames", 128))

        # Optional resize from config (dataset.resize: [H, W]).
        resize = dataset_cfg.get("resize", None)
        if resize is not None and len(resize) == 2:
            self.resize_h, self.resize_w = int(resize[0]), int(resize[1])
        else:
            self.resize_h = self.resize_w = None

        # Fixed split fractions for this baseline.
        train_frac = 0.8
        val_frac = 0.1  # test gets the remaining fraction.

        video_dir = self.root / "Video"
        if not video_dir.is_dir():
            raise FileNotFoundError(f"[PianoVAM] Video directory not found: {video_dir}")

        # Collect all video files.
        exts = {".mp4", ".mkv", ".avi", ".mov"}
        all_files: List[Path] = sorted(
            p for p in video_dir.rglob("*")
            if p.suffix.lower() in exts and p.is_file()
        )

        if not all_files:
            raise RuntimeError(f"[PianoVAM] No video files found under {video_dir}")

        n = len(all_files)
        n_train = int(n * train_frac)
        n_val = int(n * (train_frac + val_frac))

        if split == "train":
            sel_files = all_files[:n_train]
        elif split in ("val", "validation"):
            sel_files = all_files[n_train:n_val]
        elif split == "test":
            sel_files = all_files[n_val:]
        else:
            raise ValueError(f"[PianoVAM] Unknown split: {split}")

        # Optional max_clips limit from config.
        max_clips = dataset_cfg.get("max_clips")
        if max_clips is not None:
            sel_files = sel_files[: int(max_clips)]

        if not sel_files:
            raise RuntimeError(
                f"[PianoVAM] No files selected for split '{split}' "
                f"(total_videos={n}, train <= {n_train}, val <= {n_val})"
            )

        # Build minimal records for __getitem__.
        self.records: List[Dict[str, Any]] = [
            {
                "video_path": str(p.relative_to(self.root)),
                "id": p.stem,
            }
            for p in sel_files
        ]

        print(
            f"[PianoVAMDataset] root={self.root} | split={split} | "
            f"total_videos={n} | using={len(self.records)} | "
            f"frames={self.frames}, resize=({self.resize_h}, {self.resize_w})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_video(self, rec: Dict[str, Any]) -> torch.Tensor:
        """
        Load a single video and return a clip of shape (T, C, H, W)
        using uniform frame sampling and optional resizing.
        """
        rel_path = rec["video_path"]
        video_path = self.root / rel_path

        if not video_path.is_file():
            raise FileNotFoundError(f"[PianoVAM] Missing video: {video_path}")

        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        if total <= 1:
            raise RuntimeError(f"[PianoVAM] Empty or corrupt video: {video_path}")

        # Uniform sampling of self.frames indices in [0, total-1].
        idxs = np.linspace(0, total - 1, self.frames, dtype=np.int64)
        batch = vr.get_batch(idxs)   # (T, H, W, C) torch tensor (via decord bridge)

        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)

        video = batch.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        # Optional resize.
        if self.resize_h is not None and self.resize_w is not None:
            video = torch.nn.functional.interpolate(
                video,
                size=(self.resize_h, self.resize_w),
                mode="bilinear",
                align_corners=False,
            )

        return video

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        video = self._load_video(rec)

        sample: Dict[str, Any] = {
            "video": video,  # (T, C, H, W)
            "path": str(self.root / rec["video_path"]),
            "id": rec.get("id", str(idx)),
        }
        return sample


def make_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    drop_last: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Factory function used by src/data/loader.py to construct a
    PyTorch DataLoader for PianoVAM.
    """
    dataset = PianoVAMDataset(cfg, split=split)

    dataset_cfg = cfg.get("dataset", {})
    batch_size = int(dataset_cfg.get("batch_size", 2))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    shuffle = split == "train"

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )
    return loader
