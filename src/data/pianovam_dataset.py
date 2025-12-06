"""
PianoVAM dataset loader for TiViT-Piano (supervised version).

This dataset:
- Reads metadata_v2.json to get the official split (train / test / etc.).
- Matches each recording to:
    Video/{record_time}.mp4
    TSV/{record_time}.tsv
- Loads video as a single clip with T frames (uniform sampling).
- Builds frame-level labels for an 88-key piano:
    pitch[t, k]  = 1 if key k is active at time t
    onset[t, k]  = 1 if a note with key k starts at time t
    offset[t, k] = 1 if a note with key k ends at time t

Each sample is a dict:
  {
    "video":  (T, C, H, W) float32 in [0,1],
    "path":   str, full video path,
    "id":     str, recording id (record_time),
    "pitch":  (T, 88) float32 {0,1},
    "onset":  (T, 88) float32 {0,1},
    "offset": (T, 88) float32 {0,1},
  }

This is intended for the first supervised PianoVAM experiment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import decord

# Use the PyTorch bridge so decord returns torch tensors directly.
decord.bridge.set_bridge("torch")


# Piano range (88-key piano).
NOTE_MIN = 21
NOTE_MAX = 108
N_PITCHES = NOTE_MAX - NOTE_MIN + 1  # 88


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
    Supervised video+label dataset for PianoVAM.

    Splits (train / val / test) are read from metadata_v2.json ("split" field).
    For each recording with id=record_time, this dataset expects:

      Video/{id}.mp4
      TSV/{id}.tsv

    Handskeleton / Audio / MIDI are currently ignored.
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

        # Normalization options (if used elsewhere).
        self.normalize = bool(dataset_cfg.get("normalize", True))

        # Map config split names to metadata "split" values if needed.
        # For now we keep them the same.
        self.split = split

        # ------------------------------------------------------------------
        # Load metadata_v2.json
        # ------------------------------------------------------------------
        metadata_path = self.root / "metadata_v2.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"[PianoVAM] metadata_v2.json not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            meta_raw = json.load(f)

        # meta_raw is a dict: { "0": {...}, "1": {...}, ... }
        # We only keep entries that belong to the desired split and
        # for which both Video and TSV files exist.
        records: List[Dict[str, Any]] = []
        video_dir = self.root / "Video"
        tsv_dir = self.root / "TSV"

        for key, rec in meta_raw.items():
            if not isinstance(rec, dict):
                continue

            meta_split = rec.get("split")
            if meta_split is None:
                continue

            # We allow "validation" to map to "val", etc., if needed later.
            if self._split_matches(split, meta_split):
                record_time = rec.get("record_time")
                if not record_time:
                    continue

                if record_time == "2024-02-14_19-27-45":
                    continue
                
                video_path = video_dir / f"{record_time}.mp4"
                tsv_path = tsv_dir / f"{record_time}.tsv"

                if not video_path.is_file() or not tsv_path.is_file():
                    # Skip incomplete recordings.
                    continue

                rec_copy = dict(rec)
                rec_copy["id"] = record_time
                rec_copy["video_path"] = video_path
                rec_copy["tsv_path"] = tsv_path
                records.append(rec_copy)

        if not records:
            raise RuntimeError(
                f"[PianoVAM] No usable records found for split='{split}' "
                f"in {metadata_path}"
            )

        # Optional max_clips from config.
        max_clips = dataset_cfg.get("max_clips")
        if max_clips is not None:
            records = records[: int(max_clips)]

        self.records = records

        print(
            f"[PianoVAMDataset] root={self.root} | split={split} | "
            f"metadata_entries={len(meta_raw)} | using={len(self.records)} | "
            f"frames={self.frames}, resize=({self.resize_h}, {self.resize_w})"
        )

    # ------------------------------------------------------------------ #
    # Helper: split matching
    # ------------------------------------------------------------------ #
    def _split_matches(self, requested: str, meta_split: str) -> bool:
        """Return True if the metadata split should be included for this requested split."""
        requested = requested.lower()
        meta_split = meta_split.lower()

        if requested == "train":
            return meta_split == "train"
        if requested in ("val", "validation"):
            return meta_split in ("val", "validation")
        if requested == "test":
            return meta_split == "test"
        # Fallback: exact match.
        return requested == meta_split

    # ------------------------------------------------------------------ #
    # Core Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def _load_video_and_times(self, rec: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load video frames and return:
          video: tensor(T, C, H, W)
          times: np.ndarray shape (T,) with timestamps in seconds
                 aligned with the sampled frames.
        """
        video_path: Path = rec["video_path"]
        if not video_path.is_file():
            raise FileNotFoundError(f"[PianoVAM] Missing video: {video_path}")

        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)
        if total_frames <= 1:
            raise RuntimeError(f"[PianoVAM] Empty or corrupt video: {video_path}")

        # Estimate duration in seconds using the native FPS.
        try:
            native_fps = float(vr.get_avg_fps())
            if native_fps <= 0:
                raise ValueError
        except Exception:
            # Fallback if FPS is not available.
            native_fps = 30.0

        duration_sec = total_frames / native_fps

        # Sample self.frames indices uniformly over the entire duration.
        T = self.frames
        idxs = np.linspace(0, total_frames - 1, T, dtype=np.int64)
        frames = vr.get_batch(idxs)  # (T, H, W, C) torch tensor (via decord bridge)

        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)

        video = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        # Optional resize.
        if self.resize_h is not None and self.resize_w is not None:
            video = torch.nn.functional.interpolate(
                video,
                size=(self.resize_h, self.resize_w),
                mode="bilinear",
                align_corners=False,
            )

        # Build a time grid aligned with the T sampled frames.
        # Times go from 0 up to (duration_sec), excluding the endpoint.
        frame_times = np.linspace(0.0, duration_sec, T, endpoint=False, dtype=np.float32)

        return video, frame_times

    def _load_labels(
        self,
        rec: Dict[str, Any],
        frame_times: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load TSV labels for this recording and build frame-level targets.

        TSV format (tab-separated), with a header line starting with '#':
          onset    key_offset   frame_offset   note   velocity

        - onset/key_offset: times in seconds
        - note: MIDI pitch
        - velocity: MIDI velocity (ignored for now, but could be used for weighting)
        """
        tsv_path: Path = rec["tsv_path"]
        if not tsv_path.is_file():
            raise FileNotFoundError(f"[PianoVAM] Missing TSV: {tsv_path}")

        T = frame_times.shape[0]
        pitch_grid = np.zeros((T, N_PITCHES), dtype=np.float32)
        onset_grid = np.zeros((T, N_PITCHES), dtype=np.float32)
        offset_grid = np.zeros((T, N_PITCHES), dtype=np.float32)

        with tsv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    onset_sec = float(parts[0])
                    key_off_sec = float(parts[1])
                    # frame_offset = float(parts[2])  # unused for now
                    note = int(parts[3])
                    # velocity = int(parts[4])        # unused for now
                except ValueError:
                    continue

                if note < NOTE_MIN or note > NOTE_MAX:
                    continue

                pitch_idx = note - NOTE_MIN

                # Determine which frame-times fall inside [onset, key_off).
                mask = (frame_times >= onset_sec) & (frame_times < key_off_sec)
                if not np.any(mask):
                    # No frame center falls inside the active interval; we still mark
                    # onset/offset at the closest frames.
                    pass
                else:
                    pitch_grid[mask, pitch_idx] = 1.0

                # Onset frame: first frame-time >= onset_sec.
                onset_frame = int(np.searchsorted(frame_times, onset_sec, side="left"))
                if 0 <= onset_frame < T:
                    onset_grid[onset_frame, pitch_idx] = 1.0
                    pitch_grid[onset_frame, pitch_idx] = 1.0  # ensure active

                # Offset frame: first frame-time >= key_off_sec.
                offset_frame = int(np.searchsorted(frame_times, key_off_sec, side="left"))
                if 0 <= offset_frame < T:
                    offset_grid[offset_frame, pitch_idx] = 1.0

        pitch = torch.from_numpy(pitch_grid)
        onset = torch.from_numpy(onset_grid)
        offset = torch.from_numpy(offset_grid)
        return pitch, onset, offset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rec_id = rec.get("id")

        # Video frames and time grid
        video, frame_times = self._load_video_and_times(rec)   # video: (T,C,H,W)
        pitch, onset, offset = self._load_labels(rec, frame_times)  # (T, 88) each

        # Sanity check on shapes
        assert pitch.shape == onset.shape == offset.shape
        T, P = pitch.shape

        # ----- frame-level aliases για το training loop -----
        # train.py περιμένει:
        #   pitch_roll / onset_roll / offset_roll (B,T,P)
        #   hand_frame / clef_frame               (B,T)
        # εδώ είμαστε single-sample (T,P), το DataLoader θα προσθέσει B.
        pitch_roll = pitch          # (T,88)
        onset_roll = onset          # (T,88)
        offset_roll = offset        # (T,88)

        # Dummy hand/clef labels (δεν τα χρησιμοποιούμε, loss weights=0.0)
        hand_frame = torch.zeros(T, dtype=torch.long)
        clef_frame = torch.zeros(T, dtype=torch.long)

        # Clip-level labels (max over time)
        pitch_clip = pitch_roll.max(dim=0).values       # (88,)
        onset_clip = onset_roll.max(dim=0).values       # (88,)
        offset_clip = offset_roll.max(dim=0).values     # (88,)

        sample: Dict[str, Any] = {
            "video": video,                         # (T, C, H, W)
            "path": str(rec["video_path"]),
            "id": rec_id,

            # frame-level labels (used by compute_loss_frame)
            "pitch_roll":  pitch_roll,             # (T, 88)
            "onset_roll":  onset_roll,             # (T, 88)
            "offset_roll": offset_roll,            # (T, 88)
            "hand_frame":  hand_frame,             # (T,)
            "clef_frame":  clef_frame,             # (T,)

            # clip-level labels
            "pitch":  pitch_clip,                  # (88,)
            "onset":  onset_clip,                  # (88,)
            "offset": offset_clip,                 # (88,)

            # για το custom evaluation
            "tsv_path": str(rec["tsv_path"]),      # GT labels
            "frame_times": torch.from_numpy(frame_times).float(),  # (T,)
        }

        # frame_targets για συμβατότητα με generic κώδικα (δεν πειράζει αν δεν τα χρησιμοποιεί όλος ο κώδικας ακόμα)
        sample["frame_targets"] = {
            "pitch": pitch_roll,     # (T,88)
            "onset": onset_roll,     # (T,88)
            "offset": offset_roll,   # (T,88)
            "hand": hand_frame,      # (T,)
            "clef": clef_frame,      # (T,),
        }

        sample["clip_targets"] = {
            "pitch": pitch_clip,     # (88,)
            "onset": onset_clip,     # (88,)
            "offset": offset_clip,   # (88,),
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
