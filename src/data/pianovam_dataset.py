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


def _parse_point(s: Optional[str]):
    """
    Parse 'x, y' string from metadata σε (x, y) ints.
    Αν κάτι πάει στραβά, επιστρέφει None.
    """
    if not s:
        return None
    try:
        x_str, y_str = s.split(",")
        return int(x_str.strip()), int(y_str.strip())
    except Exception:
        return None


def _compute_crop_box(rec: Mapping[str, Any], margin: int = 0):
    """
    Υπολογίζει bounding box (x0, y0, x1, y1) από Point_LT/RT/RB/LB.

    Επιστρέφει None αν λείπει κάποιο σημείο.
    """
    pts = []
    for key in ("Point_LT", "Point_RT", "Point_RB", "Point_LB"):
        p = _parse_point(rec.get(key))
        if p is not None:
            pts.append(p)

    if len(pts) != 4:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)

    if margin > 0:
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin

    return x0, y0, x1, y1



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
        # ---------------------------------------------------------------
        # Handskeleton supervision (MediaPipe Hands landmarks).
        # We use Handskeleton/{record_time}.json to derive a frame-level
        # left/right label at note onsets (hand_frame ∈ {0:left, 1:right}).
        # ---------------------------------------------------------------
        hands_cfg = dataset_cfg.get("handskeleton", {})
        self.use_handskeleton = bool(hands_cfg.get("enabled", True))
        self.handskeleton_dir = self.root / str(hands_cfg.get("dir", "Handskeleton"))
        self.handskeleton_fps = float(hands_cfg.get("fps", 60.0))
        self._handskel_cache: Dict[str, Dict[int, Dict[str, Optional[np.ndarray]]]] = {}
        self._handskel_cache_order: List[str] = []
        self._handskel_cache_size: int = int(hands_cfg.get("cache_size", 64))
        # MediaPipe fingertips indices: thumb/index/middle/ring/pinky tips
        self._handskel_fingertips = [4, 8, 12, 16, 20]

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
                # --- ΝΕΟ: crop box από τα σημεία του metadata ---
                margin = int(dataset_cfg.get("crop_margin", 0))  # μπορείς να το αλλάξεις στο yaml
                crop_box = _compute_crop_box(rec_copy, margin=margin)
                rec_copy["crop_box"] = crop_box
                # -----------------------------------------------

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
           return meta_split in ("train", "ext-train")
        if requested in ("val", "validation"):
            return meta_split in ("val", "valid", "validation")
        if requested == "test":
            return meta_split == "test"
        if meta_split.startswith("special"):
            return False
        # Fallback: exact match.
        return requested == meta_split

    # ------------------------------------------------------------------ #
    # Core Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def _load_video_and_times(self, rec: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Load video frames and return:
          video: tensor(T, C, H, W)
          times: np.ndarray shape (T,) with timestamps in seconds
                 aligned with the sampled frames.
           idxs:  np.ndarray shape (T,) with the sampled frame indices (in the original video)
           full_hw: (H_full, W_full) of the original decoded frames (before crop/resize)
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

        # Original decoded frame size (before crop/resize)
        full_h = int(frames.shape[1])
        full_w = int(frames.shape[2])

        # --- ΝΕΟ: πραγματικά timestamps για ΚΑΘΕ sampled frame ---
        # χρόνος = frame_index / fps
        frame_times = (idxs.astype(np.float32) / native_fps).astype(np.float32)
        # ---------------------------------------------------------

        # --- ΝΕΟ: crop γύρω από το keyboard αν έχουμε crop_box ---
        crop_box = rec.get("crop_box")
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box

            H, W = frames.shape[1], frames.shape[2]
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            x1 = max(x0 + 1, min(x1, W))
            y1 = max(y0 + 1, min(y1, H))

            print(
                f"[PianoVAM crop] id={rec.get('id')} "
                f"full={H}x{W} → crop=({x0},{y0})-({x1},{y1}) "
                f"→ {y1 - y0}x{x1 - x0}"
            )

            frames = frames[:, y0:y1, x0:x1, :]

        # --------------------------------------------------------

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
        # frame_times = np.linspace(0.0, duration_sec, T, endpoint=False, dtype=np.float32)

        return video, frame_times, idxs, (full_h, full_w)

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
        # Προσεγγιστικό χρονικό βήμα μεταξύ δύο διαδοχικών frame_times
        if T > 1:
            approx_frame_dt = float(frame_times[1] - frame_times[0])
        else:
            approx_frame_dt = 0.0

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

                # --- ΝΕΟ: αγνόησε πολύ μικρές νότες που δεν "χωράνε" στο frame grid ---
                duration = key_off_sec - onset_sec
                if approx_frame_dt > 0.0 and duration < 0.5 * approx_frame_dt:
                    # π.χ. αν frame step είναι ~2s, αγνοούμε νότες <1s
                    continue
                # -----------------------------------------------------------------------

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
    
    # ------------------------------------------------------------------ #
    # Handskeleton -> hand_frame supervision
    # ------------------------------------------------------------------ #
    def _load_handskeleton(self, rec_id: str) -> Optional[Dict[int, Dict[str, Optional[np.ndarray]]]]:
        """Load Handskeleton/{rec_id}.json and cache it.

        Returns:
          dict: frame_idx -> {"Left": (21,3) float32 or None, "Right": (21,3) float32 or None}
          or None if file missing / disabled.
        """
        if not self.use_handskeleton:
            return None

        if rec_id in self._handskel_cache:
            return self._handskel_cache[rec_id]

        path = self.handskeleton_dir / f"{rec_id}.json"
        if not path.is_file():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))
        parsed: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        for k, v in raw.items():
            try:
                fi = int(k)
            except Exception:
                continue

            left = v.get("Left")
            right = v.get("Right")

            left_arr = None
            right_arr = None
            if left is not None:
                left_arr = np.asarray(left, dtype=np.float32)  # (21,3)
            if right is not None:
                right_arr = np.asarray(right, dtype=np.float32)

            parsed[fi] = {"Left": left_arr, "Right": right_arr}

        # simple LRU-ish cache
        self._handskel_cache[rec_id] = parsed
        self._handskel_cache_order.append(rec_id)
        if len(self._handskel_cache_order) > self._handskel_cache_size:
            evict = self._handskel_cache_order.pop(0)
            self._handskel_cache.pop(evict, None)

        return parsed

    def _precompute_key_centers(self, rec: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return (88,2) array of key centers in FULL-frame pixel coords.

        Uses the 4 keyboard corner points from metadata (Point_LT/RT/RB/LB).
        """
        lt = _parse_point(rec.get("Point_LT"))
        rt = _parse_point(rec.get("Point_RT"))
        rb = _parse_point(rec.get("Point_RB"))
        lb = _parse_point(rec.get("Point_LB"))
        if lt is None or rt is None or rb is None or lb is None:
            return None

        lt = np.asarray(lt, dtype=np.float32)
        rt = np.asarray(rt, dtype=np.float32)
        rb = np.asarray(rb, dtype=np.float32)
        lb = np.asarray(lb, dtype=np.float32)

        centers = np.zeros((N_PITCHES, 2), dtype=np.float32)
        for p in range(N_PITCHES):
            u = (p + 0.5) / float(N_PITCHES)  # 0..1 across keyboard
            top = lt + u * (rt - lt)
            bot = lb + u * (rb - lb)
            c = 0.5 * (top + bot)
            centers[p, 0] = c[0]
            centers[p, 1] = c[1]
        return centers

    def _hand_distance_to_key(self, hand_lm: np.ndarray, key_xy: np.ndarray, full_hw: Tuple[int, int]) -> float:
        """Min fingertip distance (in pixels) from this hand to the key center."""
        full_h, full_w = full_hw
        # Convert normalized -> pixels
        tips = hand_lm[self._handskel_fingertips, :2].copy()  # (5,2) in [0,1]
        tips[:, 0] *= float(full_w)
        tips[:, 1] *= float(full_h)
        d = tips - key_xy[None, :]
        dist = float(np.sqrt((d * d).sum(axis=1)).min())
        return dist

    def _build_hand_frame_targets(
        self,
        rec: Dict[str, Any],
        onset_roll: torch.Tensor,
        frame_idxs: np.ndarray,
        full_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Derive hand_frame[t] ∈ {0:left, 1:right} for this clip.

        Strategy:
          - For frames that contain onset(s), assign each onset note to the
            closest hand (min fingertip distance to key center).
          - Frame label = majority vote across onset notes in that frame.
          - Fill unlabeled frames by forward/back fill.
        """
        T, P = onset_roll.shape
        hand_frame = torch.full((T,), -1, dtype=torch.long)

        rec_id = str(rec.get("id", ""))
        skel = self._load_handskeleton(rec_id)
        if skel is None:
            # No skeleton available => keep dummy label (all zeros)
            return torch.zeros((T,), dtype=torch.long)

        centers = self._precompute_key_centers(rec)
        if centers is None:
            return torch.zeros((T,), dtype=torch.long)

        onset_any = onset_roll.sum(dim=1) > 0  # (T,)
        onset_frames = onset_any.nonzero(as_tuple=False).flatten().tolist()
        if not onset_frames:
            return torch.zeros((T,), dtype=torch.long)

        for t in onset_frames:
            fi = int(frame_idxs[t])  # original video frame index
            v = skel.get(fi)
            if v is None:
                continue

            left_lm = v.get("Left")
            right_lm = v.get("Right")

            # notes at this frame
            notes = (onset_roll[t] > 0).nonzero(as_tuple=False).flatten().tolist()
            if not notes:
                continue

            votes = []
            # tie-breaker: keep the smallest distance for the chosen hand
            dist_left_sum = 0.0
            dist_right_sum = 0.0
            n_used = 0

            for p_idx in notes:
                key_xy = centers[p_idx]  # (2,) in pixels

                if left_lm is None and right_lm is None:
                    continue
                elif left_lm is None:
                    votes.append(1)
                    # approximate distance for tie-breaker
                    dist_right_sum += 0.0
                elif right_lm is None:
                    votes.append(0)
                    dist_left_sum += 0.0
                else:
                    dl = self._hand_distance_to_key(left_lm, key_xy, full_hw)
                    dr = self._hand_distance_to_key(right_lm, key_xy, full_hw)
                    if dl <= dr:
                        votes.append(0)
                        dist_left_sum += dl
                    else:
                        votes.append(1)
                        dist_right_sum += dr
                n_used += 1

            if n_used == 0 or not votes:
                continue

            # Majority vote
            n_right = sum(1 for x in votes if x == 1)
            n_left = len(votes) - n_right
            if n_left > n_right:
                hand_frame[t] = 0
            elif n_right > n_left:
                hand_frame[t] = 1
            else:
                # tie: prefer the hand with smaller avg distance
                avg_l = dist_left_sum / max(1, n_left)
                avg_r = dist_right_sum / max(1, n_right)
                hand_frame[t] = 0 if avg_l <= avg_r else 1

        # Fill unknowns by forward/back fill
        arr = hand_frame.numpy()
        last = None
        for i in range(T):
            if arr[i] >= 0:
                last = int(arr[i])
            elif last is not None:
                arr[i] = last

        nxt = None
        for i in range(T - 1, -1, -1):
            if arr[i] >= 0:
                nxt = int(arr[i])
            elif nxt is not None:
                arr[i] = nxt

        return torch.from_numpy(arr.astype(np.int64))


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rec_id = rec.get("id")

        # Video frames and time grid
        video, frame_times, frame_idxs, full_hw = self._load_video_and_times(rec)   # video: (T,C,H,W)
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

        # Hand labels from Handskeleton (0=Left, 1=Right). Falls back to zeros if unavailable.
        hand_frame = self._build_hand_frame_targets(rec, onset_roll, frame_idxs, full_hw)
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
            "frame_idxs": torch.from_numpy(frame_idxs).long(),  # (T,) original frame indices
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
