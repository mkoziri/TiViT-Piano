"""Purpose:
    Provide a PianoVAM loader that mirrors the PianoYT runtime surface so
    pipelines can switch datasets via configuration only.  The loader reuses
    the PianoYT implementation for sampling, target construction, and optional
    augmentations while resolving a PianoVAM-specific root directory.

Key Functions/Classes:
    - PianoVAMDataset: Thin subclass of :class:`PianoYTDataset` that points to
      the local PianoVAM folder (PianoYT-style layout on disk).
    - make_dataloader(): Factory matching :func:`pianoyt_dataset.make_dataloader`
      but defaulting to PianoVAM roots and leaving the rest of the interface
      unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from utils.determinism import DEFAULT_SEED, make_loader_components
from utils.frame_targets import resolve_frame_target_spec, resolve_soft_target_config
from utils.identifiers import canonical_video_id

import src.data.pianoyt_dataset as yt
from src.data.pianoyt_dataset import SampleBuildResult
from .sampler_utils import build_onset_balanced_sampler

LOGGER = logging.getLogger(__name__)


def _safe_expanduser(path: os.PathLike[str] | str) -> Path:
    """Expand user safely; mirrors PianoYT helper without importing it."""

    candidate = Path(path)
    try:
        return candidate.expanduser()
    except RuntimeError:
        return candidate

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

def _expand_root(root_dir: Optional[str]) -> Path:
    """Resolve the PianoVAM root directory with environment fallbacks."""

    candidates = []
    if root_dir:
        candidates.append(_safe_expanduser(os.path.expandvars(str(root_dir))))

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
        # How to propagate hand labels to non-onset frames.
        #   - "none": keep -1 on unlabeled frames (recommended for clean supervision)
        #   - "ffill": forward-fill from the last labeled frame
        #   - "bfill": backward-fill from the next labeled frame
        #   - "both": ffill then bfill (fills all frames if at least one label exists)
        self.handskeleton_fill = str(
            hands_cfg.get("fill", hands_cfg.get("fill_strategy", "none"))
        ).strip().lower()
        if self.handskeleton_fill not in ("none", "ffill", "bfill", "both"):
            self.handskeleton_fill = "none"
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

    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "PianoVAM_v1.0",
            project_root / "data" / "PianoVAM",
        ]
    )
    candidates.extend(
        [
            _safe_expanduser("~/datasets/PianoVAM_v1.0"),
            _safe_expanduser("~/datasets/PianoVAM"),
        ]
    )

    for cand in candidates:
        if cand and cand.exists():
            return cand

    msg = (
        "Unable to locate PianoVAM root. Set dataset.root_dir or define "
        "TIVIT_DATA_DIR/DATASETS_HOME to point at PianoVAM_v1.0 layout."
    )
    LOGGER.error(msg)
    raise FileNotFoundError(msg)


def _load_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    meta_path = root / "metadata_v2.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[PianoVAM] metadata_v2.json missing at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"[PianoVAM] metadata_v2.json is not a dict: {type(raw)}")
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _parse_point(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except (TypeError, ValueError):
                return None
    if isinstance(value, Sequence) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _crop_from_points(entry: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """Compute (min_y, max_y, min_x, max_x) from Point_LT/RT/RB/LB."""

    labels = ["Point_LT", "Point_RT", "Point_RB", "Point_LB"]
    points: list[Tuple[float, float]] = []
    for key in labels:
        pt = _parse_point(entry.get(key))
        if pt is not None:
            points.append(pt)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    try:
        return (int(round(min_y)), int(round(max_y)), int(round(min_x)), int(round(max_x)))
    except Exception:
        return None


def _split_matches(requested: str, meta_split: str) -> bool:
    req = requested.lower()
    meta = meta_split.lower()
    if req in {"val", "valid", "validation"}:
        return meta in {"val", "valid", "validation"}
    if req in {"train", "ext-train", "ext_train"}:
        return meta in {"train", "ext-train", "ext_train"}
    if req in {"test"}:
        return meta == "test"
    if req.startswith("special"):
        return meta.startswith("special")
    return req == meta


def _resolve_media_paths(root: Path, record_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return video, midi, tsv paths for ``record_id``."""

    vid = root / "Video" / f"{record_id}.mp4"
    midi_mid = root / "MIDI" / f"{record_id}.mid"
    midi_midi = root / "MIDI" / f"{record_id}.midi"
    midi = midi_mid if midi_mid.exists() else midi_midi if midi_midi.exists() else None
    tsv = root / "TSV" / f"{record_id}.tsv"
    if not vid.exists():
        vid = None
    if tsv.exists():
        tsv_path: Optional[Path] = tsv
    else:
        tsv_path = None
    # If MIDI is absent but TSV exists, use TSV as the label source.
    if midi is None and tsv_path is not None:
        midi = tsv_path
    return vid, midi, tsv_path


_LABEL_FALLBACK_LOGGED: set[str] = set()


def _read_tsv_events(tsv_path: Path) -> torch.Tensor:
    """Parse PianoVAM TSV and return (N,3) tensor [onset_sec, key_offset_sec, pitch]."""

    if not tsv_path or not tsv_path.exists():
        return torch.zeros((0, 3), dtype=torch.float32)

    rows: list[tuple[float, float, float]] = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                # fallback to whitespace split if tabs missing
                parts = line.split()
            if len(parts) < 5:
                continue
            try:
                onset = float(parts[0])
                key_offset = float(parts[1])  # use key_offset as the offset value
                note = float(parts[3])
            except (TypeError, ValueError):
                continue
            if onset < 0 or key_offset <= onset:
                continue
            rows.append((onset, key_offset, note))

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


class PianoVAMDataset(yt.PianoYTDataset):
    """PianoVAM dataset that mirrors PianoYT surface but uses metadata_v2 layout."""

    # ------------------------------------------------------------------ #
    # Helper: split matching
    # ------------------------------------------------------------------ #
    def _split_matches(self, requested: str, meta_split: str) -> bool:
        """Return True if the metadata split should be included for this requested split."""
        requested = requested.lower()
        meta_split = meta_split.lower()

        if requested == "train":
           return meta_split in ("train", "ext-train")
        if requested in ("val", "valid", "validation"):
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
            # No skeleton available => no supervision (ignored by CE in train.py)
            return torch.full((T,), -1, dtype=torch.long)

        centers = self._precompute_key_centers(rec)
        if centers is None:
            return torch.full((T,), -1, dtype=torch.long)

        onset_any = onset_roll.sum(dim=1) > 0  # (T,)
        onset_frames = onset_any.nonzero(as_tuple=False).flatten().tolist()
        if not onset_frames:
            return torch.full((T,), -1, dtype=torch.long)

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

        fill_mode = getattr(self, "handskeleton_fill", "none")
        if fill_mode == "none":
            return hand_frame

        arr = hand_frame.numpy()
        if fill_mode in ("ffill", "both"):
            last = None
            for i in range(T):
                if arr[i] >= 0:
                    last = int(arr[i])
                elif last is not None:
                    arr[i] = last
        if fill_mode in ("bfill", "both"):
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
        # Clip-level labels (used by clip-mode training/eval paths).
        # Keep these always defined so generic training/eval does not fall back to dummy targets.
        valid_hand = hand_frame >= 0
        if bool(valid_hand.any().item()):
            hand_clip = torch.mode(hand_frame[valid_hand]).values.to(torch.long)
        else:
            hand_clip = torch.tensor(0, dtype=torch.long)
        clef_clip = torch.tensor(0, dtype=torch.long)

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
            "hand":   hand_clip,
            "clef":   clef_clip,

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
            "hand":   hand_clip,
            "clef":   clef_clip,
        }

        return sample







def make_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    drop_last: bool = False,
    *,
    seed: Optional[int] = None,
):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))

    only_video_cfg = dcfg.get("only_video")
    avlag_disabled_cfg = bool(dcfg.get("avlag_disabled", False))

    resolved_root = _expand_root(dcfg.get("root_dir"))
    dataset_cfg = dict(dcfg)
    dataset_cfg["root_dir"] = str(resolved_root)

    dataset = PianoVAMDataset(
        root_dir=str(resolved_root),
        split=split,
        frames=int(dataset_cfg.get("frames", 32)),
        stride=stride,
        resize=tuple(dataset_cfg.get("resize", [224, 224])),
        tiles=int(dataset_cfg.get("tiles", 3)),
        channels=int(dataset_cfg.get("channels", 3)),
        normalize=bool(dataset_cfg.get("normalize", True)),
        manifest=manifest_path,
        decode_fps=decode_fps,
        dataset_cfg=dataset_cfg,
        full_cfg=cfg,
        only_video=only_video_cfg,
        avlag_disabled=avlag_disabled_cfg,
    )
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    soft_cfg_raw = training_cfg.get("soft_targets") if isinstance(training_cfg, Mapping) else None
    dataset.soft_target_cfg = resolve_soft_target_config(soft_cfg_raw)

    include_low = bool(dataset_cfg.get("include_low_res", False))
    excluded_ids = set()
    if not include_low:
        excluded_ids = yt._read_excluded(dataset.root, dataset_cfg.get("excluded_list"))
    dataset.configure(
        include_low_res=include_low,
        excluded_ids=excluded_ids,
        apply_crop=bool(dataset_cfg.get("apply_crop", True)),
        crop_rescale=str(dataset_cfg.get("crop_rescale", "auto")),
    )

    if only_video_cfg and not getattr(dataset, "_only_filter_applied", False):
        only_canon = canonical_video_id(only_video_cfg)
        if not dataset.filter_to_video(only_canon):
            LOGGER.warning("[PianoVAM] --only filter skipped; id=%s not found", only_canon)

    max_clips = dataset_cfg.get("max_clips")
    dataset.limit_max_clips(max_clips if isinstance(max_clips, int) else None)
    dataset.max_clips = max_clips
    dataset.args_max_clips_or_None = max_clips if isinstance(max_clips, int) else None

    dataset.annotations_root = dataset_cfg.get("annotations_root")
    dataset.label_format = dataset_cfg.get("label_format", "midi")
    dataset.label_targets = dataset_cfg.get("label_targets", ["pitch", "onset", "offset", "hand", "clef"])
    dataset.require_labels = bool(dataset_cfg.get("require_labels", False))
    dataset.frame_targets_cfg = dict(dataset_cfg.get("frame_targets", {}) or {})
    dataset.frame_target_spec = resolve_frame_target_spec(
        dataset.frame_targets_cfg,
        frames=dataset.frames,
        stride=stride,
        fps=decode_fps,
        canonical_hw=dataset.canonical_hw,
    )
    dataset.frame_target_summary = (
        dataset.frame_target_spec.summary()
        if dataset.frame_target_spec is not None
        else "targets_conf: disabled"
    )
    dataset._rebuild_valid_index_cache(log_summary=False)

    def _collate(batch):
        vids = [b["video"] for b in batch]
        paths = [b["path"] for b in batch]
        if not vids:
            raise RuntimeError("Empty batch supplied to PianoVAM collate function")

        dims = vids[0].dim()
        if any(v.dim() != dims for v in vids):
            raise RuntimeError("Inconsistent tensor ranks in PianoVAM batch")

        max_shape = tuple(max(v.shape[d] for v in vids) for d in range(dims))
        x = vids[0].new_zeros((len(vids),) + max_shape)
        for idx, vid in enumerate(vids):
            slices = tuple(slice(0, size) for size in vid.shape)
            x[(idx,) + slices] = vid

        out = {"video": x, "path": paths}
        extra_keys = set().union(*[set(d.keys()) for d in batch]) - {"video", "path"}
        for k in extra_keys:
            vals = [d[k] for d in batch if k in d]
            if len(vals) != len(batch):
                continue
            if k == "labels":
                out[k] = vals
            else:
                v0 = vals[0]
                if torch.is_tensor(v0):
                    try:
                        out[k] = torch.stack(vals, dim=0)
                    except Exception:
                        out[k] = vals
                else:
                    out[k] = vals
        return out

    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    persistent_workers_cfg = bool(dataset_cfg.get("persistent_workers", False))
    persistent_workers = persistent_workers_cfg if num_workers > 0 else False
    prefetch_factor_cfg = dataset_cfg.get("prefetch_factor")
    prefetch_factor: Optional[int] = None
    if num_workers > 0 and prefetch_factor_cfg is not None:
        try:
            prefetch_factor_candidate = int(prefetch_factor_cfg)
        except (TypeError, ValueError):
            prefetch_factor_candidate = None
        if prefetch_factor_candidate is not None and prefetch_factor_candidate > 0:
            prefetch_factor = prefetch_factor_candidate

    base_seed = seed if seed is not None else getattr(dataset, "data_seed", None)
    if base_seed is None:
        base_seed = DEFAULT_SEED
    generator, worker_init_fn = make_loader_components(
        int(base_seed), namespace=f"{dataset.__class__.__name__}:{split}"
    )

    sampler = build_onset_balanced_sampler(
        dataset,
        dataset_cfg.get("sampler"),
        base_seed=int(base_seed),
    )
    shuffle_flag = bool(dataset_cfg.get("shuffle", True)) if split == "train" else False
    if sampler is not None:
        shuffle_flag = False

    loader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(dataset_cfg.get("batch_size", 2)),
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader


__all__ = ["PianoVAMDataset", "make_dataloader"]
