"""Purpose:
    Implement the OMAPS dataset loader used by TiViT-Piano, including video
    decoding, label parsing, and frame-level target generation.

Key Functions/Classes:
    - OMAPSDataset: PyTorch dataset that yields tiled video clips, timing
      metadata, and structured label targets.
    - make_dataloader(): Factory that builds dataloaders with appropriate
      collate functions for clip or frame objectives.
    - Helper utilities such as ``_load_clip_decord`` and ``_build_frame_targets``
      perform decoding, tiling, and pianoroll construction.

CLI:
    Not a standalone CLI.  The dataset is consumed by scripts like
    :mod:`scripts.train`, :mod:`scripts.eval_thresholds`, and diagnostic tools.
"""

import os
import glob
import math
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.time_grid import frame_to_sec, sec_to_frame
from utils.tiling import tile_vertical_token_aligned

# Try decord first (fast), else fallback to OpenCV
HAVE_DECORD = False
try:
    import decord
    decord.bridge.set_bridge('torch')
    HAVE_DECORD = True
except Exception:
    import cv2
    CV2_INTER = cv2.INTER_AREA

def _expand_root(root_dir: Optional[str]) -> Path:
    """
    Picks dataset root in this priority:
    1) explicit root_dir (YAML)
    2) $TIVIT_DATA_DIR/OMAPS or $DATASETS_HOME/OMAPS
    3) ~/datasets/OMAPS
    """
    # explicit from YAML
    if root_dir:
        p = Path(root_dir).expanduser()
        if p.exists():
            return p
    # env fallbacks
    env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env:
        cand = Path(env).expanduser().joinpath("OMAPS")
        if cand.exists():
            return cand
    # default
    return Path("~/datasets/OMAPS").expanduser()

def _list_videos(root: Path, split: str) -> List[Path]:
    split_dir = root.joinpath(split)
    pattern = str((split_dir if split_dir.exists() else root).joinpath("**/*.mp4"))
    vids = [Path(p) for p in glob.glob(pattern, recursive=True)]
    vids.sort()
    return vids

def _read_manifest(path: str) -> set:
    """Read manifest file listing filename stems; '#' comments allowed."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    ids = set()
    with open(p, "r") as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                ids.add(line)
    return ids

def _load_clip_decord(path: Path, frames: int, stride: int,
                      resize_hw: Tuple[int, int], channels: int) -> torch.Tensor:
    """
    Returns: T, C, H, W in [0,1], float32
    """
    vr = decord.VideoReader(str(path))
    total = len(vr)
    idxs = list(range(0, total, stride))[:frames]
    if len(idxs) == 0:
        idxs = [0]
    if len(idxs) < frames:
        idxs += [idxs[-1]] * (frames - len(idxs))
    batch = vr.get_batch(idxs)  # T,H,W,C (uint8)
    x = batch.to(torch.float32) / 255.0  # T,H,W,C

    # RGB->gray if requested
    if channels == 1 and x.shape[-1] == 3:
        x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)

    # resize with torch interpolate
    import torch.nn.functional as F
    x = x.permute(0, 3, 1, 2)  # T,C,H,W
    H, W = x.shape[-2:]
    if (H, W) != tuple(resize_hw):
        x = F.interpolate(x, size=resize_hw, mode="area")
    return x

def _load_clip_cv2(path: Path, frames: int, stride: int,
                   resize_hw: Tuple[int, int], channels: int) -> torch.Tensor:
    """
    Returns: T, C, H, W in [0,1], float32
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    imgs = []
    i = 0
    got = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=CV2_INTER)
            imgs.append(frame)
            got = True
            if len(imgs) == frames:
                break
        i += 1
    cap.release()
    if not got:
        raise RuntimeError(f"Failed to read frames from {path}")
    while len(imgs) < frames:
        imgs.append(imgs[-1])
    x = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32) / 255.0  # T,H,W,C
    if channels == 1:
        x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)
    x = x.permute(0, 3, 1, 2)  # T,C,H,W
    return x

def _labels_from_txt(txt_path: Path, t0: float, t1: float):
    """
    Parse a .txt sidecar file with lines: onset_sec  offset_sec  pitch
    Returns dict with pitch, onset, offset, hand, clef
    """
    pitches = []
    had_onset, had_offset = 0.0, 0.0

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
            # Does this note overlap with [t0, t1)?
            if onset < t1 and offset > t0:
                pitches.append(pitch)
                if t0 <= onset < t1:
                    had_onset = 1.0
                if t0 < offset <= t1:
                    had_offset = 1.0

    if len(pitches) == 0:
        pitch_class = 60  # default to middle C if no notes
        hand = 0
        clef = 2  # ambiguous
    else:
        # simple baseline: choose most frequent pitch
        from collections import Counter
        pitch_class, _ = Counter(pitches).most_common(1)[0]
        hand = 0 if pitch_class < 60 else 1
        clef = 0 if pitch_class < 60 else (1 if pitch_class > 64 else 2)

    return {
        "pitch": pitch_class,
        "onset": had_onset,
        "offset": had_offset,
        "hand": hand,
        "clef": clef,
    }

def _read_txt_events(txt_path: Path):
    """
    Returns a torch.FloatTensor of shape (N, 3) with columns:
    [onset_sec, offset_sec, pitch]
    If no valid lines, returns shape (0, 3).
    """
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                onset = float(parts[0])
                offset = float(parts[1])
                pitch = float(int(parts[2]))  # store as float for a single dtype tensor
                rows.append((onset, offset, pitch))
            except ValueError:
                continue
    if len(rows) == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)

def _load_clip_with_random_start(path: Path,
                                 frames: int,
                                 stride: int,
                                 channels: int,
                                 training: bool,
                                 decode_fps: float):
    """
    Load a clip using a randomized start (training=True) or start=0 (else).
    Decoding is performed on a fixed "decode_fps" grid so that all clips
    align temporally, regardless of their native fps.  The returned
    ``start_idx`` is expressed in units of ``decode_fps``.

    Returns:
        clip (T,C,H,W) float32 in [0,1], start_idx (int)
    """
    
    hop_seconds = float(stride) / float(decode_fps)
    # --- decord fast path ---
    if HAVE_DECORD:
        import decord, torch
        import torch.nn.functional as F
        vr = decord.VideoReader(str(path))
        native_fps = float(vr.get_avg_fps())
        total = len(vr)
        duration = total / native_fps if native_fps > 0 else 0.0

        clip_span = (frames - 1) * hop_seconds
        max_start_sec = max(0.0, duration - clip_span)
        if training and max_start_sec > 0:
            import random
            start_idx = random.randint(0, int(max_start_sec * decode_fps))
        else:
            start_idx = 0
        start_sec = frame_to_sec(start_idx, 1.0 / decode_fps)

        times = [frame_to_sec(start_idx + k * stride, 1.0 / decode_fps)
                 for k in range(frames)]
        idxs = [sec_to_frame(t, 1.0 / native_fps, max_idx=total - 1)
                for t in times]
        batch = vr.get_batch(idxs)  # T,H,W,C (uint8)
        x = batch.to(torch.float32) / 255.0  # T,H,W,C

        # RGB->gray if requested
        if channels == 1 and x.shape[-1] == 3:
            x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)

        # to (T,C,H,W) and resize like existing decord helper
        x = x.permute(0, 3, 1, 2)  # T,C,H,W

        return x, start_idx

    # --- OpenCV fallback (keeps your behavior) ---
    else:
        import cv2, numpy as np, torch
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or decode_fps
        duration = total / native_fps if native_fps > 0 else 0.0

        clip_span = (frames - 1) * hop_seconds
        max_start_sec = max(0.0, duration - clip_span)
        if training and max_start_sec > 0:
            import random
            start_idx = random.randint(0, int(max_start_sec * decode_fps))
        else:
            start_idx = 0
        start_sec = frame_to_sec(start_idx, 1.0 / decode_fps)

        times = [frame_to_sec(start_idx + k * stride, 1.0 / decode_fps)
                 for k in range(frames)]
        frame_idxs = [sec_to_frame(t, 1.0 / native_fps, max_idx=total - 1)
                      for t in times]

        imgs = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
        cap.release()

        if len(imgs) == 0:
            raise RuntimeError(f"Failed to read frames from {path}")

        # pad by repeating last if short
        while len(imgs) < frames:
            imgs.append(imgs[-1])

        x = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32) / 255.0  # T,H,W,C
        if channels == 1:
            x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)  # T,C,H,W
        return x, start_idx

def _extract_crop_values(meta: Union[None, Sequence[float], torch.Tensor, Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    """Normalize metadata describing a crop box into ``(min_y, max_y, min_x, max_x)``."""

    if meta is None:
        return None

    if isinstance(meta, torch.Tensor):
        if meta.numel() < 4:
            return None
        vals = meta.flatten()[:4].tolist()
    elif isinstance(meta, dict):
        keys = [
            ("min_y", "max_y", "min_x", "max_x"),
            ("y0", "y1", "x0", "x1"),
            ("top", "bottom", "left", "right"),
        ]
        vals = None
        for candidate in keys:
            if all(k in meta for k in candidate):
                vals = [meta[k] for k in candidate]  # type: ignore[index]
                break
        if vals is None and "crop" in meta:
            crop_val = meta["crop"]
            if isinstance(crop_val, (list, tuple)) and len(crop_val) >= 4:
                vals = list(crop_val[:4])
        if vals is None:
            return None
    elif isinstance(meta, Sequence):
        if len(meta) < 4:
            return None
        vals = list(meta[:4])
    else:
        return None

    try:
        return tuple(float(v) for v in vals[:4])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def apply_registration_crop(frames: torch.Tensor,
                            meta: Union[None, Sequence[float], torch.Tensor, Dict[str, Any]],
                            cfg: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """Crop ``frames`` (T,C,H,W) according to registration metadata."""

    _ = cfg  # reserved for future configuration options

    coords = _extract_crop_values(meta)
    if coords is None:
        return frames

    min_y, max_y, min_x, max_x = coords
    T, C, H, W = frames.shape

    is_normalized = all(0.0 <= v <= 1.0 for v in (min_y, max_y, min_x, max_x))
    if is_normalized:
        min_y *= H
        max_y *= H
        min_x *= W
        max_x *= W

    y0 = int(math.floor(min_y))
    y1 = int(math.ceil(max_y))
    x0 = int(math.floor(min_x))
    x1 = int(math.ceil(max_x))

    y0 = max(0, min(y0, H - 1))
    x0 = max(0, min(x0, W - 1))
    y1 = max(y0 + 1, min(y1, H))
    x1 = max(x0 + 1, min(x1, W))

    return frames[..., y0:y1, x0:x1]


def resize_to_canonical(frames: torch.Tensor,
                        canonical_hw: Optional[Sequence[int]],
                        interp: str = "bilinear") -> torch.Tensor:
    """Resize clip ``frames`` (T,C,H,W) to canonical ``(H,W)`` if provided."""

    if not canonical_hw or len(canonical_hw) < 2:
        return frames

    target_h = int(round(float(canonical_hw[0])))
    target_w = int(round(float(canonical_hw[1])))
    if target_h <= 0 or target_w <= 0:
        return frames

    h, w = frames.shape[-2:]
    if (h, w) == (target_h, target_w):
        return frames

    mode = str(interp or "bilinear").lower()
    alignable = {"linear", "bilinear", "bicubic", "trilinear"}
    if mode in alignable:
        return F.interpolate(frames, size=(target_h, target_w), mode=mode, align_corners=False)
    return F.interpolate(frames, size=(target_h, target_w), mode=mode)


def _apply_color_jitter(frames: torch.Tensor, cfg: Dict[str, Any], rng: np.random.Generator) -> torch.Tensor:
    if not cfg:
        return frames

    brightness = float(cfg.get("brightness", 0.0))
    if brightness > 0.0:
        factor = float(rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
        frames = torch.clamp(frames * factor, 0.0, 1.0)

    contrast = float(cfg.get("contrast", 0.0))
    if contrast > 0.0:
        mean = frames.mean(dim=(-1, -2), keepdim=True)
        factor = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        frames = torch.clamp((frames - mean) * factor + mean, 0.0, 1.0)

    saturation = float(cfg.get("saturation", 0.0))
    if saturation > 0.0 and frames.shape[1] > 1:
        gray = frames.mean(dim=1, keepdim=True)
        factor = float(rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
        frames = torch.clamp((frames - gray) * factor + gray, 0.0, 1.0)

    return frames


def apply_global_augment(frames: torch.Tensor,
                         aug_cfg: Optional[Dict[str, Any]],
                         *,
                         base_seed: Optional[int] = None,
                         sample_index: Optional[int] = None,
                         start_idx: Optional[int] = None,
                         interp: str = "bilinear",
                         id_key: Optional[Union[str, int]] = None) -> torch.Tensor:
    """Apply clip-consistent global augmentations prior to tiling."""

    if not aug_cfg or not bool(aug_cfg.get("enabled", False)):
        return frames

    seed_val = int(base_seed) if base_seed is not None else 0
    if sample_index is not None:
        seed_val = (seed_val + int(sample_index) * 1000003) % (2 ** 32)
    if start_idx is not None:
        seed_val = (seed_val + int(start_idx) * 9173) % (2 ** 32)
    if id_key is not None:
        key_hash = zlib.crc32(str(id_key).encode("utf-8")) & 0xFFFFFFFF
        seed_val = (seed_val + key_hash) % (2 ** 32)

    rng = np.random.default_rng(seed_val)
    mode = str(aug_cfg.get("interp", interp or "bilinear")).lower()
    alignable = {"linear", "bilinear", "bicubic", "trilinear"}

    def _resize_if_needed(tensor: torch.Tensor, size_hw: Sequence[float]) -> torch.Tensor:
        target_h = int(round(float(size_hw[0])))
        target_w = int(round(float(size_hw[1])))
        if target_h <= 0 or target_w <= 0:
            return tensor
        if tensor.shape[-2:] == (target_h, target_w):
            return tensor
        if mode in alignable:
            return F.interpolate(tensor, size=(target_h, target_w), mode=mode, align_corners=False)
        return F.interpolate(tensor, size=(target_h, target_w), mode=mode)

    resize_hw = aug_cfg.get("resize_jitter")
    if isinstance(resize_hw, Sequence) and len(resize_hw) >= 2:
        frames = _resize_if_needed(frames, resize_hw)

    crop_hw = aug_cfg.get("random_crop_hw")
    if isinstance(crop_hw, Sequence) and len(crop_hw) >= 2:
        crop_h = int(round(float(crop_hw[0])))
        crop_w = int(round(float(crop_hw[1])))
        crop_h = max(1, min(frames.shape[-2], crop_h))
        crop_w = max(1, min(frames.shape[-1], crop_w))
        max_y = frames.shape[-2] - crop_h
        max_x = frames.shape[-1] - crop_w
        y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        frames = frames[..., y0:y0 + crop_h, x0:x0 + crop_w]

    if bool(aug_cfg.get("hflip", False)):
        prob = float(aug_cfg.get("hflip_prob", 0.5))
        if float(rng.random()) < prob:
            frames = torch.flip(frames, dims=[-1])

    color_cfg = aug_cfg.get("color_jitter")
    if isinstance(color_cfg, dict):
        frames = _apply_color_jitter(frames, color_cfg, rng)

    return frames
  
  
def _build_frame_targets(labels: torch.Tensor,
                         T: int, stride: int, fps: float,
                         note_min: int = 21, note_max: int = 108,
                         tol: float = 0.025,
                         fill_mode: str = "overlap",
                         hand_from_pitch: bool = True,
                         clef_thresholds=(60, 64),
                         dilate_active_frames: int = 0,
                         targets_sparse: bool = False):
    """Build per-frame targets aligned to sampled frames."""
    hop_seconds = stride / max(1.0, float(fps))
    
    T = int(T)
    P = int(note_max - note_min + 1)
    pitch_roll = torch.zeros((T, P), dtype=torch.float32)
    onset_roll = torch.zeros((T, P), dtype=torch.float32)
    offset_roll = torch.zeros((T, P), dtype=torch.float32)

    if labels is None or labels.numel() == 0:
        hand_frame = torch.zeros((T,), dtype=torch.long)
        clef_frame = torch.full((T,), 2, dtype=torch.long)  # ambiguous
        return {
            "pitch_roll": pitch_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "hand_frame": hand_frame,
            "clef_frame": clef_frame,
        }

    on = labels[:, 0]
    off = labels[:, 1]
    pit = labels[:, 2].to(torch.int64)

    # optional pitch range clamp  and shift to 0-index
    mask_pitch = (pit >= int(note_min)) & (pit <= int(note_max))
    on, off, pit = on[mask_pitch], off[mask_pitch], pit[mask_pitch] - int(note_min)

    on_frames = sec_to_frame(on, hop_seconds)
    off_frames = sec_to_frame(off, hop_seconds)
    on_frames = on_frames.clamp(0, T - 1)
    off_frames = off_frames.clamp(0, T)

    for s, e, p in zip(on_frames, off_frames, pit):
        s = int(s)
        e = int(e)
        if e <= s:
            e = min(s + 1, T)
        pitch_roll[s:e, p] = 1.0
        onset_roll[s, p] = 1.0
        off_idx = min(e, T - 1)
        offset_roll[off_idx, p] = 1.0
        
    # optional temporal dilation by N frames
    if dilate_active_frames and dilate_active_frames > 0:
        import torch.nn.functional as F
        ksz = 2 * dilate_active_frames + 1
        ker = torch.ones((1, 1, ksz), dtype=torch.float32)
        for roll in (pitch_roll, onset_roll, offset_roll):
            # (T, P) -> (P, 1, T), conv, then back
            x = roll.transpose(0, 1).unsqueeze(1)  # (P,1,T)
            x = F.conv1d(x, ker, padding=dilate_active_frames)
            roll.copy_((x.squeeze(1) > 0).transpose(0, 1).float())

    # simple per-frame hand/clef heuristic from active pitches
    if hand_from_pitch:
        active = (pitch_roll > 0)
        # average pitch per frame (fallback to 60 if none active)
        pitch_ids = torch.arange(P, dtype=torch.float32) + float(note_min)
        sums = (pitch_roll * pitch_ids[None, :]).sum(dim=1)
        counts = active.sum(dim=1).clamp(min=1)
        avg_pitch = torch.where(active.any(dim=1), sums / counts, torch.full((T,), 60.0))

        lh_thr, rh_thr = int(clef_thresholds[0]), int(clef_thresholds[1])
        hand_frame = (avg_pitch >= lh_thr).long()  # 0=left (<lh_thr), 1=right (>=lh_thr)
        clef_frame = torch.where(
            avg_pitch < lh_thr, torch.zeros_like(hand_frame),
            torch.where(avg_pitch > rh_thr, torch.ones_like(hand_frame),
                        torch.full_like(hand_frame, 2))
        )
    else:
        hand_frame = torch.zeros((T,), dtype=torch.long)
        clef_frame = torch.full((T,), 2, dtype=torch.long)

    if targets_sparse:
        # (optional) return sparse indices instead of dense rolls (not used now)
        pass

    return {
        "pitch_roll":  pitch_roll,
        "onset_roll":  onset_roll,
        "offset_roll": offset_roll,
        "hand_frame":  hand_frame,
        "clef_frame":  clef_frame,
    }


class OMAPSDataset(Dataset):
    """
    Visual-only OMAPS loader.
    Yields dict:
      - video: T, tiles, C, H, W (float32 in [0,1])
      - path:  str
    """
    def __init__(self,
                 root_dir: Optional[str],
                 split: str = "test",
                 frames: int = 32,
                 stride: int = 2,
                 resize: Tuple[int, int] = (224, 224),
                 tiles: int = 3,
                 channels: int = 3,
                 normalize: bool = True,
                 manifest: Optional[str] = None,
                 decode_fps: float = 30.0,
                 *,
                 dataset_cfg: Optional[Dict[str, Any]] = None,
                 full_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.root = _expand_root(root_dir)
        self.split = split
        self.frames = int(frames)
        self.stride = int(stride)
        self.resize = tuple(resize)
        self.tiles = int(tiles)
        self.channels = int(channels)
        self.normalize = bool(normalize)
        self.decode_fps = float(decode_fps)
        self.dataset_cfg = dict(dataset_cfg or {})
        self.full_cfg = dict(full_cfg or {})

        reg_cfg = dict(self.dataset_cfg.get("registration", {}) or {})
        self.registration_cfg = reg_cfg
        self.registration_enabled = bool(reg_cfg.get("enabled", False))
        self.registration_interp = str(reg_cfg.get("interp", "bilinear"))

        canonical_cfg = self.dataset_cfg.get("canonical_hw", self.resize)
        if isinstance(canonical_cfg, Sequence) and len(canonical_cfg) >= 2:
            self.canonical_hw = (int(round(float(canonical_cfg[0]))), int(round(float(canonical_cfg[1]))))
        else:
            self.canonical_hw = tuple(self.resize)

        global_aug_cfg = self.dataset_cfg.get("global_aug")
        if not isinstance(global_aug_cfg, dict):
            global_aug_cfg = reg_cfg.get("global_aug") if isinstance(reg_cfg.get("global_aug"), dict) else {}
        self.global_aug_cfg = dict(global_aug_cfg or {})
        self.global_aug_enabled = bool(self.global_aug_cfg.get("enabled", False))

        tiling_cfg = {}
        if isinstance(self.full_cfg, dict):
            tiling_cfg = dict(self.full_cfg.get("tiling", {}) or {})
        self.tiling_cfg = tiling_cfg
        patch_w_cfg = tiling_cfg.get("patch_w")
        if patch_w_cfg is None:
            model_cfg = self.full_cfg.get("model", {}) if isinstance(self.full_cfg, dict) else {}
            trans_cfg = model_cfg.get("transformer", {}) if isinstance(model_cfg, dict) else {}
            patch_w_cfg = trans_cfg.get("input_patch_size")
        self.tiling_patch_w = int(patch_w_cfg) if patch_w_cfg is not None else None
        tokens_split_cfg = tiling_cfg.get("tokens_split", "auto")
        if isinstance(tokens_split_cfg, Sequence) and not isinstance(tokens_split_cfg, str):
            self.tiling_tokens_split = [int(v) for v in tokens_split_cfg]
        else:
            self.tiling_tokens_split = tokens_split_cfg
        self.tiling_overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))
        self._tiling_log_once = False
        self._registration_off_logged = False
        self.apply_crop = True

        if self.tiling_patch_w is None:
            raise ValueError(
                "tiling.patch_w or model.transformer.input_patch_size required for token-aligned tiling"
            )

        data_cfg = self.full_cfg.get("data", {}) if isinstance(self.full_cfg, dict) else {}
        experiment_cfg = self.full_cfg.get("experiment", {}) if isinstance(self.full_cfg, dict) else {}
        seed_val = data_cfg.get("seed", experiment_cfg.get("seed"))
        self.data_seed = int(seed_val) if seed_val is not None else None
        
        self.videos = _list_videos(self.root, split)
        if manifest:
            ids = _read_manifest(manifest)
            self.videos = [v for v in self.videos if v.stem in ids]
        #for overfit/debug runs: dataset will load only max_clips number of videos.
        max_clips = getattr(self, "max_clips", None)
        if max_clips is None:
            max_clips = dcfg.get("max_clips", None) if "dcfg" in locals() else None
        if max_clips is not None and len(self.videos) > max_clips:
            self.videos = self.videos[:max_clips]
        #End
        if len(self.videos) == 0:
            raise FileNotFoundError(f"No .mp4 files found under {self.root} (split='{split}').")
        


    def __len__(self):
        return len(self.videos)


    def __getitem__(self, idx: int):
        path = self.videos[idx]

        # --- load video clip (existing logic) ---
        #if HAVE_DECORD:
        #    clip = _load_clip_decord(path, self.frames, self.stride, self.resize, self.channels)
        #else:
        #    clip = _load_clip_cv2(path, self.frames, self.stride, self.resize, self.channels)
        #changed so it should always start from frame 0 when training.
        # --- load video clip with randomized start for train, deterministic for val/test ---
        is_train = (self.split == "train")
        clip, start_idx = _load_clip_with_random_start(
            path=path,
            frames=self.frames,
            stride=self.stride,
            channels=self.channels,
            training=is_train,
            decode_fps=self.decode_fps,
        )
        native_h, native_w = clip.shape[-2:]
        print(f"decode {path.name}: native {native_h}x{native_w}", flush=True)
        if self.registration_enabled and self.apply_crop:
            before_h, before_w = clip.shape[-2:]
            meta = self.registration_cfg.get("crop")
            clip = apply_registration_crop(clip, meta, self.registration_cfg)
            after_h, after_w = clip.shape[-2:]
            print(
                f"registration crop {path.name}: {before_h}x{before_w} -> {after_h}x{after_w}",
                flush=True,
            )
        elif not self.registration_enabled and not self._registration_off_logged:
            print("registration=off → decode→resize→aug→tile", flush=True)
            self._registration_off_logged = True

        before_h, before_w = clip.shape[-2:]
        clip = resize_to_canonical(clip, self.canonical_hw, self.registration_interp)
        after_h, after_w = clip.shape[-2:]
        target_h, target_w = self.canonical_hw
        print(
            f"canonical resize {path.name}: {before_h}x{before_w} -> {after_h}x{after_w} (target={target_h}x{target_w})",
            flush=True,
        )
        if self.global_aug_enabled and self.split == "train":
            before_h, before_w = clip.shape[-2:]
            clip = apply_global_augment(
                clip,
                self.global_aug_cfg,
                base_seed=self.data_seed,
                sample_index=idx,
                start_idx=start_idx,
                interp=self.global_aug_cfg.get("interp", self.registration_interp),
                id_key=path.stem,
            )
            after_h, after_w = clip.shape[-2:]
            print(
                f"global aug {path.name}: {before_h}x{before_w} -> {after_h}x{after_w}",
                flush=True,
            )

        _, tokens_per_tile, widths_px, _, aligned_w, original_w = tile_vertical_token_aligned(
            clip,
            self.tiles,
            patch_w=self.tiling_patch_w,
            tokens_split=self.tiling_tokens_split,
            overlap_tokens=self.tiling_overlap_tokens,
        )
        if aligned_w != original_w:
            clip = clip[..., :aligned_w]
        if not self._tiling_log_once:
            width_sum = sum(widths_px)
            print(
                f"tiles(tokens)={tokens_per_tile} widths_px={widths_px} "
                f"sum={width_sum} orig_W={original_w} overlap_tokens={self.tiling_overlap_tokens}",
                flush=True,
            )
            self._tiling_log_once = True

        # --- compute the clip's time window [t0, t1) in seconds ---
        T = self.frames
        fps = self.decode_fps
        t0 = frame_to_sec(start_idx, 1.0 / fps)
        t1 = frame_to_sec(start_idx + ((T - 1) * self.stride + 1), 1.0 / fps)

        sample = {"video": clip, "path": str(path)}  # <- create up-front
        labels_tensor = None                          # <- define up-front
    
        # --- locate sidecar annotation file ---
        ann_path = None
        try:
            ann_path = _find_annotation_for_video(path, getattr(self, "annotations_root", None))
            print(f"[DEBUG] video={path.name} -> ann_path={ann_path}", flush=True)  #just for debugging. this should be removed.
            print(f"[DEBUG] video={path}  stem={path.stem}  dir={path.parent}  ann_path={ann_path}", flush=True)
        except NameError:
            from pathlib import Path
            ann_root = getattr(self, "annotations_root", None)
            stem = path.stem
            candidates = []
            if ann_root:
                candidates.append(Path(ann_root).expanduser() / f"{stem}.txt")
            candidates.extend([
                path.with_suffix(".txt"),
                path.with_suffix(".mid"),
                path.with_suffix(".midi"),
                path.with_suffix(".json"),
                path.with_suffix(".csv"),
            ])
            for c in candidates:
                if Path(c).exists():
                    ann_path = Path(c)
                    break
        if ann_path is not None and ann_path.suffix.lower() == ".txt":
            labels_tensor = _read_txt_events(ann_path)  # (N,3) or (0,3)
            if labels_tensor is not None:
                sample["labels"] = labels_tensor

        # --- parse raw labels if .txt is present: labels := (N,3) [onset, offset, pitch] ---
        labels_tensor = None
        if ann_path is not None and ann_path.suffix.lower() == ".txt":
            labels_tensor = _read_txt_events(ann_path)  # (N,3) float32
        
        # --- derive clip-level targets from events overlapping [t0, t1) ---
        clip_targets = None
        if labels_tensor is not None and labels_tensor.numel() > 0:
            # Select events that overlap window
            onset = labels_tensor[:, 0]
            offset = labels_tensor[:, 1]
            pitch = labels_tensor[:, 2].to(torch.int64)  # integer MIDI pitch

            # overlap condition: onset < t1 and offset > t0
            mask = (onset < t1) & (offset > t0)
            sel_pitches = pitch[mask]
            sel_onsets  = onset[mask]
            sel_offsets = offset[mask]

            P = 88
            note_min_clip = 21
            pitch_vec  = torch.zeros(P, dtype=torch.float32)
            onset_vec  = torch.zeros(P, dtype=torch.float32)
            offset_vec = torch.zeros(P, dtype=torch.float32)
            if sel_pitches.numel() == 0:
                pitch_class = 60
                onset_flag, offset_flag = 0.0, 0.0
            else:
                uniq, counts = sel_pitches.unique(return_counts=True)
                pitch_class = int(uniq[counts.argmax()].item())
                onset_flag  = 1.0 if ((sel_onsets >= t0) & (sel_onsets < t1)).any().item() else 0.0
                offset_flag = 1.0 if ((sel_offsets >  t0) & (sel_offsets <= t1)).any().item() else 0.0

            idx = int(pitch_class - note_min_clip)
            if 0 <= idx < P:
                pitch_vec[idx] = 1.0
                if onset_flag:  onset_vec[idx] = 1.0
                if offset_flag: offset_vec[idx] = 1.0

            hand = 0 if pitch_class < 60 else 1
            clef = 0 if pitch_class < 60 else (1 if pitch_class > 64 else 2)
            
            clip_targets = {
                "pitch":  pitch_vec,
                "onset":  onset_vec,
                "offset": offset_vec,
                "hand":   torch.tensor(hand, dtype=torch.long),
                "clef":   torch.tensor(clef, dtype=torch.long),
            }

        # --- build sample dict ---
        sample = {"video": clip, "path": str(path)}
        if labels_tensor is not None:
            sample["labels"] = labels_tensor  # (N,3): raw events for this video

        if clip_targets is not None:
            sample.update(clip_targets)
        else:
            # honor strict mode if requested
            if getattr(self, "require_labels", False):
                raise FileNotFoundError(f"No usable labels for {path}")

        # --- NOW attach frame-level targets (optional) ---
        ft_cfg = getattr(self, "frame_targets_cfg", None)
        if ft_cfg and bool(ft_cfg.get("enable", False)):
            labels_ft = None
            if labels_tensor is not None:
                labels_ft = labels_tensor.clone()
                # Shift label times so frame 0 corresponds to the clip start.
                labels_ft[:, 0:2] -= t0
            ft = _build_frame_targets(
                labels=labels_ft,
                T=T, stride=self.stride, fps=fps,
                note_min=int(ft_cfg.get("note_min", 21)),
                note_max=int(ft_cfg.get("note_max", 108)),
                tol=float(ft_cfg.get("tolerance", 0.025)),
                fill_mode=str(ft_cfg.get("fill_mode", "overlap")),
                hand_from_pitch=bool(ft_cfg.get("hand_from_pitch", True)),
                clef_thresholds=tuple(ft_cfg.get("clef_thresholds", [60, 64])),
                dilate_active_frames=int(ft_cfg.get("dilate_active_frames", 0)),
                targets_sparse=bool(ft_cfg.get("targets_sparse", False)),
            )
            sample.update({
                "pitch_roll":  ft["pitch_roll"],
                "onset_roll":  ft["onset_roll"],
                "offset_roll": ft["offset_roll"],
                "hand_frame":  ft["hand_frame"],
                "clef_frame":  ft["clef_frame"],
            })
        
        return sample


def make_dataloader(cfg: dict, split: str, drop_last: bool = False):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)
    
    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))

    dataset = OMAPSDataset(
        root_dir=dcfg.get("root_dir"),
        split=split,
        frames=int(dcfg.get("frames", 32)),
        stride=stride,
        resize=tuple(dcfg.get("resize", [224, 224])),
        tiles=int(dcfg.get("tiles", 3)),
        channels=int(dcfg.get("channels", 3)),
        normalize=bool(dcfg.get("normalize", True)),
        manifest=manifest_path,
        decode_fps=decode_fps,
        dataset_cfg=dcfg,
        full_cfg=cfg,
    )
    # attach annotation config if present
    max_clips = dcfg.get("max_clips")
    if isinstance(max_clips, int) and len(dataset.videos) > max_clips:
        dataset.videos = dataset.videos[:max_clips]
    dataset.annotations_root = dcfg.get("annotations_root")
    dataset.label_format = dcfg.get("label_format", "txt")
    dataset.label_targets = dcfg.get("label_targets", ["pitch","onset","offset","hand","clef"])
    dataset.require_labels = bool(dcfg.get("require_labels", False))
    dataset.frame_targets_cfg = dcfg.get("frame_targets", {})
    dataset.max_clips = max_clips

    # --- robust collate that preserves extra fields (labels, targets) ---
    def _collate(batch):
        vids  = [b["video"] for b in batch]     # each: T,tiles,C,H,W
        paths = [b["path"]  for b in batch]
        x = torch.stack(vids, dim=0)            # B,T,tiles,C,H,W

        out = {"video": x, "path": paths}

        # pass through any extra keys
        extra_keys = set().union(*[set(d.keys()) for d in batch]) - {"video", "path"}
        for k in extra_keys:
            vals = [d[k] for d in batch if k in d]
            if len(vals) != len(batch):
                # some samples missing the key; skip to keep shapes consistent
                continue
            if k == "labels":
                # variable length; keep list of tensors (one per sample)
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

    loader = DataLoader(
        dataset,
        batch_size=int(dcfg.get("batch_size", 2)),
        shuffle=bool(dcfg.get("shuffle", True)) if split == "train" else False,
        num_workers=int(dcfg.get("num_workers", 0)),  # 0 while debugging prints
        pin_memory=False,
        drop_last=drop_last,
        collate_fn=_collate,
    )
    return loader


