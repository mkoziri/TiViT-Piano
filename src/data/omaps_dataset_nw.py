# src/data/omaps_dataset.py
import os, glob, json, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from decord import VideoReader
    HAVE_DECORD = True
except Exception:
    HAVE_DECORD = False
import cv2


class OMAPSDataset(Dataset):
    """
    OMAPS-style piano videos (.mp4) + paired .txt labels:
      onset_sec  offset_sec  pitch
    Returns video clip + frame-level targets (pitch/onset/offset) if enabled.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        frames: int = 32,
        stride: int = 2,
        resize: Tuple[int, int] = (224, 224),
        tiles: int = 1,  # keep axis for compatibility with your model (T, tiles, C, H, W)
        # Frame-target settings from cfg["dataset"]["frame_targets"]
        frame_targets_cfg: Dict[str, Any] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.frames = int(frames)
        self.stride = int(stride)
        self.resize = resize
        self.tiles = int(tiles)

        ft = frame_targets_cfg or {}
        self.enable_frame_targets = bool(ft.get("enable", False))
        self.onset_tolerance = float(ft.get("tolerance", 0.025))
        self.note_min = int(ft.get("note_min", 0))
        self.note_max = int(ft.get("note_max", 127))
        self.P = self.note_max - self.note_min + 1
        self.cache_labels = bool(ft.get("cache_labels", True))
        cache_dir = ft.get("cache_dir", None)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.root / ".cache_labels")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dilate_active_frames = int(ft.get("dilate_active_frames", 0))
        self.targets_sparse = bool(ft.get("targets_sparse", False))

        self.items = self._scan()

    # -------------- discovery & parsing ----------------

    def _scan(self) -> List[Dict[str, str]]:
        vids = sorted(glob.glob(str(self.root / self.split / "**" / "*.mp4"), recursive=True))
        items = []
        for v in vids:
            stem = os.path.splitext(v)[0]
            lbl = stem + ".txt"
            if os.path.exists(lbl):
                items.append({"video": v, "label": lbl})
        if not items:
            raise FileNotFoundError(f"No .mp4+.txt pairs under {self.root / self.split}")
        return items

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _read_events_txt(self, txt_path: str) -> List[Tuple[float, float, int]]:
        ev = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    t_on = float(parts[0]); t_off = float(parts[1]); p = int(parts[2])
                    ev.append((t_on, t_off, p))
                except Exception:
                    continue
        return ev

    # -------------- labels: events -> frame rolls ----------------

    def _events_to_frame_targets(
        self,
        events: List[Tuple[float, float, int]],
        frame_times: torch.Tensor,
        onset_tol: float,
        dilate: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dense uint8 tensors:
          pitch  (T,P): 1 while note is active in [on,off)
          onset  (T,P): 1 at nearest frame to onset within tolerance
          offset (T,P): 1 at nearest frame to offset within tolerance
        """
        T = int(frame_times.numel())
        P = self.P
        pitch = torch.zeros(T, P, dtype=torch.uint8)
        onset = torch.zeros(T, P, dtype=torch.uint8)
        offset = torch.zeros(T, P, dtype=torch.uint8)

        for (t_on, t_off, p) in events:
            if p < self.note_min or p > self.note_max or (t_off <= t_on):
                continue
            col = p - self.note_min

            # active frames via exact timestamp membership
            in_seg = (frame_times >= t_on) & (frame_times < t_off)
            if in_seg.any():
                if dilate > 0:
                    idx = torch.nonzero(in_seg, as_tuple=False).squeeze(1)
                    if idx.numel() > 0:
                        lo = max(int(idx[0]) - dilate, 0)
                        hi = min(int(idx[-1]) + dilate, T - 1)
                        in_seg = torch.zeros(T, dtype=torch.bool)
                        in_seg[lo:hi + 1] = True
                pitch[in_seg, col] = 1

            # onset/offset at nearest frame (within tolerance)
            k_on = int(torch.argmin(torch.abs(frame_times - t_on)))
            if abs(float(frame_times[k_on]) - t_on) <= onset_tol:
                onset[k_on, col] = 1

            k_off = int(torch.argmin(torch.abs(frame_times - t_off)))
            if abs(float(frame_times[k_off]) - t_off) <= onset_tol:
                offset[k_off, col] = 1

        return pitch, onset, offset

    # -------------- main item ----------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        vpath = rec["video"]
        lpath = rec["label"]

        # --- read + sample ---
        if HAVE_DECORD:
            vr = VideoReader(vpath)
            fps_raw = float(vr.get_avg_fps())
            total = len(vr)
            max_start = max(0, total - self.frames * self.stride)
            start_idx = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
            idxs = np.arange(start_idx, start_idx + self.frames * self.stride, self.stride)
            idxs = np.clip(idxs, 0, total - 1)
            frames_np = vr.get_batch(idxs).asnumpy()  # (T,H,W,C)
        else:
            cap = cv2.VideoCapture(vpath)
            fps_raw = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            max_start = max(0, total - self.frames * self.stride)
            start_idx = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
            idxs = np.arange(start_idx, start_idx + self.frames * self.stride, self.stride)
            idxs = np.clip(idxs, 0, total - 1)
            frames_list = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    frame = frames_list[-1] if frames_list else np.zeros((self.resize[0], self.resize[1], 3), np.uint8)
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_np = np.stack(frames_list, axis=0)
            cap.release()

        # --- resize + tensor ---
        T = frames_np.shape[0]
        assert T == self.frames, f"Expected T={self.frames}, got {T}"
        h, w = self.resize
        frames_np = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames_np], axis=0)
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # (T,C,H,W)
        video = video.unsqueeze(1)  # (T, tiles=1, C, H, W) — keep tiles axis

        # --- timebase ---
        fps_eff = fps_raw / float(self.stride)
        first_src_idx = int(idxs[0])
        clip_start = float(first_src_idx) / max(fps_raw, 1e-6)
        clip_end = clip_start + (self.frames - 1) / max(fps_eff, 1e-6)
        frame_times = clip_start + torch.arange(self.frames, dtype=torch.float32) / max(fps_eff, 1e-6)

        # --- events ---
        events = self._read_events_txt(lpath)
        clip_events = []
        for (t_on, t_off, p) in events:
            if t_off <= t_on:
                continue
            if (t_on < clip_end) and (t_off > clip_start):
                clip_events.append((t_on, t_off, int(p)))

        # --- frame targets ---
        if self.enable_frame_targets:
            if self.cache_labels:
                key = json.dumps({
                    "v": str(vpath), "s": float(clip_start), "T": int(self.frames),
                    "fps_eff": float(fps_eff), "tol": float(self.onset_tolerance),
                    "nr": [self.note_min, self.note_max], "dilate": int(self.dilate_active_frames)
                }, sort_keys=True)
                fname = self.cache_dir / f"{self._hash_key(key)}.npz"
                if fname.exists():
                    with np.load(fname) as npz:
                        pitch = torch.from_numpy(npz["pitch"])
                        onset = torch.from_numpy(npz["onset"])
                        offset = torch.from_numpy(npz["offset"])
                else:
                    pitch, onset, offset = self._events_to_frame_targets(
                        clip_events, frame_times, self.onset_tolerance, self.dilate_active_frames
                    )
                    np.savez_compressed(fname, pitch=pitch.numpy(), onset=onset.numpy(), offset=offset.numpy())
            else:
                pitch, onset, offset = self._events_to_frame_targets(
                    clip_events, frame_times, self.onset_tolerance, self.dilate_active_frames
                )

            frame_mask = torch.ones(self.frames, dtype=torch.uint8)

            if self.targets_sparse:
                pitch_idx = torch.nonzero(pitch, as_tuple=False)   # (N,2): (t,p)
                onset_idx = torch.nonzero(onset, as_tuple=False)
                offset_idx = torch.nonzero(offset, as_tuple=False)
                targets = {
                    "sparse": True,
                    "shape": torch.tensor([self.frames, self.P], dtype=torch.int32),
                    "frame_mask": frame_mask,
                    "pitch_idx": pitch_idx,
                    "onset_idx": onset_idx,
                    "offset_idx": offset_idx,
                }
            else:
                targets = {
                    "sparse": False,
                    "frame_mask": frame_mask,       # (T,)
                    "pitch": pitch,                 # (T,P) uint8
                    "onset": onset,                 # (T,P) uint8
                    "offset": offset,               # (T,P) uint8
                }
        else:
            # legacy fallback (keeps shapes consistent)
            targets = {
                "sparse": False,
                "frame_mask": torch.ones(self.frames, dtype=torch.uint8),
                "pitch": torch.zeros(self.frames, self.P, dtype=torch.uint8),
                "onset": torch.zeros(self.frames, self.P, dtype=torch.uint8),
                "offset": torch.zeros(self.frames, self.P, dtype=torch.uint8),
            }

        meta = {
            "fps_raw": float(fps_raw),
            "fps_eff": float(fps_eff),
            "clip_start": float(clip_start),
            "clip_end": float(clip_end),
            "path": str(vpath),
            "events": clip_events,
            "note_min": int(self.note_min),
            "note_max": int(self.note_max),
        }

        return {
            "video": video,                 # (T,1,C,H,W)
            "frame_times": frame_times,     # (T,)
            "targets": targets,
            "meta": meta,
        }


# -------------- simple collate (kept local, not in __init__) ----------------

def collate_framewise(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    vids = torch.stack([b["video"] for b in batch], dim=0)            # (B,T,tiles,C,H,W)
    frame_times = torch.stack([b["frame_times"] for b in batch], dim=0)  # (B,T)
    metas = [b["meta"] for b in batch]

    t0 = batch[0]["targets"]
    if not t0.get("sparse", False):
        pitch = torch.stack([b["targets"]["pitch"] for b in batch], dim=0)    # (B,T,P)
        onset = torch.stack([b["targets"]["onset"] for b in batch], dim=0)    # (B,T,P)
        offset = torch.stack([b["targets"]["offset"] for b in batch], dim=0)  # (B,T,P)
        fmask = torch.stack([b["targets"]["frame_mask"] for b in batch], dim=0)  # (B,T)
        targets = {"sparse": False, "pitch": pitch, "onset": onset, "offset": offset, "frame_mask": fmask}
    else:
        targets = {
            "sparse": True,
            "shape": batch[0]["targets"]["shape"],
            "frame_mask": torch.stack([b["targets"]["frame_mask"] for b in batch], dim=0),
            "pitch_idx": [b["targets"]["pitch_idx"] for b in batch],
            "onset_idx": [b["targets"]["onset_idx"] for b in batch],
            "offset_idx": [b["targets"]["offset_idx"] for b in batch],
        }

    return {"video": vids, "frame_times": frame_times, "targets": targets, "meta": metas}


# -------------- keep make_dataloader here (your style) ----------------

def make_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    ds_root = cfg["dataset"]["root_dir"]
    ft_cfg = cfg["dataset"].get("frame_targets", {"enable": False})

    dataset = OMAPSDataset(
        root=ds_root,
        split=split,
        frames=cfg["dataset"].get("frames", 32),
        stride=cfg["dataset"].get("stride", 2),
        resize=tuple(cfg["dataset"].get("resize", [224, 224])),
        tiles=cfg["dataset"].get("tiles", 1),
        frame_targets_cfg=ft_cfg,
    )

    use_frame_collate = bool(ft_cfg.get("enable", False))
    collate = collate_framewise if use_frame_collate else None

    return DataLoader(
        dataset,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=(split == "train"),
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate,
    )

