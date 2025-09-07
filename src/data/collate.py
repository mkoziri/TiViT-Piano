# src/data/collate.py
from typing import List, Dict, Any
import torch

def collate_framewise(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    vids = torch.stack([b["video"] for b in batch], dim=0)            # (B,T,tiles,C,H,W)
    frame_times = torch.stack([b["frame_times"] for b in batch], 0)   # (B,T)
    metas = [b["meta"] for b in batch]

    t0 = batch[0]["targets"]
    if not t0.get("sparse", False):
        pitch = torch.stack([b["targets"]["pitch"] for b in batch], 0)    # (B,T,P)
        onset = torch.stack([b["targets"]["onset"] for b in batch], 0)    # (B,T,P)
        offset = torch.stack([b["targets"]["offset"] for b in batch], 0)  # (B,T,P)
        fmask = torch.stack([b["targets"]["frame_mask"] for b in batch], 0)  # (B,T)
        targets = {"sparse": False, "pitch": pitch, "onset": onset, "offset": offset, "frame_mask": fmask}
    else:
        targets = {
            "sparse": True,
            "shape": batch[0]["targets"]["shape"],
            "frame_mask": torch.stack([b["targets"]["frame_mask"] for b in batch], 0),
            "pitch_idx": [b["targets"]["pitch_idx"] for b in batch],
            "onset_idx": [b["targets"]["onset_idx"] for b in batch],
            "offset_idx": [b["targets"]["offset_idx"] for b in batch],
        }

    return {"video": vids, "frame_times": frame_times, "targets": targets, "meta": metas}

