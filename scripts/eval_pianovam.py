import torch
import numpy as np
from dataclasses import dataclass
from typing import List

# ============================================================
# Configurable thresholds and tolerances
# ============================================================

ONSET_THR = 0.20
OFFSET_THR = 0.20

# With 128 frames / full video, timestamps differ by ~5–20 sec.
# So we MUST allow large tolerance.
ONSET_TOL = 5.0
OFFSET_TOL = 5.0

MIN_NOTE_LEN = 0.05

@dataclass
class NoteEvent:
    onset: float
    offset: float
    pitch: int


# ============================================================
# Decode logits → events using REAL frame_times
# ============================================================

def decode_events_from_logits(
    onset_logits: torch.Tensor,
    offset_logits: torch.Tensor,
    frame_times: torch.Tensor,
):
    """
    onset_logits, offset_logits: (T, 88) raw logits
    frame_times: (T,) times in seconds for each frame
    Επιστρέφει: List[NoteEvent]
    """
    assert onset_logits.shape == offset_logits.shape
    assert frame_times.shape[0] == onset_logits.shape[0]

    T, P = onset_logits.shape

    onset_prob  = torch.sigmoid(onset_logits)
    offset_prob = torch.sigmoid(offset_logits)

    onset_bin  = (onset_prob  > ONSET_THR)
    offset_bin = (offset_prob > OFFSET_THR)

    frame_times = torch.as_tensor(frame_times, dtype=torch.float32)
    T, P = onset_bin.shape

    events = []

    for pitch in range(P):
        active = False
        onset_time = 0.0

        for t in range(T):
            if onset_bin[t, pitch] and not active:
                active = True
                onset_time = float(frame_times[t])

            if active and offset_bin[t, pitch]:
                off_t = float(frame_times[t])
                if off_t >= onset_time:
                    events.append(NoteEvent(onset_time, off_t, pitch + 21))
                    active = False

        if active:
            events.append(NoteEvent(onset_time, float(frame_times[-1]), pitch + 21))

    return events


# ============================================================
# Load GT events from TSV
# ============================================================

def load_tsv_events(tsv_path: str):
    events = []
    with open(tsv_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            onset, offset, _, pitch, _ = line.strip().split("\t")
            events.append(
                NoteEvent(float(onset), float(offset), int(pitch))
            )
    return events


# ============================================================
# Event matching with pitch + onset tolerance + offset tolerance
# ============================================================

def match_events(pred, gt):
    used = set()
    tp = fp = fn = 0

    for p in pred:
        best = None
        best_dist = 9999

        for j, g in enumerate(gt):
            if j in used:
                continue
            if p.pitch != g.pitch:
                continue

            do = abs(p.onset - g.onset)
            df = abs(p.offset - g.offset)

            if do <= ONSET_TOL and df <= OFFSET_TOL:
                dist = do + df
                if dist < best_dist:
                    best_dist = dist
                    best = j

        if best is not None:
            used.add(best)
            tp += 1
        else:
            fp += 1

    fn = len(gt) - len(used)

    return tp, fp, fn


# ============================================================
# High-level evaluation per clip
# ============================================================

def evaluate_clip(on_logits, off_logits, tsv_path, frame_times):

    pred_events = decode_events_from_logits(on_logits, off_logits, frame_times)
    gt_events   = load_tsv_events(tsv_path)

    tp, fp, fn = match_events(pred_events, gt_events)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred_events),
        "n_gt": len(gt_events),
    }
