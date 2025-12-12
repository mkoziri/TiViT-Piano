import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


# ============================================================
# CONFIG — evaluation tolerance
# ============================================================

ONSET_TOL = 0.050  # 50 ms
OFFSET_TOL = 0.100 # 100 ms is standard for offsets


# ============================================================
# events Dataclass
# ============================================================

@dataclass
class NoteEvent:
    onset: float
    offset: float
    pitch: int


# ============================================================
# Convert frame-wise logits → binary event predictions
# ============================================================

def decode_events_from_logits(onset_logits, offset_logits, hop_seconds):
    """
    onset_logits: (T, 88)
    offset_logits: (T, 88)
    hop_seconds: e.g. 1/30
    """
    onset_prob = torch.sigmoid(onset_logits)
    offset_prob = torch.sigmoid(offset_logits)

    onset_bin = (onset_prob > 0.5).float()
    offset_bin = (offset_prob > 0.5).float()

    T = onset_bin.shape[0]
    events: List[NoteEvent] = []

    for pitch in range(88):
        active = False
        onset_time = None

        for t in range(T):
            if onset_bin[t, pitch] == 1 and not active:
                active = True
                onset_time = t * hop_seconds

            if active and offset_bin[t, pitch] == 1:
                offset_time = t * hop_seconds
                if offset_time > onset_time:
                    events.append(NoteEvent(onset_time, offset_time, pitch + 21))
                active = False

        # If note never received an offset → close at end
        if active:
            events.append(
                NoteEvent(onset_time, (T - 1) * hop_seconds, pitch + 21)
            )

    return events


# ============================================================
# Load ground truth piano events from TSV
# ============================================================

def load_tsv_events(tsv_path: str) -> List[NoteEvent]:
    events = []
    with open(tsv_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            onset, offset, _, pitch, _ = line.strip().split("\t")
            events.append(
                NoteEvent(float(onset), float(offset), int(pitch))
            )
    return events


# ============================================================
# Match predicted ↔ GT events
# ============================================================

def match_events(pred: List[NoteEvent], gt: List[NoteEvent]):
    used_gt = set()
    tp = 0
    fp = 0
    fn = 0

    for p in pred:
        best_j = None
        best_dist = 999

        for j, g in enumerate(gt):
            if j in used_gt:
                continue
            if g.pitch != p.pitch:
                continue

            onset_diff = abs(g.onset - p.onset)
            offset_diff = abs(g.offset - p.offset)

            if onset_diff <= ONSET_TOL and offset_diff <= OFFSET_TOL:
                dist = onset_diff + offset_diff
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

        if best_j is not None:
            tp += 1
            used_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt) - len(used_gt)

    return tp, fp, fn


# ============================================================
# Compute precision/recall/F1
# ============================================================

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# ============================================================
# High-level eval function (per-clip)
# ============================================================

def evaluate_clip(pred_onset_logits, pred_offset_logits,
                  tsv_path, hop_seconds):
    pred_events = decode_events_from_logits(
        pred_onset_logits, pred_offset_logits, hop_seconds
    )
    gt_events = load_tsv_events(tsv_path)

    tp, fp, fn = match_events(pred_events, gt_events)
    P, R, F1 = compute_f1(tp, fp, fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": P,
        "recall": R,
        "f1": F1,
        "n_pred": len(pred_events),
        "n_gt": len(gt_events),
    }
