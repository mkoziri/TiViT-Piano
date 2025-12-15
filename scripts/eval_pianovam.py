import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple
# ============================================================
# Configurable thresholds and tolerances
# ============================================================

ONSET_THR = 0.35
OFFSET_THR = 0.35
TOPK = 8 

# With 128 frames / full video, timestamps differ by ~5–20 sec.
# So we MUST allow large tolerance.
ONSET_TOL = 5.0
OFFSET_TOL = 10.0

MIN_NOTE_LEN = 0.00

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
    onset_thr: float = ONSET_THR,
    offset_thr: float = OFFSET_THR,
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
    if TOPK and TOPK > 0:
        # TOPK ΜΟΝΟ στο onset (για να περιορίσεις υποψήφια starts)
        _, idx = torch.topk(onset_prob, k=TOPK, dim=1)
        mask = torch.zeros_like(onset_prob, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        onset_prob = onset_prob * mask

        # ΜΗΝ κόβεις το offset με TOPK (αλλιώς δεν "κλείνουν" τα events)
        # offset_prob μένει όπως είναι



    # --- DEBUG: ρίξε μια ματιά στα probs ---
    print(
        "[DEBUG-probs] onset min=%.3f max=%.3f mean=%.3f | "
        "offset min=%.3f max=%.3f mean=%.3f"
        % (
            onset_prob.min().item(), onset_prob.max().item(), onset_prob.mean().item(),
            offset_prob.min().item(), offset_prob.max().item(), offset_prob.mean().item(),
        )
    )
    # optional: πόσα bins περνάνε το threshold
    onset_bin_tmp = (onset_prob > onset_thr)
    offset_bin_tmp = (offset_prob > offset_thr)
    print(
        "[DEBUG-bins] onset_thr=%.3f onset>thr=%d/%d | offset_thr=%.3f offset>thr=%d/%d"
        % (
            float(onset_thr), onset_bin_tmp.sum().item(), onset_bin_tmp.numel(),
            float(offset_thr), offset_bin_tmp.sum().item(), offset_bin_tmp.numel(),
        )
    )


    # ----------------------------------------

    onset_bin  = (onset_prob  > onset_thr)
    offset_bin = (offset_prob > offset_thr)

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

    return [
        e for e in events
        if (e.offset - e.onset) >= MIN_NOTE_LEN
    ]



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

def evaluate_clip(on_logits, off_logits, tsv_path, frame_times, onset_thr: float = ONSET_THR, offset_thr: float = OFFSET_THR):

    pred_events = decode_events_from_logits(on_logits, off_logits, frame_times, onset_thr=onset_thr, offset_thr=offset_thr)
    gt_events   = load_tsv_events(tsv_path)

    print(f"[DEBUG] {tsv_path}  n_pred={len(pred_events)}  n_gt={len(gt_events)}")
    if len(pred_events) > 0:
        print("[DEBUG] first pred:", pred_events[0])
    if len(gt_events) > 0:
        print("[DEBUG] first gt:", gt_events[0])

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
