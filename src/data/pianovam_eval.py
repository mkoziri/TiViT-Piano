import torch
from typing import Dict, Tuple

# -----------------------------------------
# Convert frame-wise (T, 88) rolls → events
# -----------------------------------------
def roll_to_events(roll: torch.Tensor, fps: float, tol: float = 0.050):
    """
    Converts binary frame-rolls into list of (pitch, onset_time, offset_time)
    """
    T, P = roll.shape
    events = []

    for pitch in range(P):
        active = roll[:, pitch] > 0.5
        if not active.any():
            continue

        idx = torch.where(active)[0]

        # group contiguous indices
        groups = []
        current = [idx[0].item()]
        for i in idx[1:]:
            if i == current[-1] + 1:
                current.append(i.item())
            else:
                groups.append(current)
                current = [i.item()]
        groups.append(current)

        for g in groups:
            on_t = g[0] / fps
            off_t = (g[-1] + 1) / fps
            events.append((pitch, on_t, off_t))

    return events


# -----------------------------------------
# Event-level evaluation (F1)
# -----------------------------------------
def event_f1(pred_events, gt_events, tol=0.05):
    """
    pred_events, gt_events: list of (pitch, onset, offset)
    tolerance: onset matching window
    """
    if len(pred_events) == 0 and len(gt_events) == 0:
        return 1.0, 0, 0, 0

    tp = 0
    used = set()

    for (p_pitch, p_on, p_off) in pred_events:
        for j, (g_pitch, g_on, g_off) in enumerate(gt_events):
            if j in used:
                continue
            if p_pitch != g_pitch:
                continue
            if abs(p_on - g_on) <= tol:
                tp += 1
                used.add(j)
                break

    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    if tp == 0:
        return 0.0, tp, fp, fn

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return f1, tp, fp, fn


# -----------------------------------------
# Main evaluation function used by train.py
# -----------------------------------------
def evaluate_clip(pred: Dict[str, torch.Tensor],
                  target: Dict[str, torch.Tensor],
                  fps: float):

    pitch_pred = pred["pitch_roll"].cpu()
    onset_pred = pred["onset_roll"].cpu()
    offset_pred = pred["offset_roll"].cpu()

    pitch_gt = target["pitch_roll"].cpu()
    onset_gt = target["onset_roll"].cpu()
    offset_gt = target["offset_roll"].cpu()

    # Convert to events
    pitch_p = roll_to_events(pitch_pred, fps)
    pitch_t = roll_to_events(pitch_gt, fps)

    onset_p = roll_to_events(onset_pred, fps)
    onset_t = roll_to_events(onset_gt, fps)

    offset_p = roll_to_events(offset_pred, fps)
    offset_t = roll_to_events(offset_gt, fps)

    # F1 per head
    f1_pitch, *_ = event_f1(pitch_p, pitch_t)
    f1_on, *_    = event_f1(onset_p, onset_t)
    f1_off, *_   = event_f1(offset_p, offset_t)

    return {
        "pitch_f1": f1_pitch,
        "onset_f1": f1_on,
        "offset_f1": f1_off
    }
