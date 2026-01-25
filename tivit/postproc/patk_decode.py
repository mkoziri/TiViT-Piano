"""Post-processing helpers for PATK-style evaluation.

Purpose:
    - Apply time-domain Gaussian smoothing to onset/frame probabilities.
    - Threshold smoothed probabilities and extract onset peaks.
    - Build note events (onset/offset/pitch) for evaluation metrics.
Key Functions/Classes:
    - PatkDecodeConfig: Configuration container for smoothing and tolerances.
    - smooth_time_probs(): Depthwise temporal Gaussian smoothing.
    - extract_onset_peaks(): Extract onset indices from thresholded onset masks.
    - build_notes_from_peaks(): Build notes from onset peaks + frame masks.
    - build_notes_from_rolls(): Build reference notes from onset/pitch rolls.
CLI Arguments:
    (none)
Usage:
    cfg = PatkDecodeConfig.from_config(metrics_cfg, dataset_cfg)
    onset_smooth = smooth_time_probs(onset_probs, sigma=cfg.onset_sigma, radius=cfg.onset_radius)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from tivit.decoder.decode import pool_roll_BT


@dataclass(frozen=True)
class PatkDecodeConfig:
    """Configuration for PATK-style decoding and evaluation."""

    fps: float
    onset_sigma: float
    onset_radius: int
    frame_sigma: float
    frame_radius: int
    threshold: float
    ignore_tail: int
    onset_tolerance: float
    offset_ratio: float
    offset_min_tolerance: float

    @classmethod
    def from_config(cls, metrics_cfg: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> "PatkDecodeConfig":
        patk_cfg = metrics_cfg.get("patk", {}) if isinstance(metrics_cfg, Mapping) else {}
        if not isinstance(patk_cfg, Mapping):
            patk_cfg = {}
        fps = float(patk_cfg.get("fps", 30.0) or 30.0)
        onset_tol_ms = float(patk_cfg.get("onset_tolerance_ms", 100.0) or 100.0)
        return cls(
            fps=fps if fps > 0 else 30.0,
            onset_sigma=float(patk_cfg.get("onset_sigma", 0.2) or 0.2),
            onset_radius=int(patk_cfg.get("onset_radius", 8) or 8),
            frame_sigma=float(patk_cfg.get("frame_sigma", 0.8) or 0.8),
            frame_radius=int(patk_cfg.get("frame_radius", 4) or 4),
            threshold=float(patk_cfg.get("threshold", 0.5) or 0.5),
            ignore_tail=int(patk_cfg.get("ignore_tail", 4) or 4),
            onset_tolerance=max(0.0, onset_tol_ms / 1000.0),
            offset_ratio=float(patk_cfg.get("offset_ratio", 0.2) or 0.2),
            offset_min_tolerance=float(patk_cfg.get("offset_min_tolerance", 0.05) or 0.05),
        )


def _gaussian_kernel(sigma: float, radius: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0 or radius <= 0:
        return torch.ones((1, 1, 1), device=device, dtype=dtype)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, -1)


def smooth_time_probs(probs: torch.Tensor, *, sigma: float, radius: int) -> torch.Tensor:
    """Apply depthwise Gaussian smoothing along time (T) for each pitch."""
    if sigma <= 0 or radius <= 0:
        return probs
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    if probs.dim() != 3:
        raise ValueError("Expected probs with shape (B,T,P) or (T,P).")
    bsz, t_len, pitches = probs.shape
    kernel = _gaussian_kernel(sigma, radius, device=probs.device, dtype=probs.dtype)
    kernel = kernel.repeat(pitches, 1, 1)
    x = probs.permute(0, 2, 1).contiguous()
    smoothed = F.conv1d(x, kernel, padding=radius, groups=pitches)
    return smoothed.permute(0, 2, 1).contiguous()


def resample_probs_btP(probs: torch.Tensor, t_out: int) -> torch.Tensor:
    """Resample probabilities to a target time length using linear interpolation."""
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    if probs.dim() != 3:
        raise ValueError("Expected probs with shape (B,T,P) or (T,P).")
    if t_out <= 0 or probs.shape[1] == t_out:
        return probs
    x = probs.permute(0, 2, 1)
    x = F.interpolate(x, size=t_out, mode="linear", align_corners=False)
    return x.permute(0, 2, 1).contiguous()


def resample_roll_btP(roll: torch.Tensor, t_out: int) -> torch.Tensor:
    """Resample binary rolls to a target time length using max pooling."""
    if roll.dim() == 2:
        roll = roll.unsqueeze(0)
    if roll.dim() != 3:
        raise ValueError("Expected roll with shape (B,T,P) or (T,P).")
    if t_out <= 0 or roll.shape[1] == t_out:
        return roll
    return pool_roll_BT(roll, t_out)


def extract_onset_peaks(onset_probs: torch.Tensor, onset_mask: torch.Tensor) -> list[list[list[int]]]:
    """Return onset peak indices per batch/pitch from thresholded onset masks."""
    if onset_probs.dim() == 2:
        onset_probs = onset_probs.unsqueeze(0)
    if onset_mask.dim() == 2:
        onset_mask = onset_mask.unsqueeze(0)
    if onset_probs.shape != onset_mask.shape:
        raise ValueError("onset_probs and onset_mask must share the same shape.")
    bsz, t_len, pitches = onset_probs.shape
    peaks: list[list[list[int]]] = [[[] for _ in range(pitches)] for _ in range(bsz)]
    probs_cpu = onset_probs.detach().cpu()
    mask_cpu = onset_mask.detach().cpu()
    for b in range(bsz):
        for p in range(pitches):
            mask = mask_cpu[b, :, p]
            if not bool(mask.any().item()):
                continue
            active_idx = mask.nonzero(as_tuple=False).view(-1).tolist()
            start = active_idx[0]
            prev = active_idx[0]
            for idx in active_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                segment = probs_cpu[b, start : prev + 1, p]
                peak = int(start + segment.argmax().item())
                peaks[b][p].append(peak)
                start = idx
                prev = idx
            segment = probs_cpu[b, start : prev + 1, p]
            peak = int(start + segment.argmax().item())
            peaks[b][p].append(peak)
    return peaks


def build_notes_from_peaks(
    peaks: Sequence[Sequence[Sequence[int]]],
    frame_mask: torch.Tensor,
    *,
    hop_seconds: float,
    note_min: int,
    ignore_tail: int,
) -> list[list[tuple[float, float, int]]]:
    """Build note events from onset peaks and frame masks."""
    if frame_mask.dim() == 2:
        frame_mask = frame_mask.unsqueeze(0)
    bsz, t_len, pitches = frame_mask.shape
    mask_cpu = frame_mask.detach().cpu()
    notes_by_batch: list[list[tuple[float, float, int]]] = []
    for b in range(bsz):
        notes: list[tuple[float, float, int]] = []
        for p in range(pitches):
            for onset_idx in peaks[b][p]:
                if onset_idx < 0 or onset_idx >= t_len:
                    continue
                if not bool(mask_cpu[b, onset_idx, p].item()):
                    continue
                end_idx = onset_idx
                while end_idx + 1 < t_len and bool(mask_cpu[b, end_idx + 1, p].item()):
                    end_idx += 1
                if ignore_tail > 0:
                    end_idx = max(onset_idx, end_idx - ignore_tail)
                onset_sec = onset_idx * hop_seconds
                offset_sec = (end_idx + 1) * hop_seconds
                notes.append((float(onset_sec), float(offset_sec), int(note_min + p)))
        notes_by_batch.append(notes)
    return notes_by_batch


def _derive_onsets_from_pitch(pitch_roll: torch.Tensor) -> torch.Tensor:
    """Fallback onset extraction from pitch roll rising edges."""
    if pitch_roll.dim() != 3:
        raise ValueError("pitch_roll must have shape (B,T,P).")
    prev = torch.zeros_like(pitch_roll[:, :1, :])
    shifted = torch.cat([prev, pitch_roll[:, :-1, :]], dim=1)
    return (pitch_roll > 0.5) & (shifted <= 0.5)


def build_notes_from_rolls(
    onset_roll: torch.Tensor,
    pitch_roll: torch.Tensor,
    *,
    hop_seconds: float,
    note_min: int,
) -> list[list[tuple[float, float, int]]]:
    """Build reference note events from onset + pitch rolls."""
    if onset_roll.dim() == 2:
        onset_roll = onset_roll.unsqueeze(0)
    if pitch_roll.dim() == 2:
        pitch_roll = pitch_roll.unsqueeze(0)
    if onset_roll.shape != pitch_roll.shape:
        raise ValueError("onset_roll and pitch_roll must share shape (B,T,P).")
    bsz, t_len, pitches = pitch_roll.shape
    onset_roll = onset_roll > 0.5
    pitch_roll = pitch_roll > 0.5
    if not bool(onset_roll.any().item()):
        onset_roll = _derive_onsets_from_pitch(pitch_roll)
    onset_cpu = onset_roll.detach().cpu()
    pitch_cpu = pitch_roll.detach().cpu()
    notes_by_batch: list[list[tuple[float, float, int]]] = []
    for b in range(bsz):
        notes: list[tuple[float, float, int]] = []
        for p in range(pitches):
            onset_idx = onset_cpu[b, :, p].nonzero(as_tuple=False).view(-1).tolist()
            for start in onset_idx:
                if not bool(pitch_cpu[b, start, p].item()):
                    continue
                end_idx = start
                while end_idx + 1 < t_len and bool(pitch_cpu[b, end_idx + 1, p].item()):
                    end_idx += 1
                onset_sec = start * hop_seconds
                offset_sec = (end_idx + 1) * hop_seconds
                notes.append((float(onset_sec), float(offset_sec), int(note_min + p)))
        notes_by_batch.append(notes)
    return notes_by_batch


def resolve_note_range(dataset_cfg: Mapping[str, Any]) -> tuple[int, int]:
    frame_cfg = dataset_cfg.get("frame_targets") if isinstance(dataset_cfg, Mapping) else None
    note_min = 21
    note_max = 108
    if isinstance(frame_cfg, Mapping):
        note_min = int(frame_cfg.get("note_min", note_min))
        note_max = int(frame_cfg.get("note_max", note_max))
    return note_min, note_max


def compute_target_length(t_in: int, hop_seconds: float, *, fps: float) -> int:
    """Compute target time length for a new FPS while preserving duration."""
    if t_in <= 1:
        return t_in
    duration = (t_in - 1) * float(hop_seconds)
    hop_out = 1.0 / max(float(fps), 1e-6)
    t_out = int(round(duration / hop_out)) + 1
    return max(1, t_out)


def clamp_probs(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a boolean mask from probabilities using a fixed threshold."""
    return probs >= float(threshold)

