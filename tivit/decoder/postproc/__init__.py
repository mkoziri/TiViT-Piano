"""Purpose:
    Shared event structures and helpers for decoder post-processing modules.

Key Functions/Classes:
    - NoteEvent: Lightweight dataclass representing a decoded onset event.
    - DecoderEventSet: Container bundling the pianoroll mask, per-frame
      probabilities, and derived events for downstream post-processing.
    - build_event_set(): Factory that normalizes tensors and extracts events
      from a raw boolean mask / probability pair.

CLI:
    Not a standalone module; imported by decoder post-processing stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class NoteEvent:
    clip_idx: int
    key: int
    onset_frame: int
    offset_frame: int
    confidence: float
    onset_ms: float
    offset_ms: float

    def clamp_frames(self, frame_count: int) -> None:
        self.onset_frame = max(0, min(self.onset_frame, frame_count - 1))
        self.offset_frame = max(self.onset_frame + 1, min(self.offset_frame, frame_count))


@dataclass
class DecoderEventSet:
    fps: float
    mask: torch.Tensor
    probs: torch.Tensor
    events: List[NoteEvent]
    clip_count: int
    frames: int
    pitches: int
    device: torch.device
    dtype: torch.dtype
    stats: Dict[str, Any] = field(default_factory=dict)

    def clone_events(self) -> List[NoteEvent]:
        return [NoteEvent(**vars(ev)) for ev in self.events]

    def rebuild_mask(self) -> None:
        updated = torch.zeros((self.clip_count, self.frames, self.pitches), dtype=torch.bool)
        for ev in self.events:
            updated[ev.clip_idx, ev.onset_frame:ev.offset_frame, ev.key] = True
        self.mask = updated

    def refresh_events(self) -> None:
        self.events = _extract_events(self.mask, self.probs, self.fps)

    def to_tensor(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        out_dtype = dtype or self.dtype
        out_device = device or self.device
        return self.mask.to(device=out_device, dtype=out_dtype)


def _extract_events(mask: torch.Tensor, probs: torch.Tensor, fps: float) -> List[NoteEvent]:
    clip_count, frames, pitches = mask.shape
    events: List[NoteEvent] = []
    mask_cpu = mask.cpu().numpy()
    probs_cpu = probs.cpu().numpy()
    ms_per_frame = 1000.0 / max(fps, 1e-6)
    for clip_idx in range(clip_count):
        for key in range(pitches):
            seq = mask_cpu[clip_idx, :, key]
            if not seq.any():
                continue
            starts = []
            current = None
            for frame_idx, active in enumerate(seq):
                if active and current is None:
                    current = frame_idx
                elif not active and current is not None:
                    starts.append((current, frame_idx))
                    current = None
            if current is not None:
                starts.append((current, frames))
            for onset, offset in starts:
                window = probs_cpu[clip_idx, onset:offset, key]
                confidence = float(window.max()) if window.size > 0 else 0.0
                onset_ms = onset * ms_per_frame
                offset_ms = offset * ms_per_frame
                events.append(
                    NoteEvent(
                        clip_idx=clip_idx,
                        key=key,
                        onset_frame=onset,
                        offset_frame=offset,
                        confidence=confidence,
                        onset_ms=onset_ms,
                        offset_ms=offset_ms,
                    )
                )
    return events


def build_event_set(mask: torch.Tensor, probs: torch.Tensor, fps: float) -> DecoderEventSet:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    if mask.shape != probs.shape:
        raise ValueError(f"Mask/prob tensors must align, got {mask.shape} vs {probs.shape}")
    mask_bool = mask.to(device="cpu", dtype=torch.bool).contiguous()
    probs_float = probs.to(device="cpu", dtype=torch.float32).contiguous()
    clip_count, frames, pitches = mask_bool.shape
    events = _extract_events(mask_bool, probs_float, fps)
    return DecoderEventSet(
        fps=fps,
        mask=mask_bool,
        probs=probs_float,
        events=events,
        clip_count=clip_count,
        frames=frames,
        pitches=pitches,
        device=mask.device,
        dtype=mask.dtype,
    )
