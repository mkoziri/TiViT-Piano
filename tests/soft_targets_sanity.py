"""Quick sanity checks for train-only soft targets.

Run directly as ``python tests/soft_targets_sanity.py`` to validate the target
builder on a tiny synthetic clip.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from utils.frame_target_cache import FrameTargetCache
from utils.frame_targets import (
    FrameTargetSpec,
    prepare_frame_targets,
    resolve_soft_target_config,
)


NOTE_PITCH = 60
ONSET_SEC = 0.3
OFFSET_SEC = 0.6

SPEC = FrameTargetSpec(
    frames=8,
    stride=1,
    fps=10.0,
    canonical_hw=(145, 1024),
    tolerance=0.025,
    dilation=0,
    note_min=21,
    note_max=108,
    fill_mode="overlap",
    hand_from_pitch=False,
    clef_thresholds=(60, 64),
    targets_sparse=False,
)


def _build_payload(split: str, soft_enabled: bool):
    labels = torch.tensor([[ONSET_SEC, OFFSET_SEC, NOTE_PITCH]], dtype=torch.float32)
    if soft_enabled:
        soft_cfg = resolve_soft_target_config(
            {
                "enabled": True,
                "apply_to": {"onset": True, "pitch": True},
                "onset_kernel": [0.5, 1.0, 0.5],
                "frame_kernel": [0.5, 1.0, 0.5],
            }
        )
    else:
        soft_cfg = None

    with TemporaryDirectory() as tmpdir:
        cache = FrameTargetCache(cache_dir=Path(tmpdir))
        result = prepare_frame_targets(
            labels=labels,
            lag_result=None,
            spec=SPEC,
            cache=cache,
            split=split,
            video_id="unit_test_clip",
            clip_start=0.0,
            soft_targets=soft_cfg,
        )
        if result.payload is None:
            raise AssertionError("prepare_frame_targets returned no payload")
        return result.payload


def _frame_indices():
    hop = SPEC.stride / SPEC.fps
    onset_frame = int(round(ONSET_SEC / hop))
    offset_frame = int(round(OFFSET_SEC / hop))
    pitch_idx = NOTE_PITCH - SPEC.note_min
    return onset_frame, offset_frame, pitch_idx


def _assert_close(value: float, expected: float, name: str) -> None:
    if not torch.isclose(torch.tensor(value), torch.tensor(expected), atol=1e-6):
        raise AssertionError(f"{name}: expected {expected}, got {value}")


def main() -> None:
    onset_frame, offset_frame, pitch_idx = _frame_indices()

    base_payload = _build_payload("train", soft_enabled=False)
    onset_roll = base_payload["onset_roll"]
    pitch_roll = base_payload["pitch_roll"]
    _assert_close(onset_roll[onset_frame, pitch_idx].item(), 1.0, "hard onset center")
    _assert_close(onset_roll[onset_frame - 1, pitch_idx].item(), 0.0, "hard onset prev")
    _assert_close(onset_roll[onset_frame + 1, pitch_idx].item(), 0.0, "hard onset next")
    for t in range(onset_frame, offset_frame):
        _assert_close(pitch_roll[t, pitch_idx].item(), 1.0, f"hard frame interior {t}")
    _assert_close(pitch_roll[onset_frame - 1, pitch_idx].item(), 0.0, "hard frame pre")
    _assert_close(pitch_roll[offset_frame, pitch_idx].item(), 0.0, "hard frame post")

    soft_train_payload = _build_payload("train", soft_enabled=True)
    onset_soft = soft_train_payload["onset_roll"]
    pitch_soft = soft_train_payload["pitch_roll"]
    _assert_close(onset_soft[onset_frame, pitch_idx].item(), 1.0, "soft onset center")
    _assert_close(onset_soft[onset_frame - 1, pitch_idx].item(), 0.5, "soft onset prev")
    _assert_close(onset_soft[onset_frame + 1, pitch_idx].item(), 0.5, "soft onset next")
    for t in range(onset_frame, offset_frame):
        _assert_close(pitch_soft[t, pitch_idx].item(), 1.0, f"soft frame interior {t}")
    _assert_close(pitch_soft[onset_frame - 1, pitch_idx].item(), 0.5, "soft frame pre")
    _assert_close(pitch_soft[offset_frame, pitch_idx].item(), 0.5, "soft frame post")

    soft_val_payload = _build_payload("val", soft_enabled=True)
    onset_val = soft_val_payload["onset_roll"]
    pitch_val = soft_val_payload["pitch_roll"]
    _assert_close(onset_val[onset_frame - 1, pitch_idx].item(), 0.0, "val onset prev")
    _assert_close(onset_val[onset_frame + 1, pitch_idx].item(), 0.0, "val onset next")
    _assert_close(pitch_val[onset_frame - 1, pitch_idx].item(), 0.0, "val frame pre")
    _assert_close(pitch_val[offset_frame, pitch_idx].item(), 0.0, "val frame post")

    print("Soft target sanity checks passed.")


if __name__ == "__main__":
    main()
