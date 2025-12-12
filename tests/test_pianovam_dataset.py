from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data.pianoyt_dataset as yt  # noqa: E402
from src.data.pianovam_dataset import make_dataloader  # noqa: E402


def _make_stub_tree(root: Path) -> None:
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train.txt").write_text("video_001\n", encoding="utf-8")

    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "video_001.mp4").write_bytes(b"\x00")
    (train_dir / "audio_001.midi").write_bytes(b"\x00")


@pytest.fixture()
def stubbed_io(monkeypatch):
    """Patch heavy dependencies so dataset materialization stays lightweight."""

    monkeypatch.setattr(
        yt,
        "_read_midi_events",
        lambda path: torch.tensor([[0.0, 0.5, 60.0]], dtype=torch.float32),
    )

    def _fake_clip(
        path,
        frames: int,
        stride: int,
        channels: int,
        training: bool,
        decode_fps: float,
        preferred_start_idx=None,
    ):
        clip = torch.ones(frames, channels, 16, 32)
        return clip, int(preferred_start_idx or 0)

    monkeypatch.setattr(yt, "_load_clip_with_random_start", _fake_clip)
    monkeypatch.setattr(
        yt.RegistrationRefiner, "transform_clip", lambda self, clip, **_: clip
    )


def test_pianovam_loader_matches_pianoyt_surface(tmp_path: Path, stubbed_io) -> None:
    root = tmp_path / "PianoVAM"
    _make_stub_tree(root)

    cfg = {
        "experiment": {"seed": 123, "deterministic": True},
        "model": {"transformer": {"input_patch_size": 16}},
        "tiling": {"patch_w": 16, "tokens_split": "auto", "overlap_tokens": 0},
        "training": {"soft_targets": {"enabled": False}},
        "dataset": {
            "name": "PianoVAM",
            "root_dir": str(root),
            "frames": 4,
            "hop_seconds": 1 / 30.0,
            "decode_fps": 30.0,
            "resize": [16, 32],
            "tiles": 1,
            "channels": 3,
            "normalize": True,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "apply_crop": False,
            "include_low_res": True,
            "registration": {"enabled": False},
            "frame_targets": {
                "enable": True,
                "tolerance": 0.01,
                "dilate_active_frames": 0,
                "hand_from_pitch": True,
                "clef_thresholds": [60, 64],
                "note_min": 21,
                "note_max": 108,
                "cache_labels": False,
                "targets_sparse": False,
            },
            "label_targets": ["pitch", "onset", "offset", "hand", "clef"],
            "require_labels": True,
            "sampler": {"mode": None},
            "avlag_disabled": True,
            "canonical_hw": [16, 32],
        },
    }

    loader = make_dataloader(cfg, split="train", drop_last=False, seed=7)
    dataset = loader.dataset
    assert dataset.split == "train"
    assert dataset.require_labels

    batch = next(iter(loader))
    assert batch["video"].shape[0] == 1
    assert batch["video"].shape[1] == cfg["dataset"]["frames"]
    assert batch["video"].shape[2:] == (3, 16, 32)

    assert isinstance(batch.get("labels"), list)
    assert batch["labels"][0].shape[1] == 3  # onset, offset, pitch

    for key in ("pitch", "onset", "offset", "hand", "clef", "clip_id"):
        assert key in batch, f"missing clip-level target {key}"

    # Frame-level payload mirrors PianoYT
    for key in ("pitch_roll", "onset_roll", "offset_roll", "frame_mask"):
        assert key in batch
        assert batch[key].shape[1] == cfg["dataset"]["frames"]
