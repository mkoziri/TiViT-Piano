#!/usr/bin/env python3
"""
Lightweight sanity checks for hand label utilities.

Run directly without pytest:
    python scripts/check/check_hand_label_utils.py

Each check raises AssertionError on failure; successful runs print a short
summary and exit 0.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from hand_labels import (
    AlignedHandLandmarks,
    CanonicalLandmarks,
    EventHandLabelConfig,
    build_event_hand_labels,
    key_centers_from_geometry,
    load_pianovam_hand_landmarks,
    map_landmarks_to_canonical,
)


def _aligned_single_point(x: float, y: float, conf: float = 1.0) -> AlignedHandLandmarks:
    pts = torch.zeros((1, 2, 21, 3), dtype=torch.float32)
    pts[..., 0] = x
    pts[..., 1] = y
    pts[..., 2] = conf
    mask = torch.ones((1, 2, 21), dtype=torch.bool)
    return AlignedHandLandmarks(
        landmarks=pts,
        mask=mask,
        frame_times=torch.tensor([0.0]),
        source_fps=30.0,
        clip_start_sec=0.0,
        metadata={},
    )


def _make_canonical(hand_x_left: float, hand_x_right: float) -> CanonicalLandmarks:
    xy = torch.zeros((2, 2, 21, 3), dtype=torch.float32)
    xy_norm = torch.zeros((2, 2, 21, 2), dtype=torch.float32)
    mask = torch.zeros((2, 2, 21), dtype=torch.bool)
    xy_norm[:, 0, :, 0] = hand_x_left
    xy_norm[:, 1, :, 0] = hand_x_right
    mask[:] = True
    return CanonicalLandmarks(xy=xy, xy_norm=xy_norm, mask=mask, metadata={})


def _make_frame(idx: int, left_val: float, right_val: float, *, conf: float = 0.9):
    pt_left = {"x": left_val, "y": left_val + 0.1, "conf": conf}
    pt_right = {"x": right_val, "y": right_val + 0.1, "conf": conf}
    return {
        "frame": idx,
        "left_hand": [dict(pt_left) for _ in range(21)],
        "right_hand": [dict(pt_right) for _ in range(21)],
    }


def check_event_hand_labels() -> None:
    canonical = _make_canonical(0.2, 0.8)
    onsets = torch.tensor([0.0, 0.033], dtype=torch.float32)
    key_indices = torch.tensor([0, 1], dtype=torch.int64)
    key_centers = torch.tensor([0.15, 0.85], dtype=torch.float32)
    frame_times = torch.tensor([0.0, 0.033], dtype=torch.float32)
    cfg = EventHandLabelConfig(time_tolerance=0.05, max_dx=0.2, min_points=1)

    labels = build_event_hand_labels(
        onsets_sec=onsets,
        key_indices=key_indices,
        key_centers_norm=key_centers,
        frame_times=frame_times,
        canonical=canonical,
        config=cfg,
    )

    assert labels.mask.tolist() == [True, True]
    assert labels.labels.tolist() == [0, 1]
    assert labels.coverage == 1.0

    # Distance/tolerance rejection
    canonical2 = _make_canonical(0.2, 0.4)
    onsets2 = torch.tensor([1.0], dtype=torch.float32)
    key_indices2 = torch.tensor([0], dtype=torch.int64)
    key_centers2 = torch.tensor([0.95], dtype=torch.float32)
    frame_times2 = torch.tensor([0.0, 0.033], dtype=torch.float32)
    cfg2 = EventHandLabelConfig(time_tolerance=0.01, max_dx=0.1, min_points=1)
    labels2 = build_event_hand_labels(
        onsets_sec=onsets2,
        key_indices=key_indices2,
        key_centers_norm=key_centers2,
        frame_times=frame_times2,
        canonical=canonical2,
        config=cfg2,
    )
    assert labels2.mask.sum() == 0
    assert labels2.labels.numel() == 1

    meta = {"key_bounds_px": [[0.0, 1.0], [1.0, 3.0]], "target_hw": [100, 200]}
    centers = key_centers_from_geometry(meta)
    assert centers is not None
    torch.testing.assert_close(
        centers,
        torch.tensor([0.0025, 0.01]),
        atol=1e-4,
        rtol=1e-4,
    )


def check_coordinate_transforms() -> None:
    aligned = _aligned_single_point(5.0, 10.0)
    registration = {
        "homography": torch.eye(3).reshape(-1).tolist(),
        "source_hw": [20, 30],
        "target_hw": [20, 30],
        "x_warp_ctrl": None,
    }
    out = map_landmarks_to_canonical(aligned, registration=registration, source_hw=(20, 30))
    assert out.mask.all()
    torch.testing.assert_close(out.xy[0, 0, 0, 0], torch.tensor(5.0))
    torch.testing.assert_close(out.xy[0, 0, 0, 1], torch.tensor(10.0))
    assert (out.xy_norm >= 0).all() and (out.xy_norm <= 1).all()

    aligned2 = _aligned_single_point(65.0, 30.0)
    crop = {"min_y": 10, "max_y": 60, "min_x": 15, "max_x": 115}
    reg2 = {
        "homography": torch.eye(3).reshape(-1).tolist(),
        "source_hw": [50, 100],
        "target_hw": [50, 100],
        "x_warp_ctrl": None,
    }
    out2 = map_landmarks_to_canonical(aligned2, registration=reg2, source_hw=(80, 130), crop_meta=crop)
    assert out2.mask.all()
    torch.testing.assert_close(out2.xy[0, 0, 0, 0], torch.tensor(50.0))
    torch.testing.assert_close(out2.xy[0, 0, 0, 1], torch.tensor(20.0))

    reg3 = {
        "homography": torch.eye(3).reshape(-1).tolist(),
        "source_hw": [10, 20],
        "target_hw": [10, 20],
        "x_warp_ctrl": [[0.0, 5.0], [20.0, 15.0]],
    }
    out3 = map_landmarks_to_canonical(_aligned_single_point(10.0, 0.0), registration=reg3, source_hw=(10, 20))
    torch.testing.assert_close(out3.xy[0, 0, 0, 0], torch.tensor(10.0))
    assert out3.metadata.get("used_warp_ctrl")


def check_pianovam_loader() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        frames = [_make_frame(i, 0.1 * i, 0.2 * i) for i in range(3)]
        path = tmp / "hands.json"
        path.write_text(json.dumps({"fps": 30.0, "frames": frames}), encoding="utf-8")

        result = load_pianovam_hand_landmarks(
            path,
            clip_start_sec=1.0 / 30.0,
            frames=3,
            stride=1,
            decode_fps=30.0,
        )

        assert result.landmarks.shape == (3, 2, 21, 3)
        assert result.mask.shape == (3, 2, 21)
        assert result.mask.all()
        # t0 should align to source frame 1, t1/t2 land on nearest source frames.
        torch.testing.assert_close(result.landmarks[0, 0, 0, 0], torch.tensor(0.1))
        torch.testing.assert_close(result.landmarks[1, 1, 0, 0], torch.tensor(0.4))
        torch.testing.assert_close(result.landmarks[2, 0, 0, 0], torch.tensor(0.2))
        # t0 should align to source frame 1, t1/t2 land on nearest source frames.
        torch.testing.assert_close(result.landmarks[0, 0, 0, 0], torch.tensor(0.1))
        torch.testing.assert_close(result.landmarks[1, 1, 0, 0], torch.tensor(0.4))
        if result.mask[2, 0, 0]:
            torch.testing.assert_close(result.landmarks[2, 0, 0, 0], torch.tensor(0.2))

        far_path = tmp / "hands_far.json"
        far_path.write_text(json.dumps({"fps": 30.0, "frames": [_make_frame(0, 0.0, 0.0)]}), encoding="utf-8")
        far_res = load_pianovam_hand_landmarks(
            far_path,
            clip_start_sec=2.0,
            frames=2,
            stride=1,
            decode_fps=30.0,
            time_tolerance=0.01,
        )
        assert far_res.mask.sum() == 0

        conf_path = tmp / "hands_conf.json"
        conf_path.write_text(json.dumps({"fps": 30.0, "frames": [_make_frame(0, 0.5, 0.6, conf=0.05)]}), encoding="utf-8")
        conf_res = load_pianovam_hand_landmarks(
            conf_path,
            clip_start_sec=0.0,
            frames=1,
            stride=1,
            decode_fps=30.0,
            min_confidence=0.1,
        )
        assert conf_res.mask.sum() == 0


def main() -> None:
    check_event_hand_labels()
    check_coordinate_transforms()
    check_pianovam_loader()
    print("hand label checks: ok")


if __name__ == "__main__":
    main()
