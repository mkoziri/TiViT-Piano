"""Quick checks for PianoVAM time alignment and label integrity.

Run this script on a few clips to verify that:
- The video duration (from decord) matches the last note offset in the TSV.
- The sampled frame hop implied by ``--frames`` is compatible with
  ``--decode-fps``.
- Pitches stay within the 88-key range and there are no empty TSV files.

Example:
    python scripts/debug_pianovam_alignment.py \
        --config configs/config_pianovam_supervised.yaml \
        --split val --frames 128 --decode-fps 30 --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import decord

from src.utils.config import load_config
from src.data.pianovam_dataset import _resolve_root, NOTE_MIN, NOTE_MAX


decord.bridge.set_bridge("native")


def _load_metadata(root: Path) -> Dict[str, Dict]:
    meta_path = root / "metadata_v2.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(meta: Dict[str, Dict], split: str) -> Iterable[Dict]:
    for rec in meta.values():
        if not isinstance(rec, dict):
            continue
        if rec.get("split", "").lower() != split.lower():
            continue
        record_time = rec.get("record_time")
        if not record_time:
            continue
        yield rec


def _tsv_stats(tsv_path: Path) -> Tuple[float, float, int, int, int]:
    max_onset = 0.0
    max_offset = 0.0
    n_events = 0
    min_pitch = 999
    max_pitch = -999

    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            onset, key_off, *_rest = line.split("\t")
            pitch = int(_rest[1])

            onset_f = float(onset)
            offset_f = float(key_off)

            max_onset = max(max_onset, onset_f)
            max_offset = max(max_offset, offset_f)
            min_pitch = min(min_pitch, pitch)
            max_pitch = max(max_pitch, pitch)
            n_events += 1

    return max_onset, max_offset, n_events, min_pitch, max_pitch


def _video_duration(video_path: Path) -> Tuple[float, int, float]:
    reader = decord.VideoReader(str(video_path))
    total_frames = len(reader)
    try:
        native_fps = float(reader.get_avg_fps())
        if native_fps <= 0:
            raise ValueError
    except Exception:
        native_fps = 30.0
    duration = total_frames / native_fps if native_fps else 0.0
    return duration, total_frames, native_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="PianoVAM alignment sanity checks")
    parser.add_argument("--config", type=str, default="configs/config_pianovam_supervised.yaml")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset.root_dir")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--frames", type=int, default=128, help="Frames sampled per clip")
    parser.add_argument("--decode-fps", type=float, default=30.0, help="Intended decode_fps")
    parser.add_argument("--limit", type=int, default=10, help="Number of clips to inspect")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_dataset = cfg.get("dataset", {}) or {}
    if args.dataset_root:
        cfg_dataset = {**cfg_dataset, "root_dir": args.dataset_root}
    else:
        cfg_dataset = cfg.get("dataset", {}) or {}

    root = _resolve_root({"dataset": cfg_dataset})
    meta = _load_metadata(root)

    print(f"[pianovam-debug] root={root} split={args.split} frames={args.frames} decode_fps={args.decode_fps}")

    sample_count = 0
    decode_hop = 1.0 / args.decode_fps if args.decode_fps > 0 else 0.0

    for rec in _iter_records(meta, args.split):
        rec_id = rec.get("record_time")
        video_path = root / "Video" / f"{rec_id}.mp4"
        tsv_path = root / "TSV" / f"{rec_id}.tsv"

        if not video_path.is_file() or not tsv_path.is_file():
            continue

        duration, total_frames, native_fps = _video_duration(video_path)
        max_onset, max_offset, n_events, min_pitch, max_pitch = _tsv_stats(tsv_path)

        sampled_hop = duration / max(args.frames - 1, 1)
        alignment_gap = abs(max_offset - duration)

        print(
            f"[{rec_id}] video={duration:.3f}s ({total_frames}f @ {native_fps:.2f}fps) | "
            f"tsv_last_onset={max_onset:.3f}s tsv_last_offset={max_offset:.3f}s n_events={n_events} | "
            f"sampled_hop={sampled_hop:.4f}s decode_hop={decode_hop:.4f}s gap(video-tsv)={alignment_gap:.3f}s | "
            f"pitch_range=[{min_pitch}, {max_pitch}]"
        )

        if min_pitch < NOTE_MIN or max_pitch > NOTE_MAX:
            print(f"  ⚠️ pitch outside 88-key range ({NOTE_MIN}-{NOTE_MAX})")
        if n_events == 0:
            print("  ⚠️ empty TSV (no events parsed)")

        sample_count += 1
        if sample_count >= args.limit:
            break


if __name__ == "__main__":
    main()