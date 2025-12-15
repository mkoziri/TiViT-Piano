#!/usr/bin/env python3
"""Purpose:
    Light-weight integrity check for the local PianoVAM dataset using the
    official PianoVAM_v1.0 layout (Video/MIDI/TSV plus metadata_v2.json).
    Mirrors the CLI style of the other dataset checkers without relying on
    pytest.

What it does:
    - Resolves the PianoVAM root (explicit --root or env/data fallbacks).
    - Loads split ID lists and verifies video/MIDI pairs can be located.
    - Optionally parses MIDI to count events and probes media metadata via
      ffprobe for quick smoke diagnostics.

Usage:
    python scripts/check/check_pianovam_integrity.py --root ~/datasets/PianoVAM_v1.0
    python scripts/check/check_pianovam_integrity.py --probe-media --parse-midi --max-per-split 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.data.pianovam_dataset import _expand_root, _load_metadata, _split_matches, _resolve_media_paths
import src.data.pianoyt_dataset as yt
from utils.identifiers import canonical_video_id


def ffprobe_info(media_path: Path) -> Dict[str, object]:
    """Return dict with r_frame_rate,width,height,audio_sr or {} if ffprobe missing."""

    if shutil.which("ffprobe") is None:
        return {}
    try:
        v = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,width,height",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        a_sr = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        out: Dict[str, object] = {}
        if len(v) >= 3:
            out.update({"r_frame_rate": v[0], "width": int(v[1]), "height": int(v[2])})
        if a_sr:
            out.update({"audio_sr": a_sr})
        return out
    except Exception:
        return {}


def scan_split(
    root: Path,
    split: str,
    *,
    max_items: Optional[int],
    probe_media: bool,
    parse_midi: bool,
) -> Tuple[List[Mapping[str, object]], List[str]]:
    """Scan a split for video/MIDI pairs based on metadata_v2.json."""

    metadata = _load_metadata(root)
    ids: List[str] = []
    id_to_rec: Dict[str, str] = {}
    for rec in metadata.values():
        rec_id_raw = rec.get("record_time") or rec.get("id")
        meta_split = rec.get("split")
        if not rec_id_raw or not meta_split:
            continue
        if not _split_matches(split, str(meta_split)):
            continue
        rec_id = str(rec_id_raw)
        canon = canonical_video_id(rec_id)
        ids.append(canon)
        id_to_rec[canon] = rec_id
    if not ids:
        return [], [f"[EMPTY SPLIT] No entries match split={split} in metadata_v2.json"]

    records: List[Mapping[str, object]] = []
    anomalies: List[str] = []

    limit = len(ids) if max_items is None else min(max_items, len(ids))
    for idx, vid in enumerate(ids[:limit]):
        canon = canonical_video_id(vid)
        rec_id = id_to_rec.get(canon, canon)
        video_path, midi_path, tsv_path = _resolve_media_paths(root, rec_id)
        rec: Dict[str, object] = {"id": canon}
        if video_path is not None:
            rec["video"] = str(video_path)
        else:
            anomalies.append(f"[MISSING VIDEO] split={split} id={canon}")

        if midi_path is not None:
            rec["midi"] = str(midi_path)
            if parse_midi:
                try:
                    midi_events = yt._read_midi_events(midi_path)
                    rec["midi_events"] = int(midi_events.shape[0])
                except Exception as exc:
                    anomalies.append(f"[MIDI PARSE FAIL] split={split} id={canon} error={exc}")
        else:
            anomalies.append(f"[MISSING MIDI] split={split} id={canon}")

        if tsv_path is not None:
            rec["tsv"] = str(tsv_path)

        if probe_media and video_path is not None:
            rec.update(ffprobe_info(video_path))

        records.append(rec)
    return records, anomalies


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check PianoVAM layout and annotations.")
    parser.add_argument("--root", type=str, default=None, help="Root of PianoVAM (defaults to env/data fallbacks).")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to scan (metadata_v2.json names: train, valid, test, ext-train, special*).",
    )
    parser.add_argument("--max-per-split", type=int, default=20, help="Limit items per split (0 for all).")
    parser.add_argument("--probe-media", action="store_true", help="Probe ffprobe metadata for videos.")
    parser.add_argument("--parse-midi", action="store_true", help="Parse MIDI to count events.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write JSON summary.")
    args = parser.parse_args(argv)

    try:
        root = _expand_root(args.root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[PianoVAM] root={root}")
    max_items = None if args.max_per_split <= 0 else args.max_per_split
    all_records: Dict[str, List[Mapping[str, object]]] = {}
    all_anomalies: List[str] = []

    for split in args.splits:
        try:
            records, anomalies = scan_split(
                root,
                split,
                max_items=max_items,
                probe_media=args.probe_media,
                parse_midi=args.parse_midi,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to scan split '{split}': {exc}")
            all_anomalies.append(f"[SCAN FAIL] split={split} error={exc}")
            continue

        all_records[split] = records
        all_anomalies.extend(anomalies)

        ok = sum(1 for r in records if "video" in r and "midi" in r)
        print(f"  split={split:5s} items={len(records):4d} ok_pairs={ok:4d} anomalies={len(anomalies)}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {"root": str(root), "splits": all_records, "anomalies": all_anomalies}
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"[PianoVAM] wrote JSON summary to {out_path}")

    if all_anomalies:
        print("\nAnomalies:")
        for line in all_anomalies:
            print("  -", line)
        return 1

    print("PianoVAM integrity check completed without anomalies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
