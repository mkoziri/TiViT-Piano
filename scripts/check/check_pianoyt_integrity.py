#!/usr/bin/env python3
"""Purpose:
    Validate the on-disk PianoYT dataset by checking video/MIDI pairs, reporting
    anomalies, and collecting metadata summaries.

This mirrors the UX of ``check_omaps_integrity.py`` so that day-to-day workflows
remain familiar while adjusting for PianoYT's flat directory structure and MIDI
annotations.

Key Functions/Classes:
    - read_midi(): Parses MIDI annotations with pretty_midi first and mido as a
      fallback while enforcing onset/pitch validations.
    - scan_split(): Loads PianoYT split lists, resolves videos/MIDI files, and
      aggregates per-piece statistics plus metadata crops.
    - main(): Handles CLI arguments, orchestrates split scanning, and writes
      JSON/TSV/anomaly reports alongside console summaries.

CLI:
    Execute ``python scripts/check/check_pianoyt_integrity.py --root ~/dev/tivit/data/PianoYT``
    with optional ``--probe-media`` for ffprobe enrichment.
    Use ``--meta-dir`` / ``--report-dir`` to control output destinations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


VIDEO_EXT_PREFERENCE: Sequence[str] = (".mp4", ".mkv", ".webm")
MIDI_EXT = ".midi"

# Populated via ``load_crop_metadata`` prior to scanning splits.
PIANOYT_CROPS: Dict[str, Sequence[int]] = {}
EXCLUDED_IDS: Set[str] = set()


def read_midi(midi_path: Path) -> List[Tuple[float, float, int]]:
    """Return [(onset_sec, offset_sec, pitch)] events from ``midi_path``."""

    def _validate(rows: Iterable[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        out: List[Tuple[float, float, int]] = []
        for onset, offset, pitch in rows:
            if onset < 0 or onset >= offset:
                raise ValueError(f"Invalid onset/offset pair: onset={onset} offset={offset}")
            if not (0 <= pitch <= 127):
                raise ValueError(f"Pitch out of MIDI range: {pitch}")
            out.append((float(onset), float(offset), int(pitch)))
        if not out:
            raise ValueError(f"No note events found in {midi_path}")
        out.sort(key=lambda r: (r[0], r[1], r[2]))
        return out

    errors: List[Exception] = []

    # Preferred reader: pretty_midi (handles tempo maps & controllers).
    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        rows = _validate((note.start, note.end, note.pitch)
                         for inst in pm.instruments for note in inst.notes)
        return rows
    except ImportError as exc:  # pragma: no cover - library optional in CI
        errors.append(exc)
    except Exception as exc:
        errors.append(exc)

    # Fallback: mido -- manually track tempo to convert ticks to seconds.
    try:
        import mido

        mid = mido.MidiFile(str(midi_path))
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000  # default microseconds per beat
        current_time = 0.0
        stacks: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        rows: List[Tuple[float, float, int]] = []

        for msg in mido.merge_tracks(mid.tracks):
            delta = mido.tick2second(msg.time, ticks_per_beat, tempo)
            current_time += delta
            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                stacks[(msg.channel, msg.note)].append(current_time)
            elif msg.type in {"note_off", "note_on"}:
                key = (getattr(msg, "channel", 0), msg.note)
                if not stacks[key]:
                    raise ValueError(f"Dangling note_off for pitch={msg.note}")
                onset = stacks[key].pop()
                rows.append((onset, current_time, msg.note))

        dangling = [key for key, starts in stacks.items() if starts]
        if dangling:
            raise ValueError(f"Unterminated note_on events: {dangling[:5]}")

        return _validate(rows)
    except ImportError as exc:  # pragma: no cover - library optional in CI
        errors.append(exc)
    except Exception as exc:
        errors.append(exc)

    message = ", ".join(f"{type(err).__name__}: {err}" for err in errors)
    raise ValueError(f"Failed to parse MIDI {midi_path}: {message}")


def ffprobe_info(media_path: Path) -> Dict[str, object]:
    """Return dict with r_frame_rate,width,height,audio_sr or {} if ffprobe missing."""

    if shutil.which("ffprobe") is None:
        return {}
    import subprocess

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


def load_crop_metadata(root: Path) -> Dict[str, Sequence[int]]:
    csv_path = root / "metadata" / "pianoyt.csv"
    if not csv_path.exists():
        print(f"[WARN] metadata/pianoyt.csv missing under {root}; crop lookups disabled")
        return {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"videoID", "min_y", "max_y", "min_x", "max_x"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"metadata/pianoyt.csv missing columns: {sorted(missing)}")
        crops: Dict[str, Sequence[int]] = {}
        for row in reader:
            try:
                crops[row["videoID"].strip()] = [
                    int(float(row["min_y"])),
                    int(float(row["max_y"])),
                    int(float(row["min_x"])),
                    int(float(row["max_x"])),
                ]
            except Exception as exc:
                raise SystemExit(f"Invalid crop row for videoID={row.get('videoID')}: {exc}") from exc
        return crops


def load_excluded_ids(root: Path) -> Set[str]:
    excl_path = root / "splits" / "excluded_low.txt"
    if not excl_path.exists():
        return set()
    with excl_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def scan_split(split_name: str, root: Path, probe_media: bool, anomalies: List[str]) -> List[Dict[str, object]]:
    """Return list of piece dicts for a PianoYT split."""

    ids_path = root / "splits" / f"{split_name}.txt"
    if not ids_path.exists():
        if split_name == "val":
            print(f"[WARN] splits/{split_name}.txt missing; treating VAL as empty")
            anomalies.append(f"[EMPTY SPLIT] split={split_name} (missing split file)")
            return []
        raise SystemExit(f"Expected split list missing: {ids_path}")

    with ids_path.open("r", encoding="utf-8") as f:
        ids = []
        for line in f:
            ident = line.strip()
            if not ident or ident.startswith("#"):
                continue
            if ident in EXCLUDED_IDS:
                continue
            ids.append(ident)

    if not ids:
        anomalies.append(f"[EMPTY SPLIT] split={split_name} ids=0")
        return []

    split_dir = root / split_name
    if not split_dir.exists() and split_name == "val":
        split_dir = root / "train"
    if not split_dir.exists():
        raise SystemExit(f"Expected directory missing for split '{split_name}': {root / split_name}")

    pieces: List[Dict[str, object]] = []
    for piece_id in ids:
        midi_path = split_dir / f"audio_{piece_id}.0{MIDI_EXT}"
        if not midi_path.exists():
            anomalies.append(
                f"[MISSING MIDI] split={split_name} id={piece_id} path={midi_path}"
            )
            continue

        try:
            rows = read_midi(midi_path)
        except Exception as exc:
            anomalies.append(
                f"[BAD MIDI] split={split_name} id={piece_id} midi={midi_path} :: {exc}"
            )
            continue

        duration = max(offset for _, offset, _ in rows)
        n_notes = len(rows)
        speed = n_notes / duration if duration > 0 else 0.0
        pitch_min = min(pitch for *_, pitch in rows)
        pitch_max = max(pitch for *_, pitch in rows)

        piece: Dict[str, object] = {
            "id": piece_id,
            "duration_sec": duration,
            "n_notes": n_notes,
            "notes_per_sec": speed,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "has_video": False,
            "midi": str(midi_path.resolve()),
        }

        video_path: Optional[Path] = None
        for ext in VIDEO_EXT_PREFERENCE:
            candidate = split_dir / f"video_{piece_id}.0{ext}"
            if candidate.exists():
                video_path = candidate
                break
        if video_path is not None:
            piece["has_video"] = True
            piece["video"] = str(video_path.resolve())
            if probe_media:
                piece.update(ffprobe_info(video_path))
        else:
            anomalies.append(
                f"[MISSING VIDEO] split={split_name} id={piece_id} candidates={[split_dir / f'video_{piece_id}.0{ext}' for ext in VIDEO_EXT_PREFERENCE]}"
            )

        crop = PIANOYT_CROPS.get(piece_id)
        if crop:
            piece["crop"] = list(crop)
        else:
            anomalies.append(f"[MISSING CROP] split={split_name} id={piece_id}")

        pieces.append(piece)

    if not pieces:
        anomalies.append(f"[EMPTY SPLIT] split={split_name} (no valid pieces)")

    return pieces


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_tsv(path: Path, pieces: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "duration_sec",
        "n_notes",
        "notes_per_sec",
        "pitch_min",
        "pitch_max",
        "has_video",
        "video",
        "midi",
        "crop",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for rec in sorted(pieces, key=lambda x: x["id"]):
            row = [str(rec.get(col, "")) for col in cols]
            f.write("\t".join(row) + "\n")


def summarize(name: str, pieces: Sequence[Dict[str, object]]) -> None:
    if not pieces:
        print(f"[{name}] 0 pieces")
        return
    speeds = [p["notes_per_sec"] for p in pieces]
    durs = [p["duration_sec"] for p in pieces]
    with_video = sum(1 for p in pieces if p.get("has_video"))
    print(
        f"[{name}] pieces={len(pieces)} "
        f"| notes/s median={median(speeds):.3f} "
        f"| dur median={median(durs):.2f}s "
        f"| with_video={with_video}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PianoYT Step 0: integrity + metadata for ~/dev/tivit/data/PianoYT/{train,val?,test}."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root that contains 'train', optional 'val', and 'test' subfolders",
    )
    ap.add_argument(
        "--probe-media",
        action="store_true",
        help="Use ffprobe to log FPS/resolution/audio rate if available",
    )
    ap.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("metadata"),
        help="Directory to write pianoyt_*.json reports",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write pianoyt_*.tsv and anomaly listings",
    )
    args = ap.parse_args()

    root = Path(os.path.expanduser(str(args.root))).resolve()

    global PIANOYT_CROPS, EXCLUDED_IDS
    PIANOYT_CROPS = load_crop_metadata(root)
    EXCLUDED_IDS = load_excluded_ids(root)

    anomalies: List[str] = []

    train_pieces = scan_split("train", root, args.probe_media, anomalies)
    val_list_path = root / "splits" / "val.txt"
    val_exists = val_list_path.exists()
    val_pieces = scan_split("val", root, args.probe_media, anomalies)
    if not val_exists:
        val_pieces = []
    test_pieces = scan_split("test", root, args.probe_media, anomalies)

    meta_dir = args.meta_dir
    report_dir = args.report_dir

    write_json(
        meta_dir / "pianoyt_train.json",
        {"root": str(root), "split": "train", "n": len(train_pieces), "pieces": train_pieces},
    )
    if val_exists:
        write_json(
            meta_dir / "pianoyt_val.json",
            {"root": str(root), "split": "val", "n": len(val_pieces or []), "pieces": val_pieces or []},
        )
    write_json(
        meta_dir / "pianoyt_test.json",
        {"root": str(root), "split": "test", "n": len(test_pieces), "pieces": test_pieces},
    )
    write_json(
        meta_dir / "pianoyt_all.json",
        {
            "root": str(root),
            "split": "all",
            "n": len(train_pieces)
            + len(test_pieces)
            + (len(val_pieces or []) if val_exists else 0),
            "pieces": train_pieces + (val_pieces or []) + test_pieces,
        },
    )

    write_tsv(report_dir / "pianoyt_train.tsv", train_pieces)
    if val_exists:
        write_tsv(report_dir / "pianoyt_val.tsv", val_pieces or [])
    write_tsv(report_dir / "pianoyt_test.tsv", test_pieces)

    if anomalies:
        anomaly_path = report_dir / "pianoyt_anomalies.txt"
        anomaly_path.parent.mkdir(parents=True, exist_ok=True)
        anomaly_path.write_text("\n".join(anomalies), encoding="utf-8")
        print(f"[WARN] {len(anomalies)} anomalies → {anomaly_path}")

    summarize("TRAIN", train_pieces)
    summarize("VAL", val_pieces or [])
    summarize("TEST", test_pieces)
    print(f"[OK] JSON → {meta_dir}/pianoyt_*.json")
    print(f"[OK] TSV  → {report_dir}/pianoyt_*.tsv")


if __name__ == "__main__":
    main()

