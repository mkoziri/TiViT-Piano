#!/usr/bin/env python3
"""Purpose:
    Build a calibration/validation split from any TiViT dataset manifest by
    sampling IDs from an input list.  The script replaces the previous
    OMAPS-specific flow with a generic splitter that understands the PianoYT
    layout as well as legacy datasets.

Key Functions/Classes:
    - read_ids(): Load newline-delimited IDs.
    - select_ids(): Deterministically choose validation IDs via hash or random
      strategies.
    - materialize_val_split(): Optionally create a ``val/`` directory containing
      the selected media using symlinks or hardlinks.

CLI:
    python scripts/make_calib_split.py \
      --root ~/dev/tivit/data/PianoYT \
      --train-file splits/train.txt \
      --out-val splits/val.txt \
      --out-train-minus-val splits/train_minus_val.txt \
      --fraction 0.1 --seed 1337 --method hash \
      --respect-excluded splits/excluded_low.txt \
      --create-val-dir --copy-mode symlink
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS: Sequence[str] = (".mp4", ".mkv", ".webm")


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Missing ID list: {path}")
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def read_optional_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def select_ids(ids: Sequence[str], fraction: float, seed: int, method: str) -> List[str]:
    if not ids:
        return []
    n = len(ids)
    k = min(n, max(0, math.ceil(float(fraction) * n)))
    if k <= 0:
        return []

    if method == "hash":
        keyed = []
        for vid in ids:
            h = hashlib.sha256(f"{vid}:{seed}".encode("utf-8")).hexdigest()
            keyed.append((h, vid))
        keyed.sort()
        chosen = [vid for _, vid in keyed[:k]]
    elif method == "random":
        rng = random.Random(seed)
        chosen = rng.sample(list(ids), k)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return sorted(chosen)


def write_list(path: Path, ids: Iterable[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vid in sorted(ids):
            handle.write(f"{vid}\n")


def resolve_media(train_dir: Path, video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    midi_path = train_dir / f"audio_{video_id}.0.midi"
    video = None
    for ext in VIDEO_EXTS:
        cand = train_dir / f"video_{video_id}.0{ext}"
        if cand.exists():
            video = cand
            break
    midi = midi_path if midi_path.exists() else None
    return video, midi


def materialize_val_split(root: Path, ids: Sequence[str], mode: str):
    val_dir = root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    train_dir = root / "train"

    for vid in ids:
        video_src, midi_src = resolve_media(train_dir, vid)
        if video_src is None or midi_src is None:
            print(f"[WARN] Missing media for {vid}; skipping filesystem copy")
            continue

        if mode == "none":
            continue

        video_dst = val_dir / video_src.name
        midi_dst = val_dir / midi_src.name
        for dst in (video_dst, midi_dst):
            if dst.exists():
                dst.unlink()

        if mode == "symlink":
            video_dst.symlink_to(video_src)
            midi_dst.symlink_to(midi_src)
        elif mode == "hardlink":
            os.link(video_src, video_dst)
            os.link(midi_src, midi_dst)
        else:
            raise ValueError(f"Unsupported copy mode: {mode}")
            
            
def main():
    ap = argparse.ArgumentParser(description="Create calibration/validation splits")
    ap.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    ap.add_argument("--train-file", type=Path, required=True, help="Path to train ID list")
    ap.add_argument("--out-val", type=Path, required=True, help="Output path for validation IDs")
    ap.add_argument("--out-train-minus-val", type=Path, required=True, help="Output path for remaining IDs")
    ap.add_argument("--fraction", type=float, default=0.1, help="Fraction of train IDs for validation")
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    ap.add_argument("--method", choices=["hash", "random"], default="hash", help="Selection strategy")
    ap.add_argument("--respect-excluded", type=Path, help="Optional list of IDs to exclude")
    ap.add_argument("--create-val-dir", action="store_true", help="Materialize val/ directory")
    ap.add_argument("--copy-mode", choices=["symlink", "hardlink", "none"], default="symlink", help="Copy strategy when creating val dir")
    args = ap.parse_args()

    root = args.root.expanduser()
    ids = read_ids(args.train_file.expanduser())

    excluded = read_optional_ids(args.respect_excluded.expanduser()) if args.respect_excluded else set()
    if excluded:
        ids = [vid for vid in ids if vid not in excluded]
        print(f"[INFO] Excluded {len(excluded)} IDs listed in {args.respect_excluded}")

    chosen = select_ids(ids, args.fraction, args.seed, args.method)
    remaining = sorted(set(ids) - set(chosen))

    write_list(args.out_val.expanduser(), chosen)
    write_list(args.out_train_minus_val.expanduser(), remaining)

    report = {
        "total_train": len(ids),
        "val_count": len(chosen),
        "train_minus_val_count": len(remaining),
        "fraction": args.fraction,
        "seed": args.seed,
        "method": args.method,
        "excluded": len(excluded),
    }

    report_path = args.out_val.expanduser().parent / "val_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[OK] Selected {len(chosen)} validation IDs (fraction={args.fraction:.3f}, method={args.method})")
    print(f"[OK] Wrote {args.out_val} and {args.out_train_minus_val}")
    print(f"[OK] Report → {report_path}")

    if args.create_val_dir and chosen:
        materialize_val_split(root, chosen, args.copy_mode)
        print(f"[OK] Materialized {len(chosen)} items under {root/'val'} using mode={args.copy_mode}")


if __name__ == "__main__":
    main()

