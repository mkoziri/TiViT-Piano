#!/usr/bin/env python3
"""Build a PianoVAM manifest JSON from metadata_v2.json.

Usage:
  python tivit/data/datasets/build_pianovam_manifest.py --root data/PianoVAM_v1.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _parse_point(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except (TypeError, ValueError):
                return None
    if isinstance(value, Sequence) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _crop_from_points(entry: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    labels = ["Point_LT", "Point_RT", "Point_RB", "Point_LB"]
    points: list[Tuple[float, float]] = []
    for key in labels:
        pt = _parse_point(entry.get(key))
        if pt is not None:
            points.append(pt)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    try:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (int(round(min_y)), int(round(max_y)), int(round(min_x)), int(round(max_x)))
    except Exception:
        return None


def _resolve_root(root_arg: Optional[str]) -> Path:
    candidates: list[Path] = []
    if root_arg:
        candidates.append(Path(root_arg).expanduser())
    env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env:
        base = Path(env).expanduser()
        candidates.append(base / "PianoVAM_v1.0")
        candidates.append(base / "PianoVAM")
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "data" / "PianoVAM_v1.0")
    candidates.append(repo_root / "data" / "PianoVAM")
    candidates.append(Path("~/datasets/PianoVAM_v1.0").expanduser())
    candidates.append(Path("~/datasets/PianoVAM").expanduser())

    for cand in candidates:
        if cand and cand.exists():
            return cand
    raise FileNotFoundError(
        "Unable to locate PianoVAM root. Pass --root or set TIVIT_DATA_DIR/DATASETS_HOME."
    )


def _normalize_split(
    meta_split: Any,
    *,
    include_ext_train: bool,
    include_special: bool,
) -> Optional[str]:
    if meta_split is None:
        return None
    split = str(meta_split).strip().lower()
    if split in {"val", "valid", "validation"}:
        return "val"
    if split == "train":
        return "train"
    if split in {"ext-train", "ext_train"}:
        return "train" if include_ext_train else None
    if split == "test":
        return "test"
    if split.startswith("special"):
        return "train" if include_special else None
    return None


def _resolve_label(root: Path, rec_id: str, *, prefer_tsv: bool) -> Optional[str]:
    tsv_rel = f"TSV/{rec_id}.tsv"
    midi_mid_rel = f"MIDI/{rec_id}.mid"
    midi_midi_rel = f"MIDI/{rec_id}.midi"
    tsv_path = root / tsv_rel
    midi_mid = root / midi_mid_rel
    midi_midi = root / midi_midi_rel
    midi_path = midi_mid if midi_mid.exists() else midi_midi if midi_midi.exists() else None

    if prefer_tsv:
        if tsv_path.exists():
            return tsv_rel
        if midi_path is not None:
            return midi_mid_rel if midi_path == midi_mid else midi_midi_rel
    else:
        if midi_path is not None:
            return midi_mid_rel if midi_path == midi_mid else midi_midi_rel
        if tsv_path.exists():
            return tsv_rel
    return None


def build_manifest(
    root: Path,
    *,
    include_ext_train: bool,
    include_special: bool,
    prefer_tsv: bool,
    include_hands: bool,
    include_crop: bool,
) -> Dict[str, list[Dict[str, Any]]]:
    meta_path = root / "metadata_v2.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_v2.json not found at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"metadata_v2.json must be a dict, got {type(raw)}")

    manifest: Dict[str, list[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    missing_video = 0
    missing_label = 0

    for entry in raw.values():
        if not isinstance(entry, Mapping):
            continue
        rec_id = entry.get("record_time") or entry.get("id")
        if not rec_id:
            continue
        split = _normalize_split(entry.get("split"), include_ext_train=include_ext_train, include_special=include_special)
        if split is None:
            continue
        rec_id = str(rec_id)
        video_rel = f"Video/{rec_id}.mp4"
        video_path = root / video_rel
        if not video_path.exists():
            missing_video += 1
            continue
        label_rel = _resolve_label(root, rec_id, prefer_tsv=prefer_tsv)
        if label_rel is None:
            missing_label += 1
            continue
        item: Dict[str, Any] = {"video_id": rec_id, "video": video_rel, "label": label_rel}
        if include_hands:
            hand_rel = f"Handskeleton/{rec_id}.json"
            if (root / hand_rel).exists():
                item["hand_skeleton"] = hand_rel
        if include_crop:
            crop = _crop_from_points(entry)
            if crop is not None:
                item["crop"] = list(crop)
        manifest[split].append(item)

    total = sum(len(v) for v in manifest.values())
    print(f"[manifest] total={total} train={len(manifest['train'])} val={len(manifest['val'])} test={len(manifest['test'])}")
    if missing_video:
        print(f"[manifest] skipped_missing_video={missing_video}")
    if missing_label:
        print(f"[manifest] skipped_missing_label={missing_label}")
    return manifest


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build PianoVAM manifest from metadata_v2.json.")
    ap.add_argument("--root", type=str, default=None, help="PianoVAM root (defaults to env/data fallbacks).")
    ap.add_argument("--output", type=str, default=None, help="Output manifest JSON path.")
    ap.add_argument("--include-ext-train", action="store_true", help="Map ext-train to train.")
    ap.add_argument("--include-special", action="store_true", help="Map special* splits to train.")
    ap.add_argument("--prefer-midi", action="store_true", help="Prefer MIDI labels over TSV when both exist.")
    ap.add_argument("--no-hands", action="store_true", help="Do not include hand skeleton paths.")
    ap.add_argument("--no-crop", action="store_true", help="Do not include crop metadata.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    root = _resolve_root(args.root)
    output = Path(args.output).expanduser() if args.output else root / "manifest.json"
    manifest = build_manifest(
        root,
        include_ext_train=bool(args.include_ext_train),
        include_special=bool(args.include_special),
        prefer_tsv=not bool(args.prefer_midi),
        include_hands=not bool(args.no_hands),
        include_crop=not bool(args.no_crop),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[manifest] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
