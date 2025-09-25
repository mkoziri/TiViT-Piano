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


def _clean_cell(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip().lstrip("\ufeff").rstrip("\ufeff")


def _normalise_header(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = _clean_cell(name)
    if not cleaned:
        return ""
    key = "".join(ch for ch in cleaned.lower() if ch.isalnum())
    alias_map = {
        "videoid": "videoID",
        "miny": "min_y",
        "maxy": "max_y",
        "minx": "min_x",
        "maxx": "max_x",
        "crop": "crop",
    }
    return alias_map.get(key, cleaned)


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
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)

        try:
            first_row = next(reader)
        except StopIteration:
            raise SystemExit("metadata/pianoyt.csv is empty")

        if first_row is None:
            raise SystemExit("metadata/pianoyt.csv is empty")

        first_row = [_clean_cell(cell) for cell in first_row]
        normalised = [_normalise_header(cell) for cell in first_row]
        header_fields = {name for name in normalised if name}
        known_header = {"videoID", "min_y", "max_y", "min_x", "max_x", "crop"}
        has_header = bool(header_fields & known_header)

        crops: Dict[str, Sequence[int]] = {}
        
        def parse_crop_sequence(values: Sequence[str], video_id: str) -> Sequence[int]:
            try:
                return [int(float(v)) for v in values]
            except Exception as exc:
                raise SystemExit(
                    f"Invalid crop values for videoID={video_id}: {values} ({exc})"
                ) from exc

        if has_header:
            column_map: Dict[str, int] = {}
            for idx, name in enumerate(normalised):
                if name and name not in column_map:
                    column_map[name] = idx

            fields = set(column_map)
            required = {"videoID", "min_y", "max_y", "min_x", "max_x"}
            alt_required = {"videoID", "crop"}

            video_idx = column_map.get("videoID")
            if video_idx is None:
                raise SystemExit("metadata/pianoyt.csv missing column: videoID")

            if required.issubset(fields):
                for line_no, raw_row in enumerate(reader, start=2):
                    if not raw_row or not any(cell.strip() for cell in raw_row):
                        continue
                    row = [_clean_cell(cell) for cell in raw_row]
                    if video_idx >= len(row):
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing videoID (line {line_no})"
                        )
                    video_id = row[video_idx]
                    if not video_id:
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing videoID (line {line_no})"
                        )
                    values = []
                    for key in ("min_y", "max_y", "min_x", "max_x"):
                        idx = column_map[key]
                        if idx >= len(row):
                            raise SystemExit(
                                f"metadata/pianoyt.csv row missing {key} for videoID={video_id}"
                            )
                        values.append(row[idx])
                    crops[video_id] = parse_crop_sequence(values, video_id)
                return crops

            if alt_required.issubset(fields):
                crop_idx = column_map["crop"]
                for line_no, raw_row in enumerate(reader, start=2):
                    if not raw_row or not any(cell.strip() for cell in raw_row):
                        continue
                    row = [_clean_cell(cell) for cell in raw_row]
                    if video_idx >= len(row):
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing videoID (line {line_no})"
                        )
                    video_id = row[video_idx]
                    if not video_id:
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing videoID (line {line_no})"
                        )
                    if crop_idx >= len(row):
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing crop for videoID={video_id}"
                        )
                    raw_crop = row[crop_idx]
                    if not raw_crop:
                        raise SystemExit(
                            f"metadata/pianoyt.csv row missing crop for videoID={video_id}"
                        )
                    stripped = raw_crop.strip().strip("()[]\"' ")
                    parts = [
                        part.strip()
                        for part in stripped.replace(";", ",").split(",")
                        if part.strip()
                    ]
                    if len(parts) != 4:
                        raise SystemExit(
                            "metadata/pianoyt.csv crop should contain four values "
                            f"for videoID={video_id}: got {raw_crop!r}"
                        )
                    crops[video_id] = parse_crop_sequence(parts, video_id)
                return crops

            missing = (
                alt_required - fields if alt_required.issubset(fields) else required - fields
            )
            raise SystemExit(f"metadata/pianoyt.csv missing columns: {sorted(missing)}")

        def parse_headerless_row(raw_row: List[str], line_no: int) -> None:
            row = [_clean_cell(cell) for cell in raw_row]
            if not row or not any(row):
                return
            if len(row) < 7:
                raise SystemExit(
                    "metadata/pianoyt.csv headerless layout requires 7 columns "
                    "(videoID,url,split,min_y,max_y,min_x,max_x) "
                    f"(line {line_no})"
                )
            video_id = row[0]
            if not video_id:
                raise SystemExit(
                    f"metadata/pianoyt.csv row missing videoID (line {line_no})"
                )
            crops[video_id] = parse_crop_sequence(row[3:7], video_id)

        parse_headerless_row(first_row, 1)
        for line_no, raw_row in enumerate(reader, start=2):
            parse_headerless_row(raw_row, line_no)
        return crops


def load_excluded_ids(root: Path) -> Set[str]:
    excl_path = root / "splits" / "excluded_low.txt"
    if not excl_path.exists():
        return set()
    with excl_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def _save_stage_frame(base: Path, stage: str, tensor):  # type: ignore[no-untyped-def]
    """Persist ``tensor``'s first frame to ``base/stage/frame0.png`` if possible."""

    try:
        from PIL import Image
    except Exception:
        print(f"[PIPE][WARN] pillow not available; skipping dump for stage '{stage}'")
        return

    stage_dir = base / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    frame_path = stage_dir / "frame0.png"

    if tensor.ndim == 4:
        frame = tensor[0]
    elif tensor.ndim == 3:
        frame = tensor
    else:
        raise ValueError(f"Unexpected tensor rank {tensor.ndim} for stage '{stage}' dump")

    frame = frame.detach().cpu().clamp(0.0, 1.0)
    if frame.shape[0] == 1:
        array = (frame.squeeze(0) * 255.0).round().byte().numpy()
        mode = "L"
    else:
        array = (frame.permute(1, 2, 0) * 255.0).round().byte().numpy()
        mode = "RGB"

    Image.fromarray(array, mode=mode).save(frame_path)

def verify_pipeline(
    *,
    root: Path,
    config_path: Path,
    splits: Dict[str, Sequence[Dict[str, object]]],
    n_samples: int,
    dump_dir: Optional[Path],
    strict: bool,
) -> List[str]:
    """Run a light-weight pipeline verification using the token-aligned pipeline."""

    if n_samples <= 0:
        return []

    from utils import load_config

    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        message = f"[PIPE][ERROR] config not found: {config_path}"
        print(message)
        return [message]
    except Exception as exc:  # pragma: no cover - defensive logging
        message = f"[PIPE][ERROR] failed to load config {config_path}: {exc}"
        print(message)
        return [message]

    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    if not dataset_cfg:
        message = "[PIPE][ERROR] dataset configuration missing from config file"
        print(message)
        return [message]

    cfg_root = dataset_cfg.get("root_dir")
    if cfg_root:
        cfg_root_path = Path(os.path.expanduser(str(cfg_root))).resolve()
        if cfg_root_path != root:
            print(
                f"[PIPE][INFO] overriding config root_dir {cfg_root_path} with CLI root {root}"
            )
    dataset_cfg["root_dir"] = str(root)

    from data.omaps_dataset import (
        _load_clip_with_random_start,
        apply_global_augment,
        apply_registration_crop,
        resize_to_canonical,
    )
    from utils.tiling import tile_vertical_token_aligned
    import torch

    tiles = int(dataset_cfg.get("tiles", 1))
    frames = int(dataset_cfg.get("frames", 32))
    decode_fps = float(dataset_cfg.get("decode_fps", 30.0))
    hop_seconds = float(dataset_cfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))
    channels = int(dataset_cfg.get("channels", 3))
    apply_crop_flag = bool(dataset_cfg.get("apply_crop", True))

    reg_cfg = dict(dataset_cfg.get("registration", {}) or {})
    registration_enabled = bool(reg_cfg.get("enabled", False))
    registration_interp = str(reg_cfg.get("interp", "bilinear"))

    canonical_cfg = dataset_cfg.get("canonical_hw", resize_hw)
    if isinstance(canonical_cfg, Sequence) and len(canonical_cfg) >= 2:
        canonical_hw = (int(round(float(canonical_cfg[0]))), int(round(float(canonical_cfg[1]))))
    else:
        canonical_hw = tuple(resize_hw)

    global_aug_cfg = dataset_cfg.get("global_aug")
    if not isinstance(global_aug_cfg, dict):
        candidate = reg_cfg.get("global_aug") if isinstance(reg_cfg, dict) else {}
        global_aug_cfg = candidate if isinstance(candidate, dict) else {}
    global_aug_enabled = bool(global_aug_cfg.get("enabled", False))

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    experiment_cfg = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    seed_val = data_cfg.get("seed", experiment_cfg.get("seed"))
    data_seed = int(seed_val) if seed_val is not None else None

    tiling_cfg = cfg.get("tiling", {}) if isinstance(cfg, dict) else {}
    patch_w = tiling_cfg.get("patch_w")
    if patch_w is None:
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        transformer_cfg = model_cfg.get("transformer", {}) if isinstance(model_cfg, dict) else {}
        patch_w = transformer_cfg.get("input_patch_size")
    if patch_w is None:
        message = "[PIPE][ERROR] tiling.patch_w or model.transformer.input_patch_size required"
        print(message)
        return [message]
    patch_w = int(patch_w)

    tokens_split = tiling_cfg.get("tokens_split", "auto")
    overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))

    dump_dir = dump_dir.resolve() if dump_dir else None

    violations: List[str] = []

    for split_name, pieces in splits.items():
        if not pieces:
            continue
        taken = 0
        for index, piece in enumerate(pieces):
            if taken >= n_samples:
                break
            video_path_str = piece.get("video") if isinstance(piece, dict) else None
            if not video_path_str:
                continue
            video_path = Path(video_path_str)
            if not video_path.exists():
                msg = f"[PIPE][WARN] split={split_name} id={piece.get('id')} missing video {video_path}"
                print(msg)
                violations.append(msg)
                continue

            try:
                clip_native, start_idx = _load_clip_with_random_start(
                    path=video_path,
                    frames=frames,
                    stride=stride,
                    channels=channels,
                    training=False,
                    decode_fps=decode_fps,
                )
            except Exception as exc:
                msg = f"[PIPE][ERROR] decode failed for split={split_name} id={piece.get('id')}: {exc}"
                print(msg)
                violations.append(msg)
                continue

            video_id = str(piece.get("id", video_path.stem))
            stage_names = ["decode"]
            stage_hws = [clip_native.shape[-2:]]
            stage_tensors = [clip_native]

            clip_stage = clip_native
            if registration_enabled and apply_crop_flag:
                try:
                    clip_stage = apply_registration_crop(
                        clip_stage, PIANOYT_CROPS.get(video_id), reg_cfg
                    )
                except Exception as exc:
                    msg = (
                        f"[PIPE][ERROR] registration crop failed split={split_name} id={video_id}: {exc}"
                    )
                    print(msg)
                    violations.append(msg)
                    continue

                stage_names.append("registration")
                stage_hws.append(clip_stage.shape[-2:])
                stage_tensors.append(clip_stage)

            clip_resized = resize_to_canonical(
                clip_stage, canonical_hw, registration_interp
            )
            canon_hw = clip_resized.shape[-2:]
            stage_names.append("resize")
            stage_hws.append(canon_hw)
            stage_tensors.append(clip_resized)

            if canonical_hw and len(canonical_hw) == 2:
                expected_hw = (int(canonical_hw[0]), int(canonical_hw[1]))
                if canon_hw != expected_hw:
                    msg = (
                        f"[PIPE][VIOLATION] split={split_name} id={video_id}: "
                        f"resize produced {canon_hw} expected {expected_hw}"
                    )
                    print(msg)
                    violations.append(msg)

            aug_clip = clip_resized
            if global_aug_enabled and split_name == "train":
                try:
                    aug_clip = apply_global_augment(
                        clip_resized,
                        global_aug_cfg,
                        base_seed=data_seed,
                        sample_index=index,
                        start_idx=start_idx,
                        interp=global_aug_cfg.get("interp", registration_interp),
                        id_key=video_id,
                    )
                except Exception as exc:
                    msg = (
                        f"[PIPE][ERROR] global_aug failed split={split_name} id={video_id}: {exc}"
                    )
                    print(msg)
                    violations.append(msg)
                    continue

                if data_seed is not None:
                    aug_again = apply_global_augment(
                        clip_resized,
                        global_aug_cfg,
                        base_seed=data_seed,
                        sample_index=index,
                        start_idx=start_idx,
                        interp=global_aug_cfg.get("interp", registration_interp),
                        id_key=video_id,
                    )
                    if not torch.allclose(aug_clip, aug_again):
                        msg = (
                            f"[PIPE][VIOLATION] split={split_name} id={video_id}: "
                            "global_aug not deterministic for fixed seed"
                        )
                        print(msg)
                        violations.append(msg)
            stage_names.append("aug")
            stage_hws.append(aug_clip.shape[-2:])
            stage_tensors.append(aug_clip)

            try:
                (
                    _tile_slices,
                    tokens_per_tile,
                    widths_px,
                    _,
                    aligned_w,
                    original_w,
                ) = tile_vertical_token_aligned(
                    aug_clip,
                    tiles,
                    patch_w=patch_w,
                    tokens_split=tokens_split,
                    overlap_tokens=overlap_tokens,
                )
            except Exception as exc:
                msg = f"[PIPE][ERROR] tiling failed split={split_name} id={video_id}: {exc}"
                print(msg)
                violations.append(msg)
                continue
            tile_clip = aug_clip if aligned_w == original_w else aug_clip[..., :aligned_w]
            stage_names.append("tile")
            stage_hws.append(tile_clip.shape[-2:])
            stage_tensors.append(tile_clip)
            tile_width_violations = [w for w in widths_px if w % patch_w != 0]
            if tile_width_violations:
                msg = (
                    f"[PIPE][VIOLATION] split={split_name} id={video_id}: "
                    f"tile widths {tile_width_violations} not multiples of patch_w={patch_w}"
                )
                print(msg)
                violations.append(msg)

            total_tokens = sum(tokens_per_tile)
            if patch_w > 0:
                expected_tokens = aligned_w // patch_w
                if total_tokens != expected_tokens:
                    msg = (
                        f"[PIPE][VIOLATION] split={split_name} id={video_id}: "
                        f"tokens sum {total_tokens} != aligned width {aligned_w}//patch_w"
                    )
                    print(msg)
                    violations.append(msg)

            hw_str = "→".join(f"{h}x{w}" for (h, w) in stage_hws)
            order_str = "→".join(stage_names)
            print(
                "[PIPE] split={split_name} id="
                f"{video_id} order={order_str} | hw: {hw_str} | "
                f"tiles(tokens)={tokens_per_tile} widths_px={widths_px} "
                f"aligned={aligned_w} orig={original_w}"
            )

            if dump_dir is not None:
                sample_dir = dump_dir / split_name / video_id
                for stage_name, tensor in zip(stage_names, stage_tensors):
                    _save_stage_frame(sample_dir, stage_name, tensor)

            taken += 1

    if strict and violations:
        raise SystemExit(1)

    return violations
    
    
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
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Configuration file used for pipeline verification",
    )
    ap.add_argument(
        "--verify-pipeline",
        action="store_true",
        help="Alias retained for compatibility; pipeline verification now runs by default",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per split for --verify-pipeline",
    )
    ap.add_argument(
        "--dump-dir",
        type=Path,
        help="If set, dump first-frame visualisations of each pipeline stage",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when --verify-pipeline detects violations",
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
    
    splits = {"train": train_pieces, "val": val_pieces or [], "test": test_pieces}
    violations = verify_pipeline(
        root=root,
        config_path=args.config,
        splits=splits,
        n_samples=args.n_samples,
        dump_dir=args.dump_dir,
        strict=args.strict,
    )
    if violations and not args.strict:
        print(f"[PIPE][WARN] {len(violations)} issues detected (strict mode disabled)")

if __name__ == "__main__":
    main()

