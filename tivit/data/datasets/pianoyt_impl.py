"""
PianoYT dataset loader.

Purpose:
    - Resolve PianoYT roots/manifests and parse labels plus crop metadata.
    - Reuse shared decode/tiling/target logic from BasePianoDataset.
Key Functions/Classes:
    - PianoYTDataset
CLI Arguments:
    - (none)
Usage:
    - ds = PianoYTDataset(cfg, split="train", full_cfg=cfg)
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import csv
import torch

from tivit.data.datasets.base import BasePianoDataset, DatasetEntry
from tivit.data.targets.identifiers import canonical_video_id, id_aliases, log_legacy_id_hit

LOGGER = logging.getLogger(__name__)
_CROP_FILE = "metadata/pianoyt.csv"


def _load_crop_table(root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """Load per-video crop coords (min_y, max_y, min_x, max_x) from pianoyt.csv."""

    meta_path = root / _CROP_FILE
    table: Dict[str, Tuple[int, int, int, int]] = {}
    if not meta_path.exists():
        return table
    with meta_path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        try:
            first_row = next(reader)
        except StopIteration:
            return table
        rows = [first_row] + list(reader)

    def _maybe_int(val: str) -> Optional[int]:
        try:
            return int(float(val))
        except Exception:
            return None

    # Detect header
    header = [cell.strip().strip("'\"") for cell in first_row]
    header_norm = [h.lower() for h in header]
    has_header = any(name in {"videoid", "min_y", "max_y", "min_x", "max_x", "crop"} for name in header_norm)
    start_idx = 1 if has_header else 0

    def _parse_row(row: List[str]) -> None:
        if not row:
            return
        cells = [cell.strip().strip("'\"") for cell in row]
        if has_header:
            try:
                col_map = {name: idx for idx, name in enumerate(header_norm) if name}
                vid = cells[col_map.get("videoid", 0)]
            except Exception:
                return
            if not vid:
                return
            if all(k in col_map for k in ("min_y", "max_y", "min_x", "max_x")):
                vals = [
                    _maybe_int(cells[col_map["min_y"]]),
                    _maybe_int(cells[col_map["max_y"]]),
                    _maybe_int(cells[col_map["min_x"]]),
                    _maybe_int(cells[col_map["max_x"]]),
                ]
            elif "crop" in col_map:
                raw = cells[col_map["crop"]]
                parts = [p for p in raw.replace(";", ",").replace("(", "").replace(")", "").split(",") if p.strip()]
                vals = [_maybe_int(p) for p in parts]
            else:
                return
        else:
            if len(cells) < 7:
                return
            vid = cells[0]
            vals = [_maybe_int(cells[3]), _maybe_int(cells[4]), _maybe_int(cells[5]), _maybe_int(cells[6])]
        if not vid or any(v is None for v in vals):
            return
        int_vals = [int(v) for v in vals if v is not None]
        if len(int_vals) != 4:
            return
        y0, y1, x0, x1 = int_vals
        canon = canonical_video_id(vid)
        table[canon] = (y0, y1, x0, x1)

    for row in rows[start_idx:]:
        _parse_row(row)
    return table


def _safe_expanduser(path: Path) -> Path:
    try:
        return path.expanduser()
    except Exception:
        return path


def _load_manifest(manifest_path: Optional[str]) -> Mapping[str, Any]:
    if not manifest_path:
        return {}
    path = Path(manifest_path).expanduser()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, Mapping):
        return {}
    # Flatten {pieces: [{id, midi, video, ...}]} into id->metadata for easy lookup.
    if "pieces" in data and isinstance(data.get("pieces"), list):
        pieces = data.get("pieces", []) or []
        mapped = {}
        for item in pieces:
            if not isinstance(item, Mapping):
                continue
            vid_raw = item.get("id") or item.get("video") or item.get("name")
            vid = canonical_video_id(vid_raw) if vid_raw else None
            if vid:
                mapped[vid] = dict(item)
        return mapped
    return data


def _read_split_ids(root: Path, split: str) -> List[str]:
    split_file = root / "splits" / f"{split}.txt"
    if split_file.exists():
        ids: List[str] = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    ids.append(canonical_video_id(line))
        # Preserve ordering but drop duplicates introduced by canonicalisation.
        seen: set[str] = set()
        ordered: List[str] = []
        for vid in ids:
            if vid not in seen:
                ordered.append(vid)
                seen.add(vid)
        return ordered
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split list missing: {split_file} and directory missing: {split_dir}")
    ids_set: set[str] = set()
    for video_path in split_dir.glob("video_*"):
        ids_set.add(canonical_video_id(video_path.stem))
    for midi_path in split_dir.glob("audio_*"):
        ids_set.add(canonical_video_id(midi_path.stem))
    return sorted(ids_set)


def _read_excluded(root: Path, path: Optional[str]) -> set:
    file_path = _safe_expanduser(Path(path)) if path else root / "splits" / "excluded_low.txt"
    if not file_path.exists():
        return set()
    ids = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(canonical_video_id(line))
    return ids


def _resolve_media_paths(root: Path, split: str, video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    canon_id = canonical_video_id(video_id)
    aliases = id_aliases(canon_id)
    search_dirs: List[Path] = []
    split_dir = root / split
    if split_dir.exists():
        search_dirs.append(split_dir)
    if split == "val" and not split_dir.exists():
        train_dir = root / "train"
        if train_dir.exists():
            search_dirs.append(train_dir)
    if not search_dirs:
        search_dirs.append(split_dir)

    video_path: Optional[Path] = None
    video_alias: Optional[str] = None
    midi_path: Optional[Path] = None
    midi_alias: Optional[str] = None

    for base in search_dirs:
        if video_path is None:
            for alias in aliases:
                for ext in (".mp4", ".mkv", ".webm"):
                    cand = base / f"{alias}{ext}"
                    if cand.exists():
                        video_path = cand
                        video_alias = alias
                        break
                if video_path is not None:
                    break
        if midi_path is None:
            for alias in aliases:
                audio_name = f"audio_{alias[6:]}" if alias.startswith("video_") else alias
                for ext in (".midi", ".mid"):
                    cand = base / f"{audio_name}{ext}"
                    if cand.exists():
                        midi_path = cand
                        midi_alias = alias
                        break
                if midi_path is not None:
                    break
        if video_path is not None and midi_path is not None:
            break

    if video_alias and video_alias != canon_id:
        log_legacy_id_hit(video_alias, canon_id, logger=LOGGER)
    if midi_alias and midi_alias != canon_id:
        log_legacy_id_hit(midi_alias, canon_id, logger=LOGGER)

    return video_path, midi_path


def _read_midi_events(midi_path: Path) -> List[Tuple[float, float, float]]:
    """Return (N,3) events [onset, offset, pitch] parsed from MIDI."""

    if not midi_path or not midi_path.exists():
        return []

    events: List[Tuple[float, float, float]] = []

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API", category=UserWarning)
            import pretty_midi  # type: ignore
    except Exception:
        pretty_midi = None

    if pretty_midi is not None:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            pm = None
        if pm is not None:
            for inst in pm.instruments:
                for note in inst.notes:
                    events.append((float(note.start), float(note.end), float(note.pitch)))

    if not events:
        try:
            import mido
            from mido import MidiFile, merge_tracks
            from mido.midifiles.midifiles import tick2second
        except Exception:
            return events
        mid = MidiFile(str(midi_path))
        tempo = 500000  # default microseconds per beat
        ticks_per_beat = mid.ticks_per_beat or 480
        current_sec = 0.0
        active: dict[int, List[float]] = {}
        for msg in merge_tracks(mid.tracks):
            delta = msg.time
            current_sec += tick2second(delta, ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append(current_sec)
            elif msg.type in {"note_off", "note_on"}:
                if msg.type == "note_on" and msg.velocity > 0:
                    continue
                pitches = active.get(msg.note)
                if pitches:
                    start = pitches.pop()
                    events.append((float(start), float(current_sec), float(msg.note)))

    events.sort(key=lambda x: (x[0], x[1], x[2]))
    return events


class PianoYTDataset(BasePianoDataset):
    """PianoYT dataset using BasePianoDataset heavy logic."""

    def _resolve_root(self, root_dir: Optional[str]) -> Path:
        """Resolve dataset root (default: data/PianoYT)."""
        if root_dir:
            return _safe_expanduser(Path(root_dir))
        env_dir = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
        if env_dir:
            cand = _safe_expanduser(Path(env_dir)) / "PianoYT"
            if cand.exists():
                return cand
        repo_default = Path(__file__).resolve().parents[2] / "data" / "PianoYT"
        if repo_default.exists():
            return repo_default
        return _safe_expanduser(Path("~/datasets/PianoYT"))

    def _resolve_manifest(self) -> Optional[Mapping[str, Any]]:
        """Load JSON manifest when provided."""
        manifest_cfg = self.dataset_cfg.get("manifest", {}) or {}
        raw = manifest_cfg.get(self.split)
        return _load_manifest(raw)

    def _list_entries(self, root: Path, split: str, manifest: Optional[Mapping[str, Any]]) -> List[DatasetEntry]:
        """List video/label entries for PianoYT split."""
        entries: List[DatasetEntry] = []
        raw_meta = dict(manifest or {})
        meta_map = {canonical_video_id(k): v for k, v in raw_meta.items()}
        crop_table = _load_crop_table(root)
        excluded = _read_excluded(root, self.dataset_cfg.get("excluded_list"))
        try:
            ids = list(meta_map.keys()) if meta_map else _read_split_ids(root, split)
        except FileNotFoundError as exc:
            LOGGER.warning("Split list missing for %s: %s", split, exc)
            ids = []
        if not ids:
            LOGGER.warning("No ids found for split %s under %s", split, root)
            return entries

        for vid in ids:
            if vid in excluded:
                continue
            meta = meta_map.get(vid, {}) if isinstance(meta_map, Mapping) else {}
            video_path_raw = meta.get("video")
            label_path_raw = meta.get("label") or meta.get("midi")
            video_path = _safe_expanduser(Path(video_path_raw)) if video_path_raw else None
            label_path = _safe_expanduser(Path(label_path_raw)) if label_path_raw else None
            if video_path is None or not video_path.exists():
                video_path, label_path = _resolve_media_paths(root, split, vid)
            if video_path is None or not video_path.exists():
                LOGGER.debug("Skipping %s in split %s (video not found)", vid, split)
                continue
            if label_path is not None and not label_path.exists():
                label_path = None
            metadata = dict(meta) if isinstance(meta, Mapping) else {}
            if vid in crop_table and "crop" not in metadata:
                metadata["crop"] = crop_table[vid]
            entries.append(
                DatasetEntry(
                    video_path=video_path,
                    label_path=label_path,
                    video_id=canonical_video_id(vid),
                    metadata=metadata,
                )
            )
        return entries

    def _read_labels(self, entry: DatasetEntry) -> Mapping[str, Any]:
        """Parse events plus optional hand/clef columns from labels."""
        if entry.label_path is None:
            if self.require_labels:
                raise FileNotFoundError(f"Missing label for {entry.video_path}")
            return {}
        if entry.label_path.suffix.lower() in {".mid", ".midi"}:
            events = _read_midi_events(entry.label_path)
        else:
            events: List[Tuple[float, float, float]] = []
            hands: list[int] = []
            clefs: list[int] = []
            try:
                with entry.label_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
                        hand_val = int(parts[3]) if len(parts) >= 4 else None
                        clef_val = int(parts[4]) if len(parts) >= 5 else None
                        events.append((onset, offset, pitch))
                        if hand_val is not None:
                            hands.append(hand_val)
                        if clef_val is not None:
                            clefs.append(clef_val)
            except Exception as exc:
                if self.require_labels:
                    raise
            payload: Dict[str, Any] = {"events": events}
            if hands:
                payload["hand_seq"] = hands
            if clefs:
                payload["clef_seq"] = clefs
            return payload
        if not events and self.require_labels:
            raise FileNotFoundError(f"Missing label for {entry.video_path}")
        return {"events": events}


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False, *, seed: Optional[int] = None):
    dataset = PianoYTDataset(cfg, split, full_cfg=cfg)
    return torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        dataset,
        batch_size=int(cfg.get("dataset", {}).get("batch_size", 1)),
        shuffle=bool(cfg.get("dataset", {}).get("shuffle", True)) if split == "train" else False,
        num_workers=int(cfg.get("dataset", {}).get("num_workers", 0)),
        drop_last=drop_last,
        pin_memory=True,
        prefetch_factor=cfg.get("dataset", {}).get("prefetch_factor", None),
    )
