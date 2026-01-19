"""
PianoVAM dataset loader.

Purpose:
    - Resolve PianoVAM roots/manifests, parse labels/hand metadata, and crops.
    - Reuse shared decode/tiling/target logic from BasePianoDataset.
Key Functions/Classes:
    - PianoVAMDataset
CLI Arguments:
    - (none)
Usage:
    - ds = PianoVAMDataset(cfg, split="train", full_cfg=cfg)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from tivit.data.datasets.base import BasePianoDataset, DatasetEntry, safe_collate_fn
from tivit.data.targets.identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)


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
    """Compute (min_y, max_y, min_x, max_x) from Point_LT/RT/RB/LB metadata when available."""

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


def _load_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    """Load PianoVAM metadata_v2.json (if present) and normalise keys."""

    meta_path = root / "metadata_v2.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    table: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        canon = canonical_video_id(str(key))
        table[canon] = dict(value)
    return table


class PianoVAMDataset(BasePianoDataset):
    """PianoVAM dataset using shared decoding/target logic."""

    def _resolve_root(self, root_dir: Optional[str]) -> Path:
        """Resolve dataset root with env fallbacks."""
        if root_dir:
            return Path(root_dir).expanduser()
        env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
        if env:
            return Path(env).expanduser() / "PianoVAM_v1.0"
        return Path("~/datasets/PianoVAM_v1.0").expanduser()

    def _resolve_manifest(self) -> Optional[Mapping[str, Any]]:
        """Load manifest JSON when provided."""
        manifest_cfg = self.dataset_cfg.get("manifest", {}) or {}
        path = manifest_cfg.get(self.split)
        if not path:
            return None
        p = Path(path).expanduser()
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        return data if isinstance(data, Mapping) else None

    def _list_entries(self, root: Path, split: str, manifest: Optional[Mapping[str, Any]]) -> List[DatasetEntry]:
        """List video/label entries for PianoVAM split."""
        entries: List[DatasetEntry] = []
        meta = manifest or {}
        meta_table = _load_metadata(root)
        split_meta = meta.get(split, []) if isinstance(meta, Mapping) else []
        if isinstance(split_meta, list):
            for item in split_meta:
                if not isinstance(item, Mapping):
                    continue
                vid = canonical_video_id(item.get("video_id", ""))
                if not vid:
                    continue
                video_rel = item.get("video")
                label_rel = item.get("label")
                video_path = root / video_rel if video_rel else None
                label_path = root / label_rel if label_rel else None
                if video_path is None or not video_path.exists():
                    continue
                metadata = dict(item)
                meta_entry = meta_table.get(vid, {})
                crop = metadata.get("crop") or _crop_from_points(metadata) or _crop_from_points(meta_entry)
                if crop is not None:
                    metadata["crop"] = crop
                entries.append(
                    DatasetEntry(
                        video_path=video_path,
                        label_path=label_path if label_path and label_path.exists() else None,
                        video_id=vid,
                        metadata=metadata,
                    )
                )
        else:
            # fallback: glob videos if manifest missing
            meta_table = meta_table or _load_metadata(root)
            for video_path in sorted((root / "Video").rglob("*.mp4")):
                vid = canonical_video_id(video_path.stem)
                label_path = (root / "TSV" / f"{video_path.stem}.tsv")
                meta_entry = meta_table.get(vid, {})
                crop = _crop_from_points(meta_entry)
                metadata = dict(meta_entry) if isinstance(meta_entry, Mapping) else {}
                if crop is not None:
                    metadata["crop"] = crop
                entries.append(DatasetEntry(video_path=video_path, label_path=label_path, video_id=vid, metadata=metadata))
        return entries

    def _read_labels(self, entry: DatasetEntry) -> Mapping[str, Any]:
        """Parse events and optional hand/clef/handskeleton metadata."""
        if entry.label_path is None or not entry.label_path.exists():
            if self.require_labels:
                raise FileNotFoundError(f"Missing label for {entry.video_id}")
            return {}
        events = []
        hand_meta = {}
        try:
            suffix = entry.label_path.suffix.lower()
            if suffix in {".mid", ".midi"}:
                try:
                    import pretty_midi  # type: ignore

                    pm = pretty_midi.PrettyMIDI(str(entry.label_path))
                except Exception as exc:
                    if self.require_labels:
                        raise
                    LOGGER.warning("Failed to parse MIDI for %s (%s)", entry.video_id, exc)
                else:
                    for inst in pm.instruments:
                        for note in inst.notes:
                            events.append((float(note.start), float(note.end), int(note.pitch)))
            elif suffix == ".tsv":
                with entry.label_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.strip().split("\t")
                        if len(parts) < 3:
                            parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        try:
                            onset = float(parts[0])
                            offset = float(parts[1])
                            # PianoVAM TSVs may be 5-column (pitch at index 3).
                            pitch_raw = parts[3] if len(parts) >= 5 else parts[2]
                            pitch = int(round(float(pitch_raw)))
                        except (TypeError, ValueError):
                            continue
                        events.append((onset, offset, pitch))
            else:
                with entry.label_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
                        hand_val = int(parts[3]) if len(parts) >= 4 else None
                        clef_val = int(parts[4]) if len(parts) >= 5 else None
                        events.append((onset, offset, pitch, hand_val, clef_val))
                        if hand_val is not None:
                            hand_meta.setdefault("hand_seq", []).append(hand_val)
                        if clef_val is not None:
                            hand_meta.setdefault("clef_seq", []).append(clef_val)
            # hand skeleton path if available
            if isinstance(entry.metadata, Mapping):
                hs = entry.metadata.get("hand_skeleton")
                if isinstance(hs, str):
                    hs_path = Path(hs).expanduser()
                    if hs_path.exists():
                        hand_meta["hand_skeleton_path"] = str(hs_path)
                if "hand" in entry.metadata:
                    hand_meta["hand_hint"] = entry.metadata.get("hand")
                if "clef" in entry.metadata:
                    hand_meta["clef_hint"] = entry.metadata.get("clef")
        except Exception as exc:
            if self.require_labels:
                raise
            LOGGER.warning("Failed to read labels for %s (%s)", entry.video_id, exc)
        payload: Mapping[str, Any] = {"events": events, "metadata": dict(entry.metadata)}
        if hand_meta:
            payload = {**payload, **hand_meta}
        return payload


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False, *, seed: Optional[int] = None):
    dcfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    batch_size = int(dcfg.get("batch_size", 1))
    shuffle = bool(dcfg.get("shuffle", True)) if split == "train" else False
    num_workers = int(dcfg.get("num_workers", 0))
    prefetch_factor = dcfg.get("prefetch_factor")

    ds = PianoVAMDataset(cfg, split, full_cfg=cfg)
    return torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        collate_fn=safe_collate_fn,
    )
