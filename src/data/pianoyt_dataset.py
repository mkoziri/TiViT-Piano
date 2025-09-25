"""Purpose:
    Implement the PianoYT dataset loader with the same runtime surface as the
    existing :mod:`omaps_dataset` module.  The loader mirrors the clip sampling
    behaviour, target construction, and collate contract used by TiViT-Piano so
    that training scripts can switch datasets only through configuration.

Key Functions/Classes:
    - PianoYTDataset: PyTorch ``Dataset`` yielding tiled clips and optional
      MIDI-derived labels/targets.
    - make_dataloader(): Factory matching :func:`omaps_dataset.make_dataloader`
      which wires runtime configuration, per-clip targets, and collate logic.
    - _read_midi_events(): Utility that converts a MIDI file into the ``(N,3)``
      event tensor used by downstream code (onset, offset, pitch).

CLI:
    Not a standalone CLI.  Use the loader via :mod:`scripts.train` or diagnostic
    scripts that already depend on :func:`data.make_dataloader`.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from utils.time_grid import frame_to_sec, sec_to_frame
from utils.tiling import tile_vertical_token_aligned

from .omaps_dataset import (  # reuse established helpers for identical behaviour
    _build_frame_targets,
    _load_clip_with_random_start,
    _read_manifest,
    _tile_vertical,
    apply_global_augment,
    apply_registration_crop,
    resize_to_canonical,
)

LOGGER = logging.getLogger(__name__)


def _safe_expanduser(path: Union[str, Path]) -> Path:
    """Expand ``~`` safely even when ``Path.expanduser`` cannot."""

    candidate = Path(path)
    try:
        return candidate.expanduser()
    except RuntimeError:
        LOGGER.debug("Failed to expand user in path '%s'; returning as-is.", candidate)
        return candidate

_VIDEO_EXTS: Sequence[str] = (".mp4", ".mkv", ".webm")


def _expand_root(root_dir: Optional[str]) -> Path:
    """Resolve the PianoYT root directory with environment fallbacks."""

    if root_dir:
        expanded = _safe_expanduser(os.path.expandvars(str(root_dir)))
        candidates = [expanded]
        if expanded.name.lower() != "pianoyt":
            candidates.append(expanded / "PianoYT")
        for cand in candidates:
            if cand.exists():
                return cand
    env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env:
        cand = _safe_expanduser(env) / "PianoYT"
        if cand.exists():
            return cand
    project_root = Path(__file__).resolve().parents[2]
    repo_data = project_root / "data" / "PianoYT"
    if repo_data.exists():
        return repo_data
    return _safe_expanduser("~/datasets/PianoYT")


def _read_split_ids(root: Path, split: str) -> List[str]:
    split_file = root / "splits" / f"{split}.txt"
    if split_file.exists():
        ids: List[str] = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ids.append(line)
        return ids

    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split list missing: {split_file} and directory missing: {split_dir}"
        )

    def _extract_id(name: str, prefix: str) -> Optional[str]:
        if not name.startswith(prefix):
            return None
        remainder = name[len(prefix) :]
        if ".0" not in remainder:
            return None
        return remainder.split(".0", 1)[0]

    ids_set = set()
    for video_path in split_dir.glob("video_*.0*"):
        vid = _extract_id(video_path.name, "video_")
        if vid:
            ids_set.add(vid)
    for midi_path in split_dir.glob("audio_*.0.midi"):
        vid = _extract_id(midi_path.name, "audio_")
        if vid:
            ids_set.add(vid)

    if not ids_set:
        raise FileNotFoundError(
            f"Split list missing: {split_file} and no media found in {split_dir}"
        )

    LOGGER.info(
        "[PianoYT] Using inferred ID list for split '%s' from directory %s", split, split_dir
    )
    return sorted(ids_set)


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


def _load_metadata(root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    meta_path = root / "metadata" / "pianoyt.csv"
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

        if first_row is None:
            return table

        first_row = [_clean_cell(cell) for cell in first_row]
        normalised = [_normalise_header(cell) for cell in first_row]
        header_fields = {name for name in normalised if name}
        known_header = {"videoID", "min_y", "max_y", "min_x", "max_x", "crop"}
        has_header = bool(header_fields & known_header)

        if has_header:
            column_map: Dict[str, int] = {}
            for idx, name in enumerate(normalised):
                if name and name not in column_map:
                    column_map[name] = idx

            video_idx = column_map.get("videoID")
            if video_idx is None:
                return table

            crop_keys = ("min_y", "max_y", "min_x", "max_x")
            has_explicit_crop = all(key in column_map for key in crop_keys)
            has_compound_crop = "crop" in column_map

            def parse_with_header(raw_row: List[str]) -> None:
                row = [_clean_cell(cell) for cell in raw_row]
                if video_idx >= len(row):
                    return
                vid = row[video_idx]
                if not vid:
                    return
                try:
                    if has_explicit_crop:
                        values = []
                        for key in crop_keys:
                            idx = column_map[key]
                            if idx >= len(row):
                                raise ValueError
                            values.append(int(float(row[idx])))
                    elif has_compound_crop:
                        idx = column_map["crop"]
                        if idx >= len(row):
                            return
                        raw_crop = row[idx].strip().strip("()[]\"'")
                        if not raw_crop:
                            return
                        parts = [part.strip() for part in raw_crop.replace(";", ",").split(",") if part.strip()]
                        if len(parts) != 4:
                            return
                        values = [int(float(part)) for part in parts]
                    else:
                        return
                except (TypeError, ValueError):
                    return
                table[vid] = (values[0], values[1], values[2], values[3])

            for raw_row in reader:
                parse_with_header(raw_row)
        else:
            def parse_headerless_row(raw_row: List[str]) -> None:
                row = [_clean_cell(cell) for cell in raw_row]
                if len(row) < 7:
                    return
                vid = row[0]
                if not vid:
                    return
                try:
                    min_y = int(float(row[3]))
                    max_y = int(float(row[4]))
                    min_x = int(float(row[5]))
                    max_x = int(float(row[6]))
                except (TypeError, ValueError):
                    return
                table[vid] = (min_y, max_y, min_x, max_x)

            parse_headerless_row(first_row)
            for raw_row in reader:
                parse_headerless_row(raw_row)
    return table


def _resolve_media_paths(root: Path, split: str, video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the video and MIDI paths for ``video_id`` within ``split``."""

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

    for base in search_dirs:
        midi = base / f"audio_{video_id}.0.midi"
        video = None
        for ext in _VIDEO_EXTS:
            cand = base / f"video_{video_id}.0{ext}"
            if cand.exists():
                video = cand
                break
        if video is not None or midi.exists():
            return video, midi if midi.exists() else None
    return None, None


def _read_excluded(root: Path, path: Optional[str]) -> set:
    file_path = _safe_expanduser(path) if path else root / "splits" / "excluded_low.txt"
    if not file_path.exists():
        return set()
    ids = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def _read_midi_events(midi_path: Path) -> torch.FloatTensor:
    """Return ``(N,3)`` tensor [onset_sec, offset_sec, pitch] parsed from MIDI."""

    if not midi_path or not midi_path.exists():
        return torch.zeros((0, 3), dtype=torch.float32)

    events: List[Tuple[float, float, float]] = []

    try:
        import pretty_midi  # type: ignore
    except Exception:  # pragma: no cover - handled by fallback
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
        except Exception as exc:  # pragma: no cover - import guard
            if pretty_midi is None:
                raise RuntimeError(
                    "Neither pretty_midi nor mido is available for MIDI parsing."
                ) from exc
        else:
            mid = MidiFile(str(midi_path))
            tempo = 500000  # default microseconds per beat
            ticks_per_beat = mid.ticks_per_beat or 480
            current_sec = 0.0
            active: Dict[int, List[float]] = {}
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

    if not events:
        return torch.zeros((0, 3), dtype=torch.float32)

    events.sort(key=lambda x: (x[0], x[1], x[2]))
    return torch.tensor(events, dtype=torch.float32)


class PianoYTDataset(Dataset):
    """Dataset that mirrors :class:`OMAPSDataset` but reads PianoYT assets."""

    def __init__(
        self,
        root_dir: Optional[str],
        split: str = "test",
        frames: int = 32,
        stride: int = 2,
        resize: Tuple[int, int] = (224, 224),
        tiles: int = 3,
        channels: int = 3,
        normalize: bool = True,
        manifest: Optional[str] = None,
        decode_fps: float = 30.0,
        *,
        pipeline_v2: bool = False,
        dataset_cfg: Optional[Dict[str, Any]] = None,
        full_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.root = _expand_root(root_dir)
        self.split = split
        self.frames = int(frames)
        self.stride = int(stride)
        self.resize = tuple(resize)
        self.tiles = int(tiles)
        self.channels = int(channels)
        self.normalize = bool(normalize)
        self.decode_fps = float(decode_fps)
        self.pipeline_v2 = bool(pipeline_v2)
        self.dataset_cfg = dict(dataset_cfg or {})
        self.full_cfg = dict(full_cfg or {})

        reg_cfg = dict(self.dataset_cfg.get("registration", {}) or {})
        self.registration_cfg = reg_cfg
        self.registration_enabled = bool(reg_cfg.get("enabled", False))
        self.registration_interp = str(reg_cfg.get("interp", "bilinear"))

        canonical_cfg = self.dataset_cfg.get("canonical_hw", self.resize)
        if isinstance(canonical_cfg, Sequence) and len(canonical_cfg) >= 2:
            self.canonical_hw = (
                int(round(float(canonical_cfg[0]))),
                int(round(float(canonical_cfg[1]))),
            )
        else:
            self.canonical_hw = tuple(self.resize)

        global_aug_cfg = self.dataset_cfg.get("global_aug")
        if not isinstance(global_aug_cfg, dict):
            aug_candidate = reg_cfg.get("global_aug")
            global_aug_cfg = aug_candidate if isinstance(aug_candidate, dict) else {}
        self.global_aug_cfg = dict(global_aug_cfg or {})
        self.global_aug_enabled = bool(self.global_aug_cfg.get("enabled", False))

        tiling_cfg = {}
        if isinstance(self.full_cfg, dict):
            tiling_cfg = dict(self.full_cfg.get("tiling", {}) or {})
        self.tiling_cfg = tiling_cfg
        patch_w_cfg = tiling_cfg.get("patch_w")
        if patch_w_cfg is None:
            model_cfg = self.full_cfg.get("model", {}) if isinstance(self.full_cfg, dict) else {}
            trans_cfg = model_cfg.get("transformer", {}) if isinstance(model_cfg, dict) else {}
            patch_w_cfg = trans_cfg.get("input_patch_size")
        self.tiling_patch_w = int(patch_w_cfg) if patch_w_cfg is not None else None
        tokens_split_cfg = tiling_cfg.get("tokens_split", "auto")
        if isinstance(tokens_split_cfg, Sequence) and not isinstance(tokens_split_cfg, str):
            self.tiling_tokens_split = [int(v) for v in tokens_split_cfg]
        else:
            self.tiling_tokens_split = tokens_split_cfg
        self.tiling_overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))
        self._tiling_log_once = False

        if self.pipeline_v2 and self.tiling_patch_w is None:
            raise ValueError(
                "pipeline_v2 requires 'tiling.patch_w' or model transformer patch size"
            )

        data_cfg = self.full_cfg.get("data", {}) if isinstance(self.full_cfg, dict) else {}
        experiment_cfg = self.full_cfg.get("experiment", {}) if isinstance(self.full_cfg, dict) else {}
        seed_val = data_cfg.get("seed", experiment_cfg.get("seed"))
        self.data_seed = int(seed_val) if seed_val is not None else None

        ids = _read_split_ids(self.root, split)
        if manifest:
            allow = _read_manifest(manifest)
            filtered = [vid for vid in ids if vid in allow]
            if filtered:
                ids = filtered
            else:
                LOGGER.warning(
                    "[PianoYT] Manifest %s removed all ids from split %s; ignoring manifest.",
                    manifest,
                    split,
                )

        self.metadata = _load_metadata(self.root)
        self.samples: List[Dict[str, Optional[Path]]] = []
        for vid in ids:
            video_path, midi_path = _resolve_media_paths(self.root, split, vid)
            if video_path is None:
                continue
            self.samples.append({"id": vid, "video": video_path, "midi": midi_path})

        # --- Robustness: drop unreadable/corrupt videos up-front ---
        try:
            import decord  # type: ignore
        except Exception:  # decord missing/unavailable: skip probing
            decord = None
        if decord is not None:
            kept = []
            for s in self.samples:
                v = s.get("video")
                try:
                    decord.VideoReader(str(v))
                except Exception as e:
                    LOGGER.warning("[PianoYT] Skipping unreadable video: %s (%s)", v, e)
                    continue
                kept.append(s)
            self.samples = kept
        
        # Now compute videos based on the filtered sample list
        self.videos = [s["video"] for s in self.samples if s.get("video") is not None]
        self._crop_warned: set = set()
        self.apply_crop = True
        self.crop_rescale = "auto"
        self.include_low_res = True
        self.excluded_ids: set = set()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No PianoYT media found under {self.root} for split '{split}'."
            )

    def configure(self, *, include_low_res: bool, excluded_ids: set, apply_crop: bool, crop_rescale: str):
        self.include_low_res = include_low_res
        self.apply_crop = apply_crop
        self.crop_rescale = crop_rescale.lower()
        self.excluded_ids = excluded_ids
        if not include_low_res and excluded_ids:
            self.samples = [s for s in self.samples if s["id"] not in excluded_ids]
            self.videos = [s["video"] for s in self.samples if s.get("video") is not None]
        if len(self.samples) == 0:
            raise RuntimeError("All PianoYT samples were filtered out by configuration.")

    def limit_max_clips(self, max_clips: Optional[int]):
        if max_clips is None:
            return
        if max_clips < len(self.samples):
            self.samples = self.samples[:max_clips]
            self.videos = [s["video"] for s in self.samples if s.get("video") is not None]

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_crop(self, clip: torch.Tensor, video_id: str) -> torch.Tensor:
        if not self.apply_crop:
            return clip
        crop = self.metadata.get(video_id)
        if not crop:
            return clip
        min_y, max_y, min_x, max_x = crop
        T, C, H, W = clip.shape
        if (H != 1080 or W != 1920) and self.crop_rescale != "auto":
            if video_id not in self._crop_warned:
                LOGGER.warning(
                    "[PianoYT] Skipping crop for %s (%dx%d vs 1080p baseline).",
                    video_id,
                    H,
                    W,
                )
                self._crop_warned.add(video_id)
            return clip
        if H != 1080 or W != 1920:
            scale_y = H / 1080.0
            scale_x = W / 1920.0
            min_y = int(round(min_y * scale_y))
            max_y = int(round(max_y * scale_y))
            min_x = int(round(min_x * scale_x))
            max_x = int(round(max_x * scale_x))
        min_y = max(0, min(min_y, H - 1))
        max_y = max(min_y + 1, min(max_y, H))
        min_x = max(0, min(min_x, W - 1))
        max_x = max(min_x + 1, min(max_x, W))
        return clip[..., min_y:max_y, min_x:max_x]

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        video_path = record.get("video")
        midi_path = record.get("midi")
        video_id = record.get("id", "")

        if video_path is None:
            raise FileNotFoundError(f"Missing video for PianoYT id={video_id}")

        is_train = self.split == "train"
        clip, start_idx = _load_clip_with_random_start(
            path=video_path,
            frames=self.frames,
            stride=self.stride,
            resize_hw=None if self.pipeline_v2 else self.resize,
            channels=self.channels,
            training=is_train,
            decode_fps=self.decode_fps,
            pipeline_v2=self.pipeline_v2,
        )
        
        if self.pipeline_v2:
            native_h, native_w = clip.shape[-2:]
            print(
                f"[pipeline_v2] decode {video_path.name}: native {native_h}x{native_w}",
                flush=True,
            )
            if self.registration_enabled:
                before_h, before_w = clip.shape[-2:]
                meta = self.metadata.get(video_id)
                clip = apply_registration_crop(clip, meta, self.registration_cfg)
                after_h, after_w = clip.shape[-2:]
                print(
                    f"[pipeline_v2] crop {video_path.name}: {before_h}x{before_w} -> {after_h}x{after_w}",
                    flush=True,
                )
            before_h, before_w = clip.shape[-2:]
            clip = resize_to_canonical(clip, self.canonical_hw, self.registration_interp)
            after_h, after_w = clip.shape[-2:]
            target_h, target_w = self.canonical_hw
            print(
                f"[pipeline_v2] canonical {video_path.name}: {before_h}x{before_w} -> {after_h}x{after_w} (target={target_h}x{target_w})",
                flush=True,
            )
            if self.global_aug_enabled and self.split == "train":
                before_h, before_w = clip.shape[-2:]
                clip = apply_global_augment(
                    clip,
                    self.global_aug_cfg,
                    base_seed=self.data_seed,
                    sample_index=idx,
                    start_idx=start_idx,
                    interp=self.global_aug_cfg.get("interp", self.registration_interp),
                    id_key=video_id,
                )
                after_h, after_w = clip.shape[-2:]
                print(
                    f"[pipeline_v2] global_aug {video_path.name}: {before_h}x{before_w} -> {after_h}x{after_w}",
                    flush=True,
                )
        else:
            clip = self._apply_crop(clip, video_id)
        clip = self._apply_crop(clip, video_id)
        if self.pipeline_v2:
            _, tokens_per_tile, widths_px, _, aligned_w, original_w = tile_vertical_token_aligned(
                clip,
                self.tiles,
                patch_w=self.tiling_patch_w,
                tokens_split=self.tiling_tokens_split,
                overlap_tokens=self.tiling_overlap_tokens,
            )
            if aligned_w != original_w:
                clip = clip[..., :aligned_w]
            if not self._tiling_log_once:
                width_sum = sum(widths_px)
                print(
                    f"[pipeline_v2] tiles(tokens)={tokens_per_tile} widths_px={widths_px} "
                    f"sum={width_sum} orig_W={original_w} overlap_tokens={self.tiling_overlap_tokens}",
                    flush=True,
                )
                self._tiling_log_once = True
        else:
            clip = _tile_vertical(clip, self.tiles)

        T = self.frames
        fps = self.decode_fps
        t0 = frame_to_sec(start_idx, 1.0 / fps)
        t1 = frame_to_sec(start_idx + ((T - 1) * self.stride + 1), 1.0 / fps)

        sample = {"video": clip, "path": str(video_path)}

        labels_tensor = None
        if midi_path is not None and midi_path.exists():
            labels_tensor = _read_midi_events(midi_path)
            sample["labels"] = labels_tensor
        elif getattr(self, "require_labels", False):
            raise FileNotFoundError(f"Missing MIDI annotations for {video_path}")

        clip_targets = None
        if labels_tensor is not None and labels_tensor.numel() > 0:
            onset = labels_tensor[:, 0]
            offset = labels_tensor[:, 1]
            pitch = labels_tensor[:, 2].to(torch.int64)

            mask = (onset < t1) & (offset > t0)
            sel_pitches = pitch[mask]
            sel_onsets = onset[mask]
            sel_offsets = offset[mask]

            P = 88
            note_min_clip = 21
            pitch_vec = torch.zeros(P, dtype=torch.float32)
            onset_vec = torch.zeros(P, dtype=torch.float32)
            offset_vec = torch.zeros(P, dtype=torch.float32)
            if sel_pitches.numel() == 0:
                pitch_class = 60
                onset_flag = 0.0
                offset_flag = 0.0
            else:
                uniq, counts = sel_pitches.unique(return_counts=True)
                pitch_class = int(uniq[counts.argmax()].item())
                onset_flag = 1.0 if ((sel_onsets >= t0) & (sel_onsets < t1)).any().item() else 0.0
                offset_flag = 1.0 if ((sel_offsets > t0) & (sel_offsets <= t1)).any().item() else 0.0

            idx_pitch = int(pitch_class - note_min_clip)
            if 0 <= idx_pitch < P:
                pitch_vec[idx_pitch] = 1.0
                if onset_flag:
                    onset_vec[idx_pitch] = 1.0
                if offset_flag:
                    offset_vec[idx_pitch] = 1.0

            hand = 0 if pitch_class < 60 else 1
            clef = 0 if pitch_class < 60 else (1 if pitch_class > 64 else 2)

            clip_targets = {
                "pitch": pitch_vec,
                "onset": onset_vec,
                "offset": offset_vec,
                "hand": torch.tensor(hand, dtype=torch.long),
                "clef": torch.tensor(clef, dtype=torch.long),
            }

        if clip_targets is not None:
            sample.update(clip_targets)
        elif getattr(self, "require_labels", False):
            raise FileNotFoundError(f"No usable labels for {video_path}")

        ft_cfg = getattr(self, "frame_targets_cfg", None)
        if ft_cfg and bool(ft_cfg.get("enable", False)):
            labels_ft = None
            if labels_tensor is not None and labels_tensor.numel() > 0:
                labels_ft = labels_tensor.clone()
                labels_ft[:, 0:2] -= t0
            ft = _build_frame_targets(
                labels=labels_ft,
                T=T,
                stride=self.stride,
                fps=fps,
                note_min=int(ft_cfg.get("note_min", 21)),
                note_max=int(ft_cfg.get("note_max", 108)),
                tol=float(ft_cfg.get("tolerance", 0.025)),
                fill_mode=str(ft_cfg.get("fill_mode", "overlap")),
                hand_from_pitch=bool(ft_cfg.get("hand_from_pitch", True)),
                clef_thresholds=tuple(ft_cfg.get("clef_thresholds", [60, 64])),
                dilate_active_frames=int(ft_cfg.get("dilate_active_frames", 0)),
                targets_sparse=bool(ft_cfg.get("targets_sparse", False)),
            )
            sample.update(
                {
                    "pitch_roll": ft["pitch_roll"],
                    "onset_roll": ft["onset_roll"],
                    "offset_roll": ft["offset_roll"],
                    "hand_frame": ft["hand_frame"],
                    "clef_frame": ft["clef_frame"],
                }
            )

        return sample


def make_dataloader(cfg: dict, split: str, drop_last: bool = False):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))
    
    pipeline_v2 = bool(dcfg.get("pipeline_v2", False))

    dataset = PianoYTDataset(
        root_dir=dcfg.get("root_dir"),
        split=split,
        frames=int(dcfg.get("frames", 32)),
        stride=stride,
        resize=tuple(dcfg.get("resize", [224, 224])),
        tiles=int(dcfg.get("tiles", 3)),
        channels=int(dcfg.get("channels", 3)),
        normalize=bool(dcfg.get("normalize", True)),
        manifest=manifest_path,
        decode_fps=decode_fps,
        pipeline_v2=pipeline_v2,
        dataset_cfg=dcfg,
        full_cfg=cfg,
    )

    include_low = bool(dcfg.get("include_low_res", False))
    excluded_ids = set()
    if not include_low:
        excluded_ids = _read_excluded(dataset.root, dcfg.get("excluded_list"))
    dataset.configure(
        include_low_res=include_low,
        excluded_ids=excluded_ids,
        apply_crop=bool(dcfg.get("apply_crop", True)),
        crop_rescale=str(dcfg.get("crop_rescale", "auto")),
    )

    max_clips = dcfg.get("max_clips")
    dataset.limit_max_clips(max_clips if isinstance(max_clips, int) else None)
    dataset.max_clips = max_clips

    dataset.annotations_root = dcfg.get("annotations_root")
    dataset.label_format = dcfg.get("label_format", "midi")
    dataset.label_targets = dcfg.get("label_targets", ["pitch", "onset", "offset", "hand", "clef"])
    dataset.require_labels = bool(dcfg.get("require_labels", False))
    dataset.frame_targets_cfg = dcfg.get("frame_targets", {})

    def _collate(batch):
        vids = [b["video"] for b in batch]
        paths = [b["path"] for b in batch]
        if not vids:
            raise RuntimeError("Empty batch supplied to PianoYT collate function")

        dims = vids[0].dim()
        if any(v.dim() != dims for v in vids):
            raise RuntimeError("Inconsistent tensor ranks in PianoYT batch")

        max_shape = tuple(max(v.shape[d] for v in vids) for d in range(dims))
        x = vids[0].new_zeros((len(vids),) + max_shape)
        for idx, vid in enumerate(vids):
            slices = tuple(slice(0, size) for size in vid.shape)
            x[(idx,) + slices] = vid

        out = {"video": x, "path": paths}
        extra_keys = set().union(*[set(d.keys()) for d in batch]) - {"video", "path"}
        for k in extra_keys:
            vals = [d[k] for d in batch if k in d]
            if len(vals) != len(batch):
                continue
            if k == "labels":
                out[k] = vals
            else:
                v0 = vals[0]
                if torch.is_tensor(v0):
                    try:
                        out[k] = torch.stack(vals, dim=0)
                    except Exception:
                        out[k] = vals
                else:
                    out[k] = vals
        return out

    loader = DataLoader(
        dataset,
        batch_size=int(dcfg.get("batch_size", 2)),
        shuffle=bool(dcfg.get("shuffle", True)) if split == "train" else False,
        num_workers=int(dcfg.get("num_workers", 0)),
        pin_memory=False,
        drop_last=drop_last,
        collate_fn=_collate,
    )
    return loader


__all__ = ["PianoYTDataset", "make_dataloader"]
