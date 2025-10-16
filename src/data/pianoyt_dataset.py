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
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
try:
    from typing import TypedDict
except ImportError:  # Python <3.8 fallback (not expected in TiViT)
    from typing_extensions import TypedDict  # pragma: no cover

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.av_sync import AVLagCache, estimate_av_lag, shift_label_events
from utils.frame_target_cache import FrameTargetCache, make_frame_target_cache_key
from utils.time_grid import frame_to_sec, sec_to_frame
from utils.tiling import tile_vertical_token_aligned

from .omaps_dataset import (  # reuse established helpers for identical behaviour
    _build_frame_targets,
    _load_clip_with_random_start,
    _read_manifest,
    apply_global_augment,
    apply_registration_crop,
    resize_to_canonical,
)

LOGGER = logging.getLogger(__name__)

class SampleRecord(TypedDict):
    id: str
    video: Path
    midi: Optional[Path]


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


def _read_midi_events(midi_path: Path) -> Tensor:
    """Return ``(N,3)`` tensor [onset_sec, offset_sec, pitch] parsed from MIDI."""

    if not midi_path or not midi_path.exists():
        return torch.zeros((0, 3), dtype=torch.float32)

    events: List[Tuple[float, float, float]] = []

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API",
                category=UserWarning,
            )
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
        dataset_cfg: Optional[Mapping[str, Any]] = None,
        full_cfg: Optional[Mapping[str, Any]] = None,
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
        self.dataset_cfg = dict(dataset_cfg or {})
        self.full_cfg = dict(full_cfg or {})

        # Optional hooks configured after construction (mirrors OMAPSDataset).
        # Provide defaults so static analyzers are aware of these attributes and
        # so runtime code can safely assume their existence prior to external
        # configuration.
        self.annotations_root: Optional[str] = self.dataset_cfg.get("annotations_root")
        self.label_format: str = str(self.dataset_cfg.get("label_format", "midi"))
        self.label_targets: Sequence[str] = tuple(
            self.dataset_cfg.get(
                "label_targets", ["pitch", "onset", "offset", "hand", "clef"]
            )
        )
        self.require_labels: bool = bool(self.dataset_cfg.get("require_labels", False))
        self.frame_targets_cfg: Dict[str, Any] = {}
        self.max_clips: Optional[int] = None

        self._av_sync_cache = AVLagCache()
        self._av_sync_warned = False
        self._valid_indices: List[int] = []
        self._label_warned: set = set()
        self._num_windows: int = 0
        self._frame_target_cache = FrameTargetCache()
        self._frame_target_log_once: set[str] = set()
        self._frame_target_failures: set[str] = set()

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
        if patch_w_cfg is None:
            raise ValueError(
                "tiling.patch_w or model.transformer.input_patch_size required for token-aligned tiling"
            )
        self.tiling_patch_w = int(patch_w_cfg)
        tokens_split_cfg = tiling_cfg.get("tokens_split", "auto")
        if isinstance(tokens_split_cfg, Sequence) and not isinstance(tokens_split_cfg, str):
            self.tiling_tokens_split = [int(v) for v in tokens_split_cfg]
        else:
            self.tiling_tokens_split = tokens_split_cfg
        self.tiling_overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))
        self._tiling_log_once = True
        self._registration_off_logged = False

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
        self.samples: List[SampleRecord] = []
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
        self.videos = [s["video"] for s in self.samples]
        self.apply_crop = True
        self.crop_rescale = "auto"
        self.include_low_res = True
        self.excluded_ids: set = set()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No PianoYT media found under {self.root} for split '{split}'."
            )
        
        self._rebuild_valid_index_cache(log_summary=True)

    def configure(self, *, include_low_res: bool, excluded_ids: set, apply_crop: bool, crop_rescale: str):
        self.include_low_res = include_low_res
        self.apply_crop = apply_crop
        self.crop_rescale = crop_rescale.lower()
        self.excluded_ids = excluded_ids
        if not include_low_res and excluded_ids:
            self.samples = [s for s in self.samples if s["id"] not in excluded_ids]
            self.videos = [s["video"] for s in self.samples]
        if len(self.samples) == 0:
            raise RuntimeError("All PianoYT samples were filtered out by configuration.")
        self._rebuild_valid_index_cache(log_summary=False)

    def limit_max_clips(self, max_clips: Optional[int]):
        if max_clips is None:
            return
        if max_clips < len(self.samples):
            self.samples = self.samples[:max_clips]
            self.videos = [s["video"] for s in self.samples]
        self._rebuild_valid_index_cache(log_summary=False)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        if not self._valid_indices:
            raise RuntimeError("PianoYTDataset has no valid labeled windows to sample.")

        max_attempts = len(self._valid_indices)
        attempt = 0
        logical_idx = idx % len(self._valid_indices)

        while attempt < max_attempts and self._valid_indices:
            record_idx = self._valid_indices[logical_idx]
            sample = self._load_sample_for_index(record_idx, idx)
            if sample is not None:
                return sample
            attempt += 1
            if not self._valid_indices:
                break
            logical_idx = random.randrange(len(self._valid_indices))

        raise RuntimeError("PianoYTDataset: exhausted attempts to fetch a valid sample.")

    def _load_sample_for_index(self, record_idx: int, dataset_index: int) -> Optional[Dict[str, Any]]:
        record = self.samples[record_idx]
        video_path = record["video"]
        midi_path = record["midi"]
        video_id = record["id"]

        is_train = self.split == "train"
        clip, start_idx = _load_clip_with_random_start(
            path=video_path,
            frames=self.frames,
            stride=self.stride,
            channels=self.channels,
            training=is_train,
            decode_fps=self.decode_fps,
        )

        if self.registration_enabled and self.apply_crop:
            meta = self.metadata.get(video_id)
            clip = apply_registration_crop(clip, meta, self.registration_cfg)
        elif not self.registration_enabled and not self._registration_off_logged:
            self._registration_off_logged = True

        clip = resize_to_canonical(clip, self.canonical_hw, self.registration_interp)
        
        if self.global_aug_enabled and is_train:
            clip = apply_global_augment(
                clip,
                self.global_aug_cfg,
                base_seed=self.data_seed,
                sample_index=dataset_index,
                start_idx=start_idx,
                interp=self.global_aug_cfg.get("interp", self.registration_interp),
                id_key=video_id,
            )
            
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
                f"tiles(tokens)={tokens_per_tile} widths_px={widths_px} "
                f"sum={width_sum} orig_W={original_w} overlap_tokens={self.tiling_overlap_tokens}",
                flush=True,
            )
            self._tiling_log_once = True

        T = self.frames
        fps = self.decode_fps
        hop_seconds = self.stride / max(fps, 1e-6)
        t0: float = float(frame_to_sec(start_idx, 1.0 / fps))
        t1: float = float(frame_to_sec(start_idx + ((T - 1) * self.stride + 1), 1.0 / fps))

        sample = {"video": clip, "path": str(video_path)}

        labels_tensor: Optional[torch.Tensor] = None
        if midi_path is not None and midi_path.exists():
            labels_tensor = _read_midi_events(midi_path)
        else:
            self._log_missing_labels_once(video_path)
            self._invalidate_sample_index(record_idx)
            return None

        if labels_tensor is None or labels_tensor.numel() == 0:
            self._log_missing_labels_once(video_path)
            self._invalidate_sample_index(record_idx)
            return None

        lag_result = estimate_av_lag(
            video_id=video_id,
            frames=clip,
            labels=labels_tensor,
            clip_start=t0,
            clip_end=t1,
            hop_seconds=hop_seconds,
            cache=self._av_sync_cache,
        ) if labels_tensor.numel() > 0 else None

        if lag_result is not None:
            if not lag_result.success and not self._av_sync_warned:
                LOGGER.warning(
                    "Unable to compute A/V lag for clip %s; using lag=0", video_id
                )
                self._av_sync_warned = True
            lag_seconds = (lag_result.lag_frames * hop_seconds) if lag_result.success else 0.0
            labels_tensor = shift_label_events(
                labels_tensor,
                lag_seconds,
                clip_start=t0,
                clip_end=t1,
            )
            lag_ms_display = lag_result.lag_ms if lag_result.success else 0.0
            corr_val = lag_result.corr if lag_result.success else float("nan")
            corr_str = f"{corr_val:.2f}" if math.isfinite(corr_val) else "nan"
            flags = []
            if lag_result.used_video_median:
                flags.append("used_video_median")
            if lag_result.low_corr_zero:
                flags.append("low_corr_zero")
            if lag_result.hit_bound:
                flags.append("hit_bound")
            if lag_result.clamped:
                flags.append("clamped")
            flags_str = ",".join(flags) if flags else "-"
            LOGGER.info(
                "clip=%s av_lag_ms=%+d corr=%s frames=%d flags=%s",
                video_id,
                int(round(lag_ms_display)),
                corr_str,
                T,
                flags_str,
            )

        if labels_tensor is not None and labels_tensor.numel() > 0:
            sample["labels"] = labels_tensor
        
        clip_targets: Optional[Dict[str, torch.Tensor]] = None
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
            self._log_missing_labels_once(video_path)
            return None

        ft_cfg = getattr(self, "frame_targets_cfg", None)
        if (
            labels_tensor is not None
            and labels_tensor.numel() > 0
            and ft_cfg
            and bool(ft_cfg.get("enable", False))
        ):
            tolerance = float(ft_cfg.get("tolerance", 0.025))
            dilation = int(ft_cfg.get("dilate_active_frames", 0))
            lag_ms_final = (
                lag_result.lag_ms if lag_result is not None and lag_result.success else 0.0
            )
            key_hash, key_meta = make_frame_target_cache_key(
                split=self.split,
                video_id=video_id,
                lag_ms=lag_ms_final,
                fps=fps,
                frames=T,
                tolerance=tolerance,
                dilation=dilation,
                canonical_hw=self.canonical_hw,
            )
            cached_targets, _ = self._frame_target_cache.load(key_hash)
            required_keys = (
                "pitch_roll",
                "onset_roll",
                "offset_roll",
                "hand_frame",
                "clef_frame",
            )
            if cached_targets is not None and all(k in cached_targets for k in required_keys):
                sample.update(cached_targets)
                self._log_frame_target_status(
                    video_id, "reused", key_hash, int(key_meta["lag_ms"])
                )
            else:
                labels_ft = labels_tensor.clone()
                labels_ft[:, 0:2] -= t0
                try:
                    ft = _build_frame_targets(
                        labels=labels_ft,
                        T=T,
                        stride=self.stride,
                        fps=fps,
                        note_min=int(ft_cfg.get("note_min", 21)),
                        note_max=int(ft_cfg.get("note_max", 108)),
                        tol=tolerance,
                        fill_mode=str(ft_cfg.get("fill_mode", "overlap")),
                        hand_from_pitch=bool(ft_cfg.get("hand_from_pitch", True)),
                        clef_thresholds=tuple(ft_cfg.get("clef_thresholds", [60, 64])),
                        dilate_active_frames=dilation,
                        targets_sparse=bool(ft_cfg.get("targets_sparse", False)),
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning(
                        "Failed to build frame targets for %s (key=%s): %s",
                        video_id,
                        key_hash,
                        exc,
                    )
                    self._log_frame_target_status(
                        video_id, "failed", key_hash, int(key_meta["lag_ms"])
                    )
                    self._mark_frame_target_failure(record_idx, video_id)
                    return None

                target_payload = {
                    "pitch_roll": ft["pitch_roll"],
                    "onset_roll": ft["onset_roll"],
                    "offset_roll": ft["offset_roll"],
                    "hand_frame": ft["hand_frame"],
                    "clef_frame": ft["clef_frame"],
                }
                sample.update(target_payload)
                self._frame_target_cache.save(key_hash, key_meta, target_payload)
                self._log_frame_target_status(
                    video_id, "built", key_hash, int(key_meta["lag_ms"])
                )

        return sample

    def _log_missing_labels_once(self, video_path: Path) -> None:
        name = video_path.name
        if name in self._label_warned:
            return
        self._label_warned.add(name)
        LOGGER.warning("skip_no_labels %s", name)

    def _log_frame_target_status(
        self, video_id: str, status: str, key_hash: str, lag_ms: int
    ) -> None:
        if video_id in self._frame_target_log_once:
            return
        self._frame_target_log_once.add(video_id)
        LOGGER.info(
            "targets: %s key=%s lag_ms=%+d video=%s",
            status,
            key_hash,
            lag_ms,
            video_id,
        )

    def _mark_frame_target_failure(self, record_idx: int, video_id: str) -> None:
        if video_id in self._frame_target_failures:
            return
        self._frame_target_failures.add(video_id)
        self._invalidate_sample_index(record_idx)
    
    def _invalidate_sample_index(self, record_idx: int) -> None:
        if record_idx >= len(self.samples):
            return
        self._valid_indices = [i for i in self._valid_indices if i != record_idx]
        self._num_windows = len(self._valid_indices)

    def _rebuild_valid_index_cache(self, *, log_summary: bool) -> None:
        self._valid_indices = []
        total = len(self.samples)
        skipped = 0
        for idx, record in enumerate(self.samples):
            midi_path = record.get("midi")
            labels_tensor: Optional[torch.Tensor] = None
            if midi_path is not None and midi_path.exists():
                labels_tensor = _read_midi_events(midi_path)
            if labels_tensor is None or labels_tensor.numel() == 0:
                self._log_missing_labels_once(record["video"])
                skipped += 1
                continue
            self._valid_indices.append(idx)

        self._num_windows = len(self._valid_indices)
        if log_summary:
            LOGGER.info(
                "videos: %d, N_skipped_no_labels: %d, windows: %d",
                total,
                skipped,
                self._num_windows,
            )
        if not self._valid_indices:
            LOGGER.warning(
                "[PianoYT] No valid labeled samples remain for split %s.", self.split
            )


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))
    
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
    dataset._rebuild_valid_index_cache(log_summary=False)

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
