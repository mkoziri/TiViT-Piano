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
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
try:
    from typing import TypedDict
except ImportError:  # Python <3.8 fallback (not expected in TiViT)
    from typing_extensions import TypedDict  # pragma: no cover

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.av_sync import AVLagCache, compute_av_lag, shift_label_events
from utils.identifiers import canonical_video_id
from utils.frame_target_cache import FrameTargetCache
from utils.frame_targets import (
    FrameTargetResult,
    FrameTargetSpec,
    prepare_frame_targets,
    resolve_lag_ms,
    resolve_frame_target_spec,
)
from utils.time_grid import frame_to_sec, sec_to_frame
from utils.tiling import tile_vertical_token_aligned

from .omaps_dataset import (  # reuse established helpers for identical behaviour
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


@dataclass(frozen=True)
class WindowEntry:
    record_idx: int
    start_idx: Optional[int]
    has_events: bool = True


@dataclass
class SampleBuildResult:
    sample: Optional[Dict[str, Any]]
    status: str
    lag_ms: Optional[int]
    lag_source: str
    events_on: int
    events_off: int
    has_events: bool
    start_idx: int
    video_id: str


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
                    ids.append(canonical_video_id(line))
        # preserve ordering but drop duplicates introduced by canonicalisation
        seen = set()
        ordered = []
        for vid in ids:
            if vid not in seen:
                ordered.append(vid)
                seen.add(vid)
        return ordered

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
            ids_set.add(canonical_video_id(f"video_{vid}"))
    for midi_path in split_dir.glob("audio_*.0.midi"):
        vid = _extract_id(midi_path.name, "audio_")
        if vid:
            ids_set.add(canonical_video_id(f"video_{vid}"))

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
                canon = canonical_video_id(vid)
                table[canon] = (values[0], values[1], values[2], values[3])

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
                canon = canonical_video_id(vid)
                table[canon] = (min_y, max_y, min_x, max_x)

            parse_headerless_row(first_row)
            for raw_row in reader:
                parse_headerless_row(raw_row)
    return table


def _media_token_from_id(video_id: str) -> str:
    canon = canonical_video_id(video_id)
    if canon.startswith("video_"):
        token = canon[6:]
    elif canon.startswith("video"):
        token = canon[5:]
    else:
        token = canon
    token = token.strip("_")
    return token or canon


def _resolve_media_paths(root: Path, split: str, video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the video and MIDI paths for ``video_id`` within ``split``."""

    canon_id = canonical_video_id(video_id)
    token = _media_token_from_id(canon_id)
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
        midi = base / f"audio_{token}.0.midi"
        video = None
        for ext in _VIDEO_EXTS:
            cand = base / f"video_{token}.0{ext}"
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
                ids.add(canonical_video_id(line))
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
        self.frame_targets_cfg: Dict[str, Any] = dict(
            self.dataset_cfg.get("frame_targets", {}) or {}
        )
        self.max_clips: Optional[int] = None

        self._av_sync_cache = AVLagCache()
        self._av_sync_cache.preload()
        self._av_sync_warned = False
        self._lag_log_once: Set[str] = set()
        self._valid_entries: List[WindowEntry] = []
        self._label_warned: Set[str] = set()
        self._num_windows: int = 0
        self._frame_target_cache = FrameTargetCache()
        self._frame_target_log_once: Dict[str, Set[str]] = {}
        self._frame_target_failures: Set[str] = set()

        canonical_cfg = self.dataset_cfg.get("canonical_hw", self.resize)
        if isinstance(canonical_cfg, Sequence) and len(canonical_cfg) >= 2:
            self.canonical_hw = (
                int(round(float(canonical_cfg[0]))),
                int(round(float(canonical_cfg[1]))),
            )
        else:
            self.canonical_hw = tuple(self.resize)

        self.frame_target_spec: Optional[FrameTargetSpec] = resolve_frame_target_spec(
            self.frame_targets_cfg,
            frames=self.frames,
            stride=self.stride,
            fps=self.decode_fps,
            canonical_hw=self.canonical_hw,
        )
        self.frame_target_summary: Optional[str] = (
            self.frame_target_spec.summary()
            if self.frame_target_spec is not None
            else "targets_conf: disabled"
        )

        reg_cfg = dict(self.dataset_cfg.get("registration", {}) or {})
        self.registration_cfg = reg_cfg
        self.registration_enabled = bool(reg_cfg.get("enabled", False))
        self.registration_interp = str(reg_cfg.get("interp", "bilinear"))

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
            canon = canonical_video_id(vid)
            video_path, midi_path = _resolve_media_paths(self.root, split, canon)
            if video_path is None:
                LOGGER.warning("[PianoYT] Missing video file for id=%s", canon)
                continue
            self.samples.append({"id": canon, "video": video_path, "midi": midi_path})

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
        return len(self._valid_entries)

    def __getitem__(self, idx: int):
        if not self._valid_entries:
            if self.split == "val":
                raise RuntimeError(
                    "Val split has 0 valid videos after audit; check labels/lag or widen search."
                )
            raise RuntimeError("PianoYTDataset has no valid labeled windows to sample.")

        if idx < 0 or idx >= len(self._valid_entries):
            raise IndexError(idx)

        entry = self._valid_entries[idx]
        result = self._build_sample(
            entry.record_idx,
            dataset_index=idx,
            preferred_start_idx=entry.start_idx,
            audit=False,
        )
        if result.sample is None:
            raise RuntimeError(
                f"PianoYTDataset: unable to fetch sample for {result.video_id} ({result.status})"
            )
        return result.sample

    def _build_sample(
        self,
        record_idx: int,
        dataset_index: int,
        *,
        preferred_start_idx: Optional[int] = None,
        audit: bool = False,
    ) -> SampleBuildResult:
        record = self.samples[record_idx]
        video_path = record["video"]
        midi_path = record["midi"]
        video_id = canonical_video_id(record["id"])
        start_hint = int(preferred_start_idx or 0)

        if video_path is None or not video_path.exists():
            if not audit:
                self._invalidate_sample_index(record_idx)
            return SampleBuildResult(None, "no_file", None, "", 0, 0, False, start_hint, video_id)

        if midi_path is None or not midi_path.exists():
            self._log_missing_labels_once(video_path)
            if not audit:
                self._invalidate_sample_index(record_idx)
            return SampleBuildResult(None, "no_labels", None, "", 0, 0, False, start_hint, video_id)

        labels_tensor = _read_midi_events(midi_path)
        if labels_tensor is None:
            labels_tensor = torch.zeros((0, 3), dtype=torch.float32)

        is_train = self.split == "train" and preferred_start_idx is None and not audit
        clip, start_idx = _load_clip_with_random_start(
            path=video_path,
            frames=self.frames,
            stride=self.stride,
            channels=self.channels,
            training=is_train,
            decode_fps=self.decode_fps,
            preferred_start_idx=preferred_start_idx,
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

        if audit:
            sample: Dict[str, Any] = {"path": str(video_path)}
        else:
            sample = {"video": clip, "path": str(video_path)}

        lag_result = None
        lag_ms_int: Optional[int] = None
        lag_source = "guardrail"
        shifted_labels = labels_tensor

        if labels_tensor.numel() > 0:
            if audit:
                cached_lag_ms = self._av_sync_cache.get(video_id)
                if cached_lag_ms is not None and math.isfinite(float(cached_lag_ms)):
                    lag_seconds = float(cached_lag_ms) / 1000.0
                    shifted_labels = shift_label_events(
                        labels_tensor,
                        lag_seconds,
                        clip_start=t0,
                        clip_end=t1,
                    )
                    lag_ms_int = int(round(float(cached_lag_ms)))
                    lag_source = "cache"
                else:
                    shifted_labels = labels_tensor
                    lag_source = "audit_skip"
            else:
                lag_result = compute_av_lag(
                    video_id=video_id,
                    frames=clip,
                    events=labels_tensor,
                    hop_seconds=hop_seconds,
                    clip_start=t0,
                    clip_end=t1,
                    cache=self._av_sync_cache,
                )
                if lag_result is not None:
                    if not lag_result.success and not self._av_sync_warned:
                        LOGGER.warning(
                            "Unable to compute A/V lag for clip %s; using lag=0",
                            video_id,
                        )
                        self._av_sync_warned = True
                    lag_seconds = (
                        lag_result.lag_frames * hop_seconds
                    ) if lag_result.success else 0.0
                    shifted_labels = shift_label_events(
                        labels_tensor,
                        lag_seconds,
                        clip_start=t0,
                        clip_end=t1,
                    )
                    lag_ms_display = lag_result.lag_ms if lag_result.success else 0.0
                    corr_val = float(lag_result.corr)
                    corr_str = f"{corr_val:.2f}" if math.isfinite(corr_val) else "nan"
                    flags_set: Set[str] = set(lag_result.flags or set())
                    flags_str = ",".join(sorted(flags_set)) if flags_set else "-"
                    if not audit and video_id not in self._lag_log_once:
                        LOGGER.info(
                            "clip=%s av_lag_ms=%+d corr=%s frames=%d flags=%s",
                            video_id,
                            int(round(lag_ms_display)),
                            corr_str,
                            T,
                            flags_str,
                        )
                        self._lag_log_once.add(video_id)
                else:
                    shifted_labels = labels_tensor.reshape(0, 3)
        else:
            shifted_labels = labels_tensor.reshape(0, 3)

        has_events = shifted_labels.numel() > 0
        sample["labels"] = shifted_labels

        if not has_events and self.require_labels and self.split == "train" and not audit:
            self._log_missing_labels_once(video_path)
            self._invalidate_sample_index(record_idx)
            return SampleBuildResult(None, "no_labels", None, "", 0, 0, False, start_idx, video_id)

        clip_targets: Optional[Dict[str, torch.Tensor]] = None
        if has_events:
            onset = shifted_labels[:, 0]
            offset = shifted_labels[:, 1]
            pitch = shifted_labels[:, 2].to(torch.int64)

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
        elif self.require_labels and self.split == "train" and not audit:
            self._log_missing_labels_once(video_path)
            self._invalidate_sample_index(record_idx)
            return SampleBuildResult(None, "no_labels", None, "", 0, 0, False, start_idx, video_id)

        if not audit:
            lag_ms_value, lag_source = resolve_lag_ms(lag_result)
            lag_ms_int = int(round(lag_ms_value)) if math.isfinite(lag_ms_value) else None

            spec = getattr(self, "frame_target_spec", None)
            if spec is not None:
                try:
                    ft_result: FrameTargetResult = prepare_frame_targets(
                        labels=shifted_labels,
                        lag_result=lag_result,
                        spec=spec,
                        cache=self._frame_target_cache,
                        split=self.split,
                        video_id=video_id,
                        clip_start=t0,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning(
                        "Failed to build frame targets for %s: %s",
                        video_id,
                        exc,
                    )
                    if not audit:
                        self._log_frame_target_status(video_id, "failed", "-", 0)
                        self._mark_frame_target_failure(record_idx, video_id)
                    return SampleBuildResult(None, "build_fail", lag_ms_int, lag_source, 0, 0, has_events, start_idx, video_id)

                if ft_result.cache_key is not None and not audit:
                    self._log_frame_target_status(
                        video_id,
                        ft_result.status,
                        ft_result.cache_key,
                        ft_result.lag_ms,
                    )

                if ft_result.payload is not None:
                    sample.update(ft_result.payload)
                elif self.require_labels and self.split == "train" and not audit:
                    self._log_missing_labels_once(video_path)
                    self._mark_frame_target_failure(record_idx, video_id)
                    return SampleBuildResult(None, "build_fail", ft_result.lag_ms, ft_result.lag_source, 0, 0, has_events, start_idx, video_id)

                lag_ms_int = ft_result.lag_ms
                lag_source = ft_result.lag_source

        events_on = int(((shifted_labels[:, 0] >= t0) & (shifted_labels[:, 0] < t1)).sum().item()) if has_events else 0
        events_off = int(((shifted_labels[:, 1] > t0) & (shifted_labels[:, 1] <= t1)).sum().item()) if has_events else 0
        sample["lag_ms"] = lag_ms_int
        sample["lag_source"] = lag_source

        return SampleBuildResult(sample, "ok", lag_ms_int, lag_source, events_on, events_off, has_events, start_idx, video_id)

    def _load_sample_for_index(
        self,
        record_idx: int,
        dataset_index: int,
        *,
        preferred_start_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        result = self._build_sample(
            record_idx,
            dataset_index,
            preferred_start_idx=preferred_start_idx,
            audit=False,
        )
        return result.sample

    def _log_missing_labels_once(self, video_ref: Union[str, Path]) -> None:
        video_id = canonical_video_id(video_ref)
        if not video_id:
            video_id = str(video_ref)
        if video_id in self._label_warned:
            return
        self._label_warned.add(video_id)
        LOGGER.warning("skip_no_labels %s", video_id)

    def _log_frame_target_status(
        self, video_id: str, status: str, key_hash: str, lag_ms: int
    ) -> None:
        if not key_hash:
            return
        tickets = self._frame_target_log_once.setdefault(video_id, set())
        ticket = f"{status}:{key_hash[:8]}"
        if ticket in tickets:
            return
        tickets.add(ticket)
        LOGGER.info(
            "targets: %s key=%s lag_ms=%+d split=%s id=%s",
            status,
            key_hash[:8],
            lag_ms,
            self.split,
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
        self._valid_entries = [
            entry for entry in self._valid_entries if entry.record_idx != record_idx
        ]
        self._num_windows = len(self._valid_entries)

    def _candidate_start_indices(self, labels_tensor: Optional[torch.Tensor]) -> List[int]:
        if labels_tensor is None or labels_tensor.numel() == 0:
            return [0]
        fps = max(self.decode_fps, 1e-6)
        window_frames = (self.frames - 1) * self.stride + 1
        window_span = window_frames / fps
        starts: set[int] = {0}
        for onset, offset in labels_tensor[:, 0:2].tolist():
            midpoint = (onset + offset) * 0.5
            candidate_starts = [
                max(0.0, onset - 0.5 * window_span),
                max(0.0, onset),
                max(0.0, midpoint - 0.5 * window_span),
                max(0.0, midpoint),
                max(0.0, offset - window_span),
            ]
            for start_sec in candidate_starts:
                starts.add(int(round(start_sec * fps)))
        ordered = sorted(starts)
        max_candidates = 8
        return ordered[:max_candidates]

    def _record_has_valid_window(self, record_idx: int) -> bool:
        record = self.samples[record_idx]
        midi_path = record.get("midi")
        video_path = record.get("video")
        if midi_path is None or not midi_path.exists():
            if video_path is not None:
                self._log_missing_labels_once(video_path)
            return False

        labels_tensor = _read_midi_events(midi_path)
        if labels_tensor is None or labels_tensor.numel() == 0:
            if video_path is not None:
                self._log_missing_labels_once(video_path)
            return False

        start_indices = self._candidate_start_indices(labels_tensor)
        for start_idx in start_indices:
            try:
                result = self._build_sample(
                    record_idx,
                    dataset_index=0,
                    preferred_start_idx=start_idx,
                    audit=True,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning(
                    "Failed to validate clip %s at start_idx=%d (%s)",
                    record.get("id"),
                    start_idx,
                    exc,
                )
                result = SampleBuildResult(None, "error", None, "", 0, 0, False, start_idx, record.get("id", ""))
            if result.sample is not None and result.has_events:
                return True

        if video_path is not None:
            self._log_missing_labels_once(video_path)
        return False
    
    def _plan_eval_entries(
        self, record_idx: int
    ) -> Tuple[List[WindowEntry], str, int, int, Optional[int]]:
        record = self.samples[record_idx]
        video_path = record.get("video")
        midi_path = record.get("midi")
        video_id = record.get("id", "")

        if video_path is None or not video_path.exists():
            return [], "no_file", 0, 0, None

        if midi_path is None or not midi_path.exists():
            self._log_missing_labels_once(video_path)
            return [], "no_labels", 0, 0, None

        labels_tensor = _read_midi_events(midi_path)
        if labels_tensor is None:
            labels_tensor = torch.zeros((0, 3), dtype=torch.float32)

        start_indices = self._candidate_start_indices(labels_tensor)
        seen: set[int] = set()
        event_entries: List[Tuple[WindowEntry, SampleBuildResult]] = []
        neg_entries: List[Tuple[WindowEntry, SampleBuildResult]] = []
        lag_ms_value: Optional[int] = None

        for start_idx in start_indices:
            if start_idx in seen:
                continue
            seen.add(start_idx)
            try:
                result = self._build_sample(
                    record_idx,
                    dataset_index=0,
                    preferred_start_idx=start_idx,
                    audit=True,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning(
                    "Failed to audit clip %s at start_idx=%d (%s)",
                    video_id,
                    start_idx,
                    exc,
                )
                continue

            if result.sample is None:
                continue

            if lag_ms_value is None and result.lag_ms is not None:
                lag_ms_value = result.lag_ms

            entry = WindowEntry(record_idx=record_idx, start_idx=result.start_idx, has_events=result.has_events)
            if result.has_events:
                event_entries.append((entry, result))
            else:
                neg_entries.append((entry, result))

        if not event_entries and not neg_entries:
            return [], "build_fail", 0, 0, lag_ms_value

        entries: List[WindowEntry] = []
        events_on_total = 0
        events_off_total = 0

        for entry, res in event_entries:
            entries.append(entry)
            events_on_total += res.events_on
            events_off_total += res.events_off

        if event_entries:
            allowed_neg = min(len(neg_entries), len(event_entries))
        else:
            allowed_neg = min(len(neg_entries), 1)

        for entry, res in neg_entries[:allowed_neg]:
            entries.append(WindowEntry(entry.record_idx, entry.start_idx, res.has_events))
            events_on_total += res.events_on
            events_off_total += res.events_off

        status = "ok" if entries else "build_fail"
        return entries, status, events_on_total, events_off_total, lag_ms_value

    def _rebuild_valid_index_cache(self, *, log_summary: bool) -> None:
        self._valid_entries = []
        total = len(self.samples)
        skipped = 0
        if self.split == "train":
            for idx in range(total):
                if self._record_has_valid_window(idx):
                    self._valid_entries.append(WindowEntry(idx, None, True))
                else:
                    skipped += 1
        else:
            for idx, record in enumerate(self.samples):
                entries, status, events_on, events_off, lag_ms = self._plan_eval_entries(idx)
                if self.split == "val":
                    lag_display = lag_ms if lag_ms is not None else "?"
                    LOGGER.info(
                        "val_audit | vid=%s lag_ms=%s events_on=%d events_off=%d status=%s",
                        record.get("id", ""),
                        lag_display,
                        events_on,
                        events_off,
                        status,
                    )
                if status == "ok" and entries:
                    self._valid_entries.extend(entries)
                else:
                    skipped += 1

        self._num_windows = len(self._valid_entries)
        if log_summary:
            LOGGER.info(
                "videos: %d, N_skipped_no_labels: %d, windows: %d",
                total,
                skipped,
                self._num_windows,
            )
        if not self._valid_entries:
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
    dataset.frame_targets_cfg = dict(dcfg.get("frame_targets", {}) or {})
    dataset.frame_target_spec = resolve_frame_target_spec(
        dataset.frame_targets_cfg,
        frames=dataset.frames,
        stride=stride,
        fps=decode_fps,
        canonical_hw=dataset.canonical_hw,
    )
    dataset.frame_target_summary = (
        dataset.frame_target_spec.summary()
        if dataset.frame_target_spec is not None
        else "targets_conf: disabled"
    )
    dataset.frame_target_spec = resolve_frame_target_spec(
        dataset.frame_targets_cfg,
        frames=dataset.frames,
        stride=stride,
        fps=decode_fps,
        canonical_hw=dataset.canonical_hw,
    )
    dataset.frame_target_summary = (
        dataset.frame_target_spec.summary()
        if dataset.frame_target_spec is not None
        else "targets_conf: disabled"
    )
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
