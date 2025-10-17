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
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
try:
    from typing import TypedDict
except ImportError:  # Python <3.8 fallback (not expected in TiViT)
    from typing_extensions import TypedDict  # pragma: no cover

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.av_sync import AVLagCache, compute_av_lag, shift_label_events
from utils.identifiers import canonical_video_id, id_aliases, log_legacy_id_hit
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


_EVAL_ALT_WINDOW_ATTEMPTS = 3


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


def _resolve_media_paths(root: Path, split: str, video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the video and MIDI paths for ``video_id`` within ``split``."""

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
                for ext in _VIDEO_EXTS:
                    cand = base / f"{alias}{ext}"
                    if cand.exists():
                        video_path = cand
                        video_alias = alias
                        break
                if video_path is not None:
                    break
        if midi_path is None:
            for alias in aliases:
                if alias.startswith("video_"):
                    audio_name = "audio_" + alias[6:]
                else:
                    audio_name = alias
                cand = base / f"{audio_name}.midi"
                if cand.exists():
                    midi_path = cand
                    midi_alias = alias
                    break
        if video_path is not None and midi_path is not None:
            break

    if video_alias and video_alias != canon_id:
        log_legacy_id_hit(video_alias, canon_id, logger=LOGGER)
    if midi_alias and midi_alias != canon_id:
        log_legacy_id_hit(midi_alias, canon_id, logger=LOGGER)

    return video_path, midi_path


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
        only_video: Optional[str] = None,
        avlag_disabled: Optional[bool] = None,
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
        default_audit_timeout = 10.0
        timeout_override = self.dataset_cfg.get("audit_timeout_sec")
        timeout_env = os.environ.get("DATASET_AUDIT_TIMEOUT_SEC")
        self._audit_timeout_sec = float(timeout_override) if timeout_override else default_audit_timeout
        if self._audit_timeout_sec <= 0:
            LOGGER.warning(
                "Invalid dataset audit_timeout_sec %.2f; using %.1fs default",
                self._audit_timeout_sec,
                default_audit_timeout,
            )
            self._audit_timeout_sec = default_audit_timeout
        if timeout_env:
            try:
                parsed = float(timeout_env)
                if parsed > 0:
                    self._audit_timeout_sec = parsed
                else:
                    raise ValueError("timeout must be positive")
            except ValueError:
                LOGGER.warning(
                    "Invalid DATASET_AUDIT_TIMEOUT_SEC value '%s'; using %.1fs default",
                    timeout_env,
                    self._audit_timeout_sec,
                )

        env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
        env_disabled = env_disable in {"1", "true", "yes", "on"}
        cfg_disabled = bool(self.dataset_cfg.get("avlag_disabled", False))
        self._av_sync_disabled = bool(avlag_disabled) or cfg_disabled or env_disabled
        self._av_sync_disabled_logged = False
        self._av_sync_cache = AVLagCache() if not self._av_sync_disabled else None
        self._av_sync_warned = False
        self._lag_log_once: Set[str] = set()
        self._valid_entries: List[WindowEntry] = []
        self.eval_indices_snapshot: List[Tuple[int, int]] = []
        self._eval_snapshot_flags: List[bool] = []
        self._eval_snapshot_by_video: Dict[int, List[int]] = {}
        self._eval_candidates_by_video: Dict[int, List[int]] = {}
        self._eval_materialize_stats: Dict[str, Any] = {}
        self._audit_ok_records: Set[int] = set()
        self._audit_entry_starts: Dict[int, List[int]] = {}
        self._label_warned: Set[str] = set()
        self._num_windows: int = 0
        self._frame_target_cache = FrameTargetCache()
        self._frame_target_log_once: Dict[str, Set[str]] = {}
        self._frame_target_failures: Set[str] = set()
        self._bad_clips: Set[str] = set()
        self._only_filter_applied: bool = False
        self._only_video_target: Optional[str] = None
        self.args_max_clips_or_None: Optional[int] = None
        self._last_materialize_duration: float = 0.0
        self._log_av_disabled_once()

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
        requested_only = only_video if only_video is not None else self.dataset_cfg.get("only_video")
        if requested_only:
            self._only_video_target = canonical_video_id(requested_only)
            self._only_filter_applied = True

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
        total_videos = len(ids)
        print(f"[dataset] enter PianoYT(split={split}) total_videos={total_videos}", flush=True)
        if self._only_video_target:
            ids = [
                vid
                for vid in ids
                if canonical_video_id(vid) == self._only_video_target
            ]

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
        print(f"[dataset] after filter videos={len(self.samples)}", flush=True)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No PianoYT media found under {self.root} for split '{split}'."
            )
        
        if self._av_sync_cache is not None:
            self._av_sync_cache.preload()
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
        self.args_max_clips_or_None = max_clips if isinstance(max_clips, int) else None
        if max_clips is None:
            return
        if max_clips < len(self.samples):
            self.samples = self.samples[:max_clips]
            self.videos = [s["video"] for s in self.samples]
        self._rebuild_valid_index_cache(log_summary=False)

    def filter_to_video(self, video_id: str) -> bool:
        """Restrict the dataset to clips originating from ``video_id``."""

        target = canonical_video_id(video_id)
        filtered = [
            sample
            for sample in self.samples
            if canonical_video_id(sample.get("id", "")) == target
        ]
        if not filtered:
            return False
        self.samples = filtered
        self.videos = [s.get("video") for s in self.samples if s.get("video") is not None]
        self._frame_target_log_once.clear()
        self._frame_target_failures.clear()
        self._lag_log_once.clear()
        self._valid_entries = []
        self._num_windows = 0
        self._only_filter_applied = True
        self._only_video_target = target
        self._rebuild_valid_index_cache(log_summary=False)
        return True

    def __len__(self) -> int:
        if self._uses_eval_snapshot():
            return len(self.eval_indices_snapshot)
        return len(self._valid_entries)

    def __getitem__(self, idx: int):
        if self._uses_eval_snapshot():
            total = len(self.eval_indices_snapshot)
            if total == 0:
                raise RuntimeError(
                    "PianoYTDataset: eval snapshot empty; rerun audit or refresh dataset."
                )
            idx = idx % max(total, 1)
            sample, failure_reason = self._fetch_eval_sample(idx)
            if sample is not None:
                return sample
            reason = failure_reason or "unknown"
            for step in range(1, total):
                fallback_idx = (idx - step) % total
                sample, fallback_reason = self._fetch_eval_sample(fallback_idx)
                if sample is not None:
                    print(
                        f"[dataset] fallback idx={idx} -> idx={fallback_idx} reason={reason}",
                        flush=True,
                    )
                    if fallback_reason and fallback_reason != reason:
                        reason = fallback_reason
                    return sample
            raise RuntimeError(
                "PianoYTDataset: exhausted eval fallbacks; no valid samples available."
            )

        if not self._valid_entries:
            if self.split == "val":
                raise RuntimeError(
                    "Val split has 0 valid videos after audit; check labels/lag or widen search."
                )
            raise RuntimeError("PianoYTDataset has no valid labeled windows to sample.")

        if idx < 0 or idx >= len(self._valid_entries):
            raise IndexError(idx)

        attempts = 0
        while self._valid_entries:
            if idx < 0 or idx >= len(self._valid_entries):
                idx = idx % len(self._valid_entries)
            entry = self._valid_entries[idx]
            result = self._build_sample(
                entry.record_idx,
                dataset_index=idx,
                preferred_start_idx=entry.start_idx,
                audit=False,
            )
            if result.sample is not None:
                return result.sample
            attempts += 1
            if attempts >= max(len(self._valid_entries), 1):
                break
        raise RuntimeError(
            "PianoYTDataset: unable to fetch a valid sample after filtering bad clips."
        )

    def _uses_eval_snapshot(self) -> bool:
        return self.split != "train"

    def _fetch_eval_sample(self, snapshot_idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if snapshot_idx < 0 or snapshot_idx >= len(self.eval_indices_snapshot):
            return None, "snapshot_oob"

        record_idx, start_idx = self.eval_indices_snapshot[snapshot_idx]
        result = self._build_sample(
            record_idx,
            dataset_index=snapshot_idx,
            preferred_start_idx=start_idx,
            audit=False,
        )
        if result.sample is not None:
            return result.sample, None

        failure_reason = result.status or "unknown"
        alt_sample, alt_reason = self._try_alternative_windows(
            record_idx,
            snapshot_idx=snapshot_idx,
            failure_reason=failure_reason,
            original_start=start_idx,
        )
        if alt_sample is not None:
            return alt_sample, alt_reason
        return None, alt_reason or failure_reason

    def _try_alternative_windows(
        self,
        record_idx: int,
        *,
        failure_reason: str,
        original_start: int,
        snapshot_idx: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        candidates = self._eval_candidates_by_video.get(record_idx)
        if not candidates:
            return None, failure_reason

        attempts = 0
        last_reason: Optional[str] = None
        ordered = sorted(candidates, key=lambda start: (abs(start - original_start), start))
        for start_idx in ordered:
            if attempts >= _EVAL_ALT_WINDOW_ATTEMPTS:
                break
            if start_idx == original_start:
                continue
            attempts += 1
            result = self._build_sample(
                record_idx,
                dataset_index=snapshot_idx,
                preferred_start_idx=start_idx,
                audit=False,
            )
            if result.sample is not None:
                return result.sample, None
            last_reason = result.status or failure_reason
        return None, last_reason or failure_reason

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
                self._mark_bad_clip(record_idx, video_id, "no_labels")
            return SampleBuildResult(None, "no_labels", None, "", 0, 0, False, start_hint, video_id)

        labels_tensor = _read_midi_events(midi_path)
        if labels_tensor is None:
            labels_tensor = torch.zeros((0, 3), dtype=torch.float32)

        is_train = self.split == "train" and preferred_start_idx is None and not audit
        clip: Optional[Tensor] = None
        if audit:
            start_idx = start_hint
        else:
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
            assert clip is not None
            clip_tensor = cast(Tensor, clip)
            sample = {"video": clip_tensor, "path": str(video_path)}

        lag_result = None
        lag_ms_int: Optional[int] = None
        lag_source = "guardrail"
        shifted_labels = labels_tensor

        if labels_tensor.numel() > 0:
            if self._av_sync_disabled:
                self._log_av_disabled_once()
                lag_source = "av_disabled"
                shifted_labels = labels_tensor
                lag_ms_int = 0
            elif audit:
                cache_obj = self._av_sync_cache
                cached_lag_ms = cache_obj.get(video_id) if cache_obj is not None else None
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
                cache_obj = self._av_sync_cache
                lag_result = compute_av_lag(
                    video_id=video_id,
                    frames=clip_tensor,
                    events=labels_tensor,
                    hop_seconds=hop_seconds,
                    clip_start=t0,
                    clip_end=t1,
                    cache=cache_obj,
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

        if not has_events and self.require_labels and not audit:
            self._log_missing_labels_once(video_path)
            self._mark_bad_clip(record_idx, video_id, "no_labels")
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
            if self._av_sync_disabled:
                lag_ms_value = 0.0
                lag_source = "av_disabled"
                lag_ms_int = 0
            else:
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
                        self._log_frame_target_status(video_id, "failed", "-", lag_frames=None)
                        self._mark_frame_target_failure(record_idx, video_id)
                    return SampleBuildResult(None, "build_failed", lag_ms_int, lag_source, 0, 0, has_events, start_idx, video_id)

                if ft_result.cache_key is not None and not audit:
                    self._log_frame_target_status(
                        video_id,
                        ft_result.status,
                        ft_result.cache_key,
                        lag_frames=ft_result.lag_frames,
                    )

                if ft_result.payload is not None:
                    sample.update(ft_result.payload)
                else:
                    if not audit:
                        status_lower = (ft_result.status or "").lower()
                        if status_lower in {"failed", "build_failed"}:
                            self._mark_frame_target_failure(record_idx, video_id)
                        elif self.require_labels:
                            self._log_missing_labels_once(video_path)
                            self._mark_bad_clip(record_idx, video_id, "no_labels")
                    return SampleBuildResult(None, "build_failed", ft_result.lag_ms, ft_result.lag_source, 0, 0, has_events, start_idx, video_id)

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

    def _log_av_disabled_once(self) -> None:
        if self._av_sync_disabled and not self._av_sync_disabled_logged:
            print("[debug] AV-lag disabled (lag_ms=0 for all clips)", flush=True)
            self._av_sync_disabled_logged = True

    def _log_missing_labels_once(self, video_ref: Union[str, Path]) -> None:
        video_id = canonical_video_id(video_ref)
        if not video_id:
            video_id = str(video_ref)
        if video_id in self._label_warned:
            return
        self._label_warned.add(video_id)
        LOGGER.warning("skip_no_labels %s", video_id)

    def _log_frame_target_status(
        self,
        video_id: str,
        status: str,
        key_hash: str,
        *,
        lag_frames: Optional[int],
    ) -> None:
        if not key_hash:
            return
        tickets = self._frame_target_log_once.setdefault(video_id, set())
        ticket = f"{status}:{key_hash[:8]}"
        if ticket in tickets:
            return
        tickets.add(ticket)
        frames_display = lag_frames if lag_frames is not None else "?"
        LOGGER.info(
            "targets: %s split=%s id=%s key=%s lag_frames=%s",
            status,
            self.split,
            video_id,
            key_hash[:8],
            frames_display,
        )

    def _mark_bad_clip(self, record_idx: Optional[int], video_id: str, reason: str) -> None:
        canon = canonical_video_id(video_id)
        if canon in self._bad_clips:
            return
        self._bad_clips.add(canon)
        print(f"[data_pass] mark_bad id={canon} reason={reason}", flush=True)
        if record_idx is not None and not self._uses_eval_snapshot():
            self._invalidate_sample_index(record_idx)

    def _mark_frame_target_failure(self, record_idx: int, video_id: str) -> None:
        canon = canonical_video_id(video_id)
        if canon in self._frame_target_failures:
            return
        self._frame_target_failures.add(canon)
        self._mark_bad_clip(record_idx, canon, "build_failed")
    
    def _invalidate_sample_index(self, record_idx: int) -> None:
        if self._uses_eval_snapshot():
            return
        if record_idx >= len(self.samples):
            return
        self._valid_entries = [
            entry for entry in self._valid_entries if entry.record_idx != record_idx
        ]
        self._num_windows = len(self._valid_entries)
        self.eval_indices_snapshot = []

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

    def _materialize_eval_entries_from_labels(
        self,
        max_total: Optional[int] = None,
        target_T: Optional[int] = 96,
        *,
        fps: Optional[float] = None,
        stride: Optional[int] = None,
        tol_s: Optional[float] = None,
        dilate: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Build a list of (video_idx, start_frame) entries for eval ONLY from labels,
        without decoding frames. Use:
          - onset times -> center windows on events (positive-centric)
          - plus a few uniform negatives per video
        Respect clip bounds and T. Return a list and store in self.eval_indices_snapshot.
        If max_total is given, cap the total entries globally (round-robin per video).
        """

        t_start = time.perf_counter()
        self.eval_indices_snapshot = []
        self._eval_snapshot_flags = []
        self._eval_snapshot_by_video = {}
        self._eval_candidates_by_video = {}
        self._eval_materialize_stats = {}

        if not self._uses_eval_snapshot():
            self._last_materialize_duration = time.perf_counter() - t_start
            return []

        fps_val = float(fps) if fps is not None else float(self.decode_fps)
        if not math.isfinite(fps_val) or fps_val <= 0:
            fps_val = max(self.decode_fps, 1.0)
        stride_val = int(stride) if stride is not None else int(self.stride)
        stride_val = max(1, stride_val)
        target_t_val = int(target_T) if target_T is not None else int(self.frames)
        target_t_val = max(1, target_t_val)
        tol_sec = float(tol_s) if tol_s is not None else float(self.frame_targets_cfg.get("tolerance", 0.03) or 0.0)
        tol_sec = max(0.0, tol_sec)
        dilate_frames = int(dilate) if dilate is not None else int(self.frame_targets_cfg.get("dilate_active_frames", 0) or 0)
        dilate_frames = max(0, dilate_frames)

        hop_seconds = stride_val / max(fps_val, 1e-6)
        tol_frames = int(round(tol_sec * fps_val)) if fps_val > 0 else 0
        dilate_frames_abs = dilate_frames * stride_val
        clip_span_frames = max(0, (target_t_val - 1) * stride_val)
        buffer_per_video = 2

        ok_records = sorted(self._audit_ok_records)
        if not ok_records:
            ok_records = [
                idx
                for idx, record in enumerate(self.samples)
                if canonical_video_id(record.get("id", "")) not in self._bad_clips
            ]
        ok_records = [idx for idx in ok_records if 0 <= idx < len(self.samples)]
        ok_records = sorted(
            ok_records,
            key=lambda idx: canonical_video_id(self.samples[idx].get("id", "")),
        )
        if not ok_records:
            self._last_materialize_duration = time.perf_counter() - t_start
            return []

        def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if not intervals:
                return []
            intervals = sorted(intervals, key=lambda it: it[0])
            merged: List[Tuple[int, int]] = [intervals[0]]
            for start, end in intervals[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
            return merged

        per_video_data: Dict[int, Dict[str, Any]] = {}
        total_pos_candidates = 0
        total_neg_candidates = 0

        for record_idx in ok_records:
            record = self.samples[record_idx]
            midi_path = record.get("midi")
            if midi_path is None or not midi_path.exists():
                continue
            labels_tensor = _read_midi_events(midi_path)
            if labels_tensor is None:
                labels_tensor = torch.zeros((0, 3), dtype=torch.float32)
            labels_list = labels_tensor.tolist()
            if isinstance(labels_list, float):
                labels_list = [[float(labels_tensor)]]  # defensive; should not occur

            raw_existing = self._audit_entry_starts.get(record_idx, [])
            existing_starts = [int(start) for start in raw_existing]
            existing_max = max(existing_starts) if existing_starts else 0
            event_times: List[float] = []
            for row in labels_list:
                if not isinstance(row, (list, tuple)) or len(row) < 1:
                    continue
                for val in row[:2]:
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fval):
                        event_times.append(max(0.0, fval))
            max_event_sec = max(event_times) if event_times else 0.0
            max_from_labels = 0
            if fps_val > 0:
                max_frame_est = int(sec_to_frame(max_event_sec, 1.0 / fps_val))
                max_from_labels = max(0, max_frame_est - clip_span_frames)
            max_start_idx = max(existing_max, max_from_labels)
            max_start_idx = max(0, max_start_idx - (max_start_idx % stride_val))

            pos_starts: List[int] = []
            neg_starts: List[int] = []
            seen_starts: Set[int] = set()

            onset_values: Set[float] = set()
            for row in labels_list:
                if not isinstance(row, (list, tuple)) or len(row) < 1:
                    continue
                try:
                    onset_val = float(row[0])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(onset_val):
                    continue
                onset_values.add(max(0.0, onset_val))
            onset_seconds = sorted(onset_values)
            third_offset = int(round(target_t_val / 3.0))
            third_offset = max(0, min(target_t_val - 1, third_offset))
            third_offset_sec = third_offset * hop_seconds

            for onset_sec in onset_seconds:
                start_sec = max(0.0, onset_sec - third_offset_sec)
                start_idx = int(sec_to_frame(start_sec, 1.0 / fps_val)) if fps_val > 0 else 0
                if stride_val > 1:
                    start_idx -= start_idx % stride_val
                start_idx = max(0, start_idx)
                if max_start_idx > 0 and start_idx > max_start_idx:
                    start_idx = max_start_idx
                if stride_val > 1 and start_idx % stride_val != 0:
                    start_idx -= start_idx % stride_val
                start_idx = max(0, start_idx)
                if start_idx not in seen_starts:
                    seen_starts.add(start_idx)
                    pos_starts.append(start_idx)

            active_intervals: List[Tuple[int, int]] = []
            if labels_list:
                for row in labels_list:
                    if not isinstance(row, (list, tuple)) or len(row) < 1:
                        continue
                    onset_val = float(row[0])
                    offset_val = float(row[1]) if len(row) > 1 else onset_val
                    if not math.isfinite(onset_val):
                        continue
                    if not math.isfinite(offset_val):
                        offset_val = onset_val
                    onset_val = max(0.0, onset_val)
                    offset_val = max(onset_val, offset_val)
                    start_frame = int(sec_to_frame(onset_val, 1.0 / fps_val)) if fps_val > 0 else 0
                    end_frame = int(sec_to_frame(offset_val, 1.0 / fps_val)) if fps_val > 0 else start_frame
                    start_frame = max(0, start_frame - tol_frames - dilate_frames_abs)
                    end_frame = end_frame + tol_frames + dilate_frames_abs
                    if end_frame <= start_frame:
                        end_frame = start_frame + stride_val
                    active_intervals.append((start_frame, end_frame))
            merged_intervals = _merge_intervals(active_intervals)

            def _window_overlaps_active(start_idx: int) -> bool:
                if not merged_intervals:
                    return False
                win_start = max(0, start_idx)
                win_end = win_start + clip_span_frames
                for a_start, a_end in merged_intervals:
                    if win_start <= a_end and win_end >= a_start:
                        return True
                return False

            if pos_starts:
                if len(pos_starts) <= 3:
                    neg_slots = 2
                else:
                    neg_slots = 3
            else:
                neg_slots = 1 if max_start_idx == 0 else 2
            neg_slots = max(1, min(3, neg_slots))

            if max_start_idx <= 0:
                candidate_positions = [0]
            else:
                if neg_slots == 1:
                    fractions = [0.5]
                else:
                    fractions = [i / max(neg_slots - 1, 1) for i in range(neg_slots)]
                candidate_positions = [
                    int(round(frac * max_start_idx)) for frac in fractions
                ]

            neighbor_offsets = [0, -stride_val, stride_val, -2 * stride_val, 2 * stride_val]
            for base_idx in candidate_positions:
                base_idx = max(0, min(base_idx, max_start_idx))
                for offset in neighbor_offsets:
                    candidate = base_idx + offset
                    if candidate < 0 or candidate > max_start_idx:
                        continue
                    if stride_val > 1 and candidate % stride_val != 0:
                        candidate -= candidate % stride_val
                    candidate = max(0, candidate)
                    if candidate > max_start_idx:
                        continue
                    if candidate in seen_starts:
                        continue
                    if _window_overlaps_active(candidate):
                        continue
                    seen_starts.add(candidate)
                    neg_starts.append(candidate)
                    break

            if not pos_starts and not neg_starts:
                fallback_start = max_start_idx
                if stride_val > 1 and fallback_start % stride_val != 0:
                    fallback_start -= fallback_start % stride_val
                fallback_start = max(0, fallback_start)
                neg_starts.append(fallback_start)
                seen_starts.add(fallback_start)

            pos_starts = sorted(set(pos_starts))
            neg_starts = sorted(set(neg_starts))

            total_pos_candidates += len(pos_starts)
            total_neg_candidates += len(neg_starts)

            candidate_pool: Set[int] = set(pos_starts) | set(neg_starts) | set(existing_starts)
            per_video_data[record_idx] = {
                "positives": pos_starts,
                "negatives": neg_starts,
                "candidates": sorted(candidate_pool),
                "canonical_id": canonical_video_id(record.get("id", "")),
            }

        materialized_videos: List[int] = []
        for record_idx in ok_records:
            data = per_video_data.get(record_idx)
            if not data:
                continue
            if not data["positives"] and not data["negatives"]:
                continue
            materialized_videos.append(record_idx)

        ok_video_count = len(materialized_videos)
        if ok_video_count == 0:
            self._last_materialize_duration = time.perf_counter() - t_start
            return []

        per_video_cap: Optional[int] = None
        if max_total is not None and max_total > 0:
            per_video_cap = int(math.ceil(max_total / max(1, ok_video_count))) + buffer_per_video
            per_video_cap = max(1, per_video_cap)

        for record_idx in materialized_videos:
            data = per_video_data[record_idx]
            positives_sorted = data["positives"]
            negatives_sorted = data["negatives"]
            ordered_pairs: List[Tuple[int, bool]] = []
            if per_video_cap is not None:
                for start_idx in positives_sorted:
                    if len(ordered_pairs) >= per_video_cap:
                        break
                    ordered_pairs.append((start_idx, True))
                if len(ordered_pairs) < per_video_cap:
                    for start_idx in negatives_sorted:
                        if len(ordered_pairs) >= per_video_cap:
                            break
                        ordered_pairs.append((start_idx, False))
            else:
                ordered_pairs = [(start_idx, True) for start_idx in positives_sorted] + [
                    (start_idx, False) for start_idx in negatives_sorted
                ]
            ordered_pairs.sort(key=lambda item: item[0])
            data["ordered"] = ordered_pairs

        final_entries: List[Tuple[int, int]] = []
        final_flags: List[bool] = []
        snapshot_by_video: Dict[int, List[int]] = {vid: [] for vid in materialized_videos}

        if max_total is not None and max_total > 0:
            iterator_map: Dict[int, Iterator[Tuple[int, bool]]] = {}
            for record_idx in materialized_videos:
                ordered = per_video_data[record_idx].get("ordered", [])
                if ordered:
                    iterator_map[record_idx] = iter(ordered)
            while iterator_map and len(final_entries) < max_total:
                for record_idx in list(iterator_map.keys()):
                    if len(final_entries) >= max_total:
                        break
                    iterator = iterator_map[record_idx]
                    try:
                        start_idx, is_positive = next(iterator)
                    except StopIteration:
                        del iterator_map[record_idx]
                        continue
                    final_entries.append((record_idx, start_idx))
                    final_flags.append(is_positive)
                    snapshot_by_video.setdefault(record_idx, []).append(start_idx)
        else:
            for record_idx in materialized_videos:
                for start_idx, is_positive in per_video_data[record_idx].get("ordered", []):
                    final_entries.append((record_idx, start_idx))
                    final_flags.append(is_positive)
                    snapshot_by_video.setdefault(record_idx, []).append(start_idx)

        if not final_entries:
            self._last_materialize_duration = time.perf_counter() - t_start
            return []

        pos_total = sum(1 for flag in final_flags if flag)
        neg_total = len(final_flags) - pos_total
        used_video_count = sum(1 for starts in snapshot_by_video.values() if starts)
        avg_per_video = (len(final_entries) / used_video_count) if used_video_count else 0.0

        self.eval_indices_snapshot = final_entries
        self._eval_snapshot_flags = final_flags
        self._eval_snapshot_by_video = {
            vid: starts for vid, starts in snapshot_by_video.items() if starts
        }
        self._eval_candidates_by_video = {
            vid: per_video_data.get(vid, {}).get("candidates", [])
            for vid in self._eval_snapshot_by_video.keys()
        }
        self._eval_materialize_stats = {
            "positives": pos_total,
            "negatives": neg_total,
            "videos": used_video_count,
            "avg_per_video": avg_per_video,
            "pos_candidates": total_pos_candidates,
            "neg_candidates": total_neg_candidates,
            "duration": self._last_materialize_duration,
        }

        self._last_materialize_duration = time.perf_counter() - t_start

        print(
            f"[dataset] entries built: total={len(final_entries)} (avg per video ≈ {avg_per_video:.1f}, pos≈{pos_total}, neg≈{neg_total})",
            flush=True,
        )
        return final_entries

    def materialize_eval_entries_from_labels(
        self,
        *,
        max_total: Optional[int] = None,
        target_T: Optional[int] = None,
        fps: Optional[float] = None,
        stride: Optional[int] = None,
        tol_s: Optional[float] = None,
        dilate: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Public wrapper used by evaluation/calibration scripts to rebuild the eval snapshot
        without decoding frames. Mirrors the fast-eval materialization path.
        """

        max_total_val = max_total if max_total is not None else self.args_max_clips_or_None
        target_T_val = target_T if target_T is not None else self.frames
        fps_val = fps if fps is not None else self.decode_fps
        stride_val = stride if stride is not None else self.stride
        tol_val = tol_s if tol_s is not None else self.frame_targets_cfg.get("tolerance")
        dilate_val = dilate if dilate is not None else self.frame_targets_cfg.get("dilate_active_frames")

        return self._materialize_eval_entries_from_labels(
            max_total=max_total_val,
            target_T=target_T_val,
            fps=fps_val,
            stride=stride_val,
            tol_s=tol_val,
            dilate=dilate_val,
        )

    def _record_has_valid_window(self, record_idx: int) -> bool:
        record = self.samples[record_idx]
        video_id = canonical_video_id(record.get("id", ""))
        if video_id in self._bad_clips:
            return False
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
        canon_id = canonical_video_id(video_id)

        if canon_id in self._bad_clips:
            return [], "bad_clip", 0, 0, None

        if video_path is None or not video_path.exists():
            return [], "no_file", 0, 0, None

        if midi_path is None or not midi_path.exists():
            self._log_missing_labels_once(video_path)
            self._mark_bad_clip(record_idx, video_id, "no_labels")
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
            return [], "build_failed", 0, 0, lag_ms_value

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

        status = "ok" if entries else "build_failed"
        if status in {"no_labels", "build_failed"}:
            reason = "no_labels" if status == "no_labels" else "build_failed"
            self._mark_bad_clip(record_idx, video_id, reason)
        return entries, status, events_on_total, events_off_total, lag_ms_value

    def _rebuild_valid_index_cache(self, *, log_summary: bool) -> None:
        self._valid_entries = []
        self._audit_ok_records.clear()
        self._audit_entry_starts = {}
        total = len(self.samples)
        skipped = 0
        ok_videos = 0
        bad_videos = 0
        audit_start = time.perf_counter()
        if log_summary:
            print(f"[dataset] audit start (videos={total})", flush=True)
        if self.split == "train":
            for idx in range(total):
                record = self.samples[idx]
                vid_id = canonical_video_id(record.get("id", ""))
                per_start = time.perf_counter()
                has_window = self._record_has_valid_window(idx)
                elapsed = time.perf_counter() - per_start
                status_label = "ok" if has_window else "no_window"
                if log_summary:
                    print(
                        f"[dataset] audit clip={vid_id} elapsed={elapsed:.2f}s status={status_label}",
                        flush=True,
                    )
                if elapsed > self._audit_timeout_sec:
                    if log_summary:
                        print(f"[dataset] audit timeout id={vid_id}; mark_bad", flush=True)
                    self._mark_bad_clip(idx, vid_id, "audit_timeout")
                    skipped += 1
                    bad_videos += 1
                    continue
                if has_window:
                    self._valid_entries.append(WindowEntry(idx, None, True))
                    ok_videos += 1
                    if self._uses_eval_snapshot():
                        self._audit_ok_records.add(idx)
                else:
                    skipped += 1
                    bad_videos += 1
        else:
            for idx, record in enumerate(self.samples):
                vid_id = canonical_video_id(record.get("id", ""))
                if vid_id in self._bad_clips:
                    skipped += 1
                    bad_videos += 1
                    continue
                per_start = time.perf_counter()
                entries, status, events_on, events_off, lag_ms = self._plan_eval_entries(idx)
                elapsed = time.perf_counter() - per_start
                if log_summary:
                    print(
                        f"[dataset] audit clip={vid_id} elapsed={elapsed:.2f}s status={status}",
                        flush=True,
                    )
                if elapsed > self._audit_timeout_sec:
                    if log_summary:
                        print(f"[dataset] audit timeout id={vid_id}; mark_bad", flush=True)
                    self._mark_bad_clip(idx, vid_id, "audit_timeout")
                    skipped += 1
                    bad_videos += 1
                    continue
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
                    ok_videos += 1
                    self._audit_ok_records.add(idx)
                else:
                    skipped += 1
                    bad_videos += 1

        starts_map: Dict[int, List[int]] = {}
        for entry in self._valid_entries:
            if entry.start_idx is None:
                continue
            starts_map.setdefault(entry.record_idx, []).append(int(entry.start_idx))
        self._audit_entry_starts = {
            rec_idx: sorted({int(start) for start in starts})
            for rec_idx, starts in starts_map.items()
        }

        self._num_windows = len(self._valid_entries)
        if log_summary:
            elapsed_total = time.perf_counter() - audit_start
            print(
                f"[dataset] audit done ok={ok_videos} bad={bad_videos} elapsed={elapsed_total:.2f}s",
                flush=True,
            )
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
        if self._uses_eval_snapshot():
            try:
                self._materialize_eval_entries_from_labels(
                    max_total=self.args_max_clips_or_None,
                    target_T=self.frames,
                    fps=self.decode_fps,
                    stride=self.stride,
                    tol_s=self.frame_targets_cfg.get("tolerance"),
                    dilate=self.frame_targets_cfg.get("dilate_active_frames"),
                )
            except Exception as exc:
                LOGGER.warning("Failed to materialize eval entries from labels: %s", exc)


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))
    
    only_video_cfg = dcfg.get("only_video")
    avlag_disabled_cfg = bool(dcfg.get("avlag_disabled", False))

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
        only_video=only_video_cfg,
        avlag_disabled=avlag_disabled_cfg,
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

    if only_video_cfg and not getattr(dataset, "_only_filter_applied", False):
        only_canon = canonical_video_id(only_video_cfg)
        if not dataset.filter_to_video(only_canon):
            LOGGER.warning("[PianoYT] --only filter skipped; id=%s not found", only_canon)

    max_clips = dcfg.get("max_clips")
    dataset.limit_max_clips(max_clips if isinstance(max_clips, int) else None)
    dataset.max_clips = max_clips
    dataset.args_max_clips_or_None = max_clips if isinstance(max_clips, int) else None

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

    num_workers = int(dcfg.get("num_workers", 0))
    pin_memory = bool(dcfg.get("pin_memory", False))
    persistent_workers_cfg = bool(dcfg.get("persistent_workers", False))
    persistent_workers = persistent_workers_cfg if num_workers > 0 else False

    loader = DataLoader(
        dataset,
        batch_size=int(dcfg.get("batch_size", 2)),
        shuffle=bool(dcfg.get("shuffle", True)) if split == "train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate,
        persistent_workers=persistent_workers,
    )
    return loader


__all__ = ["PianoYTDataset", "make_dataloader"]
