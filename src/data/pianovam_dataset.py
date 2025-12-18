"""Purpose:
    Provide a PianoVAM loader that mirrors the PianoYT runtime surface so
    pipelines can switch datasets via configuration only.  The loader reuses
    the PianoYT implementation for sampling, target construction, and optional
    augmentations while resolving a PianoVAM-specific root directory.

Key Functions/Classes:
    - PianoVAMDataset: Thin subclass of :class:`PianoYTDataset` that points to
      the local PianoVAM folder (PianoYT-style layout on disk).
    - make_dataloader(): Factory matching :func:`pianoyt_dataset.make_dataloader`
      but defaulting to PianoVAM roots and leaving the rest of the interface
      unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from utils.determinism import DEFAULT_SEED, make_loader_components
from utils.frame_targets import resolve_frame_target_spec, resolve_soft_target_config
from utils.identifiers import canonical_video_id
from utils.registration_refinement import RegistrationRefiner

import src.data.pianoyt_dataset as yt
from src.data.pianoyt_dataset import SampleBuildResult
from .sampler_utils import build_onset_balanced_sampler
from hand_labels import (
    EventHandLabelConfig,
    build_event_hand_labels,
    key_centers_from_geometry,
    load_pianovam_hand_landmarks,
    map_landmarks_to_canonical,
)
from hand_labels.hand_reach import compute_hand_reach

LOGGER = logging.getLogger(__name__)


def _safe_expanduser(path: os.PathLike[str] | str) -> Path:
    """Expand user safely; mirrors PianoYT helper without importing it."""

    candidate = Path(path)
    try:
        return candidate.expanduser()
    except RuntimeError:
        return candidate


def _expand_root(root_dir: Optional[str]) -> Path:
    """Resolve the PianoVAM root directory with environment fallbacks."""

    candidates = []
    if root_dir:
        candidates.append(_safe_expanduser(os.path.expandvars(str(root_dir))))

    env_base = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env_base:
        env_base_path = _safe_expanduser(env_base)
        candidates.extend(
            [
                env_base_path / "PianoVAM_v1.0",
                env_base_path / "PianoVAM",
            ]
        )

    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "PianoVAM_v1.0",
            project_root / "data" / "PianoVAM",
        ]
    )
    candidates.extend(
        [
            _safe_expanduser("~/datasets/PianoVAM_v1.0"),
            _safe_expanduser("~/datasets/PianoVAM"),
        ]
    )

    for cand in candidates:
        if cand and cand.exists():
            return cand

    msg = (
        "Unable to locate PianoVAM root. Set dataset.root_dir or define "
        "TIVIT_DATA_DIR/DATASETS_HOME to point at PianoVAM_v1.0 layout."
    )
    LOGGER.error(msg)
    raise FileNotFoundError(msg)


def _load_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    meta_path = root / "metadata_v2.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[PianoVAM] metadata_v2.json missing at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"[PianoVAM] metadata_v2.json is not a dict: {type(raw)}")
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


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
    """Compute (min_y, max_y, min_x, max_x) from Point_LT/RT/RB/LB."""

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
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    try:
        return (int(round(min_y)), int(round(max_y)), int(round(min_x)), int(round(max_x)))
    except Exception:
        return None


def _split_matches(requested: str, meta_split: str) -> bool:
    req = requested.lower()
    meta = meta_split.lower()
    if req in {"val", "valid", "validation"}:
        return meta in {"val", "valid", "validation"}
    if req in {"train", "ext-train", "ext_train"}:
        return meta in {"train", "ext-train", "ext_train"}
    if req in {"test"}:
        return meta == "test"
    if req.startswith("special"):
        return meta.startswith("special")
    return req == meta


def _resolve_media_paths_local(root: Path, record_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return video, midi, tsv paths for ``record_id``."""

    vid = root / "Video" / f"{record_id}.mp4"
    midi_mid = root / "MIDI" / f"{record_id}.mid"
    midi_midi = root / "MIDI" / f"{record_id}.midi"
    midi = midi_mid if midi_mid.exists() else midi_midi if midi_midi.exists() else None
    tsv = root / "TSV" / f"{record_id}.tsv"
    if not vid.exists():
        vid = None
    if tsv.exists():
        tsv_path: Optional[Path] = tsv
    else:
        tsv_path = None
    # If MIDI is absent but TSV exists, use TSV as the label source.
    if midi is None and tsv_path is not None:
        midi = tsv_path
    return vid, midi, tsv_path


_LABEL_FALLBACK_LOGGED: set[str] = set()


def _read_tsv_events(tsv_path: Path) -> torch.Tensor:
    """Parse PianoVAM TSV and return (N,3) tensor [onset_sec, key_offset_sec, pitch]."""

    if not tsv_path or not tsv_path.exists():
        return torch.zeros((0, 3), dtype=torch.float32)

    rows: list[tuple[float, float, float]] = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                # fallback to whitespace split if tabs missing
                parts = line.split()
            if len(parts) < 5:
                continue
            try:
                onset = float(parts[0])
                key_offset = float(parts[1])  # use key_offset as the offset value
                note = float(parts[3])
            except (TypeError, ValueError):
                continue
            if onset < 0 or key_offset <= onset:
                continue
            rows.append((onset, key_offset, note))

    if not rows:
        return torch.zeros((0, 3), dtype=torch.float32)
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return torch.tensor(rows, dtype=torch.float32)


class PianoVAMDataset(yt.PianoYTDataset):
    """PianoVAM dataset that mirrors PianoYT surface but uses metadata_v2 layout."""

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
        dataset_cfg = dict(dataset_cfg or {})
        resolved_root = _expand_root(root_dir or (dataset_cfg or {}).get("root_dir"))
        metadata = _load_metadata(resolved_root)

        # Build split list and ID mapping from metadata_v2.
        id_map: Dict[str, str] = {}
        split_ids: list[str] = []
        crop_map: Dict[str, Tuple[int, int, int, int]] = {}
        for entry in metadata.values():
            rec_id = entry.get("record_time") or entry.get("id")
            meta_split = entry.get("split")
            if not rec_id or not meta_split:
                continue
            if not _split_matches(split, str(meta_split)):
                continue
            canon = canonical_video_id(rec_id)
            id_map[canon] = rec_id
            split_ids.append(canon)
            crop_box = _crop_from_points(entry)
            if crop_box is not None:
                crop_map[canon] = crop_box
        if not split_ids:
            raise RuntimeError(f"[PianoVAM] No entries for split '{split}' in metadata_v2.json")

        # Pre-compute TSV paths per canonical id for later attachment to samples.
        self._tsv_by_canon: Dict[str, Optional[Path]] = {}
        for canon_id, rec_id in id_map.items():
            _, _, tsv_path = _resolve_media_paths_local(resolved_root, rec_id)
            self._tsv_by_canon[canon_id] = tsv_path
        self._hands_by_canon: Dict[str, Optional[Path]] = {}
        for canon_id, rec_id in id_map.items():
            hand_path = resolved_root / "Handskeleton" / f"{rec_id}.json"
            self._hands_by_canon[canon_id] = hand_path if hand_path.exists() else None

        # Monkeypatch PianoYT split/media resolution to respect PianoVAM layout.
        orig_expand_root = yt._expand_root
        orig_read_split_ids = yt._read_split_ids
        orig_resolve_media_paths = yt._resolve_media_paths
        orig_read_midi_events = yt._read_midi_events
        orig_load_metadata = yt._load_metadata

        def _pv_expand_root(_: Optional[str]) -> Path:
            return resolved_root

        def _pv_read_split_ids(root: Path, split_name: str):
            return split_ids

        def _pv_resolve_media_paths(root: Path, split_name: str, video_id: str):
            rec_id = id_map.get(canonical_video_id(video_id), video_id)
            vid_path, midi_path, tsv_path = _resolve_media_paths_local(root, rec_id)
            label_path = tsv_path if tsv_path is not None else midi_path
            if label_path is midi_path and tsv_path is None and midi_path is not None:
                canon_id = canonical_video_id(rec_id)
                if canon_id not in _LABEL_FALLBACK_LOGGED:
                    LOGGER.info("[PianoVAM] Using MIDI labels for id=%s (TSV missing)", canon_id)
                    _LABEL_FALLBACK_LOGGED.add(canon_id)
            return vid_path, label_path

        def _pv_read_midi_events(path: Path) -> torch.Tensor:
            if path.suffix.lower() == ".tsv":
                return _read_tsv_events(path)
            return orig_read_midi_events(path)

        def _pv_load_metadata(_: Path):
            return {k: v for k, v in crop_map.items()}

        yt._expand_root = _pv_expand_root  # type: ignore
        yt._read_split_ids = _pv_read_split_ids  # type: ignore
        yt._resolve_media_paths = _pv_resolve_media_paths  # type: ignore
        yt._read_midi_events = _pv_read_midi_events  # type: ignore
        yt._load_metadata = _pv_load_metadata  # type: ignore
        try:
            super().__init__(
                root_dir=str(resolved_root),
                split=split,
                frames=frames,
                stride=stride,
                resize=resize,
                tiles=tiles,
                channels=channels,
                normalize=normalize,
                manifest=manifest,
                decode_fps=decode_fps,
                dataset_cfg=dataset_cfg,
                full_cfg=full_cfg,
                only_video=only_video,
                avlag_disabled=avlag_disabled,
            )
        finally:
            yt._expand_root = orig_expand_root
            yt._read_split_ids = orig_read_split_ids
            yt._resolve_media_paths = orig_resolve_media_paths
            # Keep _read_midi_events patched so later label reads during training
            # continue to parse TSV files correctly. The wrapper defers to the
            # original reader for real MIDI files, so this is safe to leave in place.
            yt._load_metadata = orig_load_metadata

        self.dataset_name = "pianovam"
        self.root = Path(resolved_root)
        self.hand_supervision_cfg: Dict[str, Any] = dict(dataset_cfg.get("hand_supervision", {}) or {})

    def _build_sample(  # type: ignore[override]
        self,
        record_idx: int,
        dataset_index: int,
        *,
        preferred_start_idx: Optional[int] = None,
        audit: bool = False,
    ) -> SampleBuildResult:
        hand_cfg = self.hand_supervision_cfg
        result = super()._build_sample(
            record_idx,
            dataset_index,
            preferred_start_idx=preferred_start_idx,
            audit=audit,
        )
        if result.sample is not None:
            record = self.samples[record_idx]
            canon = canonical_video_id(record.get("id", ""))
            tsv_path = self._tsv_by_canon.get(canon)
            if tsv_path is not None:
                result.sample["tsv_path"] = str(tsv_path)
            hand_path = self._hands_by_canon.get(canon)
            if hand_path is not None and hand_cfg.get("enable", False) and not audit:
                try:
                    self._attach_hand_targets(
                        sample=result.sample,
                        hand_json=hand_path,
                        video_id=canon,
                        clip_start_sec=float(result.start_idx) / max(float(self.decode_fps), 1e-6),
                        T=self.frames,
                        stride=self.stride,
                        decode_fps=self.decode_fps,
                    )
                except Exception as exc:  # pragma: no cover - fail soft
                    LOGGER.debug("[PianoVAM] hand label attach failed for %s: %s", canon, exc)
        return result

    def _attach_hand_targets(
        self,
        *,
        sample: Dict[str, Any],
        hand_json: Path,
        video_id: str,
        clip_start_sec: float,
        T: int,
        stride: int,
        decode_fps: float,
    ) -> None:
        """Attach hand labels/reach to a sample using Handskeleton JSON."""

        cfg = self.hand_supervision_cfg
        min_conf = float(cfg.get("min_confidence", 0.05))
        time_tol = float(cfg.get("time_tolerance", stride / max(decode_fps, 1e-6)))
        max_dx = float(cfg.get("max_dx", 0.12))
        min_pts = int(cfg.get("min_points", 4))

        # Load hands aligned to clip grid (native coords).
        aligned = load_pianovam_hand_landmarks(
            hand_json,
            clip_start_sec=clip_start_sec,
            frames=T,
            stride=stride,
            decode_fps=decode_fps,
            min_confidence=min_conf,
            time_tolerance=time_tol,
        )

        reg_refiner = getattr(self, "registration_refiner", None)
        reg_payload = None
        if isinstance(reg_refiner, RegistrationRefiner):
            reg_payload = reg_refiner.get_cache_entry_payload(video_id)

        # Estimate source_hw for crop translation: prefer registration source, else video tensor size.
        raw_hw = sample["video"].shape[-2:] if isinstance(sample.get("video"), torch.Tensor) else None
        if raw_hw:
            source_hw = (int(raw_hw[0]), int(raw_hw[1]))
        elif reg_payload and isinstance(reg_payload.get("source_hw"), list) and len(reg_payload["source_hw"]) >= 2:
            source_hw = (int(reg_payload["source_hw"][0]), int(reg_payload["source_hw"][1]))
        else:
            source_hw = (aligned.landmarks.shape[-2], aligned.landmarks.shape[-3]) if aligned.landmarks.dim() >= 3 else (0, 0)

        crop_meta = self.metadata.get(video_id)
        canonical = map_landmarks_to_canonical(
            aligned,
            registration=reg_payload,
            source_hw=source_hw,
            crop_meta=crop_meta,
        )

        key_centers = None
        if reg_payload is not None:
            key_centers = key_centers_from_geometry(reg_payload.get("geometry_meta"))

        labels_tensor = sample.get("labels")
        if key_centers is not None and torch.is_tensor(labels_tensor) and labels_tensor.numel() > 0:
            onsets = labels_tensor[:, 0]
            pitches = labels_tensor[:, 2].to(torch.int64) - 21  # MIDI 21->0
            valid_keys = (pitches >= 0) & (pitches < key_centers.numel())
            onsets = onsets[valid_keys]
            pitches = pitches[valid_keys]
            cfg_obj = EventHandLabelConfig(
                time_tolerance=time_tol,
                max_dx=max_dx,
                min_points=min_pts,
                unknown_class=0,
            )
            evt_labels = build_event_hand_labels(
                onsets_sec=onsets,
                key_indices=pitches,
                key_centers_norm=key_centers,
                frame_times=aligned.frame_times,
                canonical=canonical,
                config=cfg_obj,
            )

            # Project event labels to frame grid: closest frame wins; conflicts drop mask.
            frame_lab = torch.zeros((T,), dtype=torch.long)
            frame_mask = torch.zeros((T,), dtype=torch.bool)
            for onset, lbl, valid in zip(onsets.tolist(), evt_labels.labels.tolist(), evt_labels.mask.tolist()):
                if not valid:
                    continue
                idx = int(torch.abs(aligned.frame_times - float(onset)).argmin().item())
                if idx < 0 or idx >= T:
                    continue
                if not frame_mask[idx]:
                    frame_lab[idx] = int(lbl)
                    frame_mask[idx] = True
                elif frame_lab[idx] != int(lbl):
                    frame_lab[idx] = 0
                    frame_mask[idx] = False

            sample["hand_frame"] = frame_lab
            sample["hand_frame_mask"] = frame_mask
            sample["hand_coverage"] = evt_labels.coverage
            sample["hand_events"] = evt_labels.labels
            sample["hand_event_mask"] = evt_labels.mask

        reach_cfg = cfg.get("reach", {}) or {}
        if reach_cfg.get("enable", False) and key_centers is not None:
            reach = compute_hand_reach(
                canonical,
                key_centers_norm=key_centers,
                radius=float(reach_cfg.get("radius", 0.12)),
                dilate=float(reach_cfg.get("dilate_margin", 0.02)),
                min_points=int(reach_cfg.get("min_points", min_pts)),
            )
            sample["hand_reach"] = reach.reach
            sample["hand_reach_valid"] = reach.valid
            sample["hand_reach_coverage"] = reach.coverage


def make_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    drop_last: bool = False,
    *,
    seed: Optional[int] = None,
):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))

    only_video_cfg = dcfg.get("only_video")
    avlag_disabled_cfg = bool(dcfg.get("avlag_disabled", False))

    resolved_root = _expand_root(dcfg.get("root_dir"))
    dataset_cfg = dict(dcfg)
    dataset_cfg["root_dir"] = str(resolved_root)

    dataset = PianoVAMDataset(
        root_dir=str(resolved_root),
        split=split,
        frames=int(dataset_cfg.get("frames", 32)),
        stride=stride,
        resize=tuple(dataset_cfg.get("resize", [224, 224])),
        tiles=int(dataset_cfg.get("tiles", 3)),
        channels=int(dataset_cfg.get("channels", 3)),
        normalize=bool(dataset_cfg.get("normalize", True)),
        manifest=manifest_path,
        decode_fps=decode_fps,
        dataset_cfg=dataset_cfg,
        full_cfg=cfg,
        only_video=only_video_cfg,
        avlag_disabled=avlag_disabled_cfg,
    )
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    soft_cfg_raw = training_cfg.get("soft_targets") if isinstance(training_cfg, Mapping) else None
    dataset.soft_target_cfg = resolve_soft_target_config(soft_cfg_raw)

    include_low = bool(dataset_cfg.get("include_low_res", False))
    excluded_ids = set()
    if not include_low:
        excluded_ids = yt._read_excluded(dataset.root, dataset_cfg.get("excluded_list"))
    dataset.configure(
        include_low_res=include_low,
        excluded_ids=excluded_ids,
        apply_crop=bool(dataset_cfg.get("apply_crop", True)),
        crop_rescale=str(dataset_cfg.get("crop_rescale", "auto")),
    )

    if only_video_cfg and not getattr(dataset, "_only_filter_applied", False):
        only_canon = canonical_video_id(only_video_cfg)
        if not dataset.filter_to_video(only_canon):
            LOGGER.warning("[PianoVAM] --only filter skipped; id=%s not found", only_canon)

    max_clips = dataset_cfg.get("max_clips")
    dataset.limit_max_clips(max_clips if isinstance(max_clips, int) else None)
    dataset.max_clips = max_clips
    dataset.args_max_clips_or_None = max_clips if isinstance(max_clips, int) else None

    dataset.annotations_root = dataset_cfg.get("annotations_root")
    dataset.label_format = dataset_cfg.get("label_format", "midi")
    dataset.label_targets = dataset_cfg.get("label_targets", ["pitch", "onset", "offset", "hand", "clef"])
    dataset.require_labels = bool(dataset_cfg.get("require_labels", False))
    dataset.frame_targets_cfg = dict(dataset_cfg.get("frame_targets", {}) or {})
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
            raise RuntimeError("Empty batch supplied to PianoVAM collate function")

        dims = vids[0].dim()
        if any(v.dim() != dims for v in vids):
            raise RuntimeError("Inconsistent tensor ranks in PianoVAM batch")

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

    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    persistent_workers_cfg = bool(dataset_cfg.get("persistent_workers", False))
    persistent_workers = persistent_workers_cfg if num_workers > 0 else False
    prefetch_factor_cfg = dataset_cfg.get("prefetch_factor")
    prefetch_factor: Optional[int] = None
    if num_workers > 0 and prefetch_factor_cfg is not None:
        try:
            prefetch_factor_candidate = int(prefetch_factor_cfg)
        except (TypeError, ValueError):
            prefetch_factor_candidate = None
        if prefetch_factor_candidate is not None and prefetch_factor_candidate > 0:
            prefetch_factor = prefetch_factor_candidate

    base_seed = seed if seed is not None else getattr(dataset, "data_seed", None)
    if base_seed is None:
        base_seed = DEFAULT_SEED
    generator, worker_init_fn = make_loader_components(
        int(base_seed), namespace=f"{dataset.__class__.__name__}:{split}"
    )

    sampler = build_onset_balanced_sampler(
        dataset,
        dataset_cfg.get("sampler"),
        base_seed=int(base_seed),
    )
    shuffle_flag = bool(dataset_cfg.get("shuffle", True)) if split == "train" else False
    if sampler is not None:
        shuffle_flag = False

    loader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(dataset_cfg.get("batch_size", 2)),
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader


__all__ = ["PianoVAMDataset", "make_dataloader"]
