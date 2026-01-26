#!/usr/bin/env python3
"""Precompute registration refinement cache for all dataset splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Sequence, Type

from tivit.core.config import resolve_config_chain
from tivit.data.roi.keyboard_roi import RegistrationRefiner, resolve_registration_cache_path
from tivit.data.targets.identifiers import canonical_video_id


def _load_config(path: Path) -> Mapping[str, Any]:
    return resolve_config_chain([path], default_base=None)


def _resolve_dataset_class(name: str) -> Type[Any]:
    key = (name or "").strip().lower()
    if key == "omaps":
        from tivit.data.datasets.omaps_impl import OMAPSDataset

        return OMAPSDataset
    if key == "pianoyt":
        from tivit.data.datasets.pianoyt_impl import PianoYTDataset

        return PianoYTDataset
    if key == "pianovam":
        from tivit.data.datasets.pianovam_impl import PianoVAMDataset

        return PianoVAMDataset
    raise ValueError(f"Unsupported dataset name '{name}'. Expected omaps, pianoyt, or pianovam.")


def _collect_splits(dataset_cfg: Mapping[str, Any]) -> list[str]:
    splits: MutableSequence[str] = []

    def _add(val: Any) -> None:
        if isinstance(val, str) and val and val not in splits:
            splits.append(val)

    for key in ("split_train", "split_val", "split_test"):
        _add(dataset_cfg.get(key))

    manifest = dataset_cfg.get("manifest")
    if isinstance(manifest, Mapping):
        for key in manifest.keys():
            _add(str(key))

    _add(dataset_cfg.get("split"))
    return list(splits) if splits else ["train"]


def _resolve_canonical_hw(dataset_cfg: Mapping[str, Any]) -> tuple[int, int]:
    canonical = dataset_cfg.get("canonical_hw") or dataset_cfg.get("resize")
    if isinstance(canonical, Sequence) and len(canonical) >= 2:
        return (int(canonical[0]), int(canonical[1]))
    raise ValueError("dataset.canonical_hw (or resize) must supply (H, W)")


def _resolve_sample_frames(dataset_cfg: Mapping[str, Any], reg_cfg: Mapping[str, Any]) -> int:
    frames = dataset_cfg.get("frames", 96)
    return int(reg_cfg.get("sample_frames", frames))


def _print_summary(*, split: str, phase: str, processed: int, fallback: int, cache_path: Path) -> None:
    print(
        "[reg_refined] split={split} phase={phase} processed={processed} fallback={fallback} cache={cache}".format(
            split=split,
            phase=phase,
            processed=processed,
            fallback=fallback,
            cache=str(cache_path),
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate reg_refined.json for all dataset splits.",
    )
    parser.add_argument("config", help="Path to dataset config YAML.")
    parser.add_argument(
        "--only-ids",
        help="Comma-separated list of video IDs to process (optional).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and ROI artifact dumps.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    logging.getLogger("tivit.data.roi.keyboard_roi").setLevel(log_level)

    cfg = _load_config(Path(args.config))
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("Config must contain a 'dataset' mapping.")

    dataset_name = str(dataset_cfg.get("name", "omaps"))
    dataset_cls = _resolve_dataset_class(dataset_name)
    splits = _collect_splits(dataset_cfg)

    canonical_hw = _resolve_canonical_hw(dataset_cfg)
    reg_cfg = dataset_cfg.get("registration") or {}
    if not isinstance(reg_cfg, Mapping):
        reg_cfg = {}
    cache_path = resolve_registration_cache_path(reg_cfg.get("cache_path"))
    sample_frames = _resolve_sample_frames(dataset_cfg, reg_cfg)

    refiner = RegistrationRefiner(canonical_hw, cache_path=cache_path, sample_frames=sample_frames)

    only_ids = set()
    if args.only_ids:
        for part in str(args.only_ids).split(","):
            token = part.strip()
            if token:
                only_ids.add(canonical_video_id(token))

    total_processed = 0
    total_fallback = 0
    warmup_target = 2

    for split in splits:
        dataset = dataset_cls(cfg, split=split, full_cfg=cfg)
        entries = list(getattr(dataset, "entries", []))
        if not entries:
            _print_summary(split=split, phase="empty", processed=0, fallback=0, cache_path=cache_path)
            continue

        processed = 0
        fallback = 0
        warmup_count = min(warmup_target, len(entries))

        for idx, entry in enumerate(entries):
            canon_id = canonical_video_id(entry.video_id)
            if only_ids and canon_id not in only_ids:
                continue
            crop_meta = None
            metadata = entry.metadata if isinstance(entry.metadata, Mapping) else {}
            if isinstance(metadata, Mapping):
                crop_meta = metadata.get("crop")
            entry_fallback = False
            try:
                result = refiner.refine(
                    video_id=entry.video_id,
                    video_path=entry.video_path,
                    crop_meta=crop_meta,
                    debug_context={"split": split, "dataset_index": idx},
                )
                if only_ids:
                    only_ids.discard(canon_id)
                if str(result.status).startswith("fallback"):
                    fallback += 1
                    entry_fallback = True
            except Exception as exc:
                logging.warning("refine failed: split=%s id=%s err=%s", split, entry.video_id, exc)
                fallback += 1
                entry_fallback = True
            processed += 1
            total_processed += 1
            if entry_fallback:
                total_fallback += 1
            if processed == warmup_count:
                _print_summary(
                    split=split,
                    phase="warmup",
                    processed=processed,
                    fallback=fallback,
                    cache_path=cache_path,
                )

        if only_ids:
            logging.warning("Missing requested IDs after split=%s: %s", split, sorted(only_ids))
            break

        _print_summary(
            split=split,
            phase="full",
            processed=processed,
            fallback=fallback,
            cache_path=cache_path,
        )

    print(
        "[reg_refined] split=all phase=done processed={processed} fallback={fallback} cache={cache}".format(
            processed=total_processed,
            fallback=total_fallback,
            cache=str(cache_path),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
