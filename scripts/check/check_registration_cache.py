"""Purpose:
    Inspect the registration refinement cache outside of pytest so we can
    confirm which clips have usable geometry before firing up longer jobs.

Key Functions/Classes:
    - parse_args(): CLI argument handling for cache path, canonical HW, and IDs.
    - log_entry(): Emits per-video cache details (status/target_hw/errors).
    - main(): Instantiates RegistrationRefiner and prints the overall summary.

CLI:
    python scripts/check/check_registration_cache.py --ids video_106 video_132
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from utils.identifiers import canonical_video_id
from utils.registration_refinement import RegistrationRefiner, resolve_registration_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print registration cache summary and entry details.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Path to reg_refined.json cache (default: repository reg_refined.json)",
    )
    parser.add_argument(
        "--canonical-hw",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(145, 800),
        help="Canonical (H, W) resolution expected by RegistrationRefiner (default: %(default)s)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=("video_106", "video_132"),
        help="Video IDs to inspect inside the cache (default: %(default)s)",
    )
    parser.add_argument(
        "--preview-keys",
        type=int,
        default=10,
        help="Number of cache keys to preview when printing the cache summary (default: %(default)s)",
    )
    return parser.parse_args()


def log_entry(refiner: RegistrationRefiner, video_id: str) -> None:
    canon = canonical_video_id(video_id)
    entry = refiner._cache.get(canon)  # type: ignore[attr-defined]
    if entry is None:
        print(f"[cache] entry for {canon}: MISSING")
        return
    print(
        "[cache] entry for {canon}: status={status} target_hw={target_hw} source_hw={source_hw} "
        "keyboard_height={height:.1f} err_after={err:.2f} frames={frames}".format(
            canon=canon,
            status=entry.status,
            target_hw=tuple(entry.target_hw),
            source_hw=tuple(entry.source_hw),
            height=float(entry.keyboard_height),
            err=float(entry.err_after),
            frames=int(entry.frames),
        )
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cache_path = resolve_registration_cache_path(args.cache)
    refiner = RegistrationRefiner(
        args.canonical_hw,
        cache_path=cache_path,
    )
    total_entries = len(refiner._cache)  # type: ignore[attr-defined]
    print(f"[cache] path={refiner.cache_path}")
    print(f"[cache] total_entries={total_entries}")
    refiner.log_cache_summary(max_keys=max(int(args.preview_keys), 0))
    ids: Sequence[str] = args.ids or ()
    for vid in ids:
        log_entry(refiner, vid)


if __name__ == "__main__":
    main()
