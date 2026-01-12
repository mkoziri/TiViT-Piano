"""Purpose:
    Normalise dataset-specific identifiers so caches, metadata, and file paths
    share a consistent `video_###` naming scheme.

Key Functions/Classes:
    - canonical_video_id(): Converts arbitrary strings/paths to canonical IDs.
    - id_aliases(): Produces canonical plus legacy alias variations.
    - log_legacy_id_hit(): Emits one-time compat logs when legacy IDs resolve.

CLI:
    Not a standalone CLI; imported by caching and dataset utilities.
"""

from __future__ import annotations

import atexit
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

LOGGER = logging.getLogger(__name__)
_LEGACY_HITS: set[str] = set()
_LEGACY_TOTAL = 0
_LEGACY_UNIQUE = 0
_LEGACY_REPORTED = False


def canonical_video_id(path_or_id: Union[str, Path]) -> str:
    """Return a canonical ``video_XXX`` style identifier for a sample.

    The helper accepts arbitrary strings or paths (including filenames such as
    ``video_220.0.mp4``) and normalises them so that cache keys, metadata, and
    label lookups consistently agree on the same identifier.  Normalisation
    steps:

    * drop any directory components and file extensions (including trailing
      ``.0`` shards)
    * lower-case and collapse non-alphanumeric separators to ``_``
    * coerce "audio" prefixes to "video"
    * ensure the result carries a ``video_`` prefix with the original numeric
      suffix preserved when present
    """

    if path_or_id is None:
        return ""

    text = str(path_or_id).strip()
    if not text:
        return ""

    name = Path(text).name
    base = name
    while True:
        stem = Path(base).stem
        if stem == base or not stem:
            break
        base = stem

    base = base.lower()
    base = base.replace("-", "_").replace(" ", "_")
    base = re.sub(r"__+", "_", base).strip("_")

    if base.startswith("audio_"):
        base = base[6:]
    elif base.startswith("audio"):
        base = base[5:].lstrip("_")

    if base.startswith("video_"):
        candidate = base[6:]
    elif base.startswith("video"):
        candidate = base[5:].lstrip("_")
    else:
        candidate = base

    candidate = candidate.strip("_")

    match = re.search(r"(\d+)$", candidate)
    if match:
        digits = match.group(1)
        return f"video_{digits}"

    if candidate:
        return f"video_{candidate}"

    return "video_unknown"


def id_aliases(canon_id: str) -> List[str]:
    """Return canonical plus known legacy ID aliases for ``canon_id``."""

    base = canonical_video_id(canon_id)
    if not base:
        return []
    aliases = [base]
    legacy = f"{base}.0"
    if legacy not in aliases:
        aliases.append(legacy)
    return aliases


def _report_legacy_hits() -> None:
    """Report a summary of legacy ID normalization hits once per run."""
    global _LEGACY_REPORTED
    if _LEGACY_REPORTED or _LEGACY_TOTAL <= 0:
        return
    _LEGACY_REPORTED = True
    LOGGER.info(
        "[compat] legacy_id_hit summary normalized=%d unique=%d",
        _LEGACY_TOTAL,
        _LEGACY_UNIQUE,
    )


def log_legacy_id_hit(legacy_id: str, canon_id: str, *, logger: Optional[logging.Logger] = None) -> None:
    """Record a compat hit; summary is logged once at shutdown."""

    legacy = str(legacy_id or "").strip()
    canon = canonical_video_id(canon_id)
    if not legacy or legacy == canon:
        return
    key = f"{legacy}->{canon}"
    global _LEGACY_TOTAL, _LEGACY_UNIQUE
    _LEGACY_TOTAL += 1
    if key not in _LEGACY_HITS:
        _LEGACY_HITS.add(key)
        _LEGACY_UNIQUE += 1


atexit.register(_report_legacy_hits)


__all__ = ["canonical_video_id", "id_aliases", "log_legacy_id_hit"]
