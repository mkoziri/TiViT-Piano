"""
Purpose:
    Smoke test for sync resolution and application (no pytest needed).

Key Functions/Classes:
    - main(): Resolve/apply sync and verify event shift.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_sync.py
"""

from __future__ import annotations

from tivit.data.sync import resolve_sync, apply_sync


def main() -> None:
    """Resolve lag from metadata and apply to events."""
    sample = {"events": [(0.1, 0.2, 60)]}
    sync = resolve_sync("video_001", {"lag_ms": 50})
    apply_sync(sample, sync)
    print("sync", sync)
    print("events", sample["events"])


if __name__ == "__main__":
    main()

