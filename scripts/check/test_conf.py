"""Purpose:
    Configure ``sys.path`` so pytest discovers the repository packages during
    script-level tests.

Key Functions/Classes:
    - Module initialization: Extends ``sys.path`` with the repository root to
      enable imports of :mod:`src` modules.

CLI:
    Not a standalone CLI; pytest automatically imports this module when running
    tests in the ``scripts`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
