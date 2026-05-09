"""
Ensure import resolution prefers `src/` over top-level folders (e.g. root `evaluation/`).

Insert repo root first, then `src/` at position 0 so packages like `evaluation` and `models`
resolve to `src/evaluation`, `src/models` while `config` still loads from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
