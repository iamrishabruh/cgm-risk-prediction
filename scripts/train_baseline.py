#!/usr/bin/env python3
"""CLI: sklearn baselines (same repeated CV protocol)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from training.baseline_train import main

if __name__ == "__main__":
    main()
