#!/usr/bin/env python3
"""CLI: out-of-fold evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import run_evaluate_cli

if __name__ == "__main__":
    run_evaluate_cli()
