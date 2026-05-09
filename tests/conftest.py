from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Root must be on path for `config`; `src` must be *first* so it wins over `./evaluation/`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def patch_config_paths(monkeypatch, fixture_dir):
    from config.config import Config

    monkeypatch.setattr(Config, "RAW_DATA", fixture_dir)
    monkeypatch.setattr(Config, "DEMOGRAPHICS_PATH", fixture_dir / "demographics.xlsx")
    return Config
