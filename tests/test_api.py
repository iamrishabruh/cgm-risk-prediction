from __future__ import annotations

import importlib
import sys

import pytest
import torch
from fastapi.testclient import TestClient

from config.config import Config
from models.factory import build_torch_model


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "CHECKPOINTS", tmp_path)
    model = build_torch_model("TabTransformer")
    torch.save(model.state_dict(), tmp_path / "TabTransformer_repFold1.pt")

    sys.modules.pop("api.main", None)
    api_main = importlib.import_module("api.main")
    return TestClient(api_main.app)


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_predict_features_round_trip(api_client):
    feats = [0.1 * i for i in range(Config.NUM_FEATURES)]
    r = api_client.post(
        "/predict/features",
        json={"features": feats, "model": "TabTransformer", "fold_index": 1},
    )
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert r.json()["prediction"] in (0, 1)
