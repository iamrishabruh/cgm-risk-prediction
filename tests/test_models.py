import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from config.config import Config
from models.attention_mlp import AttentionMLP
from models.factory import build_torch_model
from models.gnn_model import PatientGNN
from models.tabtransformer import TabTransformer


@pytest.fixture
def batch_x():
    return torch.randn(4, Config.NUM_FEATURES, dtype=torch.float32)


def test_tabtransformer_forward(batch_x):
    m = TabTransformer()
    y = m(batch_x)
    assert y.shape == (4, Config.NUM_CLASSES)


def test_attention_mlp_forward(batch_x):
    m = AttentionMLP()
    y = m(batch_x)
    assert y.shape == (4, Config.NUM_CLASSES)


def test_gnn_forward():
    m = PatientGNN(num_features=Config.NUM_FEATURES)
    x = torch.randn(5, Config.NUM_FEATURES)
    ei = torch.tensor([[i, j] for i in range(5) for j in range(5) if i != j], dtype=torch.long).t()
    data = Data(x=x, edge_index=ei)
    y = m(data)
    assert y.shape == (5, Config.NUM_CLASSES)


def test_factory_build_each():
    for name in ("TabTransformer", "AttentionMLP", "GNN"):
        model = build_torch_model(name)
        assert isinstance(model, torch.nn.Module)


def test_baseline_estimators_fit():
    from models.baseline_sklearn import make_logistic_regression, make_random_forest

    rng = np.random.RandomState(0)
    X = rng.randn(40, Config.NUM_FEATURES)
    y = rng.randint(0, 2, size=40)
    lr = make_logistic_regression()
    lr.fit(X, y)
    assert lr.predict(X).shape == (40,)
    rf = make_random_forest()
    rf.fit(X, y)
    assert rf.predict_proba(X).shape == (40, 2)
