"""Shared inference helpers for FastAPI and Streamlit (no cross-import between them)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from config.config import Config
from models.factory import build_torch_model


def load_torch_fold_state(model_type: str, fold_index: int):
    path = Config.CHECKPOINTS / f"{model_type}_repFold{fold_index}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_torch_model(model_type)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model


def _single_torch_forward(model, features: list[float], model_type: str) -> torch.Tensor:
    x_tensor = torch.tensor([features], dtype=torch.float32, device=Config.DEVICE)
    if model_type == "GNN":
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=Config.DEVICE)
        data = Data(x=x_tensor, edge_index=edge_index)
        with torch.no_grad():
            return model(data)
    with torch.no_grad():
        return model(x_tensor)


def predict_proba_from_features(
    model_type: str,
    fold_index: int,
    features: list[float],
) -> tuple[float, float]:
    """Returns (p_class_0, p_class_1) for binary head."""
    model = load_torch_fold_state(model_type, fold_index)
    logits = _single_torch_forward(model, features, model_type)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return float(probs[0]), float(probs[1])


def predict_from_features(model_type: str, fold_index: int, features: list[float]) -> int:
    model = load_torch_fold_state(model_type, fold_index)
    logits = _single_torch_forward(model, features, model_type)
    return int(logits.argmax(dim=1).item())


def ensemble_predict_features(fold_index: int, features: list[float]) -> int:
    x_tensor = torch.tensor([features], dtype=torch.float32, device=Config.DEVICE)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=Config.DEVICE)
    data = Data(x=x_tensor, edge_index=edge_index)
    with torch.no_grad():
        z_tt = load_torch_fold_state("TabTransformer", fold_index)(x_tensor)
        z_mlp = load_torch_fold_state("AttentionMLP", fold_index)(x_tensor)
        z_gnn = load_torch_fold_state("GNN", fold_index)(data)
        avg = (z_tt + z_mlp + z_gnn) / 3.0
        return int(avg.argmax(dim=1).item())
