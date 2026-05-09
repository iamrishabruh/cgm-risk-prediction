from __future__ import annotations

from config.config import Config
from models.attention_mlp import AttentionMLP
from models.gnn_model import PatientGNN
from models.tabtransformer import TabTransformer


def is_torch_model(model_type: str) -> bool:
    return model_type in Config.TORCH_MODELS


def is_baseline_model(model_type: str) -> bool:
    return model_type in Config.BASELINE_MODELS


def build_torch_model(model_type: str):
    if model_type == "TabTransformer":
        return TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim=Config.TRANSFORMER_DIM,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS,
        )
    if model_type == "AttentionMLP":
        return AttentionMLP(
            input_dim=Config.NUM_FEATURES,
            hidden_dim=256,
            num_classes=Config.NUM_CLASSES,
        )
    if model_type == "GNN":
        return PatientGNN(
            num_features=Config.NUM_FEATURES,
            hidden_dim=Config.GNN_HIDDEN_DIM,
        )
    raise ValueError(f"Not a torch model type: {model_type}")
