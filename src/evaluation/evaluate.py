"""Out-of-fold evaluation for torch models, ensembles, and sklearn baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from config.config import Config
from data.processor import DataProcessor
from evaluation.metrics import classification_metrics
from models.factory import build_torch_model
from training.torch_trainer import make_gnn_loader, make_tabular_loader

logger = logging.getLogger(__name__)


def load_torch_checkpoint(model_type: str, fold_count: int):
    model = build_torch_model(model_type)
    path = Config.CHECKPOINTS / f"{model_type}_repFold{fold_count}.pt"
    state = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()
    return model


def load_baseline_checkpoint(model_type: str, fold_count: int):
    path = Config.CHECKPOINTS / f"{model_type}_repFold{fold_count}.joblib"
    return joblib.load(path)


def evaluate_torch_model_on_loader(model, loader, is_graph: bool) -> tuple[float, float, float, float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                data = batch.to(Config.DEVICE)
                outputs = model(data)
                labels = data.y
            else:
                inputs, labels = batch
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    _, acc, auc, sens, spec = classification_metrics(all_labels, all_probs)
    return avg_loss, acc, auc, sens, spec


def build_ensemble_for_fold(fold_count: int) -> dict:
    models = {}
    for mtype in ("TabTransformer", "AttentionMLP", "GNN"):
        models[mtype] = load_torch_checkpoint(mtype, fold_count)
    return models


def evaluate_ensemble(models_dict: dict, loader: GeoDataLoader) -> tuple[float, float, float, float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            data = batch.to(Config.DEVICE)
            labels = data.y
            x = data.x
            out_tt = models_dict["TabTransformer"](x)
            out_mlp = models_dict["AttentionMLP"](x)
            out_gnn = models_dict["GNN"](data)
            avg_logits = (out_tt + out_mlp + out_gnn) / 3.0
            loss = criterion(avg_logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(avg_logits, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    _, acc, auc, sens, spec = classification_metrics(all_labels, all_probs)
    return avg_loss, acc, auc, sens, spec


def evaluate_sklearn(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    if len(X) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    _, acc, auc, sens, spec = classification_metrics(y, prob)
    # report sklearn log loss optionally
    try:
        from sklearn.metrics import log_loss

        ll = log_loss(y, model.predict_proba(X))
    except Exception:
        ll = 0.0
    return float(ll), acc, auc, sens, spec


def evaluate_cv(
    model_type: str,
    n_splits: int,
    n_repeats: int,
    raw_dir: Path | None = None,
    demographics_path: Path | None = None,
) -> tuple[list[tuple], tuple[float, float, float, float, float]]:
    processor = DataProcessor(raw_dir=raw_dir, demographics_path=demographics_path)
    X_all, y_all = processor.load_dataset(augment=False)
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=Config.SEED,
    )

    fold_metrics: list[tuple] = []
    fold_count = 1

    for _train_idx, test_idx in rskf.split(X_all, y_all):
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        if model_type in Config.BASELINE_MODELS:
            model = load_baseline_checkpoint(model_type, fold_count)
            metrics = evaluate_sklearn(model, X_test, y_test)
        elif model_type in ("TabTransformer", "AttentionMLP"):
            ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )
            loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False)
            model = load_torch_checkpoint(model_type, fold_count)
            metrics = evaluate_torch_model_on_loader(model, loader, is_graph=False)
        elif model_type == "GNN":
            loader = make_gnn_loader(X_test, y_test, shuffle=False)
            model = load_torch_checkpoint("GNN", fold_count)
            metrics = evaluate_torch_model_on_loader(model, loader, is_graph=True)
        elif model_type == Config.ENSEMBLE_NAME:
            loader = make_gnn_loader(X_test, y_test, shuffle=False)
            models_dict = build_ensemble_for_fold(fold_count)
            metrics = evaluate_ensemble(models_dict, loader)
        else:
            raise ValueError(model_type)

        fold_metrics.append(metrics)
        logger.info(
            "repFold%s %s | loss=%.4f acc=%.4f auc=%.4f sens=%.4f spec=%.4f",
            fold_count,
            model_type,
            *metrics,
        )
        fold_count += 1

    avg = tuple(float(np.mean([m[i] for m in fold_metrics])) for i in range(5))
    return fold_metrics, avg


def run_evaluate_cli():
    parser = argparse.ArgumentParser(description="Evaluate models with repeated CV (test fold metrics).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(
            Config.TORCH_MODELS | Config.BASELINE_MODELS | {Config.ENSEMBLE_NAME}
        ),
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    folds, overall = evaluate_cv(args.model, args.folds, args.repeats)
    print("\nPer-fold (loss, acc, auc, sens, spec):")
    for i, row in enumerate(folds, start=1):
        print(f"  fold {i}: {row}")
    print(
        f"\nMean | loss={overall[0]:.4f} acc={overall[1]:.4f} auc={overall[2]:.4f} "
        f"sens={overall[3]:.4f} spec={overall[4]:.4f}"
    )


def evaluate_all(model_type: str, folds: int, repeats: int):
    """Streamlit-friendly wrapper."""
    return evaluate_cv(model_type, folds, repeats)


if __name__ == "__main__":
    run_evaluate_cli()
