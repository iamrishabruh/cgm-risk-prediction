"""Repeated stratified CV training for torch models with an inner train/val split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold

from config.config import Config
from data.processor import DataProcessor
from models.factory import build_torch_model
from training.splits import train_val_mask_from_indices
from training.torch_trainer import TorchTrainer, make_gnn_loader, make_tabular_loader

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / "training.log"),
            logging.StreamHandler(),
        ],
    )


def run_torch_cv(
    model_type: str,
    n_splits: int = 2,
    n_repeats: int = 1,
    raw_dir: Path | None = None,
    demographics_path: Path | None = None,
) -> None:
    """
    RepeatedStratifiedKFold on the full dataset.

    Within each training fold:
    - Hold out a stratified validation fraction for early stopping and checkpoint selection.
    - If the fold is too small for a validation split, fall back to training loss only
      (see `docs/model-card.md`).

    The held-out test fold from CV is not used during training; evaluate with
    `python -m evaluation.evaluate` or the evaluation API.
    """
    configure_logging()
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    processor = DataProcessor(raw_dir=raw_dir, demographics_path=demographics_path)
    features, labels = processor.load_dataset(augment=False)
    features = np.asarray(features)
    labels = np.asarray(labels)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=Config.SEED,
    )

    for fold_idx, (train_index, _test_index) in enumerate(rskf.split(features, labels), start=1):
        train_sub, val_sub = train_val_mask_from_indices(
            labels,
            train_index,
            val_fraction=Config.VAL_FRACTION,
            seed=Config.SEED + fold_idx,
        )

        X_tr, y_tr = features[train_sub], labels[train_sub]
        has_val = len(val_sub) > 0
        if has_val:
            X_va, y_va = features[val_sub], labels[val_sub]

        if model_type in ("TabTransformer", "AttentionMLP"):
            loader_train = make_tabular_loader(X_tr, y_tr, shuffle=True)
            loader_val = make_tabular_loader(X_va, y_va, shuffle=False) if has_val else None
            is_graph = False
        elif model_type == "GNN":
            loader_train = make_gnn_loader(X_tr, y_tr, shuffle=True)
            loader_val = make_gnn_loader(X_va, y_va, shuffle=False) if has_val else None
            is_graph = True
        else:
            raise ValueError(f"Unsupported torch model: {model_type}")

        model = build_torch_model(model_type)
        trainer = TorchTrainer(model, Config.DEVICE)
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = trainer.train_epoch(loader_train, is_graph=is_graph)
            if has_val and loader_val is not None:
                val_loss = trainer.eval_epoch(loader_val, is_graph=is_graph)
                score = val_loss
                logger.info(
                    "repFold %s | epoch %s/%s | train_loss=%.4f | val_loss=%.4f",
                    fold_idx,
                    epoch + 1,
                    Config.NUM_EPOCHS,
                    train_loss,
                    val_loss,
                )
            else:
                score = train_loss
                logger.info(
                    "repFold %s | epoch %s/%s | train_loss=%.4f (no val split)",
                    fold_idx,
                    epoch + 1,
                    Config.NUM_EPOCHS,
                    train_loss,
                )

            if np.isnan(train_loss):
                logger.error("Training halted due to NaN loss.")
                break

            if score < best_score:
                best_score = score
                patience_counter = 0
                ckpt = Config.CHECKPOINTS / f"{model_type}_repFold{fold_idx}.pt"
                torch.save(model.state_dict(), ckpt)
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOP_PATIENCE:
                    logger.info("Early stopping repFold %s (%s)", fold_idx, model_type)
                    break


def train_model_cli(model_type: str, folds: int, repeats: int) -> None:
    """Entry point for UI subprocess calls."""
    run_torch_cv(model_type, n_splits=folds, n_repeats=repeats)


def main():
    parser = argparse.ArgumentParser(description="Train CGM tabular models (repeated CV, inner val split).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(Config.TORCH_MODELS),
        help="Torch architecture",
    )
    parser.add_argument("--folds", type=int, default=2, help="CV folds")
    parser.add_argument("--repeats", type=int, default=1, help="CV repeats")
    args = parser.parse_args()
    run_torch_cv(args.model, n_splits=args.folds, n_repeats=args.repeats)


if __name__ == "__main__":
    main()
