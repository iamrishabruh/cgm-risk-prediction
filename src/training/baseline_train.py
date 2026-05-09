"""Sklearn baseline training with the same repeated CV protocol as torch models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from config.config import Config
from data.processor import DataProcessor
from models.baseline_sklearn import make_baseline_estimator

logger = logging.getLogger(__name__)


def run_baseline_cv(
    model_type: str,
    n_splits: int = 2,
    n_repeats: int = 1,
    raw_dir: Path | None = None,
    demographics_path: Path | None = None,
) -> None:
    """
    Fit classical models on each CV training fold (full training fold, no torch-style val split).

    Baselines use closed-form or bagging estimators without epochal early stopping; the CV fold
    itself provides out-of-fold generalization estimates when you run `evaluation.evaluate`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    np.random.seed(Config.SEED)

    if model_type not in Config.BASELINE_MODELS:
        raise ValueError(model_type)

    processor = DataProcessor(raw_dir=raw_dir, demographics_path=demographics_path)
    X, y = processor.load_dataset(augment=False)
    X = np.asarray(X)
    y = np.asarray(y)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=Config.SEED,
    )

    for fold_idx, (train_index, _test_index) in enumerate(rskf.split(X, y), start=1):
        est = make_baseline_estimator(model_type)
        est.fit(X[train_index], y[train_index])
        out_path = Config.CHECKPOINTS / f"{model_type}_repFold{fold_idx}.joblib"
        joblib.dump(est, out_path)
        logger.info("Saved %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Train sklearn baselines (repeated CV).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(Config.BASELINE_MODELS),
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    run_baseline_cv(args.model, n_splits=args.folds, n_repeats=args.repeats)


if __name__ == "__main__":
    main()
