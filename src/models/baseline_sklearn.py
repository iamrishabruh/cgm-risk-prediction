"""Classical baselines for comparison with neural models (same tabular features)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config.config import Config


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=Config.SEED,
        solver="lbfgs",
    )


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=Config.SEED,
        n_jobs=-1,
    )


def make_baseline_estimator(model_type: str):
    if model_type == "LogisticRegression":
        return make_logistic_regression()
    if model_type == "RandomForest":
        return make_random_forest()
    raise ValueError(f"Unknown baseline: {model_type}")
