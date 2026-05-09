"""Classification metrics for binary glycemic-control proxy tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_prob_positive: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """
    Returns (loss placeholder as 0.0, accuracy, auc, sensitivity, specificity).

    Cross-entropy is reported separately in torch evaluation; sklearn path sets loss to 0.0
    and relies on the other metrics for comparison.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob_positive, dtype=np.float64)
    y_hat = (y_prob >= threshold).astype(np.int64)

    acc = float(np.mean(y_hat == y_true)) if len(y_true) else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_hat)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        sens, spec = 0.0, 0.0

    return 0.0, acc, auc, sens, spec
