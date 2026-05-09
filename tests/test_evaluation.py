import numpy as np

from evaluation.metrics import classification_metrics


def test_classification_metrics_perfect_sep():
    y_true = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.9, 0.8]
    loss, acc, auc, sens, spec = classification_metrics(y_true, p)
    assert acc == 1.0
    assert auc == 1.0
    assert sens == 1.0
    assert spec == 1.0
    assert loss == 0.0


def test_classification_metrics_single_class_auc_fallback():
    y_true = [0, 0, 0]
    p = [0.2, 0.3, 0.1]
    _loss, _acc, auc, _sens, _spec = classification_metrics(y_true, p)
    assert auc == 0.0
