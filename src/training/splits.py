"""Train/validation splits inside a CV training fold."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from config.config import Config


def train_val_mask_from_indices(
    y: np.ndarray,
    train_fold_idx: np.ndarray,
    val_fraction: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split `train_fold_idx` into (train_sub_idx, val_idx) with stratification when possible.

    If the fold is too small or stratification fails, falls back to unstratified split.
    If still too small for any hold-out, returns (train_fold_idx, empty array) and
    callers should select checkpoints using training loss only.
    """
    vf = val_fraction if val_fraction is not None else Config.VAL_FRACTION
    rs = seed if seed is not None else Config.SEED
    n = len(train_fold_idx)
    if n < Config.VAL_MIN_SAMPLES * 2:
        return train_fold_idx, np.array([], dtype=np.int64)

    rel = np.arange(n)
    y_fold = y[train_fold_idx]
    try:
        tr_rel, va_rel = train_test_split(
            rel,
            test_size=vf,
            stratify=y_fold,
            random_state=rs,
        )
    except ValueError:
        tr_rel, va_rel = train_test_split(rel, test_size=vf, random_state=rs)

    if len(va_rel) == 0:
        return train_fold_idx, np.array([], dtype=np.int64)
    return train_fold_idx[tr_rel], train_fold_idx[va_rel]
