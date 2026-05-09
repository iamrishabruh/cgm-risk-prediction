import numpy as np

from config.config import Config
from training.splits import train_val_mask_from_indices


def test_inner_val_split_stratified():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    train_fold = np.arange(len(y))
    tr, va = train_val_mask_from_indices(y, train_fold, val_fraction=0.25, seed=0)
    assert len(tr) + len(va) == len(y)
    assert len(va) >= 1


def test_tiny_fold_skips_val():
    y = np.array([0, 1])
    train_fold = np.array([0, 1])
    tr, va = train_val_mask_from_indices(y, train_fold, val_fraction=0.2, seed=0)
    assert len(va) == 0
    assert np.array_equal(tr, train_fold)
