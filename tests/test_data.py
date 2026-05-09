import numpy as np
import pytest

from config.config import Config
from data.features import NUM_TABULAR_FEATURES, extract_cgm_features, sanitize_feature_vector
from data.processor import DataProcessor


def test_extract_cgm_features_shape_and_rmssd():
    x = np.array([100.0, 110.0, 105.0], dtype=np.float64)
    mean, std, rmssd = extract_cgm_features(x)
    assert mean == pytest.approx(np.mean(x))
    assert std == pytest.approx(np.std(x))
    assert rmssd > 0


def test_extract_cgm_features_empty():
    mean, std, rmssd = extract_cgm_features(np.array([]))
    assert (mean, std, rmssd) == (0.0, 0.0, 0.0)


def test_sanitize_feature_vector_nan_inf():
    out = sanitize_feature_vector([1.0, float("nan"), float("inf"), None])
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert out[3] == 0.0


def test_process_file_output_shape(patch_config_paths, fixture_dir):
    proc = DataProcessor(
        raw_dir=fixture_dir,
        demographics_path=fixture_dir / "demographics.xlsx",
    )
    demo = proc._load_demographics()
    feats, label = proc.process_file(fixture_dir / "Subject1.xlsx", demo)
    assert len(feats) == Config.NUM_FEATURES == NUM_TABULAR_FEATURES
    assert label in (0, 1)


def test_missing_and_dirty_cgm_handling(patch_config_paths, fixture_dir):
    proc = DataProcessor(
        raw_dir=fixture_dir,
        demographics_path=fixture_dir / "demographics.xlsx",
    )
    demo = proc._load_demographics()
    feats, _ = proc.process_file(fixture_dir / "Subject2.xlsx", demo)
    assert len(feats) == NUM_TABULAR_FEATURES
    assert all(isinstance(x, float) for x in feats)


def test_load_dataset_scaled_shape(patch_config_paths, fixture_dir):
    proc = DataProcessor(
        raw_dir=fixture_dir,
        demographics_path=fixture_dir / "demographics.xlsx",
    )
    X, y = proc.load_dataset(augment=False)
    assert X.shape[1] == NUM_TABULAR_FEATURES
    assert X.shape[0] == 3
    assert y.shape[0] == 3
