"""Tests for cepea_forecast.recalibration (no AutoGluon dependency)."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from cepea_forecast.recalibration import (
    CALIBRATION_FILENAME,
    QUANTILE_LEVELS,
    CalibrationBundle,
    apply_recalibration,
    calibration_ready,
    load_calibration,
    save_calibration,
    _fit_single_isotonic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prediction_length():
    return 4  # small for testing


@pytest.fixture
def sample_bundle(prediction_length):
    """Create a CalibrationBundle with identity-ish isotonic models."""
    bundle = CalibrationBundle(
        prediction_length=prediction_length,
        n_origins=10,
    )
    rng = np.random.default_rng(42)
    for q_col in QUANTILE_LEVELS:
        for step in range(1, prediction_length + 1):
            # Fit on data where recalibrated ~ predicted + small bias
            X = np.sort(rng.uniform(200, 350, size=20))
            bias = 5.0 if q_col == "0.1" else (-5.0 if q_col == "0.9" else 0.0)
            y = X + bias + rng.normal(0, 2, size=20)
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(X, y)
            bundle.models[(q_col, step)] = iso
    return bundle


@pytest.fixture
def sample_forecast(prediction_length):
    """Fake forecast DataFrame mimicking normalize_forecast_frame output."""
    dates = pd.date_range("2026-01-02", periods=prediction_length, freq="W-FRI")
    return pd.DataFrame({
        "timestamp": dates,
        "mean": [300.0, 305.0, 310.0, 315.0],
        "0.1": [280.0, 285.0, 290.0, 295.0],
        "0.5": [300.0, 305.0, 310.0, 315.0],
        "0.9": [320.0, 325.0, 330.0, 335.0],
    })


# ---------------------------------------------------------------------------
# Tests for _fit_single_isotonic
# ---------------------------------------------------------------------------

class TestFitSingleIsotonic:
    def test_returns_none_for_too_few_samples(self):
        result = _fit_single_isotonic([1.0, 2.0], [1.1, 2.1], 0.5)
        assert result is None

    def test_returns_isotonic_with_enough_samples(self):
        X = list(range(10))
        y = [x * 1.1 for x in X]
        result = _fit_single_isotonic(X, y, 0.5)
        assert isinstance(result, IsotonicRegression)

    def test_predictions_are_monotonic(self):
        rng = np.random.default_rng(99)
        X = sorted(rng.uniform(100, 400, 30))
        y = [x + rng.normal(0, 5) for x in X]
        iso = _fit_single_isotonic(X, y, 0.5)
        test_x = np.linspace(min(X), max(X), 50)
        preds = iso.predict(test_x)
        assert all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))


# ---------------------------------------------------------------------------
# Tests for apply_recalibration
# ---------------------------------------------------------------------------

class TestApplyRecalibration:
    def test_output_shape_matches_input(self, sample_forecast, sample_bundle):
        result = apply_recalibration(sample_forecast, sample_bundle)
        assert result.shape == sample_forecast.shape
        assert list(result.columns) == list(sample_forecast.columns)

    def test_monotonicity_enforced(self, sample_forecast, sample_bundle):
        result = apply_recalibration(sample_forecast, sample_bundle)
        for idx in result.index:
            assert result.loc[idx, "0.1"] <= result.loc[idx, "0.5"]
            assert result.loc[idx, "0.5"] <= result.loc[idx, "0.9"]

    def test_mean_within_quantile_range(self, sample_forecast, sample_bundle):
        result = apply_recalibration(sample_forecast, sample_bundle)
        for idx in result.index:
            assert result.loc[idx, "0.1"] <= result.loc[idx, "mean"]
            assert result.loc[idx, "mean"] <= result.loc[idx, "0.9"]

    def test_values_change_after_recalibration(self, sample_forecast, sample_bundle):
        result = apply_recalibration(sample_forecast, sample_bundle)
        # At least some values should differ from originals
        changed = (result[QUANTILE_LEVELS] != sample_forecast[QUANTILE_LEVELS]).any().any()
        assert changed, "Recalibration should modify at least some quantile values"

    def test_does_not_modify_original(self, sample_forecast, sample_bundle):
        original_copy = sample_forecast.copy()
        apply_recalibration(sample_forecast, sample_bundle)
        pd.testing.assert_frame_equal(sample_forecast, original_copy)

    def test_handles_missing_calibrators_gracefully(self, sample_forecast, prediction_length):
        empty_bundle = CalibrationBundle(prediction_length=prediction_length)
        result = apply_recalibration(sample_forecast, empty_bundle)
        # With no calibrators, output should match input (except possible mean clamping)
        for q in QUANTILE_LEVELS:
            pd.testing.assert_series_equal(result[q], sample_forecast[q])


# ---------------------------------------------------------------------------
# Tests for persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, sample_bundle):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            save_calibration(sample_bundle, model_dir)
            assert calibration_ready(model_dir)

            loaded = load_calibration(model_dir)
            assert loaded is not None
            assert loaded.prediction_length == sample_bundle.prediction_length
            assert loaded.n_origins == sample_bundle.n_origins
            assert loaded.n_models == sample_bundle.n_models

    def test_load_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_calibration(Path(tmpdir)) is None
            assert not calibration_ready(Path(tmpdir))

    def test_loaded_models_produce_same_predictions(self, sample_bundle):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            save_calibration(sample_bundle, model_dir)
            loaded = load_calibration(model_dir)

            test_val = np.array([300.0])
            for key, iso in sample_bundle.models.items():
                original_pred = iso.predict(test_val)[0]
                loaded_pred = loaded.models[key].predict(test_val)[0]
                assert abs(original_pred - loaded_pred) < 1e-10


# ---------------------------------------------------------------------------
# Tests for CalibrationBundle
# ---------------------------------------------------------------------------

class TestCalibrationBundle:
    def test_n_models_counts_correctly(self, sample_bundle, prediction_length):
        expected = len(QUANTILE_LEVELS) * prediction_length
        assert sample_bundle.n_models == expected

    def test_empty_bundle(self):
        bundle = CalibrationBundle(prediction_length=52)
        assert bundle.n_models == 0
        assert bundle.n_origins == 0
