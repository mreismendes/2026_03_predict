from __future__ import annotations

import numpy as np
import pandas as pd

from cepea_forecast.features import (
    FOURIER_ORDER,
    WEEKLY_KNOWN_COVARIATES,
    add_calendar_features,
    add_lag_features,
    build_future_known_covariates,
    engineer_features,
    known_covariates_for,
)


def _make_weekly_df(n_weeks: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2025-01-03", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame({"target": range(100, 100 + n_weeks)}, index=dates)


def test_known_covariates_for_weekly() -> None:
    names = known_covariates_for("weekly")
    assert len(names) == FOURIER_ORDER * 2
    assert "sin_yearly_1" in names
    assert "cos_yearly_1" in names
    assert f"sin_yearly_{FOURIER_ORDER}" in names
    assert f"cos_yearly_{FOURIER_ORDER}" in names


def test_add_calendar_features_weekly() -> None:
    df = _make_weekly_df()
    result = add_calendar_features(df, "weekly")
    for k in range(1, FOURIER_ORDER + 1):
        assert f"sin_yearly_{k}" in result.columns
        assert f"cos_yearly_{k}" in result.columns
    # Fourier values should be bounded [-1, 1]
    for col in WEEKLY_KNOWN_COVARIATES:
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


def test_add_lag_features_weekly() -> None:
    df = _make_weekly_df()
    result = add_lag_features(df, "weekly")
    expected_past = [
        "rolling_mean_4", "rolling_mean_13", "rolling_mean_26",
        "momentum_4", "momentum_13", "momentum_26",
        "ma_ratio_4_13", "ma_ratio_4_26",
        "log_return_1",
        "realized_vol_4", "realized_vol_13",
        "yoy_change",
        "range_position_52",
        "rsi_14",
    ]
    for col in expected_past:
        assert col in result.columns, f"Missing past covariate: {col}"
    # Warm-up NaNs are expected for lag features; AutoGluon handles them natively
    # But after the warm-up period (52 rows), all values should be present
    tail = result.iloc[52:]
    assert tail.isna().sum().sum() == 0, f"NaNs found after warm-up period: {tail.isna().sum()}"


def test_rsi_bounded() -> None:
    df = _make_weekly_df()
    result = add_lag_features(df, "weekly")
    assert result["rsi_14"].min() >= 0.0
    assert result["rsi_14"].max() <= 100.0


def test_range_position_bounded() -> None:
    df = _make_weekly_df()
    result = add_lag_features(df, "weekly")
    assert result["range_position_52"].min() >= 0.0
    assert result["range_position_52"].max() <= 1.0


def test_engineer_features_returns_known_cov_names() -> None:
    df = _make_weekly_df()
    enriched, known_names = engineer_features(df, "weekly")
    assert known_names == WEEKLY_KNOWN_COVARIATES
    assert all(name in enriched.columns for name in known_names)
    assert "rolling_mean_4" in enriched.columns
    assert "target" in enriched.columns


def test_build_future_known_covariates_weekly() -> None:
    last_ts = pd.Timestamp("2026-02-27")
    known_names = known_covariates_for("weekly")
    future = build_future_known_covariates(
        last_timestamp=last_ts,
        prediction_length=4,
        frequency="W-FRI",
        item_id="test_item",
        known_covariates_names=known_names,
    )
    assert len(future) == 4
    assert "item_id" in future.columns
    assert "timestamp" in future.columns
    for name in known_names:
        assert name in future.columns
    assert (future["item_id"] == "test_item").all()
    # Fourier values should be bounded
    for col in known_names:
        assert future[col].min() >= -1.0
        assert future[col].max() <= 1.0


def test_fourier_continuity_across_year_boundary() -> None:
    """Verify Fourier features are smooth across Dec->Jan (unlike week_of_year)."""
    dates = pd.date_range("2025-12-19", periods=4, freq="W-FRI")
    df = pd.DataFrame({"target": [100, 101, 102, 103]}, index=dates)
    result = add_calendar_features(df, "weekly")
    # The sin/cos values should change smoothly (no jump > 0.5 between consecutive weeks)
    for col in WEEKLY_KNOWN_COVARIATES:
        diffs = result[col].diff().abs().dropna()
        assert diffs.max() < 0.5, f"{col} has a discontinuity at year boundary"
