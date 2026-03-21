from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cepea_forecast.features import (
    FOURIER_ORDER,
    WEEKLY_KNOWN_COVARIATES,
    _hurst_rs,
    _sample_entropy,
    _transfer_entropy,
    add_calendar_features,
    add_lag_features,
    build_future_known_covariates,
    engineer_features,
    known_covariates_for,
)


def _make_weekly_df(n_weeks: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2025-01-03", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame({"target": range(100, 100 + n_weeks)}, index=dates)


def _make_realistic_weekly_df(n_weeks: int = 130, seed: int = 42) -> pd.DataFrame:
    """Create a random-walk DataFrame that exercises GARCH/Markov/FFD."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-06", periods=n_weeks, freq="W-FRI")
    returns = rng.normal(0.002, 0.03, n_weeks)
    prices = 300 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"target": prices}, index=dates)


# Module-scoped fixtures to avoid re-fitting GARCH/Markov/FFD in every test
@pytest.fixture(scope="module")
def realistic_lag_result():
    df = _make_realistic_weekly_df(130)
    return add_lag_features(df, "weekly")


@pytest.fixture(scope="module")
def realistic_with_covariates_result():
    df = _make_realistic_weekly_df(130)
    rng = np.random.default_rng(42)
    df["BOI_USD"] = df["target"] / 5.5 + rng.normal(0, 0.5, len(df))
    df["BRL_KG"] = df["target"] * 0.04 + rng.normal(0, 0.5, len(df))
    return add_lag_features(df, "weekly")


# =========================================================================
# Existing tests (unchanged)
# =========================================================================

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
    # Only check original columns (Tier 1-3 may skip on linear data)
    tail = result[expected_past].iloc[52:]
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
    for col in known_names:
        assert future[col].min() >= -1.0
        assert future[col].max() <= 1.0


def test_fourier_continuity_across_year_boundary() -> None:
    """Verify Fourier features are smooth across Dec->Jan (unlike week_of_year)."""
    dates = pd.date_range("2025-12-19", periods=4, freq="W-FRI")
    df = pd.DataFrame({"target": [100, 101, 102, 103]}, index=dates)
    result = add_calendar_features(df, "weekly")
    for col in WEEKLY_KNOWN_COVARIATES:
        diffs = result[col].diff().abs().dropna()
        assert diffs.max() < 0.5, f"{col} has a discontinuity at year boundary"


# =========================================================================
# Tier 1-3 helper function unit tests
# =========================================================================

def test_hurst_helper_bounded() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 100)
    h = _hurst_rs(x)
    assert 0.0 <= h <= 1.0


def test_hurst_helper_short_returns_nan() -> None:
    assert np.isnan(_hurst_rs(np.array([1.0, 2.0])))


def test_sample_entropy_constant_returns_zero() -> None:
    assert _sample_entropy(np.ones(50)) == 0.0


def test_sample_entropy_random_returns_positive() -> None:
    rng = np.random.default_rng(42)
    se = _sample_entropy(rng.normal(0, 1, 100))
    assert se > 0.0


def test_sample_entropy_short_returns_nan() -> None:
    assert np.isnan(_sample_entropy(np.array([1.0, 2.0])))


def test_transfer_entropy_causal_direction() -> None:
    """TE from cause to effect should be greater than reverse."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * x[i - 1] + 0.2 * rng.normal()
    te_x_to_y = _transfer_entropy(x, y)
    te_y_to_x = _transfer_entropy(y, x)
    assert te_x_to_y > te_y_to_x


def test_transfer_entropy_short_returns_nan() -> None:
    assert np.isnan(_transfer_entropy(np.array([1.0, 2.0]), np.array([3.0, 4.0])))


# =========================================================================
# Tier 1: Fractional Differencing integration tests
# =========================================================================

def test_ffd_features_present(realistic_lag_result) -> None:
    assert "ffd_boi_brl" in realistic_lag_result.columns
    valid = realistic_lag_result["ffd_boi_brl"].dropna()
    assert len(valid) > 0


def test_ffd_conditional_columns() -> None:
    df = _make_realistic_weekly_df(130)
    result = add_lag_features(df, "weekly")
    assert "ffd_boi_usd" not in result.columns
    df["BOI_USD"] = df["target"] / 5.5
    result2 = add_lag_features(df, "weekly")
    assert "ffd_boi_usd" in result2.columns


# =========================================================================
# Tier 2: GARCH, Markov Switching, Hurst integration tests
# =========================================================================

def test_garch_features_present(realistic_lag_result) -> None:
    assert "garch_cond_var" in realistic_lag_result.columns
    valid = realistic_lag_result["garch_cond_var"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all(), "Conditional variance must be non-negative"


def test_gjr_asymmetry_time_varying(realistic_lag_result) -> None:
    assert "gjr_asymmetry" in realistic_lag_result.columns
    valid = realistic_lag_result["gjr_asymmetry"].dropna()
    assert len(valid) > 0
    assert valid.min() > 0
    assert valid.nunique() > 1, "gjr_asymmetry should be time-varying, not constant"


def test_markov_switching_bounded(realistic_lag_result) -> None:
    assert "ms_high_vol_prob" in realistic_lag_result.columns
    valid = realistic_lag_result["ms_high_vol_prob"].dropna()
    assert len(valid) > 0
    assert valid.min() >= -1e-10
    assert valid.max() <= 1.0 + 1e-10
    assert valid.nunique() > 1, "ms_high_vol_prob is constant — Markov fitting likely failed"


def test_hurst_column_bounded(realistic_lag_result) -> None:
    assert "hurst_52" in realistic_lag_result.columns
    valid = realistic_lag_result["hurst_52"].dropna()
    assert len(valid) > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0


# =========================================================================
# Tier 3: Sample Entropy, Transfer Entropy, DCC Correlation integration tests
# =========================================================================

def test_sample_entropy_column_present(realistic_lag_result) -> None:
    assert "sample_entropy_52" in realistic_lag_result.columns
    valid = realistic_lag_result["sample_entropy_52"].dropna()
    assert len(valid) > 0


def test_transfer_entropy_conditional(realistic_with_covariates_result) -> None:
    assert "te_bezerro_to_boi" in realistic_with_covariates_result.columns
    assert "te_boi_to_bezerro" in realistic_with_covariates_result.columns


def test_transfer_entropy_absent_without_covariates(realistic_lag_result) -> None:
    assert "te_bezerro_to_boi" not in realistic_lag_result.columns
    assert "te_usd_to_boi" not in realistic_lag_result.columns


def test_dcc_corr_bounded(realistic_with_covariates_result) -> None:
    assert "dcc_corr_boi_usd" in realistic_with_covariates_result.columns
    valid = realistic_with_covariates_result["dcc_corr_boi_usd"].dropna()
    assert len(valid) > 0
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01


def test_dcc_corr_dev_columns_present(realistic_with_covariates_result) -> None:
    assert "dcc_corr_dev_boi_usd" in realistic_with_covariates_result.columns
    assert "dcc_corr_dev_boi_bezerro" in realistic_with_covariates_result.columns
    for col in ["dcc_corr_dev_boi_usd", "dcc_corr_dev_boi_bezerro"]:
        valid = realistic_with_covariates_result[col].dropna()
        assert len(valid) > 10
        assert abs(valid.mean()) < 0.2


def test_te_usd_columns_present(realistic_with_covariates_result) -> None:
    assert "te_usd_to_boi" in realistic_with_covariates_result.columns
    assert "te_boi_to_usd" in realistic_with_covariates_result.columns


def test_dcc_corr_conditional(realistic_lag_result) -> None:
    assert "dcc_corr_boi_usd" not in realistic_lag_result.columns
    assert "dcc_corr_boi_bezerro" not in realistic_lag_result.columns


def test_tier_1_2_3_no_nans_after_warmup(realistic_lag_result) -> None:
    """After sufficient warm-up (row 85), all new features should have values."""
    strict_cols = ["ffd_boi_brl", "hurst_52", "garch_cond_var", "gjr_asymmetry", "ms_high_vol_prob"]
    tail = realistic_lag_result[strict_cols].iloc[85:]
    nans = tail.isna().sum()
    for col in strict_cols:
        assert nans[col] == 0, f"NaNs in {col} after row 85: {nans[col]}"
    se_tail = realistic_lag_result["sample_entropy_52"].iloc[85:]
    se_nan_pct = se_tail.isna().mean()
    assert se_nan_pct < 0.10, f"Too many NaNs in sample_entropy_52: {se_nan_pct:.1%}"


# =========================================================================
# Fallback path tests
# =========================================================================

def test_garch_fallback_on_short_series() -> None:
    """GARCH should use realized_vol_13**2 fallback when < 50 returns."""
    df = _make_weekly_df(48)  # 47 valid returns < 50 threshold
    result = add_lag_features(df, "weekly")
    assert "garch_cond_var" in result.columns
    # Fallback: garch_cond_var must equal realized_vol_13 ** 2
    expected = result["realized_vol_13"] ** 2
    pd.testing.assert_series_equal(
        result["garch_cond_var"], expected, check_names=False
    )


def test_markov_switching_fallback_short_series() -> None:
    """Markov Switching should use 0.5 fallback when series < 80 rows."""
    df = _make_realistic_weekly_df(70)
    result = add_lag_features(df, "weekly")
    assert "ms_high_vol_prob" in result.columns
    assert (result["ms_high_vol_prob"] == 0.5).all()


def test_ffd_skipped_on_short_series() -> None:
    """FFD should not produce columns when series has < 50 rows."""
    df = _make_realistic_weekly_df(40)
    result = add_lag_features(df, "weekly")
    assert "ffd_boi_brl" not in result.columns
