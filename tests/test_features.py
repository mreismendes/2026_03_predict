from __future__ import annotations

import pandas as pd

from cepea_forecast.features import (
    add_calendar_features,
    add_lag_features,
    build_future_known_covariates,
    engineer_features,
    known_covariates_for,
)


def _make_weekly_df(n_weeks: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2025-01-03", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame({"target": range(100, 100 + n_weeks)}, index=dates)


def test_known_covariates_for_weekly() -> None:
    names = known_covariates_for("weekly")
    assert "month" in names
    assert "quarter" in names
    assert "week_of_year" in names


def test_add_calendar_features_weekly() -> None:
    df = _make_weekly_df()
    result = add_calendar_features(df, "weekly")
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "week_of_year" in result.columns
    assert result["month"].iloc[0] == df.index[0].month
    assert result["quarter"].iloc[0] == df.index[0].quarter


def test_add_lag_features_weekly() -> None:
    df = _make_weekly_df()
    result = add_lag_features(df, "weekly")
    assert "rolling_mean_4" in result.columns
    assert "rolling_mean_13" in result.columns
    assert "rolling_std_4" in result.columns
    assert "momentum_4" in result.columns
    assert result.isna().sum().sum() == 0


def test_engineer_features_returns_known_cov_names() -> None:
    df = _make_weekly_df()
    enriched, known_names = engineer_features(df, "weekly")
    assert known_names == ["month", "quarter", "week_of_year"]
    assert all(name in enriched.columns for name in known_names)
    assert "rolling_mean_4" in enriched.columns
    assert "target" in enriched.columns


def test_build_future_known_covariates_weekly() -> None:
    last_ts = pd.Timestamp("2026-02-27")
    future = build_future_known_covariates(
        last_timestamp=last_ts,
        prediction_length=4,
        frequency="W-FRI",
        item_id="test_item",
        known_covariates_names=["month", "quarter", "week_of_year"],
    )
    assert len(future) == 4
    assert "item_id" in future.columns
    assert "timestamp" in future.columns
    assert "month" in future.columns
    assert "quarter" in future.columns
    assert "week_of_year" in future.columns
    assert (future["item_id"] == "test_item").all()
