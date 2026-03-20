from __future__ import annotations

import pandas as pd


WEEKLY_KNOWN_COVARIATES = ["month", "quarter", "week_of_year"]


def known_covariates_for(granularity: str) -> list[str]:
    if granularity == "weekly":
        return list(WEEKLY_KNOWN_COVARIATES)
    raise ValueError(f"Unsupported granularity: {granularity}")


def add_calendar_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add calendar-based known covariates to an aggregated DataFrame with DatetimeIndex."""
    out = df.copy()
    idx = out.index
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    if granularity == "weekly":
        out["week_of_year"] = idx.isocalendar().week.astype(int)
    return out


def add_lag_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add rolling/lag past covariates to an aggregated DataFrame with DatetimeIndex."""
    out = df.copy()
    target = out["target"]

    if granularity == "weekly":
        out["rolling_mean_4"] = target.rolling(4, min_periods=1).mean()
        out["rolling_mean_13"] = target.rolling(13, min_periods=1).mean()
        out["rolling_std_4"] = target.rolling(4, min_periods=2).std()
        out["momentum_4"] = target.pct_change(4)

    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].bfill().ffill()

    return out


def engineer_features(df: pd.DataFrame, granularity: str) -> tuple[pd.DataFrame, list[str]]:
    """Full feature engineering pipeline for aggregated data.

    Returns the enriched DataFrame and the list of known covariate column names.
    """
    enriched = add_calendar_features(df, granularity)
    enriched = add_lag_features(enriched, granularity)
    return enriched, known_covariates_for(granularity)


def build_future_known_covariates(
    last_timestamp: pd.Timestamp,
    prediction_length: int,
    frequency: str,
    item_id: str,
    known_covariates_names: list[str],
) -> pd.DataFrame:
    """Generate a DataFrame of known covariate values for the forecast horizon."""
    future_dates = pd.date_range(start=last_timestamp, periods=prediction_length + 1, freq=frequency)[1:]
    future_df = pd.DataFrame({"item_id": item_id, "timestamp": future_dates})

    if "month" in known_covariates_names:
        future_df["month"] = future_dates.month
    if "quarter" in known_covariates_names:
        future_df["quarter"] = future_dates.quarter
    if "week_of_year" in known_covariates_names:
        future_df["week_of_year"] = future_dates.isocalendar().week.astype(int).values

    return future_df
