from __future__ import annotations

import numpy as np
import pandas as pd

FOURIER_ORDER = 4
_DAYS_PER_YEAR = 365.25
_WEEKLY_WINDOWS = (4, 13, 26)

WEEKLY_KNOWN_COVARIATES = [
    f"sin_yearly_{k}" for k in range(1, FOURIER_ORDER + 1)
] + [
    f"cos_yearly_{k}" for k in range(1, FOURIER_ORDER + 1)
]


def known_covariates_for(granularity: str) -> list[str]:
    if granularity == "weekly":
        return list(WEEKLY_KNOWN_COVARIATES)
    raise ValueError(f"Unsupported granularity: {granularity}")


def _day_of_year_fraction(idx: pd.DatetimeIndex) -> np.ndarray:
    """Return fractional position within the year for each timestamp."""
    return idx.dayofyear.values / _DAYS_PER_YEAR


def add_calendar_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add Fourier-based known covariates to an aggregated DataFrame with DatetimeIndex."""
    out = df.copy()
    if granularity == "weekly":
        t_frac = _day_of_year_fraction(out.index)
        for k in range(1, FOURIER_ORDER + 1):
            out[f"sin_yearly_{k}"] = np.sin(2 * np.pi * k * t_frac)
            out[f"cos_yearly_{k}"] = np.cos(2 * np.pi * k * t_frac)
    return out


def add_lag_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add rolling/lag past covariates to an aggregated DataFrame with DatetimeIndex."""
    out = df.copy()
    target = out["target"]

    if granularity == "weekly":
        # --- Trend / momentum features ---
        for w in _WEEKLY_WINDOWS:
            out[f"rolling_mean_{w}"] = target.rolling(w, min_periods=1).mean()
            out[f"momentum_{w}"] = target.pct_change(w)

        # --- MA crossover ratios (trend direction) ---
        out["ma_ratio_4_13"] = out["rolling_mean_4"] / out["rolling_mean_13"]
        out["ma_ratio_4_26"] = out["rolling_mean_4"] / out["rolling_mean_26"]

        # --- Log returns ---
        out["log_return_1"] = np.log(target / target.shift(1))

        # --- Return-based volatility (improves WQL interval calibration) ---
        log_ret = out["log_return_1"]
        out["realized_vol_4"] = log_ret.rolling(4, min_periods=2).std()
        out["realized_vol_13"] = log_ret.rolling(13, min_periods=4).std()

        # --- Year-over-year change (seasonal deviation) ---
        out["yoy_change"] = target.pct_change(52)

        # --- Range position within 52-week high/low [0,1] ---
        high_52 = target.rolling(52, min_periods=1).max()
        low_52 = target.rolling(52, min_periods=1).min()
        price_range = high_52 - low_52
        out["range_position_52"] = np.where(
            price_range > 0,
            (target - low_52) / price_range,
            0.5,
        )

        # --- RSI 14-week (overbought/oversold oscillator) ---
        delta = target.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=14, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(span=14, min_periods=1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # When only gains: rs=inf, rsi=100 naturally. When both NaN (first row),
        # fallback to 50.0 (neutral). The avg_gain>0 branch is a safety net.
        fallback = pd.Series(
            np.where(avg_gain > 0, 100.0, 50.0), index=rsi.index
        )
        out["rsi_14"] = rsi.where(rsi.notna(), fallback)

        # --- Long-horizon volatility (improves WQL at steps 26-52) ---
        out["realized_vol_26"] = log_ret.rolling(26, min_periods=8).std()
        out["realized_vol_52"] = log_ret.rolling(52, min_periods=16).std()
        out["vol_ratio_4_13"] = out["realized_vol_4"] / out["realized_vol_13"]

        # --- Acceleration (second derivative of trend) ---
        out["acceleration_4"] = out["momentum_4"] - out["momentum_4"].shift(4)
        out["acceleration_13"] = out["momentum_13"] - out["momentum_13"].shift(13)
        out["momentum_divergence"] = out["momentum_4"] - out["momentum_13"]

        # --- Cross-asset: FX features (if USD column present) ---
        if "USD" in out.columns:
            usd = out["USD"]
            out["usd_momentum_13"] = usd.pct_change(13)
            out["fx_adjusted_return_4"] = target.pct_change(4) - usd.pct_change(4)
        if "BOI_USD" in out.columns:
            out["boi_usd_momentum_13"] = out["BOI_USD"].pct_change(13)

        # --- Cross-asset: Bezerro cycle features (if BRL_KG column present) ---
        if "BRL_KG" in out.columns:
            brl_kg = out["BRL_KG"]
            out["bezerro_momentum_13"] = brl_kg.pct_change(13)
            out["bezerro_momentum_26"] = brl_kg.pct_change(26)
        if "BOI_BEZERRO_RATIO" in out.columns:
            ratio = out["BOI_BEZERRO_RATIO"]
            out["ratio_momentum_13"] = ratio.pct_change(13)
            out["ratio_momentum_26"] = ratio.pct_change(26)
            # Range position of the spread within its 52-week band
            ratio_high = ratio.rolling(52, min_periods=1).max()
            ratio_low = ratio.rolling(52, min_periods=1).min()
            ratio_range = ratio_high - ratio_low
            out["ratio_range_position_52"] = np.where(
                ratio_range > 0,
                (ratio - ratio_low) / ratio_range,
                0.5,
            )

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

    t_frac = _day_of_year_fraction(future_dates)
    for k in range(1, FOURIER_ORDER + 1):
        sin_name = f"sin_yearly_{k}"
        cos_name = f"cos_yearly_{k}"
        if sin_name in known_covariates_names:
            future_df[sin_name] = np.sin(2 * np.pi * k * t_frac)
        if cos_name in known_covariates_names:
            future_df[cos_name] = np.cos(2 * np.pi * k * t_frac)

    return future_df
