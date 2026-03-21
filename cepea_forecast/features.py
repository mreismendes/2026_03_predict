from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FOURIER_ORDER = 4
_DAYS_PER_YEAR = 365.25
_WEEKLY_WINDOWS = (4, 13, 26)
_FFD_MAPPING = {
    "target": "ffd_boi_brl",
    "BOI_USD": "ffd_boi_usd",
    "BRL_KG": "ffd_brl_kg",
}
_DCC_HALFLIFE = 26

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


def _align_to_index(
    values: np.ndarray, source_index: pd.Index, target_index: pd.Index
) -> pd.Series:
    """Create a NaN-initialized Series on target_index with values placed at source_index."""
    aligned = pd.Series(np.nan, index=target_index)
    aligned.loc[source_index] = values
    return aligned


def _ffd_weights(d: float, max_width: int = 52, threshold: float = 1e-4) -> np.ndarray:
    """Compute Fixed-width window Fractional Differencing weights (capped at max_width)."""
    weights = [1.0]
    k = 1
    while k < max_width:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])


def _ffd_transform(series: np.ndarray, d: float) -> np.ndarray:
    """Apply FFD filter with order d to a 1-D array."""
    w = _ffd_weights(d)
    width = len(w)
    n = len(series)
    if n < width:
        return np.full(n, np.nan)
    result = np.full(n, np.nan)
    result[width - 1 :] = np.convolve(series, w[::-1], mode="valid")
    return result


def _find_optimal_d(series: np.ndarray, max_d: float = 1.0, p_threshold: float = 0.05) -> float:
    """Find minimum d that makes the fractionally differenced series stationary (ADF test)."""
    from statsmodels.tsa.stattools import adfuller

    d_low, d_high = 0.0, max_d
    for _ in range(20):
        d_mid = (d_low + d_high) / 2
        transformed = _ffd_transform(series, d_mid)
        valid = transformed[~np.isnan(transformed)]
        if len(valid) < 20:
            d_low = d_mid
            continue
        try:
            adf_pvalue = adfuller(valid, maxlag=1, regression="c", autolag=None)[1]
        except Exception:
            d_low = d_mid
            continue
        if adf_pvalue < p_threshold:
            d_high = d_mid
        else:
            d_low = d_mid
    return d_high


def _hurst_rs(x: np.ndarray) -> float:
    """Estimate Hurst exponent via Rescaled Range (R/S) analysis."""
    n = len(x)
    if n < 20:
        return np.nan
    sizes = []
    rs_means = []
    size = n
    while size >= 8:
        num_blocks = n // size
        if num_blocks < 1:
            size //= 2
            continue
        rs_vals = []
        for i in range(num_blocks):
            block = x[i * size : (i + 1) * size]
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            sizes.append(size)
            rs_means.append(np.mean(rs_vals))
        size //= 2
    if len(sizes) < 3:
        return np.nan
    log_n = np.log(sizes)
    log_rs = np.log(rs_means)
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute sample entropy of a 1-D array."""
    n = len(x)
    if n < m + 2:
        return np.nan
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    r = r_factor * std

    n_templates = n - m - 1
    if n_templates < 2:
        return np.nan

    def _count_matches(dim: int) -> int:
        templates = np.array([x[i : i + dim] for i in range(n_templates)])
        # Vectorized pairwise Chebyshev distance (upper triangle only)
        dists = np.max(np.abs(templates[:, None] - templates[None, :]), axis=2)
        return int(np.sum(dists[np.triu_indices(n_templates, k=1)] < r))

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return -np.log(a / b)


def _transfer_entropy(
    source: np.ndarray, target: np.ndarray, n_bins: int = 5, lag: int = 1
) -> float:
    """Transfer entropy from source to target using quantile-binned data."""
    n = len(source)
    if n < lag + 10:
        return np.nan

    def _discretize(arr: np.ndarray) -> np.ndarray:
        edges = np.percentile(arr, np.linspace(0, 100, n_bins + 1))
        edges[-1] += 1e-10
        return np.clip(np.digitize(arr, edges[1:-1]), 0, n_bins - 1)

    sx = _discretize(source)
    sy = _discretize(target)

    y_now = sy[lag:].astype(int)
    y_past = sy[:-lag].astype(int)
    x_past = sx[:-lag].astype(int)
    nn = len(y_now)

    # Vectorized joint counts via np.add.at
    joint_xyy = np.zeros((n_bins, n_bins, n_bins))
    np.add.at(joint_xyy, (y_now, y_past, x_past), 1)
    joint_yy = joint_xyy.sum(axis=2)
    marg_y = joint_yy.sum(axis=0)

    te = 0.0
    for yn in range(n_bins):
        for yp in range(n_bins):
            if joint_yy[yn, yp] == 0:
                continue
            p_yn_given_yp = joint_yy[yn, yp] / marg_y[yp] if marg_y[yp] > 0 else 0
            for xp in range(n_bins):
                if joint_xyy[yn, yp, xp] == 0:
                    continue
                p_xyy = joint_xyy[yn, yp, xp] / nn
                p_yp_xp = np.sum(joint_xyy[:, yp, xp]) / nn
                if p_yp_xp == 0:
                    continue
                p_yn_given_yp_xp = joint_xyy[yn, yp, xp] / (p_yp_xp * nn)
                if p_yn_given_yp_xp > 0 and p_yn_given_yp > 0:
                    te += p_xyy * np.log(p_yn_given_yp_xp / p_yn_given_yp)
    return max(te, 0.0)


def _rolling_transfer_entropy(
    source_col: pd.Series,
    target_col: pd.Series,
    window: int = 52,
    min_periods: int = 26,
) -> pd.Series:
    """Compute rolling transfer entropy from source to target."""
    result = pd.Series(np.nan, index=source_col.index)
    for i in range(min_periods - 1, len(source_col)):
        start = max(0, i + 1 - window)
        s = source_col.values[start : i + 1]
        t = target_col.values[start : i + 1]
        if len(s) >= min_periods and not (np.any(np.isnan(s)) or np.any(np.isnan(t))):
            result.iloc[i] = _transfer_entropy(s, t)
    return result


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
            ratio_high = ratio.rolling(52, min_periods=1).max()
            ratio_low = ratio.rolling(52, min_periods=1).min()
            ratio_range = ratio_high - ratio_low
            out["ratio_range_position_52"] = np.where(
                ratio_range > 0,
                (ratio - ratio_low) / ratio_range,
                0.5,
            )

        # ================================================================
        # TIER 1: Fractional Differencing (long-memory preservation)
        # ================================================================
        for src_col, dst_col in _FFD_MAPPING.items():
            if src_col not in out.columns:
                continue
            series = out[src_col].dropna()
            if len(series) < 50:
                continue
            try:
                d_star = _find_optimal_d(series.values)
                if d_star >= 0.99:
                    logger.warning(
                        "FFD for %s: d*=%.3f (near full differencing); "
                        "series may be highly non-stationary", src_col, d_star
                    )
                transformed = _ffd_transform(series.values, d_star)
                out[dst_col] = _align_to_index(transformed, series.index, out.index)
            except Exception as e:
                logger.warning("FFD for %s skipped: %s", src_col, e)

        # ================================================================
        # TIER 2: GARCH Conditional Variance + GJR-GARCH Asymmetry
        # ================================================================
        returns = log_ret.dropna()

        try:
            from arch import arch_model

            if len(returns) >= 50:
                returns_pct = returns * 100

                # GARCH(1,1) — conditional variance
                try:
                    am = arch_model(
                        returns_pct, vol="GARCH", p=1, q=1,
                        dist="StudentsT", mean="Zero", rescale=False,
                    )
                    res = am.fit(disp="off", show_warning=False)
                    cond_var = (res.conditional_volatility ** 2) / 10000
                    out["garch_cond_var"] = _align_to_index(
                        cond_var.values, cond_var.index, out.index
                    )
                except Exception as e:
                    logger.warning("GARCH(1,1) failed, using fallback: %s", e)
                    out["garch_cond_var"] = out["realized_vol_13"] ** 2

                # GJR-GARCH(1,1) — leverage effect (time-varying asymmetry)
                try:
                    am_gjr = arch_model(
                        returns_pct, vol="GARCH", p=1, o=1, q=1,
                        dist="StudentsT", mean="Zero", rescale=False,
                    )
                    res_gjr = am_gjr.fit(disp="off", show_warning=False)
                    gjr_cond_var = (res_gjr.conditional_volatility ** 2) / 10000
                    gjr_aligned = _align_to_index(
                        gjr_cond_var.values, gjr_cond_var.index, out.index
                    )
                    garch_cv = out["garch_cond_var"]
                    out["gjr_asymmetry"] = np.where(
                        garch_cv > 0, gjr_aligned / garch_cv, 1.0,
                    )
                except Exception as e:
                    logger.warning("GJR-GARCH failed, using fallback: %s", e)
                    out["gjr_asymmetry"] = 1.0
            else:
                out["garch_cond_var"] = out["realized_vol_13"] ** 2
                out["gjr_asymmetry"] = 1.0
        except ImportError:
            logger.warning("arch not installed; GARCH features skipped")

        # ================================================================
        # TIER 2: Markov Switching Regime Probabilities
        # ================================================================
        try:
            from statsmodels.tsa.regime_switching.markov_regression import (
                MarkovRegression,
            )

            if len(returns) >= 80:
                try:
                    model = MarkovRegression(
                        returns.values, k_regimes=2, switching_variance=True,
                    )
                    res = model.fit(maxiter=200, disp=False)
                    probs = res.smoothed_marginal_probabilities
                    ret_vals = returns.values
                    var_by_regime = []
                    for regime in range(2):
                        w = probs[:, regime]
                        if w.sum() > 0:
                            mu = np.average(ret_vals, weights=w)
                            var_by_regime.append(
                                np.average((ret_vals - mu) ** 2, weights=w)
                            )
                        else:
                            var_by_regime.append(0.0)
                    high_vol_col = int(np.argmax(var_by_regime))
                    if len(probs) != len(returns):
                        raise ValueError(
                            f"Markov probs length {len(probs)} != returns {len(returns)}"
                        )
                    out["ms_high_vol_prob"] = _align_to_index(
                        probs[:, high_vol_col], returns.index, out.index
                    )
                except Exception as e:
                    logger.warning("Markov Switching failed, using fallback: %s", e)
                    out["ms_high_vol_prob"] = 0.5
            else:
                out["ms_high_vol_prob"] = 0.5
        except ImportError:
            logger.warning("statsmodels not available; Markov Switching skipped")

        # ================================================================
        # TIER 2: Rolling Hurst Exponent (persistence structure)
        # ================================================================
        out["hurst_52"] = log_ret.rolling(52, min_periods=40).apply(
            _hurst_rs, raw=True
        )

        # ================================================================
        # TIER 3: Sample Entropy (predictability indicator)
        # ================================================================
        out["sample_entropy_52"] = log_ret.rolling(52, min_periods=26).apply(
            _sample_entropy, raw=True
        )

        # ================================================================
        # TIER 3: Covariate log returns (computed once, reused below)
        # ================================================================
        _cov_log_rets = {
            "usd": (
                np.log(out["BOI_USD"] / out["BOI_USD"].shift(1))
                if "BOI_USD" in out.columns else None
            ),
            "bezerro": (
                np.log(out["BRL_KG"] / out["BRL_KG"].shift(1))
                if "BRL_KG" in out.columns else None
            ),
        }

        # ================================================================
        # TIER 3: Transfer Entropy + Dynamic Correlations (unified loop)
        # ================================================================
        _te_dcc_config = [
            ("usd", "boi_usd"),
            ("bezerro", "boi_bezerro"),
        ]
        for key, suffix in _te_dcc_config:
            cov_ret = _cov_log_rets[key]
            if cov_ret is None:
                continue
            # Transfer entropy (both directions)
            out[f"te_{key}_to_boi"] = _rolling_transfer_entropy(cov_ret, log_ret)
            out[f"te_boi_to_{key}"] = _rolling_transfer_entropy(log_ret, cov_ret)
            # EWMA correlation + deviation from expanding mean
            ewma_corr = log_ret.ewm(
                halflife=_DCC_HALFLIFE, min_periods=26
            ).corr(cov_ret)
            out[f"dcc_corr_{suffix}"] = ewma_corr
            out[f"dcc_corr_dev_{suffix}"] = (
                ewma_corr - ewma_corr.expanding(min_periods=26).mean()
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
