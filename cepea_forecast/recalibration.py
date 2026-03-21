"""Per-horizon quantile recalibration using isotonic regression.

Workflow
--------
1. After training an AutoGluon model, call ``fit_recalibrators`` which runs
   rolling-origin backtesting via ``predictor.backtest_predictions()`` /
   ``predictor.backtest_targets()`` to collect (predicted_quantile, actual)
   pairs for every (quantile_level, forecast_step) combination.
2. Data is POOLED across all steps per quantile level, fitting just 3
   ``IsotonicRegression`` models (one per quantile) with ~1,000+ samples each.
   This avoids the overfitting/oscillation that occurs with 156 per-step models.
3. At production time, ``apply_recalibration`` takes the raw forecast DataFrame
   and returns a corrected copy with monotonicity enforced (q0.1 <= q0.5 <= q0.9).
4. The fitted calibrators are persisted alongside the model as a single pickle file.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cepea_forecast.config import QUANTILE_LEVELS as _QUANTILE_LEVELS_FLOAT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

QUANTILE_LEVELS: list[str] = [str(q) for q in _QUANTILE_LEVELS_FLOAT]
"""Quantile column names as they appear in AutoGluon output (derived from config)."""

CALIBRATION_FILENAME = "quantile_calibrators.pkl"


@dataclass
class CalibrationBundle:
    """Container for fitted IsotonicRegression models (3 pooled, stored per step).

    Layout: ``models[(quantile_col, step)] = IsotonicRegression``
    where *quantile_col* is one of ``"0.1"``, ``"0.5"``, ``"0.9"`` and
    *step* is 1-based (1 .. prediction_length). All steps for the same
    quantile share the same pooled IsotonicRegression instance.
    """

    prediction_length: int
    quantile_levels: list[str] = field(default_factory=lambda: list(QUANTILE_LEVELS))
    models: dict[tuple[str, int], IsotonicRegression] = field(default_factory=dict)
    n_origins: int = 0
    min_samples_per_cell: int = 0

    @property
    def n_models(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Step 1: Collect (predicted, actual) pairs via rolling-origin backtest
# ---------------------------------------------------------------------------

def _collect_backtest_pairs(
    predictor: Any,
    train_data: Any,
    known_covariates_names: list[str] | None,
    num_val_windows: int,
    prediction_length: int,
) -> dict[tuple[str, int], tuple[list[float], list[float]]]:
    """Run backtest and collect per-(quantile, step) observation pairs.

    Returns
    -------
    dict mapping ``(quantile_col, step)`` to ``(predicted_values, actual_values)``
    where each is a list of floats with length == num_val_windows.
    """
    from autogluon.timeseries import TimeSeriesDataFrame

    # backtest_predictions returns list[TimeSeriesDataFrame], one per window
    # Each has columns: "mean", "0.1", "0.5", "0.9" (matching quantile_levels)
    # Index: (item_id, timestamp)
    predictions_list: list = predictor.backtest_predictions(
        data=train_data,
        num_val_windows=num_val_windows,
        use_cache=False,
    )
    targets_list: list = predictor.backtest_targets(
        data=train_data,
        num_val_windows=num_val_windows,
    )

    # Accumulate pairs keyed by (quantile_col, 1-based step)
    pairs: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    for q in QUANTILE_LEVELS:
        for step in range(1, prediction_length + 1):
            pairs[(q, step)] = ([], [])

    for window_idx, (preds_df, targets_df) in enumerate(
        zip(predictions_list, targets_list)
    ):
        # preds_df is a TimeSeriesDataFrame with multi-index (item_id, timestamp)
        # For single-item series, reset index to get a flat DataFrame
        preds_flat = preds_df.reset_index()
        targets_flat = targets_df.reset_index()

        # Sort by timestamp to ensure step ordering
        preds_flat = preds_flat.sort_values("timestamp").reset_index(drop=True)
        targets_flat = targets_flat.sort_values("timestamp").reset_index(drop=True)

        # The predictions DataFrame should have exactly prediction_length rows
        n_steps = min(len(preds_flat), prediction_length)
        if len(preds_flat) != prediction_length:
            logger.warning(
                "Backtest window %d: expected %d prediction rows, got %d",
                window_idx, prediction_length, len(preds_flat),
            )

        # Align actuals: targets_df contains the full history + holdout.
        # The last prediction_length timestamps in targets correspond to the forecast.
        # We need to match by timestamp.
        target_lookup = dict(
            zip(targets_flat["timestamp"], targets_flat["target"])
        )

        for step_idx in range(n_steps):
            step = step_idx + 1  # 1-based
            ts = preds_flat.loc[step_idx, "timestamp"]
            actual = target_lookup.get(ts)
            if actual is None or pd.isna(actual):
                continue
            for q in QUANTILE_LEVELS:
                predicted_q = preds_flat.loc[step_idx, q]
                if pd.notna(predicted_q):
                    pairs[(q, step)][0].append(float(predicted_q))
                    pairs[(q, step)][1].append(float(actual))

    return pairs


# ---------------------------------------------------------------------------
# Step 2: Fit IsotonicRegression per (quantile, step)
# ---------------------------------------------------------------------------

def _fit_single_isotonic(
    predicted: list[float],
    actual: list[float],
    quantile_level: float,
) -> IsotonicRegression | None:
    """Fit one IsotonicRegression mapping predicted quantile -> recalibrated.

    For quantile recalibration the target is:
        y_calib = actual value (what we want the quantile to better represent)
    and X = predicted quantile value.

    The isotonic model learns a monotonically increasing mapping from
    predicted -> recalibrated so that coverage matches the nominal level.

    Parameters
    ----------
    predicted : predicted quantile values from backtest
    actual : corresponding true values
    quantile_level : e.g. 0.1 -- not used directly but could inform direction

    Returns None if fewer than 5 observations (not enough to fit reliably).
    """
    if len(predicted) < 5:
        return None

    X = np.array(predicted, dtype=np.float64)
    y = np.array(actual, dtype=np.float64)

    iso = IsotonicRegression(
        increasing=True,       # predicted up => actual up (monotonic prices)
        out_of_bounds="clip",  # extrapolate by clamping to boundary values
    )
    iso.fit(X, y)
    return iso


def fit_recalibrators(
    predictor: Any,
    train_data: Any,
    prediction_length: int,
    known_covariates_names: list[str] | None = None,
    num_val_windows: int | None = None,
) -> CalibrationBundle:
    """Fit pooled quantile recalibration models from backtest residuals.

    Instead of fitting 156 per-(quantile, step) models (which overfit with
    ~20 samples each), this pools ALL steps together and fits just 3 models
    (one per quantile level). Each model gets ~20 windows x 52 steps ≈ 1,000+
    data points, producing smooth, consistent recalibration across the horizon.

    The same pooled model is then stored for every step, so apply_recalibration
    works unchanged.
    """
    if num_val_windows is None:
        try:
            n_rows = len(train_data)
        except TypeError:
            n_rows = 1500
        max_windows = max(1, (n_rows - 2 * prediction_length) // prediction_length)
        num_val_windows = min(26, max_windows)

    logger.info(
        "Collecting backtest pairs: %d windows x %d steps x %d quantiles (pooled)",
        num_val_windows,
        prediction_length,
        len(QUANTILE_LEVELS),
    )

    pairs = _collect_backtest_pairs(
        predictor=predictor,
        train_data=train_data,
        known_covariates_names=known_covariates_names,
        num_val_windows=num_val_windows,
        prediction_length=prediction_length,
    )

    bundle = CalibrationBundle(
        prediction_length=prediction_length,
        n_origins=num_val_windows,
    )

    # Pool all steps for each quantile level → 3 models instead of 156
    for q_col in QUANTILE_LEVELS:
        pooled_pred: list[float] = []
        pooled_actual: list[float] = []
        for step in range(1, prediction_length + 1):
            pred_vals, act_vals = pairs.get((q_col, step), ([], []))
            pooled_pred.extend(pred_vals)
            pooled_actual.extend(act_vals)

        q_level = float(q_col)
        iso = _fit_single_isotonic(pooled_pred, pooled_actual, q_level)
        if iso is not None:
            # Store the SAME pooled model for every step (smooth, consistent)
            for step in range(1, prediction_length + 1):
                bundle.models[(q_col, step)] = iso
            logger.info(
                "Fitted pooled calibrator for q=%s: %d samples",
                q_col, len(pooled_pred),
            )
        else:
            logger.warning(
                "Skipping calibrator for q=%s: only %d pooled samples",
                q_col, len(pooled_pred),
            )

    bundle.min_samples_per_cell = min(
        (len(pairs.get((q, 1), ([], []))[0]) for q in QUANTILE_LEVELS),
        default=0,
    )

    logger.info(
        "Fitted %d calibrators (3 pooled models x %d steps, %d total samples per quantile)",
        bundle.n_models,
        prediction_length,
        sum(len(pairs.get((QUANTILE_LEVELS[0], s), ([], []))[0]) for s in range(1, prediction_length + 1)),
    )

    return bundle


# ---------------------------------------------------------------------------
# Step 3: Apply recalibration to production forecast
# ---------------------------------------------------------------------------

def apply_recalibration(
    forecast_frame: pd.DataFrame,
    bundle: CalibrationBundle,
) -> pd.DataFrame:
    """Apply fitted isotonic calibrators to a forecast DataFrame.

    Parameters
    ----------
    forecast_frame
        DataFrame with columns ``"mean"``, ``"0.1"``, ``"0.5"``, ``"0.9"``
        and a ``"timestamp"`` column.  Rows are ordered by step (1..prediction_length).
    bundle
        The fitted ``CalibrationBundle``.

    Returns
    -------
    pd.DataFrame
        Copy of *forecast_frame* with quantile columns recalibrated and
        monotonicity enforced (q0.1 <= q0.5 <= q0.9 at every step).
    """
    out = forecast_frame.copy()

    # Apply per-(quantile, step) isotonic recalibration
    for step_idx, row_idx in enumerate(out.index):
        step = step_idx + 1
        for q_col in QUANTILE_LEVELS:
            iso = bundle.models.get((q_col, step))
            if iso is None:
                continue
            raw_val = out.at[row_idx, q_col]
            if pd.notna(raw_val):
                out.at[row_idx, q_col] = iso.predict([float(raw_val)])[0]

    # Enforce monotonicity: q0.1 <= q0.5 <= q0.9 (vectorized)
    out[QUANTILE_LEVELS] = np.sort(out[QUANTILE_LEVELS].values, axis=1)

    # Clamp mean to be within the quantile range (sanity check)
    if "mean" in out.columns:
        outside = (out["mean"] < out[QUANTILE_LEVELS[0]]) | (out["mean"] > out[QUANTILE_LEVELS[-1]])
        if outside.any():
            logger.warning(
                "Mean forecast outside [q%s, q%s] at %d/%d steps after recalibration; clamping.",
                QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1], int(outside.sum()), len(out),
            )
        out["mean"] = out["mean"].clip(lower=out[QUANTILE_LEVELS[0]], upper=out[QUANTILE_LEVELS[-1]])

    return out


# ---------------------------------------------------------------------------
# Step 4: Persistence
# ---------------------------------------------------------------------------

def save_calibration(bundle: CalibrationBundle, model_dir: Path) -> Path:
    """Pickle the CalibrationBundle next to the AutoGluon model."""
    path = model_dir / CALIBRATION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved calibration bundle to %s (%d models)", path, bundle.n_models)
    return path


def load_calibration(model_dir: Path) -> CalibrationBundle | None:
    """Load a previously saved CalibrationBundle, or return None if absent."""
    path = model_dir / CALIBRATION_FILENAME
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Loaded calibration bundle from %s (%d models)", path, bundle.n_models)
    return bundle


def calibration_ready(model_dir: Path) -> bool:
    """Check whether a saved calibration file exists."""
    return (model_dir / CALIBRATION_FILENAME).exists()
