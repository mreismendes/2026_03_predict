# CEPEA Boi Gordo Forecast

Time-series forecasting of CEPEA cattle prices using AutoGluon TimeSeries.
Trains a single weekly model (52-period forecast horizon), generates predictions
with confidence intervals, and produces a PDF report.

## Commands

```bash
source .venv/bin/activate
python -m forecast_cli predict --base-dir .   # forecast using trained models (trains if needed)
python -m forecast_cli retrain --base-dir .    # force retrain all models
pytest                                         # run tests
pytest tests/test_data_io.py tests/test_features.py tests/test_launcher.py  # fast tests only
pytest -k "not forecasting and not reporting and not pipeline"              # same, by exclusion
```

## Architecture

- `forecast_cli.py` - Entry point; auto-activates .venv, sets MPLCONFIGDIR/LOKY_MAX_CPU_COUNT
- `cepea_forecast/cli.py` - Argparse CLI (predict | retrain)
- `cepea_forecast/config.py` - Path layout (AppPaths), forecast lengths, preset (currently `fast_training` for dev)
- `cepea_forecast/data_io.py` - Reads .xls/.xlsx/.csv, auto-detects header row (tries 0-4), parses date+target+covariates
- `cepea_forecast/features.py` - Feature engineering: calendar known covariates + rolling/lag past covariates
- `cepea_forecast/forecasting.py` - AutoGluon training/prediction with known_covariates, MODEL_SPECS, aggregation
- `cepea_forecast/pipeline.py` - Orchestrates data loading, feature engineering, training, forecasting, PDF generation
- `cepea_forecast/reporting.py` - PDF report with matplotlib plots and reportlab tables

## Data Flow

`data/*.xls` -> load_source_data (daily) -> aggregate weekly (W-FRI) -> engineer_features (calendar + lag) -> AutoGluon train/predict -> CSV + PDF

## Feature Engineering

**Known covariates** (deterministic, available for future via Fourier decomposition):
- sin_yearly_1..4, cos_yearly_1..4 (smooth yearly seasonality, replaces integer week_of_year)

**Past covariates** (history only, auto-detected by AutoGluon):
- Trend: rolling_mean_4, rolling_mean_13, rolling_mean_26
- Momentum: momentum_4, momentum_13, momentum_26
- MA crossover: ma_ratio_4_13, ma_ratio_4_26
- Returns: log_return_1
- Volatility: realized_vol_4, realized_vol_13 (return-based, improves WQL interval calibration)
- Seasonal: yoy_change (52-week pct_change)
- Regime: range_position_52 (price position within 52-week high/low [0,1])
- Oscillator: rsi_14 (14-week RSI, overbought/oversold signal)

## Key Directories

- `data/` - Input: CEPEA spreadsheet (newest file by mtime is used)
- `artifacts/models/` - Trained AutoGluon models + metadata.json per model
- `artifacts/predictions/` - latest_forecast.csv
- `output/pdf/` - latest_forecast_report.pdf
- `tmp/pdfs/` - Temporary plot images during report generation

All output directories (`artifacts/`, `output/`, `tmp/`) are gitignored. `data/` is committed.

## Development Workflow

- Preset is `fast_training` during development (change to `best_quality` for production in config.py)
- Eval metric: WQL (Weighted Quantile Loss) for probabilistic forecast quality

## Gotchas

- Data file is auto-selected by newest modification time in `data/` - only .xls/.xlsx/.csv supported
- `predict` auto-trains if any model is missing (checks metadata.json existence)
- Weekly aggregation uses W-FRI (Friday end), drops incomplete trailing period
- Numeric covariate detection threshold: 60% non-null after coercion
- Brazilian number format handled: strips R$, resolves comma/dot ambiguity
- `forecast_cli.py` uses os.execv to re-exec under .venv python if not already in it
- LOKY_MAX_CPU_COUNT set to physical cores on macOS (via sysctl hw.physicalcpu)
- `known_covariates` must be provided at predict time via `build_future_known_covariates()`
- AutoGluon import is very slow (~3 min); tests that touch forecasting.py take ~3 min total
- Config constant is `DEFAULT_AUTOGUON_PRESET` (typo of AUTOGLUON) — used in config.py:6, forecasting.py:10,163; do not rename without updating all 3 references

## Dependencies

Python 3.11 | autogluon.timeseries, pandas, xlrd, openpyxl, python-dateutil, reportlab, matplotlib, pytest
