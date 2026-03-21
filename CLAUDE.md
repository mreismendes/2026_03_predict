# CEPEA Boi Gordo Forecast

Time-series forecasting of CEPEA cattle prices using AutoGluon TimeSeries.
Trains a single weekly model (52-period forecast horizon), generates predictions
with confidence intervals, and produces a PDF report.

## Commands

```bash
source .venv/bin/activate
python -m forecast_cli --base-dir .            # always trains + predicts (single command)
pytest                                         # run tests
pytest tests/test_data_io.py tests/test_features.py tests/test_launcher.py  # fast tests only
pytest -k "not forecasting and not reporting and not pipeline"              # same, by exclusion
```

## Architecture

- `forecast_cli.py` - Entry point; auto-activates .venv, sets MPLCONFIGDIR/LOKY_MAX_CPU_COUNT
- `cepea_forecast/cli.py` - Argparse CLI (--base-dir), calls pipeline.run()
- `cepea_forecast/config.py` - Path layout (AppPaths), forecast lengths, preset, AUTOGLUON_HYPERPARAMETERS (27 models)
- `cepea_forecast/data_io.py` - Loads CEPEA_BOI + CEPEA_BEZERRO files, merges on date (left join), BOI_BRL is target
- `cepea_forecast/features.py` - Feature engineering: calendar known covariates + rolling/lag past covariates
- `cepea_forecast/forecasting.py` - AutoGluon training/prediction with known_covariates, MODEL_SPECS, aggregation
- `cepea_forecast/pipeline.py` - Orchestrates data loading, feature engineering, training, forecasting, PDF generation
- `cepea_forecast/reporting.py` - PDF report with matplotlib plots and reportlab tables

## Data Flow

`data/CEPEA_BOI.xls` + `data/CEPEA_BEZERRO.xls` -> merge on date (left join) -> aggregate weekly (W-FRI) -> engineer_features (Fourier + lag) -> AutoGluon train/predict -> CSV + PDF

## Feature Engineering

**Known covariates** (deterministic, available for future via Fourier decomposition):
- sin_yearly_1..4, cos_yearly_1..4 (smooth yearly seasonality, replaces integer week_of_year)

**Source covariates** (from data files, auto-detected as past covariates):
- BOI_USD, USD, BEZERRO_BRL, BEZERRO_PESO, BRL_KG, BOI_BEZERRO_RATIO (BOI_BRL / BRL_KG)

**Engineered past covariates** (history only, auto-detected by AutoGluon):
- Trend: rolling_mean_4/13/26
- Momentum: momentum_4/13/26
- MA crossover: ma_ratio_4_13, ma_ratio_4_26
- Returns: log_return_1
- Volatility: realized_vol_4/13/26/52, vol_ratio_4_13
- Seasonal: yoy_change (52-week pct_change)
- Regime: range_position_52 (price position within 52-week high/low [0,1])
- Oscillator: rsi_14 (14-week RSI)
- Acceleration: acceleration_4/13, momentum_divergence (second derivative of trend)
- FX dynamics: usd_momentum_13, fx_adjusted_return_4, boi_usd_momentum_13 (if USD/BOI_USD columns present)
- Cattle cycle: bezerro_momentum_13/26, ratio_momentum_13/26, ratio_range_position_52 (if BEZERRO columns present)
- Long-memory (FFD): ffd_boi_brl, ffd_boi_usd, ffd_brl_kg (fractional differencing preserving memory structure; conditional on column presence; ADF-based optimal d*, max window 52)
- GARCH regime: garch_cond_var (GARCH(1,1) conditional variance), gjr_asymmetry (GJR/GARCH conditional variance ratio — leverage effect measure; arch library)
- Markov regime: ms_high_vol_prob (2-state Markov Switching smoothed probability of high-volatility state; statsmodels)
- Persistence: hurst_52 (52-week rolling Hurst exponent via R/S analysis; H>0.5=trending, H<0.5=mean-reverting)
- Predictability: sample_entropy_52 (rolling 52-week sample entropy on log returns; m=2, r=0.2×std)
- Nonlinear causality: te_bezerro_to_boi, te_boi_to_bezerro, te_usd_to_boi, te_boi_to_usd (rolling 52-week transfer entropy; conditional on column presence)
- Dynamic correlation: dcc_corr_boi_usd, dcc_corr_dev_boi_usd, dcc_corr_boi_bezerro, dcc_corr_dev_boi_bezerro (EWMA correlation halflife=26 + deviation from expanding mean; conditional on column presence)

## Key Directories

- `data/` - Input: CEPEA_BOI.xls (required) + CEPEA_BEZERRO.xls (optional), matched by name
- `artifacts/models/` - Trained AutoGluon models + metadata.json per model
- `artifacts/predictions/` - latest_forecast.csv
- `output/pdf/` - latest_forecast_report.pdf
- `tmp/pdfs/` - Temporary plot images during report generation

All output directories (`artifacts/`, `output/`, `tmp/`) are gitignored. `data/` is committed.

## Development Workflow

- Preset is `best_quality` for production; `AUTOGLUON_HYPERPARAMETERS` in config.py lists all 27 models explicitly
- `TRAINING_TIME_LIMIT` in config.py controls max training time (default None = unlimited)
- Eval metric: WQL (Weighted Quantile Loss) for probabilistic forecast quality

## Gotchas

- Data files must be named with `BOI` and `BEZERRO` in the filename (e.g., `CEPEA_BOI.xls`, `CEPEA_BEZERRO.xls`)
- BOI is required, BEZERRO is optional; merged via left join on date (BEZERRO NaN for dates before 2000)
- `run()` always retrains from scratch then predicts — no conditional skip
- Weekly aggregation uses W-FRI (Friday end), drops incomplete trailing period
- Numeric covariate detection threshold: 60% non-null after coercion
- Brazilian number format handled: strips R$, resolves comma/dot ambiguity
- `forecast_cli.py` uses os.execv to re-exec under .venv python if not already in it
- LOKY_MAX_CPU_COUNT set to physical cores on macOS (via sysctl hw.physicalcpu)
- `known_covariates` must be provided at predict time via `build_future_known_covariates()`
- Pretrained models (Chronos) download weights on first use (~1GB); first training requires internet
- Training 27 models with `best_quality` takes 30-60+ min; set `TRAINING_TIME_LIMIT` in config.py to cap it
- AutoGluon import is very slow (~3 min); tests that touch forecasting.py take ~3 min total
- Config constant is `DEFAULT_AUTOGUON_PRESET` (typo of AUTOGLUON) — used in config.py, forecasting.py (import + train_model default); do not rename without updating all references

## Dependencies

Python 3.11 | autogluon.timeseries, pandas, xlrd, openpyxl, python-dateutil, reportlab, matplotlib, scikit-learn, arch, statsmodels, pytest
