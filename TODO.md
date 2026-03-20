# TODO — Forecast Pipeline Improvements

## Priority 1: Per-Horizon Quantile Recalibration
**Expected WQL gain: 5-15% | Effort: 4-6 hrs**

Step-52 intervals are likely miscalibrated. Fix post-hoc using isotonic regression on a calibration set.
- Reserve last ~250 weeks for rolling-origin calibration
- For each (quantile_level, horizon_step): fit `sklearn.isotonic.IsotonicRegression` on predicted vs actual
- Apply recalibration to production forecasts
- Always enforce quantile monotonicity after recalibration (`np.sort([q01, q05, q09])`)
- Library: scikit-learn (already available)

## Priority 2: NeuralForecast Models (N-HiTS, iTransformer, TiDE)
**Expected WQL gain: 3-8% | Effort: 6-8 hrs**

These architectures are NOT in AutoGluon and add genuine model diversity:
- **N-HiTS**: hierarchical interpolation, excels at long horizons (52 weeks)
- **iTransformer**: inverted Transformer, best variant for multivariate data
- **TiDE**: Google's simple MLP that often beats Transformers

Implementation:
- `pip install neuralforecast`
- Train NeuralForecast models alongside AutoGluon
- Combine quantile outputs via simple averaging or QRA (quantile regression averaging)
- Library: neuralforecast (PyTorch Lightning)

## Priority 3: Quantile Monotonicity Enforcement
**Expected WQL gain: 0-2% | Effort: 30 min**

Guard against quantile crossings (q0.1 > q0.5) after any recalibration step.
```python
q01, q05, q09 = np.sort([q01, q05, q09])
```
Apply as the final post-processing step in the pipeline.

## Priority 4: Orbit Bayesian Models (DLT, LGT)
**Expected WQL gain: 2-5% | Effort: 4-6 hrs**

Structurally different from all AutoGluon models — full Bayesian posterior distributions, time-varying coefficients via Stan/Pyro.
- `pip install orbit-ml`
- DLT (Damped Local Trend) handles level shifts well
- LGT (Local Global Trend) captures regime changes
- KTR (Kernel Time-varying Regression) — unique: allows time-varying covariate effects
- Add to outer ensemble for forecast diversity

## Priority 5: MAPIE Conformal Prediction (EnbPI)
**Expected WQL gain: 3-8% | Effort: 2-3 hrs**

Coverage guarantee — ensures the 80% interval actually covers 80% of outcomes.
- `pip install mapie`
- Use `MapieTimeSeriesRegressor` with EnbPI method
- Apply as post-processing wrapper on AutoGluon point forecasts
- ACI (Adaptive Conformal Inference) variant adapts to distribution shift

---

## Not Recommended (with rationale)

| Approach | Why skip |
|----------|---------|
| Prophet / NeuralProphet | Designed for business metrics, not commodity prices |
| TimeGPT (Nixtla) | API-only, vendor lock-in, sends data to third party |
| LLM-based forecasting | No numerical precision (character-level tokenization) |
| Moirai / TimesFM / Lag-Llama / MOMENT / ForecastPFN | Not production-ready, redundant with Chronos, or wrong tool (short-context) |
| MLForecast (Nixtla) | AutoGluon TabularModels already cover this |
| StatsForecast (Nixtla) | 90% overlap with AutoGluon statistical models |
| STL decomposition as features | Lookahead bias (centered moving averages); Fourier + rolling means already decompose the signal correctly |
