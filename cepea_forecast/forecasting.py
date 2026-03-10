from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cepea_forecast.config import DEFAULT_AUTOGUON_PRESET, MONTHLY_FORECAST_LENGTHS, WEEKLY_FORECAST_LENGTHS


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    granularity: str
    frequency: str
    prediction_length: int
    seasonal_period: int
    item_id: str
    title: str


@dataclass(frozen=True)
class ForecastBundle:
    spec: ModelSpec
    history: pd.Series
    forecast_frame: pd.DataFrame
    rows: pd.DataFrame
    source_file: Path
    model_dir: Path
    metadata: dict[str, str]


MODEL_SPECS = {
    "weekly_26": ModelSpec(
        model_id="weekly_26",
        granularity="weekly",
        frequency="W-FRI",
        prediction_length=WEEKLY_FORECAST_LENGTHS[0],
        seasonal_period=52,
        item_id="target_weekly_26",
        title="Weekly forecast - 26 periods",
    ),
    "weekly_52": ModelSpec(
        model_id="weekly_52",
        granularity="weekly",
        frequency="W-FRI",
        prediction_length=WEEKLY_FORECAST_LENGTHS[1],
        seasonal_period=52,
        item_id="target_weekly_52",
        title="Weekly forecast - 52 periods",
    ),
    "monthly_6": ModelSpec(
        model_id="monthly_6",
        granularity="monthly",
        frequency="ME",
        prediction_length=MONTHLY_FORECAST_LENGTHS[0],
        seasonal_period=12,
        item_id="target_monthly_6",
        title="Monthly forecast - 6 periods",
    ),
    "monthly_12": ModelSpec(
        model_id="monthly_12",
        granularity="monthly",
        frequency="ME",
        prediction_length=MONTHLY_FORECAST_LENGTHS[1],
        seasonal_period=12,
        item_id="target_monthly_12",
        title="Monthly forecast - 12 periods",
    ),
}


def _autogluon_modules():
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon TimeSeries is not installed. Install dependencies from requirements.txt first."
        ) from exc
    return TimeSeriesDataFrame, TimeSeriesPredictor


def _resample_rule(granularity: str) -> str:
    if granularity == "weekly":
        return "W-FRI"
    if granularity == "monthly":
        return "ME"
    raise ValueError(f"Unsupported granularity: {granularity}")


def _last_non_null(series: pd.Series):
    observed = series.dropna()
    if observed.empty:
        return pd.NA
    return observed.iloc[-1]


def aggregate_period_frame(daily: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if "date" not in daily.columns or "target" not in daily.columns:
        raise ValueError("Daily data must contain 'date' and 'target' columns.")

    indexed = daily.copy()
    indexed["date"] = pd.to_datetime(indexed["date"])
    indexed = indexed.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
    indexed = indexed.loc[:, [column for column in indexed.columns if column != "date"]]

    rule = _resample_rule(granularity)
    numeric_columns = [column for column in indexed.columns if pd.api.types.is_numeric_dtype(indexed[column])]
    non_numeric_columns = [column for column in indexed.columns if column not in numeric_columns]

    aggregated_parts: list[pd.DataFrame] = []
    if numeric_columns:
        aggregated_parts.append(indexed[numeric_columns].resample(rule).mean())
    if non_numeric_columns:
        aggregated_parts.append(indexed[non_numeric_columns].resample(rule).agg(_last_non_null))

    if not aggregated_parts:
        raise ValueError("No columns are available for aggregation.")

    aggregated = pd.concat(aggregated_parts, axis=1).loc[:, indexed.columns]
    aggregated = aggregated.dropna(subset=["target"])
    if aggregated.empty:
        raise ValueError(f"No aggregated {granularity} observations are available.")

    last_observed = indexed.index.max().normalize()
    if aggregated.index.max().normalize() > last_observed:
        aggregated = aggregated.iloc[:-1]

    aggregated = aggregated.dropna(subset=["target"])
    if aggregated.empty:
        raise ValueError(f"No closed {granularity} periods are available after dropping incomplete periods.")
    return aggregated


def aggregate_period_series(daily: pd.DataFrame, granularity: str) -> pd.Series:
    return aggregate_period_frame(daily, granularity)["target"]


def build_timeseries_frame(data: pd.DataFrame, item_id: str):
    TimeSeriesDataFrame, _ = _autogluon_modules()
    frame = data.reset_index().rename(columns={data.index.name or "index": "timestamp"})
    frame.insert(0, "item_id", item_id)
    return TimeSeriesDataFrame.from_data_frame(frame, id_column="item_id", timestamp_column="timestamp")


def train_model(
    aggregated: pd.DataFrame,
    model_dir: Path,
    spec: ModelSpec,
    source_file: Path,
    data_status: str,
    target_name: str,
    covariate_names: list[str],
    covariate_labels: dict[str, str],
    preset_name: str = DEFAULT_AUTOGUON_PRESET,
) -> dict[str, str]:
    _, TimeSeriesPredictor = _autogluon_modules()
    if model_dir.exists():
        shutil.rmtree(model_dir)

    min_history = spec.prediction_length * 2
    if len(aggregated) < min_history:
        raise ValueError(
            f"Not enough {spec.granularity} history to train the model. "
            f"Need at least {min_history} closed periods and only found {len(aggregated)}."
        )

    num_val_windows = 1
    if len(aggregated) >= spec.prediction_length * 5:
        num_val_windows = 3
    elif len(aggregated) >= spec.prediction_length * 3:
        num_val_windows = 2

    predictor = TimeSeriesPredictor(
        prediction_length=spec.prediction_length,
        path=str(model_dir),
        freq=spec.frequency,
        target="target",
        quantile_levels=[0.1, 0.5, 0.9],
        eval_metric="MASE",
        eval_metric_seasonal_period=spec.seasonal_period,
        verbosity=2,
    )
    train_data = build_timeseries_frame(aggregated, spec.item_id)
    predictor.fit(
        train_data=train_data,
        presets=preset_name,
        verbosity=2,
        num_val_windows=num_val_windows,
        refit_every_n_windows=1,
        refit_full=True,
    )

    metadata = {
        "model_id": spec.model_id,
        "granularity": spec.granularity,
        "prediction_length": str(spec.prediction_length),
        "seasonal_period": str(spec.seasonal_period),
        "target_column": target_name,
        "internal_target_column": "target",
        "covariate_type": "past_covariates",
        "covariate_count": str(len(covariate_names)),
        "covariate_columns": json.dumps(covariate_names),
        "covariate_labels": json.dumps(covariate_labels, ensure_ascii=False, sort_keys=True),
        "source_file": str(source_file),
        "data_status": data_status,
        "preset": preset_name,
        "model_family": f"{preset_name} preset default models",
        "num_val_windows": str(num_val_windows),
        "model_trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_period_end": str(aggregated.index.max().date()),
    }
    write_metadata(model_dir, metadata)
    return metadata


def load_metadata(model_dir: Path) -> dict[str, str]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_metadata(model_dir: Path, metadata: dict[str, str]) -> None:
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def model_ready(model_dir: Path) -> bool:
    return model_dir.exists() and (model_dir / "metadata.json").exists()


def load_predictor(model_dir: Path):
    _, TimeSeriesPredictor = _autogluon_modules()
    return TimeSeriesPredictor.load(str(model_dir))


def normalize_forecast_frame(forecast) -> pd.DataFrame:
    frame = forecast.reset_index()
    if "item_id" in frame.columns:
        frame = frame.drop(columns=["item_id"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def forecast_to_rows(
    forecast_frame: pd.DataFrame,
    spec: ModelSpec,
    source_file: Path,
    data_status: str,
    model_dir: Path,
    metadata: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for step, row in enumerate(forecast_frame.to_dict(orient="records"), start=1):
        rows.append(
            {
                "model_id": spec.model_id,
                "granularity": spec.granularity,
                "prediction_length": spec.prediction_length,
                "step": step,
                "target_period_end": pd.Timestamp(row["timestamp"]).date().isoformat(),
                "mean": float(row["mean"]),
                "0.1": float(row["0.1"]),
                "0.5": float(row["0.5"]),
                "0.9": float(row["0.9"]),
                "source_file": str(source_file),
                "data_status": data_status,
                "model_path": str(model_dir),
                "model_trained_at": metadata["model_trained_at"],
            }
        )
    return pd.DataFrame(rows)


def forecast_model(
    aggregated: pd.DataFrame,
    model_dir: Path,
    spec: ModelSpec,
    source_file: Path,
    data_status: str,
) -> ForecastBundle:
    predictor = load_predictor(model_dir)
    metadata = load_metadata(model_dir)
    data = build_timeseries_frame(aggregated, spec.item_id)
    forecast = predictor.predict(data)
    forecast_frame = normalize_forecast_frame(forecast)
    rows = forecast_to_rows(
        forecast_frame=forecast_frame,
        spec=spec,
        source_file=source_file,
        data_status=data_status,
        model_dir=model_dir,
        metadata=metadata,
    )
    return ForecastBundle(
        spec=spec,
        history=aggregated["target"],
        forecast_frame=forecast_frame,
        rows=rows,
        source_file=source_file,
        model_dir=model_dir,
        metadata=metadata,
    )
