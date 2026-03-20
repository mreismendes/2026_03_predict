from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cepea_forecast.config import build_paths
from cepea_forecast.data_io import LoadedSourceData, find_data_files, load_source_data
from cepea_forecast.features import engineer_features
from cepea_forecast.forecasting import MODEL_SPECS, aggregate_period_frame, forecast_model, model_ready, train_model
from cepea_forecast.reporting import generate_forecast_report


@dataclass(frozen=True)
class DataSourceResult:
    status: str
    message: str


@dataclass(frozen=True)
class PipelineResult:
    command: str
    source_file: Path
    data_result: DataSourceResult
    predictions_path: Path | None = None
    report_path: Path | None = None
    predictions: pd.DataFrame | None = None


def _model_dir_for(paths, model_id: str) -> Path:
    return paths.models_dir / model_id


def _aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to weekly and apply feature engineering."""
    aggregated = aggregate_period_frame(daily, "weekly")
    enriched, _ = engineer_features(aggregated, "weekly")
    return enriched


def _train_all(paths, loaded: LoadedSourceData, source_file: Path, data_status: str, aggregated: pd.DataFrame) -> None:
    for spec in MODEL_SPECS.values():
        model_dir = _model_dir_for(paths, spec.model_id)
        train_model(
            aggregated=aggregated,
            model_dir=model_dir,
            spec=spec,
            source_file=source_file,
            data_status=data_status,
            target_name=loaded.target_name,
            covariate_names=loaded.covariate_names,
            covariate_labels=loaded.covariate_labels,
        )


def _ensure_models(paths, loaded: LoadedSourceData, source_file: Path, data_status: str, aggregated: pd.DataFrame) -> None:
    if all(model_ready(_model_dir_for(paths, spec.model_id)) for spec in MODEL_SPECS.values()):
        return
    _train_all(paths, loaded=loaded, source_file=source_file, data_status=data_status, aggregated=aggregated)


def retrain(base_dir: Path | str = ".") -> PipelineResult:
    paths = build_paths(base_dir)
    boi_path, bezerro_path = find_data_files(paths.data_dir)
    loaded = load_source_data(paths.data_dir)
    data_result = DataSourceResult(
        status="manual_upload",
        message=f"BOI: {boi_path.name}" + (f", BEZERRO: {bezerro_path.name}" if bezerro_path else ""),
    )
    aggregated = _aggregate_weekly(loaded.frame)
    _train_all(paths, loaded=loaded, source_file=boi_path, data_status=data_result.status, aggregated=aggregated)
    return PipelineResult(command="retrain", source_file=boi_path, data_result=data_result)


def predict(base_dir: Path | str = ".") -> PipelineResult:
    paths = build_paths(base_dir)
    boi_path, bezerro_path = find_data_files(paths.data_dir)
    loaded = load_source_data(paths.data_dir)
    data_result = DataSourceResult(
        status="manual_upload",
        message=f"BOI: {boi_path.name}" + (f", BEZERRO: {bezerro_path.name}" if bezerro_path else ""),
    )
    aggregated = _aggregate_weekly(loaded.frame)
    _ensure_models(paths, loaded=loaded, source_file=boi_path, data_status=data_result.status, aggregated=aggregated)

    bundles = []
    for spec in MODEL_SPECS.values():
        model_dir = _model_dir_for(paths, spec.model_id)
        bundles.append(
            forecast_model(
                aggregated=aggregated,
                model_dir=model_dir,
                spec=spec,
                source_file=boi_path,
                data_status=data_result.status,
            )
        )

    combined = bundles[0].rows if len(bundles) == 1 else pd.concat([b.rows for b in bundles], ignore_index=True)
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.predictions_dir / "latest_forecast.csv"
    combined.to_csv(output_path, index=False)
    report_path = generate_forecast_report(
        output_path=paths.pdf_output_dir / "latest_forecast_report.pdf",
        bundles=bundles,
        source_file=boi_path,
    )
    return PipelineResult(
        command="predict",
        source_file=boi_path,
        data_result=data_result,
        predictions_path=output_path,
        report_path=report_path,
        predictions=combined,
    )
