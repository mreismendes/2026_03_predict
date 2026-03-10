from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cepea_forecast.config import build_paths
from cepea_forecast.data_io import LoadedSourceData, find_latest_data_file, load_source_data
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


def resolve_latest_data_file(paths) -> tuple[Path, DataSourceResult]:
    latest_file = find_latest_data_file(paths.data_dir)
    if latest_file is not None:
        return latest_file, DataSourceResult(
            status="manual_upload",
            message="Using the newest file already present in data/.",
        )

    raise FileNotFoundError(
        f"No supported source file was found in {paths.data_dir}. Upload a .xls, .xlsx, or .csv file first."
    )


def _model_dir_for(paths, model_id: str) -> Path:
    return paths.models_dir / model_id


def _aggregate_by_granularity(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "weekly": aggregate_period_frame(daily, granularity="weekly"),
        "monthly": aggregate_period_frame(daily, granularity="monthly"),
    }


def _train_all(paths, loaded: LoadedSourceData, source_file: Path, data_status: str) -> None:
    aggregated_frames = _aggregate_by_granularity(loaded.frame)
    for spec in MODEL_SPECS.values():
        aggregated = aggregated_frames[spec.granularity]
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


def retrain(base_dir: Path | str = ".") -> PipelineResult:
    paths = build_paths(base_dir)
    source_file, data_result = resolve_latest_data_file(paths)
    loaded = load_source_data(source_file)
    _train_all(paths, loaded=loaded, source_file=source_file, data_status=data_result.status)
    return PipelineResult(command="retrain", source_file=source_file, data_result=data_result)


def _ensure_models(paths, loaded: LoadedSourceData, source_file: Path, data_status: str) -> None:
    if all(model_ready(_model_dir_for(paths, spec.model_id)) for spec in MODEL_SPECS.values()):
        return
    _train_all(paths, loaded=loaded, source_file=source_file, data_status=data_status)


def predict(base_dir: Path | str = ".") -> PipelineResult:
    paths = build_paths(base_dir)
    source_file, data_result = resolve_latest_data_file(paths)
    loaded = load_source_data(source_file)
    _ensure_models(paths, loaded=loaded, source_file=source_file, data_status=data_result.status)
    aggregated_frames = _aggregate_by_granularity(loaded.frame)

    bundles = []
    for spec in MODEL_SPECS.values():
        aggregated = aggregated_frames[spec.granularity]
        model_dir = _model_dir_for(paths, spec.model_id)
        bundles.append(
            forecast_model(
                aggregated=aggregated,
                model_dir=model_dir,
                spec=spec,
                source_file=source_file,
                data_status=data_result.status,
            )
        )

    combined = pd.concat([bundle.rows for bundle in bundles], ignore_index=True)
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.predictions_dir / "latest_forecast.csv"
    combined.to_csv(output_path, index=False)
    report_path = generate_forecast_report(
        output_path=paths.pdf_output_dir / "latest_forecast_report.pdf",
        bundles=bundles,
        source_file=source_file,
    )
    return PipelineResult(
        command="predict",
        source_file=source_file,
        data_result=data_result,
        predictions_path=output_path,
        report_path=report_path,
        predictions=combined,
    )
