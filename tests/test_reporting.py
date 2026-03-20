from __future__ import annotations

from pathlib import Path

import pandas as pd

from cepea_forecast.forecasting import ForecastBundle, MODEL_SPECS
from cepea_forecast.reporting import _forecast_table_data, generate_forecast_report


def test_generate_forecast_report_creates_pdf(tmp_path: Path) -> None:
    spec = MODEL_SPECS["weekly_52"]
    history = pd.Series(
        [300.0, 305.0, 307.5],
        index=pd.to_datetime(["2025-12-19", "2025-12-26", "2026-01-02"]),
    )
    forecast_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-09", periods=6, freq="W-FRI"),
            "mean": [308.0, 309.0, 310.0, 311.0, 312.0, 313.0],
            "0.1": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
            "0.5": [308.0, 309.0, 310.0, 311.0, 312.0, 313.0],
            "0.9": [316.0, 317.0, 318.0, 319.0, 320.0, 321.0],
        }
    )
    rows = pd.DataFrame(
        {
            "model_id": [spec.model_id] * 6,
            "granularity": [spec.granularity] * 6,
            "prediction_length": [spec.prediction_length] * 6,
            "step": [1, 2, 3, 4, 5, 6],
            "target_period_end": [value.date().isoformat() for value in forecast_frame["timestamp"]],
            "mean": forecast_frame["mean"],
            "0.1": forecast_frame["0.1"],
            "0.5": forecast_frame["0.5"],
            "0.9": forecast_frame["0.9"],
            "source_file": ["data/source.xls"] * 6,
            "data_status": ["manual_upload"] * 6,
            "model_path": ["artifacts/models/weekly_52"] * 6,
            "model_trained_at": ["2026-03-09T12:00:00+00:00"] * 6,
        }
    )
    bundle = ForecastBundle(
        spec=spec,
        history=history,
        forecast_frame=forecast_frame,
        rows=rows,
        source_file=Path("data/source.xls"),
        model_dir=Path("artifacts/models/weekly_52"),
        metadata={
            "preset": "fast_training",
            "target_column": "Settlement BRL",
            "covariate_count": "2",
            "model_family": "TemporalFusionTransformer",
        },
    )

    output_path = generate_forecast_report(tmp_path / "output" / "pdf" / "report.pdf", [bundle], Path("data/source.xls"))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_forecast_table_data_contains_all_forecast_rows() -> None:
    spec = MODEL_SPECS["weekly_52"]
    forecast_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-13", periods=3, freq="W-FRI"),
            "mean": [101.0, 102.0, 103.0],
            "0.1": [99.0, 100.0, 101.0],
            "0.5": [101.0, 102.0, 103.0],
            "0.9": [103.0, 104.0, 105.0],
        }
    )
    rows = pd.DataFrame(
        {
            "model_id": [spec.model_id] * 3,
            "granularity": [spec.granularity] * 3,
            "prediction_length": [spec.prediction_length] * 3,
            "step": [1, 2, 3],
            "target_period_end": [value.date().isoformat() for value in forecast_frame["timestamp"]],
            "mean": forecast_frame["mean"],
            "0.1": forecast_frame["0.1"],
            "0.5": forecast_frame["0.5"],
            "0.9": forecast_frame["0.9"],
            "source_file": ["data/source.xls"] * 3,
            "data_status": ["manual_upload"] * 3,
            "model_path": ["artifacts/models/weekly_52"] * 3,
            "model_trained_at": ["2026-03-09T12:00:00+00:00"] * 3,
        }
    )
    bundle = ForecastBundle(
        spec=spec,
        history=pd.Series([90.0], index=pd.to_datetime(["2026-03-06"])),
        forecast_frame=forecast_frame,
        rows=rows,
        source_file=Path("data/source.xls"),
        model_dir=Path("artifacts/models/weekly_52"),
        metadata={"target_column": "Target"},
    )

    table_data = _forecast_table_data(bundle)

    assert table_data[0] == ["Step", "Period End", "Mean", "P10", "P50", "P90", "Change %"]
    assert table_data[1][0] == "1"
    assert table_data[1][1] == "2026-03-13"
    assert "101" in table_data[1][2]  # Mean ~ 101
    assert table_data[-1][0] == "3"
    assert table_data[-1][1] == "2026-03-27"
    assert len(table_data[-1]) == 7  # 7 columns including Change %
