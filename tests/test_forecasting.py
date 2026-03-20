from __future__ import annotations

from pathlib import Path

import pandas as pd

from cepea_forecast.forecasting import MODEL_SPECS, aggregate_period_frame, aggregate_period_series, forecast_to_rows


def test_weekly_aggregation_drops_incomplete_week() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-02-23",
                    "2026-02-24",
                    "2026-02-25",
                    "2026-02-26",
                    "2026-02-27",
                    "2026-03-02",
                    "2026-03-03",
                    "2026-03-04",
                    "2026-03-05",
                ]
            ),
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0],
            "covariate_1": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0],
        }
    )

    weekly_frame = aggregate_period_frame(daily, "weekly")
    weekly = aggregate_period_series(daily, "weekly")

    assert weekly.empty is False
    assert weekly.index.tolist() == [pd.Timestamp("2026-02-27")]
    assert weekly.iloc[0] == 3.0
    assert weekly_frame["covariate_1"].iloc[0] == 30.0


def test_forecast_to_rows_returns_full_prediction_path() -> None:
    spec = MODEL_SPECS["weekly_52"]
    forecast_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-13", periods=52, freq="W-FRI"),
            "mean": list(range(52)),
            "0.1": list(range(52)),
            "0.5": list(range(52)),
            "0.9": list(range(52)),
        }
    )

    rows = forecast_to_rows(
        forecast_frame=forecast_frame,
        spec=spec,
        source_file=Path("data/latest.xls"),
        data_status="manual_upload",
        model_dir=Path("artifacts/models/weekly_52"),
        metadata={"model_trained_at": "2026-03-09T12:00:00+00:00"},
    )

    assert rows["model_id"].iloc[0] == "weekly_52"
    assert rows["prediction_length"].iloc[0] == 52
    assert rows["step"].tolist() == list(range(1, 53))
    assert rows["target_period_end"].iloc[0] == "2026-03-13"
    assert rows["target_period_end"].iloc[-1] == "2027-03-05"
