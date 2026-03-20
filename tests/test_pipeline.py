from __future__ import annotations

from pathlib import Path

import pandas as pd

from cepea_forecast.pipeline import predict, resolve_latest_data_file


def test_resolve_latest_data_file_uses_cached_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cached = data_dir / "cached.csv"
    cached.write_text("Data,Target\n01/03/2026,1\n", encoding="utf-8")
    paths = type("Paths", (), {"data_dir": data_dir})()

    source_file, result = resolve_latest_data_file(paths)

    assert source_file == cached
    assert result.status == "manual_upload"


def test_predict_trains_missing_models_and_writes_output(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "source.csv"
    source.write_text(
        "Data,Target,FX\n"
        "02/01/2026,300,5.3\n"
        "09/01/2026,310,5.4\n"
        "16/01/2026,320,5.5\n"
        "23/01/2026,330,5.6\n"
        "30/01/2026,340,5.7\n"
        "06/02/2026,350,5.8\n"
        "13/02/2026,360,5.9\n"
        "20/02/2026,370,6.0\n"
        "27/02/2026,380,6.1\n"
        "06/03/2026,390,6.2\n",
        encoding="utf-8",
    )

    trained = []

    def fake_train_model(
        aggregated,
        model_dir,
        spec,
        source_file,
        data_status,
        target_name,
        covariate_names,
        covariate_labels,
        preset_name="fast_training",
    ):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text('{"model_trained_at": "2026-03-09T12:00:00+00:00"}', encoding="utf-8")
        trained.append((spec.granularity, len(aggregated), target_name, tuple(covariate_names), source_file.name, data_status))
        return {"model_trained_at": "2026-03-09T12:00:00+00:00"}

    def fake_forecast_model(aggregated, model_dir, spec, source_file, data_status):
        steps = list(range(1, spec.prediction_length + 1))
        rows = pd.DataFrame(
            {
                "model_id": [spec.model_id] * len(steps),
                "granularity": [spec.granularity] * len(steps),
                "prediction_length": [spec.prediction_length] * len(steps),
                "step": steps,
                "target_period_end": [f"2026-12-{value:02d}" for value in steps],
                "mean": [float(value) for value in steps],
                "0.1": [float(value) for value in steps],
                "0.5": [float(value) for value in steps],
                "0.9": [float(value) for value in steps],
                "source_file": [str(source_file)] * len(steps),
                "data_status": [data_status] * len(steps),
                "model_path": [str(model_dir)] * len(steps),
                "model_trained_at": ["2026-03-09T12:00:00+00:00"] * len(steps),
            }
        )
        return type(
            "Bundle",
            (),
            {
                "rows": rows,
                "spec": spec,
                "history": aggregated["target"],
                "forecast_frame": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2026-03-31", periods=spec.prediction_length, freq=spec.frequency),
                        "mean": [float(value) for value in steps],
                        "0.1": [float(value) for value in steps],
                        "0.5": [float(value) for value in steps],
                        "0.9": [float(value) for value in steps],
                    }
                ),
                "source_file": source_file,
                "model_dir": model_dir,
                "metadata": {"preset": "fast_training", "target_column": "Target", "covariate_count": "1"},
            },
        )()

    def fake_generate_report(output_path, bundles, source_file):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF-1.4")
        return output_path

    monkeypatch.setattr("cepea_forecast.pipeline.train_model", fake_train_model)
    monkeypatch.setattr("cepea_forecast.pipeline.forecast_model", fake_forecast_model)
    monkeypatch.setattr("cepea_forecast.pipeline.generate_forecast_report", fake_generate_report)

    result = predict(base_dir=tmp_path)

    assert len(trained) == 1
    assert all(record[2] == "Target" for record in trained)
    assert all(record[3] == ("covariate_1",) for record in trained)
    assert result.predictions_path is not None
    assert result.predictions_path.exists()
    assert result.report_path is not None
    assert result.report_path.exists()
    assert len(result.predictions) == 52
