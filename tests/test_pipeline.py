from __future__ import annotations

from pathlib import Path

import pandas as pd

from cepea_forecast.pipeline import predict


def test_predict_trains_missing_models_and_writes_output(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    boi = data_dir / "CEPEA_BOI.csv"
    boi.write_text(
        "Data,BOI_BRL,BOI_USD,USD\n"
        "02/01/2026,300,57.7,5.2\n"
        "09/01/2026,310,59.6,5.2\n"
        "16/01/2026,320,61.5,5.2\n"
        "23/01/2026,330,63.5,5.2\n"
        "30/01/2026,340,65.4,5.2\n"
        "06/02/2026,350,67.3,5.2\n"
        "13/02/2026,360,69.2,5.2\n"
        "20/02/2026,370,71.2,5.2\n"
        "27/02/2026,380,73.1,5.2\n"
        "06/03/2026,390,75.0,5.2\n",
        encoding="utf-8",
    )
    bezerro = data_dir / "CEPEA_BEZERRO.csv"
    bezerro.write_text(
        "Data,BEZERRO_BRL,BEZERRO_PESO,BRL_KG\n"
        "02/01/2026,3000,200,15.0\n"
        "09/01/2026,3050,201,15.2\n"
        "16/01/2026,3100,202,15.3\n"
        "23/01/2026,3150,203,15.5\n"
        "30/01/2026,3200,204,15.7\n"
        "06/02/2026,3250,205,15.9\n"
        "13/02/2026,3300,206,16.0\n"
        "20/02/2026,3350,207,16.2\n"
        "27/02/2026,3400,208,16.3\n"
        "06/03/2026,3450,209,16.5\n",
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
        hyperparameters=None,
        time_limit=None,
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
                "metadata": {"preset": "fast_training", "target_column": "BOI_BRL", "covariate_count": "5"},
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
    assert all(record[2] == "BOI_BRL" for record in trained)
    # 6 covariates: BOI_USD, USD, BEZERRO_BRL, BEZERRO_PESO, BRL_KG, BOI_BEZERRO_RATIO
    assert all(len(record[3]) == 6 for record in trained)
    assert result.predictions_path is not None
    assert result.predictions_path.exists()
    assert result.report_path is not None
    assert result.report_path.exists()
    assert len(result.predictions) == 52
