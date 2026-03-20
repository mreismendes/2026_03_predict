from __future__ import annotations

import argparse
import os
from pathlib import Path

from cepea_forecast.pipeline import PipelineResult, predict, retrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CEPEA weekly forecasting CLI")
    parser.add_argument("command", choices=["predict", "retrain"], help="Command to execute")
    parser.add_argument("--base-dir", default=".", help="Project root containing data/ and artifacts/")
    return parser


def _print_result(result: PipelineResult) -> None:
    print(f"Command: {result.command}")
    print(f"Source file: {result.source_file}")
    print(f"Data source: {result.data_result.status} - {result.data_result.message}")
    if result.predictions_path is not None and result.predictions is not None:
        print(f"Prediction CSV: {result.predictions_path}")
        if result.report_path is not None:
            print(f"Report PDF: {result.report_path}")
        summary = (
            result.predictions.groupby(["model_id", "granularity", "prediction_length"], as_index=False)
            .agg(
                forecast_start=("target_period_end", "min"),
                forecast_end=("target_period_end", "max"),
                first_mean=("mean", "first"),
                last_mean=("mean", "last"),
            )
            .sort_values(["granularity", "prediction_length"])
        )
        summary["first_mean"] = summary["first_mean"].map(lambda value: f"{value:.4f}")
        summary["last_mean"] = summary["last_mean"].map(lambda value: f"{value:.4f}")
        print(summary.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_dir = Path(args.base_dir).resolve()
    mplconfig_dir = base_dir / "artifacts" / "mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

    try:
        if args.command == "predict":
            result = predict(base_dir=base_dir)
        else:
            result = retrain(base_dir=base_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    _print_result(result)
    return 0
