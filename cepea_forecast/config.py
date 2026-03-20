from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_AUTOGUON_PRESET = "best_quality"
WEEKLY_FORECAST_LENGTH = 52
TRAINING_TIME_LIMIT: int | None = None

AUTOGLUON_HYPERPARAMETERS: dict[str, dict] = {
    # Baseline
    "Naive": {},
    "SeasonalNaive": {},
    "Average": {},
    "SeasonalAverage": {},
    "Zero": {},
    # Statistical
    "ETS": {},
    "AutoARIMA": {},
    "AutoETS": {},
    "AutoCES": {},
    "Theta": {},
    "NPTS": {},
    # Statistical - sparse data
    "ADIDA": {},
    "Croston": {},
    "IMAPA": {},
    # Deep Learning
    "DeepAR": {},
    "DLinear": {},
    "PatchTST": {},
    "SimpleFeedForward": {},
    "TemporalFusionTransformer": {},
    "TiDE": {},
    "WaveNet": {},
    # Tabular
    "DirectTabular": {},
    "PerStepTabular": {},
    "RecursiveTabular": {},
    # Pretrained
    "Chronos2": {},
    "Chronos": {},
    "Toto": {},
}


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    predictions_dir: Path
    output_dir: Path
    pdf_output_dir: Path
    tmp_pdf_dir: Path


def build_paths(base_dir: Path | str = ".") -> AppPaths:
    root = Path(base_dir).resolve()
    artifacts_dir = root / "artifacts"
    models_dir = artifacts_dir / "models"
    predictions_dir = artifacts_dir / "predictions"
    output_dir = root / "output"
    return AppPaths(
        base_dir=root,
        data_dir=root / "data",
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        pdf_output_dir=output_dir / "pdf",
        tmp_pdf_dir=root / "tmp" / "pdfs",
    )
