from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_AUTOGUON_PRESET = "fast_training"
WEEKLY_FORECAST_LENGTH = 52


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
