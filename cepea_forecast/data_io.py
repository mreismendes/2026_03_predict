from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".xls", ".xlsx", ".csv"}
TARGET_COLUMN = "BOI_BRL"


@dataclass(frozen=True)
class LoadedSourceData:
    frame: pd.DataFrame
    target_name: str
    covariate_names: list[str]
    covariate_labels: dict[str, str]


def _find_file(data_dir: Path, pattern: str) -> Path | None:
    """Find a file in data_dir whose stem contains the pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and pattern_lower in path.stem.lower():
            return path
    return None


def find_data_files(data_dir: Path) -> tuple[Path, Path | None]:
    """Find BOI (required) and BEZERRO (optional) data files in the directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    boi_file = _find_file(data_dir, "BOI")
    if boi_file is None:
        raise FileNotFoundError(
            f"No BOI data file found in {data_dir}. Expected a file with 'BOI' in its name."
        )

    bezerro_file = _find_file(data_dir, "BEZERRO")
    return boi_file, bezerro_file


def _read_table(path: Path, header_row: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header_row, sep=None, engine="python")
    if suffix == ".xls":
        return pd.read_excel(path, header=header_row, engine="xlrd")
    if suffix == ".xlsx":
        return pd.read_excel(path, header=header_row, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"[R$\s]", "", regex=True)
    has_comma = cleaned.str.contains(",", regex=False)
    has_dot = cleaned.str.contains(".", regex=False)
    cleaned = cleaned.where(~(has_comma & has_dot), cleaned.str.replace(".", "", regex=False))
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _load_single_file(path: Path) -> pd.DataFrame:
    """Load a single data file, returning a DataFrame with parsed date index and numeric columns."""
    for header_row in range(5):
        try:
            frame = _read_table(path, header_row)
        except ImportError as exc:
            raise RuntimeError(f"Missing reader dependency for {path.suffix}: {exc}") from exc
        except Exception:
            continue

        frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if frame.empty or frame.shape[1] < 2:
            continue

        date_series = pd.to_datetime(frame.iloc[:, 0], dayfirst=True, errors="coerce")
        if float(date_series.notna().mean()) < 0.6:
            continue

        result = pd.DataFrame({"date": date_series})
        for col in frame.columns[1:]:
            numeric = _coerce_numeric(frame[col])
            if float(numeric.notna().mean()) >= 0.6:
                result[str(col).strip()] = numeric.astype(float)

        result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if not result.empty:
            return result.reset_index(drop=True)

    raise ValueError(f"Could not parse data from {path}")


def load_source_data(data_dir: Path) -> LoadedSourceData:
    """Load BOI and BEZERRO files from data_dir, merge on date, return unified frame.

    BOI_BRL is the target column. All other numeric columns are past covariates.
    BEZERRO data is joined on date; dates only in BEZERRO (not in BOI) are dropped.
    """
    boi_path, bezerro_path = find_data_files(data_dir)

    boi = _load_single_file(boi_path)
    if TARGET_COLUMN not in boi.columns:
        raise ValueError(f"BOI file missing target column '{TARGET_COLUMN}'. Found: {list(boi.columns)}")

    if bezerro_path is not None:
        bezerro = _load_single_file(bezerro_path)
        # Prefix BEZERRO columns to avoid name collisions (except date)
        bezerro_cols = {col: col for col in bezerro.columns if col != "date"}
        merged = boi.merge(bezerro, on="date", how="left", suffixes=("", "_bez"))
    else:
        merged = boi

    # Derived feature: BOI/BEZERRO price ratio (spread indicator)
    if "BRL_KG" in merged.columns:
        merged["BOI_BEZERRO_RATIO"] = merged[TARGET_COLUMN] / merged["BRL_KG"]

    # Build target + covariates
    covariate_names = [col for col in merged.columns if col not in ("date", TARGET_COLUMN)]
    covariate_labels = {col: col for col in covariate_names}

    frame = pd.DataFrame({"date": merged["date"], "target": merged[TARGET_COLUMN]})
    for col in covariate_names:
        frame[col] = merged[col]

    frame = frame.dropna(subset=["date", "target"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return LoadedSourceData(
        frame=frame.reset_index(drop=True),
        target_name=TARGET_COLUMN,
        covariate_names=covariate_names,
        covariate_labels=covariate_labels,
    )
