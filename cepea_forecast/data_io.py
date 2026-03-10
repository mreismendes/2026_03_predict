from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".xls", ".xlsx", ".csv"}


@dataclass(frozen=True)
class LoadedSourceData:
    frame: pd.DataFrame
    target_name: str
    covariate_names: list[str]
    covariate_labels: dict[str, str]


def find_latest_data_file(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    files = [path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


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


def _coerce_categorical(series: pd.Series) -> pd.Series:
    values = series.copy()
    values = values.where(~pd.isna(values), pd.NA)
    values = values.astype("string").str.strip()
    values = values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return values


def _make_covariate_name(position: int, used_names: set[str]) -> str:
    name = f"covariate_{position}"
    while name in used_names:
        position += 1
        name = f"covariate_{position}"
    return name


def load_source_data(path: Path) -> LoadedSourceData:
    read_errors: list[str] = []
    for header_row in range(5):
        try:
            frame = _read_table(path, header_row)
        except ImportError as exc:
            raise RuntimeError(f"Missing reader dependency for {path.suffix}: {exc}") from exc
        except Exception as exc:  # pragma: no cover - captured for fallback diagnostics
            read_errors.append(f"header={header_row}: {exc}")
            continue

        frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if frame.empty:
            continue

        if frame.shape[1] < 2:
            continue

        original_columns = list(frame.columns)
        date_series = pd.to_datetime(frame.iloc[:, 0], dayfirst=True, errors="coerce")
        target_series = _coerce_numeric(frame.iloc[:, 1])
        date_score = float(date_series.notna().mean())
        target_score = float(target_series.notna().mean())
        if date_score < 0.6 or target_score < 0.6:
            continue

        data = pd.DataFrame({"date": date_series, "target": target_series})
        covariate_names: list[str] = []
        covariate_labels: dict[str, str] = {}
        used_names = {"date", "target"}
        for position, original_name in enumerate(original_columns[2:], start=1):
            internal_name = _make_covariate_name(position, used_names)
            used_names.add(internal_name)
            raw_series = frame[original_name]
            numeric_series = _coerce_numeric(raw_series)
            numeric_score = float(numeric_series.notna().mean())
            if numeric_score >= 0.6:
                covariate_series = numeric_series.astype(float)
            else:
                covariate_series = _coerce_categorical(raw_series)
            data[internal_name] = covariate_series
            covariate_names.append(internal_name)
            covariate_labels[internal_name] = str(original_name).strip() or internal_name

        data = data.dropna(subset=["date", "target"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if not data.empty:
            target_name = str(original_columns[1]).strip() or "target"
            return LoadedSourceData(
                frame=data.reset_index(drop=True),
                target_name=target_name,
                covariate_names=covariate_names,
                covariate_labels=covariate_labels,
            )

    details = "; ".join(read_errors) if read_errors else "no readable header row produced a valid date/target mapping"
    raise ValueError(f"Could not parse source data from {path}: {details}")


def load_daily_series(path: Path) -> pd.DataFrame:
    return load_source_data(path).frame
