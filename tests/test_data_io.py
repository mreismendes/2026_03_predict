from __future__ import annotations

from pathlib import Path

import os
import pandas as pd

from cepea_forecast.data_io import find_latest_data_file, load_source_data


def test_find_latest_data_file_prefers_newest_mtime(tmp_path: Path) -> None:
    older = tmp_path / "older.csv"
    newer = tmp_path / "newer.csv"
    older.write_text("date,value\n01/01/2024,1\n", encoding="utf-8")
    newer.write_text("date,value\n02/01/2024,2\n", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert find_latest_data_file(tmp_path) == newer


def test_load_source_data_uses_positional_target_and_covariates(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "Trade Date,Settlement BRL,FX USD,Market Regime\n"
        "01/03/2026,320.50,5.50,bull\n"
        "02/03/2026,321.75,5.60,bear\n",
        encoding="utf-8",
    )

    loaded = load_source_data(source)

    assert loaded.target_name == "Settlement BRL"
    assert loaded.covariate_names == ["covariate_1", "covariate_2"]
    assert loaded.covariate_labels == {"covariate_1": "FX USD", "covariate_2": "Market Regime"}
    assert list(loaded.frame.columns) == ["date", "target", "covariate_1", "covariate_2"]
    assert len(loaded.frame) == 2
    assert loaded.frame["target"].tolist() == [320.5, 321.75]
    assert loaded.frame["covariate_1"].tolist() == [5.5, 5.6]
    assert loaded.frame["covariate_2"].tolist() == ["bull", "bear"]
    assert pd.Timestamp("2026-03-01") == loaded.frame.loc[0, "date"]
