from __future__ import annotations

from pathlib import Path

import pandas as pd

from cepea_forecast.data_io import find_data_files, load_source_data


def test_find_data_files_locates_boi_and_bezerro(tmp_path: Path) -> None:
    boi = tmp_path / "CEPEA_BOI.csv"
    bezerro = tmp_path / "CEPEA_BEZERRO.csv"
    boi.write_text("Data,BOI_BRL\n01/03/2026,320\n", encoding="utf-8")
    bezerro.write_text("Data,BEZERRO_BRL\n01/03/2026,3000\n", encoding="utf-8")

    boi_path, bezerro_path = find_data_files(tmp_path)
    assert boi_path == boi
    assert bezerro_path == bezerro


def test_find_data_files_works_without_bezerro(tmp_path: Path) -> None:
    boi = tmp_path / "CEPEA_BOI.csv"
    boi.write_text("Data,BOI_BRL\n01/03/2026,320\n", encoding="utf-8")

    boi_path, bezerro_path = find_data_files(tmp_path)
    assert boi_path == boi
    assert bezerro_path is None


def test_load_source_data_merges_boi_and_bezerro(tmp_path: Path) -> None:
    boi = tmp_path / "CEPEA_BOI.csv"
    boi.write_text(
        "Data,BOI_BRL,BOI_USD,USD\n"
        "01/03/2026,320.50,61.50,5.21\n"
        "02/03/2026,321.75,61.80,5.20\n"
        "03/03/2026,322.00,62.00,5.19\n",
        encoding="utf-8",
    )
    bezerro = tmp_path / "CEPEA_BEZERRO.csv"
    bezerro.write_text(
        "Data,BEZERRO_BRL,BEZERRO_PESO,BRL_KG\n"
        "01/03/2026,3200.00,200.00,16.00\n"
        "03/03/2026,3210.00,201.00,15.97\n",
        encoding="utf-8",
    )

    loaded = load_source_data(tmp_path)

    assert loaded.target_name == "BOI_BRL"
    assert "target" in loaded.frame.columns
    assert "BOI_USD" in loaded.frame.columns
    assert "USD" in loaded.frame.columns
    assert "BEZERRO_BRL" in loaded.frame.columns
    assert "BEZERRO_PESO" in loaded.frame.columns
    assert "BRL_KG" in loaded.frame.columns
    assert len(loaded.frame) == 3
    # Row with date mismatch (02/03) should have NaN for BEZERRO columns (left join)
    row_02 = loaded.frame[loaded.frame["date"] == pd.Timestamp("2026-03-02")]
    assert pd.isna(row_02["BEZERRO_BRL"].iloc[0])
    # Row with matching date should have values
    row_01 = loaded.frame[loaded.frame["date"] == pd.Timestamp("2026-03-01")]
    assert row_01["BEZERRO_BRL"].iloc[0] == 3200.0
    assert row_01["target"].iloc[0] == 320.5
    # Derived feature: BOI_BRL / BRL_KG
    assert "BOI_BEZERRO_RATIO" in loaded.frame.columns
    assert "BOI_BEZERRO_RATIO" in loaded.covariate_names
    assert row_01["BOI_BEZERRO_RATIO"].iloc[0] == 320.5 / 16.0


def test_load_source_data_boi_only(tmp_path: Path) -> None:
    boi = tmp_path / "CEPEA_BOI.csv"
    boi.write_text(
        "Data,BOI_BRL,BOI_USD\n"
        "01/03/2026,320.50,61.50\n"
        "02/03/2026,321.75,61.80\n",
        encoding="utf-8",
    )

    loaded = load_source_data(tmp_path)

    assert loaded.target_name == "BOI_BRL"
    assert loaded.covariate_names == ["BOI_USD"]
    assert len(loaded.frame) == 2
