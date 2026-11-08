"""Tests for CSV loading, schema validation and dynamic discovery."""

from __future__ import annotations

import pandas as pd
import pytest

from geolens.models.schemas import REQUIRED_COLUMNS
from geolens.services.csv_loader import (
    CSVLoadError,
    discover_clusters,
    discover_lithologies,
    load_csv,
    load_csv_with_check,
    validate_schema,
)


def test_load_csv_preserves_source_columns(well_a_bytes: bytes) -> None:
    df = load_csv(well_a_bytes, filename="well_a.csv")
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert len(df) == 8
    assert df["lithology"].iloc[0] == "Shale"


def test_load_csv_adds_source_row_index(well_a_bytes: bytes) -> None:
    df = load_csv(well_a_bytes, filename="well_a.csv")
    assert "source_row_index" in df.columns
    assert df["source_row_index"].tolist() == list(range(len(df)))


def test_validate_schema_accepts_required_columns(well_a_bytes: bytes) -> None:
    df = load_csv(well_a_bytes, filename="well_a.csv")
    check = validate_schema(df)
    assert check.is_valid
    assert check.missing_columns == ()


def test_validate_schema_reports_missing_columns() -> None:
    df = pd.DataFrame({"top_y": [1], "bottom_y": [2]})
    check = validate_schema(df)
    assert not check.is_valid
    assert "lithology" in check.missing_columns
    assert "cluster_id" in check.missing_columns


def test_load_csv_with_check_returns_error_for_empty_file() -> None:
    df, check, error = load_csv_with_check(b"", filename="empty.csv")
    assert df is None
    assert not check.is_valid
    assert error is not None


def test_load_csv_raises_on_empty() -> None:
    with pytest.raises(CSVLoadError):
        load_csv(b"   ", filename="blank.csv")


def test_missing_values_are_preserved_as_nan(messy_bytes: bytes) -> None:
    df = load_csv(messy_bytes, filename="well_d_messy.csv")
    # Row 1 has an empty lithology label; it must not be silently dropped.
    assert len(df) == 10
    assert df["lithology"].isna().sum() == 1


def test_discover_lithologies_is_dynamic(well_a_bytes: bytes, well_c_bytes: bytes) -> None:
    a = load_csv(well_a_bytes, filename="a.csv")
    c = load_csv(well_c_bytes, filename="c.csv")
    assert "Shale" in discover_lithologies(a)
    assert "Evaporite" in discover_lithologies(c)
    assert "Evaporite" not in discover_lithologies(a)


def test_discover_clusters_is_dynamic(well_a_bytes: bytes, well_b_bytes: bytes) -> None:
    a = load_csv(well_a_bytes, filename="a.csv")
    b = load_csv(well_b_bytes, filename="b.csv")
    assert discover_clusters(a) == ["C1", "C2", "C3", "C4"]
    assert discover_clusters(b) == ["K1", "K2", "K3", "K4"]


def test_new_lithology_requires_no_code_change() -> None:
    csv = (
        "top_y,bottom_y,cluster_id,lithology,top_depth,base_depth\n"
        "0,10,X9,Unobtainium,0,10\n"
    ).encode()
    df = load_csv(csv, filename="novel.csv")
    assert discover_lithologies(df) == ["Unobtainium"]
    assert discover_clusters(df) == ["X9"]
