"""Tests for alternative column-name handling.

Real extraction outputs are not always named identically. The company's own
export uses ``bottom_depth`` instead of ``base_depth``; GeoLens must accept it
without the user renaming anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geolens.models.schemas import COL_BASE_DEPTH, COL_TOP_DEPTH
from geolens.services.csv_loader import (
    load_csv,
    load_csv_with_check,
    resolve_column_aliases,
    restore_source_columns,
    validate_schema,
)
from geolens.services.data_processor import build_document
from geolens.services.statistics import overview_metrics

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def company_csv_bytes() -> bytes:
    return (SAMPLE_DIR / "company_example.csv").read_bytes()


def test_resolve_aliases_maps_bottom_depth() -> None:
    rename_map, pairs = resolve_column_aliases(
        ["top_depth", "bottom_depth", "lithology", "top_y", "bottom_y", "cluster_id"]
    )
    assert rename_map == {"bottom_depth": "base_depth"}
    assert pairs == [("bottom_depth", "base_depth")]


def test_canonical_name_wins_over_alias() -> None:
    """If base_depth is present, bottom_depth must not overwrite it."""
    rename_map, pairs = resolve_column_aliases(
        ["top_depth", "base_depth", "bottom_depth", "lithology", "top_y", "bottom_y", "cluster_id"]
    )
    assert rename_map == {}
    assert pairs == []


def test_alias_matching_is_case_and_space_insensitive() -> None:
    rename_map, _ = resolve_column_aliases(
        ["Top_Depth", " Bottom_Depth ", "Lithology", "Top_Y", "Bottom_Y", "Cluster_ID"]
    )
    assert rename_map == {" Bottom_Depth ": "base_depth"}


def test_company_csv_loads_successfully(company_csv_bytes: bytes) -> None:
    df, check, error = load_csv_with_check(company_csv_bytes, filename="example.csv")
    assert error is None
    assert df is not None
    assert check.is_valid
    assert COL_BASE_DEPTH in df.columns
    assert COL_TOP_DEPTH in df.columns
    assert "bottom_depth" not in df.columns


def test_company_csv_reports_the_mapping(company_csv_bytes: bytes) -> None:
    _, check, _ = load_csv_with_check(company_csv_bytes, filename="example.csv")
    assert check.renamed_columns == (("bottom_depth", "base_depth"),)
    assert "bottom_depth" in check.message
    assert "base_depth" in check.message


def test_company_csv_values_are_preserved(company_csv_bytes: bytes) -> None:
    df = load_csv(company_csv_bytes, filename="example.csv")
    assert len(df) == 17
    assert df[COL_TOP_DEPTH].iloc[0] == 25400
    assert df[COL_BASE_DEPTH].iloc[0] == 25420
    assert df["lithology"].iloc[0] == "Shale"
    # Cluster ids are identifiers, so they stay as strings ("0", not 0).
    assert df["cluster_id"].iloc[0] == "0"


def test_company_csv_builds_a_ready_document(company_csv_bytes: bytes) -> None:
    doc = build_document(csv_bytes=company_csv_bytes, csv_name="example.csv")
    assert doc.status == "Ready"
    assert doc.interval_count == 17
    assert doc.renamed_columns == (("bottom_depth", "base_depth"),)


def test_company_csv_statistics_are_correct(company_csv_bytes: bytes) -> None:
    doc = build_document(csv_bytes=company_csv_bytes, csv_name="example.csv")
    metrics = overview_metrics(doc.processed_df)
    assert metrics["interval_count"] == 17
    assert metrics["lithology_count"] == 3
    assert metrics["cluster_count"] == 3
    # 25400 -> 25700
    assert metrics["depth_coverage"] == 300.0
    assert metrics["cumulative_thickness"] == 300.0
    assert metrics["gap_count"] == 0


def test_company_csv_has_no_structural_findings(company_csv_bytes: bytes) -> None:
    doc = build_document(csv_bytes=company_csv_bytes, csv_name="example.csv")
    assert doc.findings == []


def test_original_source_df_restores_uploaded_headers(company_csv_bytes: bytes) -> None:
    doc = build_document(csv_bytes=company_csv_bytes, csv_name="example.csv")
    original = doc.original_source_df()
    assert "bottom_depth" in original.columns
    assert "base_depth" not in original.columns
    # Internal bookkeeping column must not leak into the original export.
    assert "source_row_index" not in original.columns


def test_restore_source_columns_without_renames() -> None:
    df = pd.DataFrame({"top_depth": [1], "base_depth": [2], "source_row_index": [0]})
    restored = restore_source_columns(df, ())
    assert list(restored.columns) == ["top_depth", "base_depth"]


def test_validate_schema_still_rejects_truly_missing_columns() -> None:
    df = pd.DataFrame({"top_depth": [1], "lithology": ["Shale"]})
    check = validate_schema(df)
    assert not check.is_valid
    assert "base_depth" in check.missing_columns


def test_unknown_extra_columns_are_reported_not_dropped() -> None:
    csv = (
        "top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id,confidence\n"
        "0,10,Shale,0,100,0,0.9\n"
    ).encode()
    df, check, _ = load_csv_with_check(csv, filename="extra.csv")
    assert check.is_valid
    assert "confidence" in check.extra_columns
    assert "confidence" in df.columns
