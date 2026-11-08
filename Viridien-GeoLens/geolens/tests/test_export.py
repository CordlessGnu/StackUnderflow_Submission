"""Tests for export generation."""

from __future__ import annotations

import io

import pandas as pd

from geolens.services.data_processor import consolidate_documents
from geolens.services.export_service import (
    build_batch_workbook,
    build_single_workbook,
    build_summary_report,
    dataframe_to_csv_bytes,
    export_filename,
    processed_intervals,
)
from geolens.services.validation import check_collection


def test_processed_intervals_adds_thickness(simple_frame: pd.DataFrame) -> None:
    result = processed_intervals(simple_frame.drop(columns=["thickness"]))
    assert "thickness" in result.columns
    assert result["thickness"].tolist() == [10.0, 10.0, 10.0]


def test_csv_bytes_roundtrip(simple_frame: pd.DataFrame) -> None:
    data = dataframe_to_csv_bytes(simple_frame)
    assert isinstance(data, bytes)
    restored = pd.read_csv(io.BytesIO(data))
    assert len(restored) == len(simple_frame)


def test_single_workbook_has_expected_sheets(simple_frame: pd.DataFrame) -> None:
    data = build_single_workbook(simple_frame, document_name="Doc 1")
    assert data[:2] == b"PK"  # xlsx is a zip archive
    workbook = pd.ExcelFile(io.BytesIO(data))
    assert "Intervals" in workbook.sheet_names
    assert "Lithology Stats" in workbook.sheet_names
    assert "Cluster Stats" in workbook.sheet_names
    assert "Data Quality" in workbook.sheet_names


def test_batch_workbook_has_expected_sheets(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    findings = check_collection(combined)
    data = build_batch_workbook(combined, findings=findings)
    workbook = pd.ExcelFile(io.BytesIO(data))
    for sheet in (
        "Consolidated Intervals",
        "Document Summaries",
        "Lithology Comparison",
        "Lithology Stats",
        "Cluster Stats",
        "Data Quality",
    ):
        assert sheet in workbook.sheet_names


def test_batch_workbook_handles_empty_frame() -> None:
    data = build_batch_workbook(pd.DataFrame())
    workbook = pd.ExcelFile(io.BytesIO(data))
    assert "Consolidated Intervals" in workbook.sheet_names


def test_summary_report_contains_disclaimer(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    report = build_summary_report(combined)
    assert "GeoLens" in report
    assert "not a verified geological interpretation" in report
    assert "Lithology distribution" in report


def test_summary_report_records_filters(document_a) -> None:
    report = build_summary_report(
        document_a.processed_df,
        filters_applied={"Lithology": ["Shale"]},
    )
    assert "Lithology: Shale" in report


def test_export_filename_is_timestamped() -> None:
    name = export_filename("test", "csv")
    assert name.startswith("geolens_test_")
    assert name.endswith(".csv")


def test_export_preserves_source_columns(document_a) -> None:
    exported = processed_intervals(document_a.processed_df)
    for column in ("top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"):
        assert column in exported.columns
