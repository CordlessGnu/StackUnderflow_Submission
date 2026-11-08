"""Tests for calculated fields, consolidation and filtering."""

from __future__ import annotations

import pandas as pd

from geolens.models.schemas import (
    COL_DOCUMENT_ID,
    COL_LITHOLOGY,
    COL_THICKNESS,
)
from geolens.services.data_processor import (
    add_calculated_fields,
    apply_filters,
    build_document,
    compress_sequence,
    consolidate_documents,
    depth_coverage,
    depth_units_consistent,
    interval_thickness,
    lithology_sequence,
)


def test_thickness_is_calculated(simple_frame: pd.DataFrame) -> None:
    assert simple_frame[COL_THICKNESS].tolist() == [10.0, 10.0, 10.0]


def test_source_columns_are_not_overwritten(simple_frame: pd.DataFrame) -> None:
    assert simple_frame["top_depth"].tolist() == [0.0, 10.0, 20.0]
    assert simple_frame["base_depth"].tolist() == [10.0, 20.0, 30.0]
    assert simple_frame["lithology"].tolist() == ["Shale", "Sandstone", "Shale"]


def test_missing_lithology_gets_placeholder() -> None:
    raw = pd.DataFrame(
        {
            "top_y": [0],
            "bottom_y": [10],
            "cluster_id": ["C1"],
            "lithology": [None],
            "top_depth": [0.0],
            "base_depth": [10.0],
            "source_row_index": [0],
        }
    )
    processed = add_calculated_fields(raw, document_id="d", document_name="D")
    assert processed[COL_LITHOLOGY].iloc[0] == "Unknown"
    assert processed["is_valid"].iloc[0] is False or processed["is_valid"].iloc[0] == False


def test_negative_thickness_flagged() -> None:
    raw = pd.DataFrame(
        {
            "top_y": [0],
            "bottom_y": [10],
            "cluster_id": ["C1"],
            "lithology": ["Shale"],
            "top_depth": [50.0],
            "base_depth": [10.0],
            "source_row_index": [0],
        }
    )
    processed = add_calculated_fields(raw, document_id="d", document_name="D")
    assert processed[COL_THICKNESS].iloc[0] == -40.0
    assert "negative thickness" in processed["issues"].iloc[0]


def test_lithology_case_is_folded() -> None:
    """'shale', 'SHALE' and 'Shale' must become one lithology label."""
    raw = pd.DataFrame(
        {
            "top_y": [0, 0, 0],
            "bottom_y": [10, 10, 10],
            "cluster_id": ["C1", "C1", "C1"],
            "lithology": ["shale", "SHALE", "Shale"],
            "top_depth": [0.0, 10.0, 20.0],
            "base_depth": [10.0, 20.0, 30.0],
            "source_row_index": [0, 1, 2],
        }
    )
    processed = add_calculated_fields(raw, document_id="d", document_name="D")
    assert processed[COL_LITHOLOGY].unique().tolist() == ["Shale"]


def test_depth_coverage(simple_frame: pd.DataFrame) -> None:
    coverage = depth_coverage(simple_frame)
    assert coverage["min_top"] == 0.0
    assert coverage["max_base"] == 30.0
    assert coverage["coverage"] == 30.0
    assert coverage["cumulative_thickness"] == 30.0
    assert coverage["gap_count"] == 0


def test_depth_coverage_detects_gaps() -> None:
    df = pd.DataFrame(
        {
            "top_depth": [0.0, 20.0],
            "base_depth": [10.0, 30.0],
            "thickness": [10.0, 10.0],
        }
    )
    coverage = depth_coverage(df)
    assert coverage["gap_count"] == 1
    assert coverage["gap_total"] == 10.0
    assert coverage["coverage"] == 30.0


def test_consolidation_preserves_documents(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    assert combined[COL_DOCUMENT_ID].nunique() == 2
    assert len(combined) == len(document_a.processed_df) + len(document_b.processed_df)
    assert set(combined[COL_LITHOLOGY]) >= {"Shale", "Dolomite"}


def test_adding_document_does_not_corrupt_existing(document_a, document_b, document_c) -> None:
    first = consolidate_documents([document_a])
    second = consolidate_documents([document_a, document_b, document_c])
    assert len(second) > len(first)
    assert len(second[second[COL_DOCUMENT_ID] == document_a.document_id]) == len(first)


def test_apply_filters_by_lithology(simple_frame: pd.DataFrame) -> None:
    filtered = apply_filters(simple_frame, lithologies=["Shale"])
    assert set(filtered[COL_LITHOLOGY]) == {"Shale"}
    assert len(filtered) == 2


def test_apply_filters_by_depth(simple_frame: pd.DataFrame) -> None:
    filtered = apply_filters(simple_frame, depth_range=(15.0, 30.0))
    assert len(filtered) == 2


def test_empty_selection_means_no_restriction(simple_frame: pd.DataFrame) -> None:
    assert len(apply_filters(simple_frame, lithologies=[])) == len(simple_frame)


def test_interval_thickness_recomputes_when_absent(simple_frame: pd.DataFrame) -> None:
    without = simple_frame.drop(columns=[COL_THICKNESS])
    assert interval_thickness(without).tolist() == [10.0, 10.0, 10.0]


def test_lithology_sequence_is_depth_ordered() -> None:
    df = pd.DataFrame(
        {
            "top_depth": [20.0, 0.0, 10.0],
            "lithology": ["Sandstone", "Shale", "Siltstone"],
        }
    )
    assert lithology_sequence(df) == ["Shale", "Siltstone", "Sandstone"]


def test_compress_sequence_collapses_repeats() -> None:
    assert compress_sequence(["Shale", "Shale", "Sandstone", "Shale"]) == [
        "Shale",
        "Sandstone",
        "Shale",
    ]


def test_build_document_keeps_source_and_processed_separate(well_a_bytes: bytes) -> None:
    record = build_document(csv_bytes=well_a_bytes, csv_name="well_a.csv")
    assert COL_THICKNESS not in record.source_df.columns
    assert COL_THICKNESS in record.processed_df.columns
    assert record.schema_check.is_valid


def test_build_document_handles_invalid_schema() -> None:
    record = build_document(csv_bytes=b"a,b\n1,2\n", csv_name="bad.csv")
    assert not record.schema_check.is_valid
    assert record.status == "Invalid schema"


def test_depth_units_consistency_heuristic(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    assert depth_units_consistent(combined) is True
