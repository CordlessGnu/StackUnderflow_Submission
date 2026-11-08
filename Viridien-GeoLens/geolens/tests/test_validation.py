"""Tests for data consistency checks."""

from __future__ import annotations

import pandas as pd

from geolens.services.data_processor import add_calculated_fields, build_document
from geolens.services.validation import (
    check_collection,
    check_document,
    findings_to_frame,
    quality_summary,
    valid_record_count,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    raw["source_row_index"] = range(len(raw))
    return add_calculated_fields(raw, document_id="doc", document_name="Doc")


def test_clean_document_has_no_errors(simple_frame: pd.DataFrame) -> None:
    findings = check_document(simple_frame, document_id="doc")
    assert not any(f.severity == "error" for f in findings)


def test_missing_lithology_reported() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": None,
                "top_depth": 0.0,
                "base_depth": 10.0,
            }
        ]
    )
    findings = check_document(df, document_id="doc")
    assert any("lithology" in f.check for f in findings)


def test_negative_thickness_reported() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": "Shale",
                "top_depth": 50.0,
                "base_depth": 10.0,
            }
        ]
    )
    findings = check_document(df, document_id="doc")
    assert any(f.check == "negative_thickness" and f.severity == "error" for f in findings)


def test_zero_thickness_reported() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": "Shale",
                "top_depth": 10.0,
                "base_depth": 10.0,
            }
        ]
    )
    findings = check_document(df, document_id="doc")
    assert any(f.check == "zero_thickness" for f in findings)


def test_overlap_and_gap_reported() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": "Shale",
                "top_depth": 0.0,
                "base_depth": 20.0,
            },
            {
                "top_y": 10,
                "bottom_y": 20,
                "cluster_id": "C2",
                "lithology": "Sandstone",
                "top_depth": 10.0,
                "base_depth": 30.0,
            },
            {
                "top_y": 20,
                "bottom_y": 30,
                "cluster_id": "C3",
                "lithology": "Siltstone",
                "top_depth": 40.0,
                "base_depth": 50.0,
            },
        ]
    )
    findings = check_document(df, document_id="doc")
    checks = {f.check for f in findings}
    assert "overlapping_intervals" in checks
    assert "depth_gaps" in checks


def test_duplicate_records_reported() -> None:
    row = {
        "top_y": 0,
        "bottom_y": 10,
        "cluster_id": "C1",
        "lithology": "Shale",
        "top_depth": 0.0,
        "base_depth": 10.0,
    }
    df = _frame([row, dict(row)])
    findings = check_document(df, document_id="doc")
    assert any(f.check == "duplicate_records" for f in findings)


def test_cluster_lithology_mismatch_is_informational() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": "Shale",
                "top_depth": 0.0,
                "base_depth": 10.0,
            },
            {
                "top_y": 10,
                "bottom_y": 20,
                "cluster_id": "C1",
                "lithology": "Sandstone",
                "top_depth": 10.0,
                "base_depth": 20.0,
            },
        ]
    )
    findings = check_document(df, document_id="doc")
    mismatch = [f for f in findings if f.check == "cluster_lithology_mismatch"]
    assert mismatch and mismatch[0].severity == "info"


def test_messy_sample_reports_findings(messy_bytes: bytes) -> None:
    record = build_document(csv_bytes=messy_bytes, csv_name="well_d_messy.csv")
    assert record.findings
    checks = {f.check for f in record.findings}
    assert "missing_lithology" in checks


def test_collection_checks(document_a, document_b) -> None:
    from geolens.services.data_processor import consolidate_documents

    combined = consolidate_documents([document_a, document_b])
    findings = check_collection(combined)
    assert isinstance(findings, list)


def test_findings_to_frame_and_summary() -> None:
    df = _frame(
        [
            {
                "top_y": 0,
                "bottom_y": 10,
                "cluster_id": "C1",
                "lithology": "Shale",
                "top_depth": 50.0,
                "base_depth": 10.0,
            }
        ]
    )
    findings = check_document(df, document_id="doc")
    frame = findings_to_frame(findings)
    assert not frame.empty
    summary = quality_summary(findings)
    assert summary["error"] >= 1


def test_valid_record_count(simple_frame: pd.DataFrame) -> None:
    assert valid_record_count(simple_frame) == 3
