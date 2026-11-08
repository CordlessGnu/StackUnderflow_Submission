"""Tests for single-document and collection statistics."""

from __future__ import annotations

import pandas as pd

from geolens.models.schemas import COL_LITHOLOGY
from geolens.services.data_processor import consolidate_documents
from geolens.services.statistics import (
    cluster_statistics,
    collection_summary,
    lithology_comparison_matrix,
    lithology_distribution,
    lithology_statistics,
    organisation_statistics,
    overview_metrics,
    trim_matrix,
)


def test_overview_metrics(simple_frame: pd.DataFrame) -> None:
    metrics = overview_metrics(simple_frame)
    assert metrics["interval_count"] == 3
    assert metrics["lithology_count"] == 2
    assert metrics["cluster_count"] == 2
    assert metrics["depth_coverage"] == 30.0
    assert metrics["average_thickness"] == 10.0
    assert metrics["thickest_interval"]["thickness"] == 10.0


def test_overview_metrics_empty() -> None:
    metrics = overview_metrics(pd.DataFrame())
    assert metrics["interval_count"] == 0
    assert metrics["thickest_interval"] is None


def test_lithology_statistics(simple_frame: pd.DataFrame) -> None:
    stats = lithology_statistics(simple_frame)
    shale = stats[stats[COL_LITHOLOGY] == "Shale"].iloc[0]
    assert shale["interval_count"] == 2
    assert shale["total_thickness"] == 20.0
    assert shale["avg_thickness"] == 10.0
    assert round(shale["percent_of_depth"], 1) == 66.7
    assert shale["cluster_count"] == 1


def test_lithology_statistics_empty() -> None:
    stats = lithology_statistics(pd.DataFrame())
    assert stats.empty
    assert "interval_count" in stats.columns


def test_cluster_statistics(simple_frame: pd.DataFrame) -> None:
    stats = cluster_statistics(simple_frame)
    c1 = stats[stats["cluster_id"] == "C1"].iloc[0]
    assert c1["interval_count"] == 2
    assert c1["lithology_count"] == 1
    assert c1["total_thickness"] == 20.0


def test_cluster_statistics_multiple_lithologies() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": ["C1", "C1"],
            "lithology": ["Shale", "Sandstone"],
            "top_depth": [0.0, 10.0],
            "base_depth": [10.0, 20.0],
            "thickness": [10.0, 10.0],
        }
    )
    stats = cluster_statistics(df)
    assert stats.iloc[0]["lithology_count"] == 2


def test_collection_summary(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    summary = collection_summary(combined)
    assert len(summary) == 2
    assert summary["interval_count"].sum() == len(combined)


def test_organisation_statistics(document_a, document_b, document_c) -> None:
    combined = consolidate_documents([document_a, document_b, document_c])
    stats = organisation_statistics(combined)
    assert stats["document_count"] == 3
    assert stats["interval_count"] == len(combined)
    assert stats["lithology_count"] >= 6
    # Clusters are document-scoped, so the total is the sum of per-document counts.
    per_doc = combined.groupby("document_id")["cluster_id"].nunique().sum()
    assert stats["cluster_count"] == per_doc
    assert stats["largest_document"] is not None


def test_lithology_distribution(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    distribution = lithology_distribution(combined)
    assert not distribution.empty
    assert distribution["interval_count"].sum() == len(combined)


def test_comparison_matrix_has_zero_for_absent_lithology(document_a, document_c) -> None:
    combined = consolidate_documents([document_a, document_c])
    matrix = lithology_comparison_matrix(combined, metric="count")
    assert "Evaporite" in matrix.index
    # Evaporite exists only in well_c, so well_a's column must be zero.
    doc_a_column = document_a.document_name
    assert matrix.loc["Evaporite", doc_a_column] == 0


def test_comparison_matrix_percent_sums_to_100(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    matrix = lithology_comparison_matrix(combined, metric="percent")
    for column in matrix.columns:
        assert abs(matrix[column].sum() - 100.0) < 0.01


def test_comparison_matrix_thickness(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    matrix = lithology_comparison_matrix(combined, metric="thickness")
    assert matrix.loc["Shale", document_a.document_name] > 0


def test_trim_matrix_drops_empty_rows_and_columns() -> None:
    matrix = pd.DataFrame(
        [[1, 0, 2, 0], [0, 0, 0, 0], [3, 1, 0, 2]],
        index=["A", "B", "C"],
        columns=["d1", "d2", "d3", "d4"],
    )
    trimmed, dropped_rows, dropped_columns = trim_matrix(matrix)
    assert list(trimmed.index) == ["A", "C"]
    assert dropped_rows == 1
    assert dropped_columns == 0
    assert (trimmed.sum(axis=1) > 0).all()


def test_trim_matrix_caps_columns_and_rows() -> None:
    matrix = pd.DataFrame(
        [[1, 5, 2, 9], [4, 1, 8, 3], [2, 7, 1, 6]],
        index=["A", "B", "C"],
        columns=["d1", "d2", "d3", "d4"],
    )
    trimmed, dropped_rows, dropped_columns = trim_matrix(matrix, max_columns=2, max_rows=2)
    assert trimmed.shape == (2, 2)
    assert dropped_rows == 1
    assert dropped_columns == 2
    # Column totals are d1=7, d2=13, d3=11, d4=18, so d4 and d2 survive.
    assert list(trimmed.columns) == ["d4", "d2"]


def test_trim_matrix_empty_input() -> None:
    trimmed, dropped_rows, dropped_columns = trim_matrix(pd.DataFrame())
    assert trimmed.empty
    assert dropped_rows == 0
    assert dropped_columns == 0
