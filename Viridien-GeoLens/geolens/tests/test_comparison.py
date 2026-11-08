"""Tests for cross-document comparison and sequence similarity."""

from __future__ import annotations

import pandas as pd

from geolens.services.comparison import (
    compare_sequences,
    comparison_matrix,
    document_sequences,
    documents_sharing_lithology,
    pairwise_sequence_comparison,
    shared_lithologies,
    similarity_matrix,
)
from geolens.services.data_processor import consolidate_documents


def test_document_sequences_are_compressed(document_a) -> None:
    sequences = document_sequences(document_a.processed_df)
    sequence = sequences[document_a.document_name]
    assert sequence[0] == "Shale"
    # Consecutive duplicates must be collapsed.
    assert all(a != b for a, b in zip(sequence, sequence[1:]))


def test_compare_identical_sequences() -> None:
    result = compare_sequences("A", ["Shale", "Sandstone"], "B", ["Shale", "Sandstone"])
    assert result.similarity == 1.0
    assert result.shared_lithologies == ("Sandstone", "Shale")


def test_compare_different_sequences() -> None:
    result = compare_sequences("A", ["Shale"], "B", ["Evaporite"])
    assert result.similarity == 0.0
    assert result.only_in_a == ("Shale",)
    assert result.only_in_b == ("Evaporite",)


def test_pairwise_comparison(document_a, document_b, document_c) -> None:
    combined = consolidate_documents([document_a, document_b, document_c])
    table = pairwise_sequence_comparison(combined)
    assert len(table) == 3  # 3 choose 2
    assert "similarity_percent" in table.columns


def test_similarity_matrix_is_symmetric(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    matrix = similarity_matrix(combined)
    names = list(matrix.index)
    assert matrix.loc[names[0], names[1]] == matrix.loc[names[1], names[0]]
    assert matrix.loc[names[0], names[0]] == 100.0


def test_documents_sharing_lithology(document_a, document_b, document_c) -> None:
    combined = consolidate_documents([document_a, document_b, document_c])
    documents = documents_sharing_lithology(combined, "Shale")
    assert len(documents) == 3
    assert documents_sharing_lithology(combined, "Evaporite") == [document_c.document_name]


def test_shared_lithologies(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    common = shared_lithologies(combined, [document_a.document_name, document_b.document_name])
    assert "Shale" in common
    assert "Dolomite" not in common


def test_comparison_matrix_wrapper(document_a, document_b) -> None:
    combined = consolidate_documents([document_a, document_b])
    matrix = comparison_matrix(combined, metric="count")
    assert isinstance(matrix, pd.DataFrame)
    assert not matrix.empty


def test_cluster_ids_are_not_compared_across_documents(document_a, document_b) -> None:
    """Cluster ids are document-scoped; the comparison matrix must be lithology-based."""
    combined = consolidate_documents([document_a, document_b])
    matrix = comparison_matrix(combined, metric="count")
    assert "C1" not in matrix.index
    assert "K1" not in matrix.index
