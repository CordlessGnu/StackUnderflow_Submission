"""Cross-document comparison services.

Scope note: these routines compare *lithology label sequences* only. They do not
establish geological equivalence, and cluster IDs are never compared across
documents because they are scoped to their source document.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

import pandas as pd

from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_DOCUMENT_ID,
    COL_DOCUMENT_NAME,
    COL_LITHOLOGY,
    COL_TOP_DEPTH,
)
from geolens.services.data_processor import compress_sequence, lithology_sequence
from geolens.services.statistics import (
    lithology_comparison_matrix,
    lithology_presence_matrix,
)


@dataclass(frozen=True)
class SequenceComparison:
    """Result of comparing two lithology sequences."""

    document_a: str
    document_b: str
    similarity: float
    shared_lithologies: tuple[str, ...]
    only_in_a: tuple[str, ...]
    only_in_b: tuple[str, ...]
    sequence_a: tuple[str, ...]
    sequence_b: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "document_a": self.document_a,
            "document_b": self.document_b,
            "similarity_percent": round(self.similarity * 100.0, 1),
            "shared_lithologies": ", ".join(self.shared_lithologies),
            "only_in_a": ", ".join(self.only_in_a),
            "only_in_b": ", ".join(self.only_in_b),
            "sequence_a": " → ".join(self.sequence_a),
            "sequence_b": " → ".join(self.sequence_b),
        }


def document_sequences(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return the compressed lithology sequence for each document."""
    sequences: dict[str, list[str]] = {}
    if df.empty:
        return sequences
    for doc_name, group in df.groupby(COL_DOCUMENT_NAME, dropna=False):
        ordered = group.sort_values(COL_TOP_DEPTH, kind="mergesort")
        sequences[str(doc_name)] = compress_sequence(lithology_sequence(ordered))
    return sequences


def compare_sequences(
    name_a: str, seq_a: Sequence[str], name_b: str, seq_b: Sequence[str]
) -> SequenceComparison:
    """Compare two lithology sequences using a simple ratio metric.

    This is an exploratory similarity indicator, not a geological correlation.
    """
    matcher = SequenceMatcher(a=list(seq_a), b=list(seq_b), autojunk=False)
    similarity = float(matcher.ratio())
    set_a, set_b = set(seq_a), set(seq_b)
    return SequenceComparison(
        document_a=name_a,
        document_b=name_b,
        similarity=similarity,
        shared_lithologies=tuple(sorted(set_a & set_b)),
        only_in_a=tuple(sorted(set_a - set_b)),
        only_in_b=tuple(sorted(set_b - set_a)),
        sequence_a=tuple(seq_a),
        sequence_b=tuple(seq_b),
    )


def pairwise_sequence_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare every pair of documents and return a similarity table."""
    sequences = document_sequences(df)
    names = sorted(sequences)
    rows: list[dict[str, object]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            result = compare_sequences(name_a, sequences[name_a], name_b, sequences[name_b])
            rows.append(result.as_dict())
    columns = [
        "document_a",
        "document_b",
        "similarity_percent",
        "shared_lithologies",
        "only_in_a",
        "only_in_b",
        "sequence_a",
        "sequence_b",
    ]
    return pd.DataFrame(rows, columns=columns)


def similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a symmetric document x document sequence similarity matrix."""
    sequences = document_sequences(df)
    names = sorted(sequences)
    if not names:
        return pd.DataFrame()
    data: dict[str, dict[str, float]] = {}
    for name_a in names:
        row: dict[str, float] = {}
        for name_b in names:
            if name_a == name_b:
                row[name_b] = 100.0
            else:
                result = compare_sequences(name_a, sequences[name_a], name_b, sequences[name_b])
                row[name_b] = round(result.similarity * 100.0, 1)
        data[name_a] = row
    return pd.DataFrame(data).T.loc[names, names]


def comparison_matrix(
    df: pd.DataFrame, *, metric: str = "count"
) -> pd.DataFrame:
    """Public wrapper for the lithology comparison matrix."""
    return lithology_comparison_matrix(df, metric=metric)


def presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the lithology presence (interval count) matrix."""
    return lithology_presence_matrix(df)


def documents_sharing_lithology(df: pd.DataFrame, lithology: str) -> list[str]:
    """Return documents in which a lithology occurs."""
    if df.empty:
        return []
    subset = df[df[COL_LITHOLOGY] == lithology]
    return sorted({str(n) for n in subset[COL_DOCUMENT_NAME].dropna().unique()})


def shared_lithologies(df: pd.DataFrame, document_names: Sequence[str]) -> list[str]:
    """Return lithologies present in every one of the selected documents."""
    if df.empty or not document_names:
        return []
    subset = df[df[COL_DOCUMENT_NAME].isin(set(document_names))]
    per_doc = subset.groupby(COL_DOCUMENT_NAME)[COL_LITHOLOGY].apply(set)
    if len(per_doc) < len(set(document_names)):
        return []
    common = set.intersection(*per_doc.tolist()) if len(per_doc) else set()
    return sorted(str(v) for v in common)


def document_depth_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return min/max depth per document for comparability checks."""
    if df.empty:
        return pd.DataFrame(columns=[COL_DOCUMENT_NAME, "min_top", "max_base", "span"])
    rows = []
    for doc_name, group in df.groupby(COL_DOCUMENT_NAME):
        top = pd.to_numeric(group[COL_TOP_DEPTH], errors="coerce").dropna()
        base = pd.to_numeric(group[COL_BASE_DEPTH], errors="coerce").dropna()
        if top.empty or base.empty:
            continue
        rows.append(
            {
                COL_DOCUMENT_NAME: doc_name,
                "min_top": float(top.min()),
                "max_base": float(base.max()),
                "span": float(base.max() - top.min()),
            }
        )
    return pd.DataFrame(rows)


def document_ids(df: pd.DataFrame) -> list[str]:
    """Return unique document ids in a consolidated dataframe."""
    if df.empty:
        return []
    return sorted({str(v) for v in df[COL_DOCUMENT_ID].dropna().unique()})
