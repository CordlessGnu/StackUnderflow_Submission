"""Statistics services for single-document and collection-level analysis."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_CLUSTER_ID,
    COL_DOCUMENT_ID,
    COL_DOCUMENT_NAME,
    COL_LITHOLOGY,
    COL_THICKNESS,
    COL_TOP_DEPTH,
)
from geolens.services.data_processor import depth_coverage, interval_thickness
from geolens.utils.formatting import safe_divide


def overview_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the single-document overview metrics.

    Returns interval count, unique lithology/cluster counts, depth coverage,
    average thickness and the thickest interval.
    """
    if df.empty:
        return {
            "interval_count": 0,
            "lithology_count": 0,
            "cluster_count": 0,
            "depth_coverage": 0.0,
            "min_top": float("nan"),
            "max_base": float("nan"),
            "average_thickness": 0.0,
            "thickest_interval": None,
            "cumulative_thickness": 0.0,
            "gap_total": 0.0,
            "gap_count": 0,
        }

    coverage = depth_coverage(df)
    thickness = interval_thickness(df)
    positive = thickness[thickness > 0]

    thickest = None
    if not positive.empty:
        idx = positive.idxmax()
        row = df.loc[idx]
        thickest = {
            "lithology": row.get(COL_LITHOLOGY),
            "cluster_id": row.get(COL_CLUSTER_ID),
            "top_depth": row.get(COL_TOP_DEPTH),
            "base_depth": row.get(COL_BASE_DEPTH),
            "thickness": float(positive.loc[idx]),
        }

    return {
        "interval_count": int(len(df)),
        "lithology_count": int(df[COL_LITHOLOGY].nunique(dropna=True)),
        "cluster_count": int(df[COL_CLUSTER_ID].nunique(dropna=True)),
        "depth_coverage": coverage["coverage"],
        "min_top": coverage["min_top"],
        "max_base": coverage["max_base"],
        "average_thickness": float(positive.mean()) if not positive.empty else 0.0,
        "thickest_interval": thickest,
        "cumulative_thickness": coverage["cumulative_thickness"],
        "gap_total": coverage["gap_total"],
        "gap_count": coverage["gap_count"],
    }


def lithology_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lithology statistics.

    Columns: lithology, interval_count, total_thickness, avg_thickness,
    percent_of_depth, cluster_count, clusters.
    """
    columns = [
        COL_LITHOLOGY,
        "interval_count",
        "total_thickness",
        "avg_thickness",
        "percent_of_depth",
        "cluster_count",
        "clusters",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work[COL_THICKNESS] = interval_thickness(work)
    total_thickness = float(work[COL_THICKNESS].clip(lower=0).sum())

    rows: list[dict[str, Any]] = []
    for lithology, group in work.groupby(COL_LITHOLOGY, dropna=False):
        thickness = group[COL_THICKNESS].clip(lower=0)
        total = float(thickness.sum())
        clusters = sorted({str(c) for c in group[COL_CLUSTER_ID].dropna().unique()})
        rows.append(
            {
                COL_LITHOLOGY: lithology,
                "interval_count": int(len(group)),
                "total_thickness": total,
                "avg_thickness": float(thickness.mean()) if len(group) else 0.0,
                "percent_of_depth": safe_divide(total, total_thickness) * 100.0,
                "cluster_count": len(clusters),
                "clusters": ", ".join(clusters),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("total_thickness", ascending=False).reset_index(drop=True)


def cluster_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster statistics scoped to a single document.

    Columns: cluster_id, interval_count, lithologies, lithology_count,
    total_thickness, min_top, max_base, depth_ranges.
    """
    columns = [
        COL_CLUSTER_ID,
        "interval_count",
        "lithologies",
        "lithology_count",
        "total_thickness",
        "min_top",
        "max_base",
        "depth_ranges",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work[COL_THICKNESS] = interval_thickness(work)

    rows: list[dict[str, Any]] = []
    for cluster_id, group in work.groupby(COL_CLUSTER_ID, dropna=False):
        lithologies = sorted({str(v) for v in group[COL_LITHOLOGY].dropna().unique()})
        top = pd.to_numeric(group[COL_TOP_DEPTH], errors="coerce").dropna()
        base = pd.to_numeric(group[COL_BASE_DEPTH], errors="coerce").dropna()
        ranges = [
            f"{float(r[COL_TOP_DEPTH]):g}–{float(r[COL_BASE_DEPTH]):g}"
            for _, r in group.dropna(subset=[COL_TOP_DEPTH, COL_BASE_DEPTH]).iterrows()
        ]
        rows.append(
            {
                COL_CLUSTER_ID: cluster_id,
                "interval_count": int(len(group)),
                "lithologies": ", ".join(lithologies),
                "lithology_count": len(lithologies),
                "total_thickness": float(group[COL_THICKNESS].clip(lower=0).sum()),
                "min_top": float(top.min()) if not top.empty else float("nan"),
                "max_base": float(base.max()) if not base.empty else float("nan"),
                "depth_ranges": "; ".join(ranges),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("interval_count", ascending=False).reset_index(drop=True)


def document_summary(document: Any) -> dict[str, Any]:
    """Build a one-row summary for a document record."""
    df = getattr(document, "processed_df", pd.DataFrame())
    metrics = overview_metrics(df)
    return {
        COL_DOCUMENT_ID: getattr(document, "document_id", "unknown"),
        COL_DOCUMENT_NAME: getattr(document, "document_name", "unknown"),
        "interval_count": metrics["interval_count"],
        "lithology_count": metrics["lithology_count"],
        "cluster_count": metrics["cluster_count"],
        "depth_coverage": metrics["depth_coverage"],
        "min_top": metrics["min_top"],
        "max_base": metrics["max_base"],
        "average_thickness": metrics["average_thickness"],
        "status": getattr(document, "status", "Unknown"),
    }


def collection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-document summary table for a consolidated dataframe."""
    columns = [
        COL_DOCUMENT_ID,
        COL_DOCUMENT_NAME,
        "interval_count",
        "lithology_count",
        "cluster_count",
        "depth_coverage",
        "min_top",
        "max_base",
        "average_thickness",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for doc_id, group in df.groupby(COL_DOCUMENT_ID, dropna=False):
        metrics = overview_metrics(group)
        rows.append(
            {
                COL_DOCUMENT_ID: doc_id,
                COL_DOCUMENT_NAME: group[COL_DOCUMENT_NAME].iloc[0],
                "interval_count": metrics["interval_count"],
                "lithology_count": metrics["lithology_count"],
                "cluster_count": metrics["cluster_count"],
                "depth_coverage": metrics["depth_coverage"],
                "min_top": metrics["min_top"],
                "max_base": metrics["max_base"],
                "average_thickness": metrics["average_thickness"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def organisation_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Organisation-wide statistics across a consolidated dataset."""
    if df.empty:
        return {
            "document_count": 0,
            "interval_count": 0,
            "lithology_count": 0,
            "cluster_count": 0,
            "average_thickness": 0.0,
            "total_thickness": 0.0,
            "top_lithologies": [],
            "largest_document": None,
            "lithology_documents": {},
        }

    thickness = interval_thickness(df)
    positive = thickness[thickness > 0]

    # Clusters are scoped per document: count distinct (document, cluster) pairs.
    cluster_pairs = df[[COL_DOCUMENT_ID, COL_CLUSTER_ID]].dropna().drop_duplicates()

    lithology_counts = (
        df.groupby(COL_LITHOLOGY)[COL_DOCUMENT_ID]
        .nunique()
        .sort_values(ascending=False)
    )
    lithology_documents: dict[str, list[str]] = {}
    for lithology, group in df.groupby(COL_LITHOLOGY):
        lithology_documents[str(lithology)] = sorted(
            {str(n) for n in group[COL_DOCUMENT_NAME].dropna().unique()}
        )

    per_doc_counts = df.groupby(COL_DOCUMENT_NAME).size().sort_values(ascending=False)
    largest = None
    if not per_doc_counts.empty:
        largest = {"document": str(per_doc_counts.index[0]), "intervals": int(per_doc_counts.iloc[0])}

    return {
        "document_count": int(df[COL_DOCUMENT_ID].nunique()),
        "interval_count": int(len(df)),
        "lithology_count": int(df[COL_LITHOLOGY].nunique()),
        "cluster_count": int(len(cluster_pairs)),
        "average_thickness": float(positive.mean()) if not positive.empty else 0.0,
        "total_thickness": float(positive.sum()),
        "top_lithologies": [
            {"lithology": str(k), "document_count": int(v)}
            for k, v in lithology_counts.head(10).items()
        ],
        "largest_document": largest,
        "lithology_documents": lithology_documents,
    }


def lithology_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Interval counts and thickness per lithology across a dataset."""
    if df.empty:
        return pd.DataFrame(columns=[COL_LITHOLOGY, "interval_count", "total_thickness"])
    work = df.copy()
    work[COL_THICKNESS] = interval_thickness(work)
    grouped = (
        work.groupby(COL_LITHOLOGY)
        .agg(
            interval_count=(COL_LITHOLOGY, "size"),
            total_thickness=(COL_THICKNESS, lambda s: float(s.clip(lower=0).sum())),
        )
        .reset_index()
        .sort_values("interval_count", ascending=False)
    )
    return grouped.reset_index(drop=True)


def depth_range_for(df: pd.DataFrame) -> tuple[float, float]:
    """Return the (min top, max base) depth range, or (0, 0) when unavailable."""
    if df.empty:
        return 0.0, 0.0
    top = pd.to_numeric(df[COL_TOP_DEPTH], errors="coerce").dropna()
    base = pd.to_numeric(df[COL_BASE_DEPTH], errors="coerce").dropna()
    if top.empty or base.empty:
        return 0.0, 0.0
    return float(top.min()), float(base.max())


def lithology_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Lithology x document matrix of interval counts (zero when absent)."""
    if df.empty:
        return pd.DataFrame()
    matrix = pd.crosstab(df[COL_LITHOLOGY], df[COL_DOCUMENT_NAME])
    return matrix.sort_index()


def lithology_thickness_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Lithology x document matrix of total thickness (zero when absent)."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work[COL_THICKNESS] = interval_thickness(work).clip(lower=0)
    matrix = work.pivot_table(
        index=COL_LITHOLOGY,
        columns=COL_DOCUMENT_NAME,
        values=COL_THICKNESS,
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.sort_index()


def lithology_percentage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Lithology x document matrix of percentage of each document's thickness."""
    thickness = lithology_thickness_matrix(df)
    if thickness.empty:
        return thickness
    totals = thickness.sum(axis=0)
    return thickness.divide(totals.replace(0, pd.NA), axis=1).fillna(0.0) * 100.0


def lithology_comparison_matrix(
    df: pd.DataFrame, *, metric: str = "count"
) -> pd.DataFrame:
    """Build the cross-document lithology comparison matrix.

    Args:
        df: consolidated processed dataframe.
        metric: one of ``count``, ``thickness`` or ``percent``.

    Returns:
        Dataframe indexed by lithology with one column per document. Absent
        lithologies are shown as zero rather than treated as an error.
    """
    if metric == "thickness":
        return lithology_thickness_matrix(df)
    if metric == "percent":
        return lithology_percentage_matrix(df)
    return lithology_presence_matrix(df)


def trim_matrix(
    matrix: pd.DataFrame,
    *,
    max_columns: int = 12,
    max_rows: int = 40,
) -> tuple[pd.DataFrame, int, int]:
    """Trim a comparison matrix down to rows and columns that carry data.

    Rows where every value is zero (a lithology that appears in none of the
    selected documents) are dropped so the table never shows blank rows.
    Columns are capped to the ``max_columns`` documents with the largest totals,
    and rows to the ``max_rows`` lithologies with the largest totals.

    Returns:
        ``(trimmed, dropped_rows, dropped_columns)`` where the counts describe
        how much was removed so the caller can tell the user.
    """
    if matrix.empty:
        return matrix, 0, 0

    work = matrix.copy()
    work = work.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Drop lithologies that are absent from every document.
    row_totals = work.sum(axis=1)
    kept_rows = work[row_totals > 0]
    dropped_rows = int(len(work) - len(kept_rows))
    work = kept_rows

    if work.empty:
        return work, dropped_rows, 0

    # Drop documents that contain none of the remaining lithologies.
    col_totals = work.sum(axis=0)
    kept_cols = work.loc[:, col_totals > 0]
    dropped_columns = int(len(work.columns) - len(kept_cols.columns))
    work = kept_cols

    if work.empty:
        return work, dropped_rows, dropped_columns

    # Cap the number of columns, keeping the documents with the most data.
    if len(work.columns) > max_columns:
        order = work.sum(axis=0).sort_values(ascending=False).index[:max_columns]
        dropped_columns += int(len(work.columns) - len(order))
        work = work.loc[:, list(order)]

    # Cap the number of rows, keeping the most prominent lithologies.
    if len(work) > max_rows:
        order = work.sum(axis=1).sort_values(ascending=False).index[:max_rows]
        dropped_rows += int(len(work) - len(order))
        work = work.loc[list(order)]

    return work, dropped_rows, dropped_columns


def lithology_document_lookup(df: pd.DataFrame, lithology: str) -> list[str]:
    """Return documents containing a given lithology."""
    if df.empty:
        return []
    subset = df[df[COL_LITHOLOGY] == lithology]
    return sorted({str(n) for n in subset[COL_DOCUMENT_NAME].dropna().unique()})


def filter_by_documents(df: pd.DataFrame, document_names: Sequence[str]) -> pd.DataFrame:
    """Filter a consolidated dataframe to the given document names."""
    if not document_names:
        return df
    return df[df[COL_DOCUMENT_NAME].isin(set(document_names))].reset_index(drop=True)
