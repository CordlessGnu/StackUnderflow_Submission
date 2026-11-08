"""Core data processing: calculated fields, consolidation and filtering.

Design rule: source columns are never modified. Every calculated value is added
as a new column so exports remain traceable back to the original CSV.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CSV_PROCESSING_VERSION = 'final-lithology-clusters-v1'

from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_CLUSTER_ID,
    COL_CLUSTER_MISSING,
    COL_DOCUMENT_ID,
    COL_DOCUMENT_NAME,
    COL_ISSUES,
    COL_IS_VALID,
    COL_LITHOLOGY,
    COL_LITHOLOGY_MISSING,
    COL_SOURCE_ROW,
    COL_THICKNESS,
    COL_TOP_DEPTH,
    PROCESSED_COLUMN_ORDER,
    UNKNOWN_CLUSTER,
    UNKNOWN_LITHOLOGY,
)


def _clean_label(series: pd.Series, fallback: str, *, fold_case: bool = False) -> pd.Series:
    """Normalise a categorical column, filling blanks with a placeholder.

    ``fold_case`` collapses labels that differ only in capitalisation ("Shale",
    "SHALE", "shale") into one canonical display form, so they are treated as
    the same lithology everywhere downstream.
    """
    cleaned = series.astype("object").where(series.notna(), fallback)
    cleaned = cleaned.apply(lambda v: str(v).strip() if v is not None else fallback)
    cleaned = cleaned.replace("", fallback)
    if fold_case:
        cleaned = cleaned.apply(lambda v: v.title() if v != fallback else v)
    return cleaned


def add_calculated_fields(
    df: pd.DataFrame,
    *,
    document_id: str,
    document_name: str,
) -> pd.DataFrame:
    """Return a copy of ``df`` with GeoLens calculated columns appended.

    Calculated columns:
        thickness, document_id, document_name, is_valid, issues
    """
    out = df.copy()

    if COL_SOURCE_ROW not in out.columns:
        out[COL_SOURCE_ROW] = range(len(out))

    top = pd.to_numeric(out.get(COL_TOP_DEPTH), errors="coerce")
    base = pd.to_numeric(out.get(COL_BASE_DEPTH), errors="coerce")
    out[COL_THICKNESS] = (base - top).astype(float)

    out[COL_DOCUMENT_ID] = document_id
    out[COL_DOCUMENT_NAME] = document_name

    # Keep pipeline provenance in derived flags; source_df is never modified.
    status = out.get('match_status', pd.Series('', index=out.index)).astype('object')
    status = status.where(status.notna(), '').astype(str).str.strip().str.casefold()
    out['is_blank_interval'] = status.eq('blank')
    if 'lithology_fallback' in out.columns:
        out['is_runner_up_fallback'] = out['lithology_fallback'].map(
            lambda value: str(value).strip().casefold() in ('true', '1', 'yes') if pd.notna(value) else False)
    else:
        out['is_runner_up_fallback'] = False

    # Record missingness before placeholders, except intentionally blank beds.
    if COL_LITHOLOGY in out.columns:
        out[COL_LITHOLOGY_MISSING] = out[COL_LITHOLOGY].isna() | (
            out[COL_LITHOLOGY].astype(str).str.strip() == ""
        )
        intentional_blank = out['is_blank_interval'] & out[COL_LITHOLOGY_MISSING]
        out[COL_LITHOLOGY_MISSING] &= ~intentional_blank
        # Case-fold real labels. "Blank" is a DISPLAY category only: original
        # extraction CSV/source_df retain the empty lithology cell exactly.
        out[COL_LITHOLOGY] = _clean_label(
            out[COL_LITHOLOGY], UNKNOWN_LITHOLOGY, fold_case=True
        )
        out.loc[intentional_blank, COL_LITHOLOGY] = 'Blank'
    else:
        out[COL_LITHOLOGY_MISSING] = True

    if COL_CLUSTER_ID in out.columns:
        out[COL_CLUSTER_MISSING] = out[COL_CLUSTER_ID].isna() | (
            out[COL_CLUSTER_ID].astype(str).str.strip() == ""
        )
        out[COL_CLUSTER_ID] = _clean_label(out[COL_CLUSTER_ID], UNKNOWN_CLUSTER)
    else:
        out[COL_CLUSTER_MISSING] = True

    from geolens.services.cluster_ids import align_pipeline_cluster_ids
    out = align_pipeline_cluster_ids(out)

    issues: list[str] = []
    valid: list[bool] = []
    for _, row in out.iterrows():
        row_issues: list[str] = []
        if pd.isna(row.get(COL_TOP_DEPTH)):
            row_issues.append("missing top_depth")
        if pd.isna(row.get(COL_BASE_DEPTH)):
            row_issues.append("missing base_depth")
        thickness = row.get(COL_THICKNESS)
        if pd.notna(thickness):
            if thickness < 0:
                row_issues.append("negative thickness")
            elif thickness == 0:
                row_issues.append("zero thickness")
        if bool(row.get(COL_LITHOLOGY_MISSING, False)):
            row_issues.append("missing lithology")
        if bool(row.get(COL_CLUSTER_MISSING, False)):
            row_issues.append("missing cluster id")
        issues.append("; ".join(row_issues))
        valid.append(not row_issues)

    out[COL_ISSUES] = issues
    out[COL_IS_VALID] = valid
    return out


def consolidate_documents(documents: Sequence[object]) -> pd.DataFrame:
    """Concatenate processed frames from multiple documents.

    Each document must expose ``document_id``, ``document_name`` and
    ``processed_df``. Original columns are preserved; documents with different
    lithologies or cluster counts are unioned safely.
    """
    frames: list[pd.DataFrame] = []
    for doc in documents:
        frame = getattr(doc, "processed_df", None)
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        copy[COL_DOCUMENT_ID] = getattr(doc, "document_id", "unknown")
        copy[COL_DOCUMENT_NAME] = getattr(doc, "document_name", "unknown")
        frames.append(copy)

    if not frames:
        return pd.DataFrame(columns=list(PROCESSED_COLUMN_ORDER))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    ordered = [c for c in PROCESSED_COLUMN_ORDER if c in combined.columns]
    remaining = [c for c in combined.columns if c not in ordered]
    return combined[ordered + remaining]


def sort_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Sort intervals by document then top depth, keeping source order as tiebreak."""
    if df.empty:
        return df
    sort_cols = [c for c in (COL_DOCUMENT_ID, COL_TOP_DEPTH, COL_SOURCE_ROW) if c in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    *,
    lithologies: Iterable[str] | None = None,
    clusters: Iterable[str] | None = None,
    documents: Iterable[str] | None = None,
    depth_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Filter a processed dataframe by lithology, cluster, document and depth.

    An empty selection means "no restriction" for that dimension.
    """
    out = df
    if lithologies:
        selected = set(lithologies)
        out = out[out[COL_LITHOLOGY].isin(selected)]
    if clusters:
        selected = set(clusters)
        out = out[out[COL_CLUSTER_ID].isin(selected)]
    if documents:
        selected = set(documents)
        out = out[out[COL_DOCUMENT_ID].isin(selected)]
    if depth_range is not None:
        low, high = depth_range
        top = pd.to_numeric(out[COL_TOP_DEPTH], errors="coerce")
        base = pd.to_numeric(out[COL_BASE_DEPTH], errors="coerce")
        out = out[(base >= low) & (top <= high)]
    return out.reset_index(drop=True)


def depth_coverage(df: pd.DataFrame) -> dict[str, float]:
    """Compute depth coverage and gap information for a set of intervals.

    Returns:
        min_top, max_base, coverage (max_base - min_top),
        cumulative_thickness (sum of interval thickness),
        gap_total (coverage minus cumulative thickness, floored at 0),
        gap_count (number of gaps between consecutive intervals).
    """
    empty = {
        "min_top": float("nan"),
        "max_base": float("nan"),
        "coverage": 0.0,
        "cumulative_thickness": 0.0,
        "gap_total": 0.0,
        "gap_count": 0,
    }
    if df.empty:
        return empty

    top = pd.to_numeric(df[COL_TOP_DEPTH], errors="coerce").dropna()
    base = pd.to_numeric(df[COL_BASE_DEPTH], errors="coerce").dropna()
    if top.empty or base.empty:
        return empty

    min_top = float(top.min())
    max_base = float(base.max())
    coverage = max(0.0, max_base - min_top)

    thickness = pd.to_numeric(df[COL_THICKNESS], errors="coerce").dropna()
    cumulative = float(thickness[thickness > 0].sum())

    ordered = df.dropna(subset=[COL_TOP_DEPTH, COL_BASE_DEPTH]).copy()
    ordered = ordered.sort_values(COL_TOP_DEPTH, kind="mergesort")
    gap_total = 0.0
    gap_count = 0
    previous_base: float | None = None
    for _, row in ordered.iterrows():
        row_top = float(row[COL_TOP_DEPTH])
        row_base = float(row[COL_BASE_DEPTH])
        if previous_base is not None and row_top > previous_base + 1e-9:
            gap_total += row_top - previous_base
            gap_count += 1
        previous_base = row_base if previous_base is None else max(previous_base, row_base)

    return {
        "min_top": min_top,
        "max_base": max_base,
        "coverage": coverage,
        "cumulative_thickness": cumulative,
        "gap_total": round(gap_total, 6),
        "gap_count": gap_count,
    }


def interval_thickness(df: pd.DataFrame) -> pd.Series:
    """Return interval thickness, recomputing from depths when absent."""
    if COL_THICKNESS in df.columns:
        return pd.to_numeric(df[COL_THICKNESS], errors="coerce")
    top = pd.to_numeric(df[COL_TOP_DEPTH], errors="coerce")
    base = pd.to_numeric(df[COL_BASE_DEPTH], errors="coerce")
    return base - top


def lithology_sequence(df: pd.DataFrame) -> list[str]:
    """Return the ordered lithology sequence for one document."""
    if df.empty or COL_LITHOLOGY not in df.columns:
        return []
    ordered = df.sort_values(COL_TOP_DEPTH, kind="mergesort")
    return [str(v) for v in ordered[COL_LITHOLOGY].tolist()]


def compress_sequence(sequence: Sequence[str]) -> list[str]:
    """Collapse consecutive duplicate lithologies into a single entry."""
    compressed: list[str] = []
    for label in sequence:
        if not compressed or compressed[-1] != label:
            compressed.append(label)
    return compressed


def depth_units_consistent(df: pd.DataFrame) -> bool:
    """Heuristic check that depth values look like a single consistent scale.

    GeoLens cannot verify units. This only flags obviously incomparable data
    (e.g. one document in pixels and another in metres) by comparing ranges.
    """
    if df.empty or COL_DOCUMENT_ID not in df.columns:
        return True
    ranges = []
    for _, group in df.groupby(COL_DOCUMENT_ID):
        top = pd.to_numeric(group[COL_TOP_DEPTH], errors="coerce").dropna()
        base = pd.to_numeric(group[COL_BASE_DEPTH], errors="coerce").dropna()
        if top.empty or base.empty:
            continue
        span = float(base.max()) - float(top.min())
        if span > 0:
            ranges.append(span)
    if len(ranges) < 2:
        return True
    largest, smallest = max(ranges), min(ranges)
    return smallest > 0 and (largest / smallest) < 1000


def numeric_summary(series: pd.Series) -> dict[str, float]:
    """Return count/min/max/mean for a numeric series, ignoring NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def to_numeric_array(series: pd.Series) -> np.ndarray:
    """Convert a series to a float array with NaN for invalid entries."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def build_document(
    *,
    csv_bytes: bytes,
    csv_name: str,
    document_id: str | None = None,
):
    """Create a :class:`DocumentRecord` from raw upload bytes.

    The source dataframe is kept untouched; a separate processed dataframe with
    calculated fields is produced. Validation findings are attached.
    """
    from geolens.models.data_models import DocumentRecord
    from geolens.services.csv_loader import load_csv_with_check
    from geolens.services.validation import check_document
    from geolens.utils.helpers import slugify, stable_id

    doc_id = document_id or stable_id(csv_name, str(len(csv_bytes)))
    display_name = slugify(csv_name, fallback=doc_id)

    source_df, schema_check, error = load_csv_with_check(csv_bytes, filename=csv_name)
    if source_df is None:
        empty = pd.DataFrame(columns=list(PROCESSED_COLUMN_ORDER))
        return DocumentRecord(
            document_id=doc_id,
            document_name=display_name,
            source_df=empty,
            processed_df=empty,
            schema_check=schema_check,
            findings=[],
            load_error=error,
        )

    processed = add_calculated_fields(
        source_df, document_id=doc_id, document_name=display_name
    )
    findings = check_document(processed, document_id=doc_id) if schema_check.is_valid else []

    return DocumentRecord(
        document_id=doc_id,
        document_name=display_name,
        source_df=source_df,
        processed_df=processed,
        schema_check=schema_check,
        findings=findings,
        load_error=None,
    )
