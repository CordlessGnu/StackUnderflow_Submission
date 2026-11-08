"""Data consistency checking.

This service reports *structural* consistency only. It never claims that the AI
extraction is geologically correct, and it never invents confidence scores.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_CLUSTER_ID,
    COL_CLUSTER_MISSING,
    COL_DOCUMENT_ID,
    COL_LITHOLOGY,
    COL_LITHOLOGY_MISSING,
    COL_SOURCE_ROW,
    COL_TOP_DEPTH,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    QualityFinding,
)
from geolens.services.data_processor import interval_thickness


def _finding(
    document_id: str,
    check: str,
    severity: str,
    count: int,
    message: str,
    rows: Sequence[int] = (),
) -> QualityFinding:
    return QualityFinding(
        document_id=document_id,
        check=check,
        severity=severity,
        count=int(count),
        message=message,
        row_indices=tuple(int(r) for r in rows),
    )


def check_document(df: pd.DataFrame, *, document_id: str) -> list[QualityFinding]:
    """Run all consistency checks for a single document's processed dataframe."""
    findings: list[QualityFinding] = []
    if df.empty:
        return [
            _finding(
                document_id,
                "empty_document",
                SEVERITY_WARNING,
                0,
                "The document contains no interval records.",
            )
        ]

    row_ids = df[COL_SOURCE_ROW] if COL_SOURCE_ROW in df.columns else pd.Series(range(len(df)))

    # Missing required values -------------------------------------------------
    for column, label in (
        (COL_TOP_DEPTH, "top depth"),
        (COL_BASE_DEPTH, "base depth"),
        (COL_LITHOLOGY, "lithology label"),
        (COL_CLUSTER_ID, "cluster id"),
    ):
        if column not in df.columns:
            findings.append(
                _finding(document_id, f"missing_column_{column}", SEVERITY_ERROR, len(df), f"Column '{column}' is absent.")
            )
            continue
        # Prefer the pre-fill missingness flags so placeholders are not mistaken
        # for real labels.
        flag_column = {
            COL_LITHOLOGY: COL_LITHOLOGY_MISSING,
            COL_CLUSTER_ID: COL_CLUSTER_MISSING,
        }.get(column)
        if flag_column and flag_column in df.columns:
            missing = df[flag_column].fillna(False).astype(bool)
        else:
            missing = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if missing.any():
            findings.append(
                _finding(
                    document_id,
                    f"missing_{column}",
                    SEVERITY_WARNING,
                    int(missing.sum()),
                    f"{int(missing.sum())} record(s) have a missing {label}.",
                    row_ids[missing].tolist(),
                )
            )

    for flag, check, message in (
        ('is_blank_interval', 'intentional_blank_intervals', 'intentionally blank interval(s); source lithology cells remain empty'),
        ('is_runner_up_fallback', 'runner_up_fallbacks', 'runner-up fallback label(s); these are not verified matches'),
    ):
        if flag in df.columns:
            present = df[flag].fillna(False).astype(bool)
            if present.any():
                findings.append(_finding(document_id, check, SEVERITY_INFO, int(present.sum()),
                                         f'{int(present.sum())} {message}.', row_ids[present].tolist()))

    # Thickness anomalies -----------------------------------------------------
    thickness = interval_thickness(df)
    negative = thickness < 0
    if negative.any():
        findings.append(
            _finding(
                document_id,
                "negative_thickness",
                SEVERITY_ERROR,
                int(negative.sum()),
                f"{int(negative.sum())} interval(s) have a base depth above the top depth (negative thickness).",
                row_ids[negative].tolist(),
            )
        )
    zero = thickness == 0
    if zero.any():
        findings.append(
            _finding(
                document_id,
                "zero_thickness",
                SEVERITY_INFO,
                int(zero.sum()),
                f"{int(zero.sum())} interval(s) have zero thickness.",
                row_ids[zero].tolist(),
            )
        )

    # Duplicate records -------------------------------------------------------
    source_cols = [c for c in (COL_TOP_DEPTH, COL_BASE_DEPTH, COL_LITHOLOGY, COL_CLUSTER_ID) if c in df.columns]
    if source_cols:
        duplicates = df.duplicated(subset=source_cols, keep="first")
        if duplicates.any():
            findings.append(
                _finding(
                    document_id,
                    "duplicate_records",
                    SEVERITY_WARNING,
                    int(duplicates.sum()),
                    f"{int(duplicates.sum())} duplicate interval record(s) detected.",
                    row_ids[duplicates].tolist(),
                )
            )

    # Ordering, overlap and gaps ---------------------------------------------
    ordered = df.dropna(subset=[COL_TOP_DEPTH, COL_BASE_DEPTH]).copy()
    if not ordered.empty:
        ordered = ordered.sort_values(COL_TOP_DEPTH, kind="mergesort")
        tops = pd.to_numeric(ordered[COL_TOP_DEPTH], errors="coerce")
        bases = pd.to_numeric(ordered[COL_BASE_DEPTH], errors="coerce")

        out_of_order = (bases.diff() < 0) & bases.notna()
        # A base decreasing relative to the previous base suggests unsorted input.
        if out_of_order.any():
            findings.append(
                _finding(
                    document_id,
                    "depth_ordering",
                    SEVERITY_INFO,
                    int(out_of_order.sum()),
                    "Some intervals are not in strictly increasing depth order in the source file; GeoLens sorts by top depth for display.",
                )
            )

        overlaps = tops.iloc[1:].to_numpy() < bases.shift(1).iloc[1:].to_numpy() - 1e-9
        if overlaps.any():
            findings.append(
                _finding(
                    document_id,
                    "overlapping_intervals",
                    SEVERITY_WARNING,
                    int(overlaps.sum()),
                    f"{int(overlaps.sum())} interval(s) overlap the preceding interval.",
                    ordered[COL_SOURCE_ROW].iloc[1:][overlaps].tolist() if COL_SOURCE_ROW in ordered else (),
                )
            )

        gaps = tops.iloc[1:].to_numpy() > bases.shift(1).iloc[1:].to_numpy() + 1e-9
        if gaps.any():
            findings.append(
                _finding(
                    document_id,
                    "depth_gaps",
                    SEVERITY_INFO,
                    int(gaps.sum()),
                    f"{int(gaps.sum())} gap(s) between consecutive intervals. Gaps may be geologically legitimate.",
                    ordered[COL_SOURCE_ROW].iloc[1:][gaps].tolist() if COL_SOURCE_ROW in ordered else (),
                )
            )

    # Cluster / lithology consistency ----------------------------------------
    if COL_CLUSTER_ID in df.columns and COL_LITHOLOGY in df.columns:
        per_cluster = df.groupby(COL_CLUSTER_ID)[COL_LITHOLOGY].nunique(dropna=True)
        inconsistent = per_cluster[per_cluster > 1]
        if not inconsistent.empty:
            findings.append(
                _finding(
                    document_id,
                    "cluster_lithology_mismatch",
                    SEVERITY_INFO,
                    int(len(inconsistent)),
                    f"{len(inconsistent)} cluster(s) map to more than one lithology label. This may be legitimate but warrants inspection.",
                )
            )

    return findings


def check_collection(df: pd.DataFrame) -> list[QualityFinding]:
    """Run consistency checks across every document in a consolidated frame."""
    findings: list[QualityFinding] = []
    if df.empty:
        return findings
    for doc_id, group in df.groupby(COL_DOCUMENT_ID, dropna=False):
        findings.extend(check_document(group, document_id=str(doc_id)))
    return findings


def findings_to_frame(findings: Sequence[QualityFinding]) -> pd.DataFrame:
    """Convert findings into a display/export dataframe."""
    columns = ["document_id", "check", "severity", "count", "message", "row_indices"]
    if not findings:
        return pd.DataFrame(columns=columns)
    rows = []
    for finding in findings:
        row = finding.as_dict()
        row["row_indices"] = ", ".join(str(i) for i in finding.row_indices)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def quality_summary(findings: Sequence[QualityFinding]) -> dict[str, int]:
    """Count findings by severity."""
    summary = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 0, SEVERITY_INFO: 0}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1
    return summary


def valid_record_count(df: pd.DataFrame) -> int:
    """Count structurally valid records (no blocking issues)."""
    if df.empty:
        return 0
    thickness = interval_thickness(df)
    top = pd.to_numeric(df.get(COL_TOP_DEPTH), errors="coerce")
    base = pd.to_numeric(df.get(COL_BASE_DEPTH), errors="coerce")
    if COL_LITHOLOGY_MISSING in df.columns:
        lithology_missing = df[COL_LITHOLOGY_MISSING].fillna(False).astype(bool)
    else:
        lithology_missing = df.get(COL_LITHOLOGY, pd.Series([""] * len(df))).astype(str).str.strip() == ""
    mask = top.notna() & base.notna() & (thickness >= 0) & (~lithology_missing)
    return int(mask.sum())
