"""Export services: CSV, Excel workbooks and summary reports.

All exports are generated from the dataframe the caller supplies, so the user's
current filters are respected. Source columns are always preserved.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from geolens.models.schemas import (
    COL_DOCUMENT_NAME,
    COL_LITHOLOGY,
    COL_THICKNESS,
    PROCESSED_COLUMN_ORDER,
    QualityFinding,
)
from geolens.services.data_processor import interval_thickness
from geolens.services.statistics import (
    cluster_statistics,
    collection_summary,
    lithology_comparison_matrix,
    lithology_statistics,
    overview_metrics,
)
from geolens.services.validation import findings_to_frame


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a dataframe to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8-sig")


def processed_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Return intervals with calculated thickness, ordered for export."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(PROCESSED_COLUMN_ORDER))
    work = df.copy()
    if COL_THICKNESS not in work.columns:
        work[COL_THICKNESS] = interval_thickness(work)
    ordered = [c for c in PROCESSED_COLUMN_ORDER if c in work.columns]
    remaining = [c for c in work.columns if c not in ordered]
    return work[ordered + remaining]


def _write_sheet(writer: pd.ExcelWriter, name: str, df: pd.DataFrame) -> None:
    """Write a dataframe to a worksheet, guarding against empty frames."""
    safe_name = name[:31]
    if df is None or df.empty:
        pd.DataFrame({"info": ["No data available for this section."]}).to_excel(
            writer, sheet_name=safe_name, index=False
        )
        return
    df.to_excel(writer, sheet_name=safe_name, index=False)


def build_single_workbook(
    df: pd.DataFrame,
    *,
    document_name: str,
    findings: Sequence[QualityFinding] = (),
    filters_applied: Mapping[str, Any] | None = None,
) -> bytes:
    """Build an Excel workbook for a single document."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _write_sheet(writer, "Intervals", processed_intervals(df))
        _write_sheet(writer, "Lithology Stats", lithology_statistics(df))
        _write_sheet(writer, "Cluster Stats", cluster_statistics(df))
        _write_sheet(writer, "Data Quality", findings_to_frame(findings))
        _write_sheet(writer, "Export Info", _export_info_frame(document_name, filters_applied))
    buffer.seek(0)
    return buffer.getvalue()


def build_batch_workbook(
    df: pd.DataFrame,
    *,
    findings: Sequence[QualityFinding] = (),
    filters_applied: Mapping[str, Any] | None = None,
    comparison_tables: Mapping[str, pd.DataFrame] | None = None,
) -> bytes:
    """Build the multi-sheet batch Excel workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _write_sheet(writer, "Consolidated Intervals", processed_intervals(df))
        _write_sheet(writer, "Document Summaries", collection_summary(df))
        _write_sheet(writer, "Lithology Comparison", lithology_comparison_matrix(df, metric="count"))
        _write_sheet(writer, "Lithology Stats", lithology_statistics(df))
        _write_sheet(writer, "Cluster Stats", _cluster_stats_all(df))
        _write_sheet(writer, "Data Quality", findings_to_frame(findings))
        for name, table in (comparison_tables or {}).items():
            _write_sheet(writer, name, table)
        _write_sheet(writer, "Export Info", _export_info_frame("Batch export", filters_applied))
    buffer.seek(0)
    return buffer.getvalue()


def _cluster_stats_all(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster statistics per document (cluster ids are document-scoped)."""
    if df.empty:
        return pd.DataFrame()
    frames = []
    for doc_name, group in df.groupby(COL_DOCUMENT_NAME):
        stats = cluster_statistics(group)
        if stats.empty:
            continue
        stats.insert(0, COL_DOCUMENT_NAME, doc_name)
        frames.append(stats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _export_info_frame(
    scope: str, filters_applied: Mapping[str, Any] | None
) -> pd.DataFrame:
    rows = [
        {"field": "Export scope", "value": scope},
        {"field": "Generated (UTC)", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")},
        {
            "field": "Filters applied",
            "value": _describe_filters(filters_applied),
        },
        {
            "field": "Note",
            "value": "Exports reflect the current GeoLens view. They are not verified geological truth.",
        },
    ]
    return pd.DataFrame(rows)


def _describe_filters(filters_applied: Mapping[str, Any] | None) -> str:
    if not filters_applied:
        return "None (full dataset)"
    parts = []
    for key, value in filters_applied.items():
        if value in (None, [], (), ""):
            continue
        if isinstance(value, (list, tuple, set)):
            parts.append(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            parts.append(f"{key}: {value}")
    return "; ".join(parts) if parts else "None (full dataset)"


def build_summary_report(
    df: pd.DataFrame,
    *,
    findings: Sequence[QualityFinding] = (),
    filters_applied: Mapping[str, Any] | None = None,
    comparison: pd.DataFrame | None = None,
) -> str:
    """Generate a concise plain-text summary report."""
    metrics = overview_metrics(df)
    lines: list[str] = []
    lines.append("GeoLens — Geological Data Summary Report")
    lines.append("=" * 48)
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Filters applied: {_describe_filters(filters_applied)}")
    lines.append("")
    lines.append("Scope")
    lines.append("-" * 48)
    lines.append(f"Documents analysed: {df[COL_DOCUMENT_NAME].nunique() if not df.empty else 0}")
    lines.append(f"Intervals: {metrics['interval_count']}")
    lines.append(f"Unique lithology labels: {metrics['lithology_count']}")
    lines.append(f"Unique clusters (per document): {metrics['cluster_count']}")
    lines.append(f"Depth coverage: {metrics['depth_coverage']:.2f}")
    lines.append(f"Cumulative interval thickness: {metrics['cumulative_thickness']:.2f}")
    lines.append(f"Average interval thickness: {metrics['average_thickness']:.2f}")
    lines.append("")

    lines.append("Lithology distribution")
    lines.append("-" * 48)
    stats = lithology_statistics(df)
    if stats.empty:
        lines.append("No lithology data available.")
    else:
        for _, row in stats.iterrows():
            lines.append(
                f"- {row[COL_LITHOLOGY]}: {int(row['interval_count'])} interval(s), "
                f"total thickness {row['total_thickness']:.2f}, "
                f"{row['percent_of_depth']:.1f}% of cumulative thickness"
            )
    lines.append("")

    if comparison is not None and not comparison.empty:
        lines.append("Cross-document lithology comparison (interval counts)")
        lines.append("-" * 48)
        lines.append(comparison.to_string())
        lines.append("")

    lines.append("Data consistency observations")
    lines.append("-" * 48)
    if not findings:
        lines.append("No structural issues detected.")
    else:
        name_by_id = _document_name_lookup(df)
        for finding in findings:
            label = name_by_id.get(finding.document_id, finding.document_id)
            lines.append(f"- [{finding.severity.upper()}] {label}: {finding.message}")
    lines.append("")
    lines.append(
        "Disclaimer: this report describes the uploaded extraction outputs. "
        "It is not a verified geological interpretation."
    )
    return "\n".join(lines)


def _document_name_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map document ids to human-readable names for report readability."""
    from geolens.models.schemas import COL_DOCUMENT_ID

    if df.empty or COL_DOCUMENT_ID not in df.columns or COL_DOCUMENT_NAME not in df.columns:
        return {}
    pairs = df[[COL_DOCUMENT_ID, COL_DOCUMENT_NAME]].dropna().drop_duplicates()
    return {str(row[COL_DOCUMENT_ID]): str(row[COL_DOCUMENT_NAME]) for _, row in pairs.iterrows()}


def export_filename(prefix: str, extension: str) -> str:
    """Build a timestamped export filename."""
    return f"geolens_{prefix}_{_timestamp()}.{extension}"
