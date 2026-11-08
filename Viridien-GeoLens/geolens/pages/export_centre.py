"""Export Centre page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from geolens.components.sidebar import render_empty_state, render_page_header
from geolens.components.uploader import get_collection, get_single_document
from geolens.models.schemas import COL_DOCUMENT_NAME, COL_LITHOLOGY
from geolens.services.comparison import comparison_matrix, pairwise_sequence_comparison
from geolens.services.export_service import (
    build_batch_workbook,
    build_single_workbook,
    build_summary_report,
    dataframe_to_csv_bytes,
    export_filename,
    processed_intervals,
)
from geolens.services.statistics import (
    cluster_statistics,
    collection_summary,
    filter_by_documents,
    lithology_statistics,
    trim_matrix,
)
from geolens.services.validation import check_collection, findings_to_frame


def render() -> None:
    """Render the Export Centre page."""
    render_page_header(
        "Outputs",
        "Export Centre",
        "Download processed data, statistics, comparison tables and summary reports for the current view.",
    )

    st.caption(
        "Exports are generated from the data currently loaded in this session. "
        "Source CSV columns are always preserved alongside calculated fields."
    )

    tabs = st.tabs(["Single document", "Batch collection", "Summary report"])

    with tabs[0]:
        _render_single_exports()

    with tabs[1]:
        _render_batch_exports()

    with tabs[2]:
        _render_summary_report()


def _render_single_exports() -> None:
    record = get_single_document()
    if record is None or record.processed_df.empty:
        render_empty_state(
            "No single document loaded",
            "Load a document in Single Analysis to enable single-document exports.",
        )
        return

    df = record.processed_df
    st.markdown(f"**{record.document_name}** · {len(df)} intervals")

    columns = st.columns(2)

    with columns[0]:
        st.markdown("##### Data")
        st.download_button(
            "Original CSV",
            data=dataframe_to_csv_bytes(record.original_source_df()),
            file_name=export_filename(f"{record.document_name}_source", "csv"),
            mime="text/csv",
            key="export_single_source",
            width='stretch',
        )
        st.download_button(
            "Processed CSV (with thickness)",
            data=dataframe_to_csv_bytes(processed_intervals(df)),
            file_name=export_filename(f"{record.document_name}_processed", "csv"),
            mime="text/csv",
            key="export_single_processed",
            width='stretch',
        )
        st.download_button(
            "Lithology statistics CSV",
            data=dataframe_to_csv_bytes(lithology_statistics(df)),
            file_name=export_filename(f"{record.document_name}_lithology_stats", "csv"),
            mime="text/csv",
            key="export_single_lith",
            width='stretch',
        )
        st.download_button(
            "Cluster statistics CSV",
            data=dataframe_to_csv_bytes(cluster_statistics(df)),
            file_name=export_filename(f"{record.document_name}_cluster_stats", "csv"),
            mime="text/csv",
            key="export_single_cluster",
            width='stretch',
        )

    with columns[1]:
        st.markdown("##### Workbook")
        st.download_button(
            "Excel workbook",
            data=build_single_workbook(
                df,
                document_name=record.document_name,
                findings=record.findings,
            ),
            file_name=export_filename(f"{record.document_name}_workbook", "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_single_xlsx",
            width='stretch',
        )

    st.markdown("##### Data quality findings")
    frame = findings_to_frame(record.findings)
    if frame.empty:
        st.success("No structural inconsistencies detected.")
    else:
        st.dataframe(frame, width='stretch', hide_index=True)


def _render_batch_exports() -> None:
    collection = get_collection()
    consolidated = collection.consolidated() if len(collection) else pd.DataFrame()

    if consolidated.empty:
        render_empty_state(
            "No collection loaded",
            "Add documents in Batch Analysis to enable consolidated exports.",
        )
        return

    st.markdown(f"**{consolidated[COL_DOCUMENT_NAME].nunique()} documents** · {len(consolidated)} intervals")

    document_options = sorted(consolidated[COL_DOCUMENT_NAME].unique().tolist())
    selected_documents = st.multiselect(
        "Restrict export to documents",
        options=document_options,
        default=[],
        placeholder="All documents",
        key="export_doc_filter",
    )
    lithology_options = sorted(consolidated[COL_LITHOLOGY].unique().tolist())
    selected_lithologies = st.multiselect(
        "Restrict export to lithologies",
        options=lithology_options,
        default=[],
        placeholder="All lithologies",
        key="export_lith_filter",
    )

    export_df = filter_by_documents(consolidated, selected_documents)
    if selected_lithologies:
        export_df = export_df[export_df[COL_LITHOLOGY].isin(selected_lithologies)]

    if export_df.empty:
        st.info("No records match the selected export filters.")
        return

    filters_applied = {
        "Documents": selected_documents or "All",
        "Lithologies": selected_lithologies or "All",
    }
    st.caption(
        "Filters applied: "
        + "; ".join(f"{k}: {v}" for k, v in filters_applied.items())
    )

    findings = check_collection(export_df)
    comparison_tables = {
        "Lithology Thickness": comparison_matrix(export_df, metric="thickness"),
        "Lithology Percent": comparison_matrix(export_df, metric="percent"),
        "Sequence Comparison": pairwise_sequence_comparison(export_df),
    }

    columns = st.columns(2)
    with columns[0]:
        st.download_button(
            "Consolidated CSV",
            data=dataframe_to_csv_bytes(processed_intervals(export_df)),
            file_name=export_filename("collection_consolidated", "csv"),
            mime="text/csv",
            key="export_batch_csv",
            width='stretch',
        )
        st.download_button(
            "Document summaries CSV",
            data=dataframe_to_csv_bytes(collection_summary(export_df)),
            file_name=export_filename("collection_summaries", "csv"),
            mime="text/csv",
            key="export_batch_summaries",
            width='stretch',
        )
        st.download_button(
            "Lithology comparison CSV",
            data=dataframe_to_csv_bytes(comparison_matrix(export_df, metric="count")),
            file_name=export_filename("lithology_comparison", "csv"),
            mime="text/csv",
            key="export_batch_comparison",
            width='stretch',
        )

    with columns[1]:
        st.download_button(
            "Batch Excel workbook",
            data=build_batch_workbook(
                export_df,
                findings=findings,
                filters_applied=filters_applied,
                comparison_tables=comparison_tables,
            ),
            file_name=export_filename("collection_workbook", "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_batch_xlsx",
            width='stretch',
        )
        st.download_button(
            "Data quality CSV",
            data=dataframe_to_csv_bytes(findings_to_frame(findings)),
            file_name=export_filename("data_quality", "csv"),
            mime="text/csv",
            key="export_batch_quality",
            width='stretch',
        )

    st.markdown("##### Preview: lithology comparison")
    preview, _, _ = trim_matrix(comparison_matrix(export_df, metric="count"))
    if preview.empty:
        st.info("No comparison data available.")
    else:
        st.dataframe(
            preview,
            width='stretch',
            height=max(120, min(420, 38 + 35 * max(len(preview), 1))),
        )


def _render_summary_report() -> None:
    collection = get_collection()
    consolidated = collection.consolidated() if len(collection) else pd.DataFrame()

    record = get_single_document()
    if consolidated.empty and (record is None or record.processed_df.empty):
        render_empty_state(
            "No data available",
            "Load a document or a collection to generate a summary report.",
        )
        return

    use_collection = not consolidated.empty
    df = consolidated if use_collection else record.processed_df
    findings = check_collection(df) if use_collection else record.findings
    comparison = comparison_matrix(df, metric="count") if use_collection else None

    report = build_summary_report(
        df,
        findings=findings,
        filters_applied={"Scope": "Full collection" if use_collection else record.document_name},
        comparison=comparison,
    )

    st.markdown("##### Report preview")
    st.code(report, language="text")

    st.download_button(
        "Download summary report (.txt)",
        data=report.encode("utf-8"),
        file_name=export_filename("summary_report", "txt"),
        mime="text/plain",
        key="export_summary_txt",
    )
