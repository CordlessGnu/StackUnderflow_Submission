"""Batch analysis page: multi-document consolidation and comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from geolens.components.charts import (
    comparison_heatmap,
    grouped_bar_chart,
    lithology_bar_chart,
    presence_matrix_chart,
    render_chart,
    sequence_strip,
)
from geolens.components.metrics import collection_cards, render_metric_row
from geolens.components.sidebar import render_empty_state, render_page_header
from geolens.components.uploader import (
    collection_dataframe,
    get_collection,
    render_batch_upload,
    render_collection_manager,
)
from geolens.models.schemas import COL_DOCUMENT_NAME, COL_LITHOLOGY, COL_THICKNESS
from geolens.services.comparison import (
    comparison_matrix,
    document_sequences,
    documents_sharing_lithology,
    pairwise_sequence_comparison,
)
from geolens.services.data_processor import depth_units_consistent
from geolens.services.statistics import (
    collection_summary,
    filter_by_documents,
    lithology_distribution,
    lithology_statistics,
    organisation_statistics,
    trim_matrix,
)
from geolens.services.validation import check_collection, findings_to_frame, quality_summary
from geolens.utils.colors import build_color_map
from geolens.utils.formatting import format_number


def render() -> None:
    """Render the Batch Analysis page."""
    render_page_header(
        "Collection",
        "Batch Analysis",
        "Consolidate multiple extraction outputs and compare lithology composition across the collection.",
    )

    collection = get_collection()
    render_batch_upload()

    if len(collection) == 0:
        render_empty_state(
            "No documents in the collection",
            "Upload several extraction CSVs. GeoLens consolidates them into one dataset "
            "and builds a cross-document lithology comparison.",
        )
        return

    render_collection_manager(collection)

    # Schema notices and the detailed overview are collapsed so the analysis
    # below is reachable without scrolling past a long list.
    renamed_docs = [doc for doc in collection if doc.renamed_columns]
    if renamed_docs:
        with st.expander(
            f"Column names matched to the GeoLens schema ({len(renamed_docs)} document(s))"
        ):
            st.caption("Only column names were mapped — values were not modified.")
            for doc in renamed_docs:
                mapping = ", ".join(f"`{orig}` → `{canon}`" for orig, canon in doc.renamed_columns)
                st.markdown(f"- **{doc.document_name}**: {mapping}")

    overview = collection_dataframe(collection)
    if not overview.empty:
        with st.expander("Collection overview (detailed)"):
            st.dataframe(overview, width='stretch', hide_index=True)

    consolidated = collection.consolidated()
    if consolidated.empty:
        st.warning("None of the uploaded documents could be processed. Check the schema errors above.")
        return

    stats = organisation_statistics(consolidated)
    render_metric_row(collection_cards(stats))

    if not depth_units_consistent(consolidated):
        st.warning(
            "Depth ranges differ by several orders of magnitude between documents. "
            "Depth units may be inconsistent — avoid comparing absolute depths across documents."
        )

    st.markdown("---")

    # ---- Filters -----------------------------------------------------------
    st.markdown('<div class="gl-panel-title">Filter statistics</div>', unsafe_allow_html=True)
    filter_columns = st.columns(2)
    with filter_columns[0]:
        document_options = sorted(consolidated[COL_DOCUMENT_NAME].unique().tolist())
        selected_documents = st.multiselect(
            "Documents",
            options=document_options,
            default=[],
            placeholder="All documents",
            key="batch_doc_filter",
        )
    with filter_columns[1]:
        lithology_options = sorted(consolidated[COL_LITHOLOGY].unique().tolist())
        selected_lithologies = st.multiselect(
            "Lithologies",
            options=lithology_options,
            default=[],
            placeholder="All lithologies",
            key="batch_lith_filter",
        )

    filtered = filter_by_documents(consolidated, selected_documents)
    if selected_lithologies:
        filtered = filtered[filtered[COL_LITHOLOGY].isin(selected_lithologies)]

    if filtered.empty:
        st.info("No records match the current filters.")
        return

    color_map = build_color_map(consolidated[COL_LITHOLOGY].astype(str).unique())

    tabs = st.tabs(
        [
            "Lithology comparison",
            "Presence & distribution",
            "Document summaries",
            "Sequence comparison",
            "Data quality",
        ]
    )

    with tabs[0]:
        _render_comparison_matrix(filtered, color_map)

    with tabs[1]:
        _render_presence(filtered, color_map)

    with tabs[2]:
        _render_summaries(filtered)

    with tabs[3]:
        _render_sequences(filtered, color_map)

    with tabs[4]:
        _render_quality(consolidated)


def _table_height(rows: int, *, row_px: int = 35, minimum: int = 120, maximum: int = 420) -> int:
    """Height that fits the rows present, capped so long tables scroll."""
    return max(minimum, min(maximum, 38 + row_px * max(rows, 1)))


def _render_trim_note(dropped_rows: int, dropped_columns: int) -> None:
    """Explain any rows/columns hidden to keep the matrix readable."""
    parts: list[str] = []
    if dropped_rows:
        parts.append(f"{dropped_rows} lithology row(s) with no data hidden")
    if dropped_columns:
        parts.append(f"{dropped_columns} document column(s) hidden")
    if parts:
        st.caption("Showing the most significant entries only — " + "; ".join(parts) + ".")


def _render_comparison_matrix(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    st.markdown("#### Cross-document lithology comparison")
    st.caption(
        "Rows are lithology labels discovered from the uploaded files; columns are documents. "
        "A zero means the lithology was not detected in that document — not an extraction error."
    )

    metric = st.radio(
        "Matrix value",
        options=["count", "thickness", "percent"],
        format_func=lambda v: {
            "count": "Number of occurrences",
            "thickness": "Total thickness",
            "percent": "Percentage of document thickness",
        }[v],
        horizontal=True,
        key="batch_matrix_metric",
    )

    matrix = comparison_matrix(df, metric=metric)
    if matrix.empty:
        st.info("No comparison data available.")
        return

    # Only keep rows and columns that actually carry data, and cap the width so
    # a collection with many lithologies stays readable.
    matrix, dropped_rows, dropped_columns = trim_matrix(matrix)
    if matrix.empty:
        st.info("No comparison data available.")
        return
    _render_trim_note(dropped_rows, dropped_columns)

    suffix = "%" if metric == "percent" else ""
    render_chart(
        comparison_heatmap(matrix, title="Lithology × document", value_suffix=suffix),
        key="batch_matrix_heatmap",
    )

    display = matrix.copy()
    display.columns = [str(c) for c in display.columns]
    if metric == "percent":
        display = display.round(1)
    else:
        display = display.round(2)
    st.dataframe(display, width='stretch', height=_table_height(len(display)))

    st.markdown("#### Grouped comparison")
    render_chart(
        grouped_bar_chart(matrix, title="Lithology counts by document", y_title="Value"),
        key="batch_matrix_grouped",
    )


def _render_presence(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    st.markdown("#### Lithology presence")
    matrix = comparison_matrix(df, metric="count")
    matrix, dropped_rows, dropped_columns = trim_matrix(matrix)
    if matrix.empty:
        st.info("No presence data available.")
        return
    _render_trim_note(dropped_rows, dropped_columns)
    render_chart(presence_matrix_chart(matrix), key="batch_presence")

    st.markdown("#### Lithology distribution across the collection")
    distribution = lithology_distribution(df)
    left, right = st.columns(2)
    with left:
        render_chart(
            lithology_bar_chart(
                distribution,
                value_column="interval_count",
                color_map=color_map,
                title="Intervals per lithology",
            ),
            key="batch_dist_count",
        )
    with right:
        render_chart(
            lithology_bar_chart(
                distribution,
                value_column="total_thickness",
                color_map=color_map,
                title="Total thickness per lithology",
            ),
            key="batch_dist_thickness",
        )

    st.markdown("#### Where does a lithology occur?")
    lithologies = sorted(df[COL_LITHOLOGY].unique().tolist())
    selected = st.selectbox("Select a lithology", options=lithologies, key="batch_lith_lookup")
    if selected:
        documents = documents_sharing_lithology(df, selected)
        st.markdown(
            f"**{selected}** appears in {len(documents)} document(s): "
            + ", ".join(f"`{d}`" for d in documents)
        )
        subset = df[df[COL_LITHOLOGY] == selected]
        per_doc = (
            subset.groupby(COL_DOCUMENT_NAME)
            .agg(intervals=(COL_LITHOLOGY, "size"), thickness=(COL_THICKNESS, "sum"))
            .reset_index()
        )
        st.dataframe(per_doc, width='stretch', hide_index=True, height=320)


def _render_summaries(df: pd.DataFrame) -> None:
    st.markdown("#### Document summaries")
    summary = collection_summary(df)
    st.dataframe(summary, width='stretch', hide_index=True, height=320)

    st.markdown("#### Lithology statistics (filtered collection)")
    stats = lithology_statistics(df)
    if stats.empty:
        st.info("No lithology statistics available.")
        return
    display = stats.copy()
    display["total_thickness"] = display["total_thickness"].map(format_number)
    display["avg_thickness"] = display["avg_thickness"].map(format_number)
    display["percent_of_depth"] = display["percent_of_depth"].map(lambda v: f"{v:.1f}%")
    st.dataframe(display, width='stretch', hide_index=True, height=420)


def _render_sequences(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    st.markdown("#### Geological sequence comparison")
    st.caption(
        "Each document is reduced to its ordered lithology sequence (consecutive duplicates collapsed). "
        "Similarity is an exploratory indicator only — it does not establish geological equivalence."
    )

    sequences = document_sequences(df)
    if not sequences:
        st.info("No sequences available.")
        return

    render_chart(sequence_strip(sequences, color_map=color_map), key="batch_sequences")

    with st.expander("Inspect original sequences"):
        for name, sequence in sequences.items():
            st.markdown(f"**{name}**")
            st.markdown(" → ".join(sequence) if sequence else "—")

    comparison = pairwise_sequence_comparison(df)
    if comparison.empty:
        st.info("Add at least two documents to compare sequences.")
        return

    st.dataframe(comparison, width='stretch', hide_index=True, height=320)


def _render_quality(df: pd.DataFrame) -> None:
    st.markdown("#### Data consistency findings")
    findings = check_collection(df)
    summary = quality_summary(findings)

    columns = st.columns(3)
    columns[0].metric("Errors", summary.get("error", 0))
    columns[1].metric("Warnings", summary.get("warning", 0))
    columns[2].metric("Observations", summary.get("info", 0))

    st.caption(
        "Structural checks only. A depth gap or a cluster mapping to several lithologies "
        "may be geologically legitimate and requires human inspection."
    )

    frame = findings_to_frame(findings)
    if frame.empty:
        st.success("No structural inconsistencies detected across the collection.")
    else:
        st.dataframe(frame, width='stretch', hide_index=True, height=320)
