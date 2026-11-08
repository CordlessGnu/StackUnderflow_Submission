"""Comparison Studio page: side-by-side document comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from geolens.components.charts import (
    comparison_heatmap,
    lithology_bar_chart,
    render_chart,
    similarity_heatmap,
)
from geolens.components.lithology_profile import build_profile_figure
from geolens.components.sidebar import render_empty_state, render_page_header
from geolens.components.uploader import get_collection
from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_DOCUMENT_NAME,
    COL_LITHOLOGY,
    COL_THICKNESS,
    COL_TOP_DEPTH,
)
from geolens.services.comparison import (
    comparison_matrix,
    document_depth_ranges,
    document_sequences,
    shared_lithologies,
    similarity_matrix,
)
from geolens.services.data_processor import depth_units_consistent
from geolens.services.statistics import (
    collection_summary,
    filter_by_documents,
    lithology_statistics,
    trim_matrix,
)
from geolens.utils.colors import build_color_map


def render() -> None:
    """Render the Comparison Studio page."""
    render_page_header(
        "Workspace",
        "Comparison Studio",
        "Compare selected documents side by side: profiles, distributions, presence and sequence similarity.",
    )

    collection = get_collection()
    if len(collection) == 0:
        render_empty_state(
            "Nothing to compare yet",
            "Load documents in Batch Analysis first, then return here to compare them side by side.",
        )
        return

    consolidated = collection.consolidated()
    if consolidated.empty:
        st.warning("No processable documents in the collection.")
        return

    document_names = sorted(consolidated[COL_DOCUMENT_NAME].unique().tolist())

    st.markdown('<div class="gl-panel-title">Select documents</div>', unsafe_allow_html=True)
    default_selection = document_names[: min(3, len(document_names))]
    selected = st.multiselect(
        "Documents to compare",
        options=document_names,
        default=default_selection,
        key="comparison_documents",
    )

    if len(selected) < 2:
        st.info("Select at least two documents to compare.")
        return

    subset = filter_by_documents(consolidated, selected)
    color_map = build_color_map(consolidated[COL_LITHOLOGY].astype(str).unique())

    if not depth_units_consistent(subset):
        st.warning(
            "Depth ranges differ substantially between the selected documents. "
            "Shared depth axes are shown for visual convenience only and do not imply "
            "that matching depths represent the same formation."
        )

    tabs = st.tabs(
        [
            "Side-by-side profiles",
            "Distribution comparison",
            "Presence matrix",
            "Sequence similarity",
            "Highlight lithology",
        ]
    )

    with tabs[0]:
        _render_profiles(subset, selected, color_map)

    with tabs[1]:
        _render_distributions(subset, color_map)

    with tabs[2]:
        _render_presence(subset)

    with tabs[3]:
        _render_similarity(subset)

    with tabs[4]:
        _render_highlight(subset, color_map)


def _render_profiles(df: pd.DataFrame, selected: list[str], color_map: dict[str, str]) -> None:
    st.markdown("#### Side-by-side geological profiles")
    st.caption(
        "Profiles share a depth axis when depth references are comparable. "
        "GeoLens does not align wells or infer stratigraphic equivalence."
    )

    # Each profile keeps a fixed width so it stays readable no matter how many
    # documents are compared; the row scrolls horizontally instead of squeezing.
    profile_width = 300
    # ``wrap=False`` keeps every profile on one row and lets the row scroll
    # horizontally instead of wrapping or squeezing the columns.
    with st.container(horizontal=True, gap="medium", width="stretch", wrap=False):
        for name in selected:
            with st.container(width=profile_width):
                doc_df = df[df[COL_DOCUMENT_NAME] == name]
                st.markdown(f"**{name}**")
                st.caption(f"{len(doc_df)} intervals")
                fig = build_profile_figure(
                    doc_df,
                    color_by="lithology",
                    color_map=color_map,
                    height=560,
                    show_labels=False,
                )
                st.plotly_chart(
                    fig,
                    width='stretch',
                    key=f"compare_profile_{name}",
                    config={"displaylogo": False, "scrollZoom": True},
                )

    st.markdown("#### Depth coverage comparison")
    ranges = document_depth_ranges(df)
    if not ranges.empty:
        st.dataframe(ranges, width='stretch', hide_index=True)


def _render_distributions(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    st.markdown("#### Lithology distribution comparison")
    stats = lithology_statistics(df)
    if stats.empty:
        st.info("No statistics available.")
        return

    left, right = st.columns(2)
    with left:
        render_chart(
            lithology_bar_chart(
                stats,
                value_column="interval_count",
                color_map=color_map,
                title="Intervals per lithology (selected documents)",
            ),
            key="compare_dist_count",
        )
    with right:
        render_chart(
            lithology_bar_chart(
                stats,
                value_column="total_thickness",
                color_map=color_map,
                title="Total thickness per lithology",
            ),
            key="compare_dist_thickness",
        )

    st.markdown("#### Interval count and depth coverage")
    summary = collection_summary(df)
    st.dataframe(summary, width='stretch', hide_index=True)

    st.markdown("#### Shared lithologies")
    common = shared_lithologies(df, sorted(df[COL_DOCUMENT_NAME].unique().tolist()))
    if common:
        st.markdown("Present in every selected document: " + ", ".join(f"`{c}`" for c in common))
    else:
        st.markdown("No lithology is present in every selected document.")


def _render_presence(df: pd.DataFrame) -> None:
    st.markdown("#### Lithology presence matrix")
    matrix = comparison_matrix(df, metric="count")
    if matrix.empty:
        st.info("No presence data available.")
        return
    matrix, dropped_rows, dropped_columns = trim_matrix(matrix)
    if matrix.empty:
        st.info("No presence data available.")
        return
    if dropped_rows or dropped_columns:
        st.caption(
            "Showing the most significant entries only — "
            f"{dropped_rows} lithology row(s) and {dropped_columns} document column(s) hidden."
        )
    render_chart(comparison_heatmap(matrix, title="Occurrences per document"), key="compare_presence")
    st.dataframe(
        matrix,
        width='stretch',
        height=max(120, min(420, 38 + 35 * max(len(matrix), 1))),
    )


def _render_similarity(df: pd.DataFrame) -> None:
    st.markdown("#### Lithology sequence similarity")
    st.caption(
        "Similarity compares ordered lithology labels only. Cluster IDs are never compared "
        "across documents because they are scoped to their source document."
    )

    matrix = similarity_matrix(df)
    if matrix.empty:
        st.info("Not enough documents to compare.")
        return
    render_chart(similarity_heatmap(matrix), key="compare_similarity")

    sequences = document_sequences(df)
    with st.expander("Inspect sequences"):
        for name, sequence in sequences.items():
            st.markdown(f"**{name}**: " + (" → ".join(sequence) if sequence else "—"))


def _render_highlight(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    st.markdown("#### Highlight a lithology across documents")
    lithologies = sorted(df[COL_LITHOLOGY].unique().tolist())
    selected = st.selectbox("Lithology", options=lithologies, key="compare_highlight_lith")

    if not selected:
        return

    subset = df[df[COL_LITHOLOGY] == selected]
    if subset.empty:
        st.info("This lithology does not occur in the selected documents.")
        return

    per_doc = (
        subset.groupby(COL_DOCUMENT_NAME)
        .agg(
            intervals=(COL_LITHOLOGY, "size"),
            total_thickness=(COL_THICKNESS, "sum"),
            min_top=(COL_TOP_DEPTH, "min"),
            max_base=(COL_BASE_DEPTH, "max"),
        )
        .reset_index()
    )
    st.dataframe(per_doc, width='stretch', hide_index=True)

    st.markdown("#### Occurrence by document")
    chart_data = per_doc.rename(columns={"interval_count": "intervals"})
    fig = lithology_bar_chart(
        chart_data.assign(lithology=chart_data[COL_DOCUMENT_NAME]),
        value_column="intervals",
        color_map={name: color_map.get(selected, "#4C6E91") for name in chart_data[COL_DOCUMENT_NAME]},
        title=f"Intervals of {selected} per document",
    )
    render_chart(fig, key="compare_highlight_chart")
