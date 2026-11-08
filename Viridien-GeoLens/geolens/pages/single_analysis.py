"""Single-document analysis page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from geolens.components.charts import (
    lithology_bar_chart,
    lithology_donut,
    render_chart,
)
from geolens.components.filters import (
    apply_filter_state,
    describe_filters,
    render_filter_controls,
)
from geolens.components.interval_details import (
    render_cluster_explorer,
    render_interval_details,
    render_interval_table,
)
from geolens.components.lithology_profile import get_selected_interval, render_profile
from geolens.components.metrics import overview_cards, render_metric_row
from geolens.components.sidebar import render_empty_state, render_page_header
from geolens.components.uploader import render_single_upload
from geolens.models.schemas import COL_CLUSTER_ID, COL_LITHOLOGY
from geolens.services.statistics import (
    cluster_statistics,
    lithology_statistics,
    overview_metrics,
)
from geolens.services.validation import findings_to_frame, quality_summary
from geolens.utils.colors import build_color_map
from geolens.utils.formatting import format_number


def render() -> None:
    """Render the Single Analysis page."""
    render_page_header(
        "Single document",
        "Single Analysis",
        "Load an extraction CSV or process a TIFF + JSON: interactive profile, lithology statistics and cluster structure.",
    )

    record = render_single_upload()

    if record is None:
        if st.session_state.get('single_input_mode') == 'TIFF + JSON' and st.session_state.get('extraction_job_id'):
            st.caption('Analytics appear here after a successful extraction. You can navigate away while it runs.')
        else:
            render_empty_state(
                "No document loaded",
                "Upload an extraction CSV or select a TIFF + bounding-box JSON to build an interactive profile. "
                "Lithology types and cluster IDs are discovered automatically.",
            )
        return

    if record.load_error:
        st.error(record.load_error)
        return

    if not record.schema_check.is_valid:
        st.error(record.schema_check.message)
        st.caption(
            "GeoLens requires the six extraction columns: top_y, bottom_y, cluster_id, "
            "lithology, top_depth, base_depth. Common alternatives such as "
            "bottom_depth are recognised automatically."
        )
        return

    if record.renamed_columns:
        mapping = ", ".join(f"`{orig}` → `{canon}`" for orig, canon in record.renamed_columns)
        st.info(f"Column names matched to the GeoLens schema: {mapping}. Values were not modified.")

    df = record.processed_df
    if df.empty:
        st.warning("The uploaded CSV contains no interval records.")
        return

    if 'is_blank_interval' in df.columns and df['is_blank_interval'].any():
        st.caption('“Blank” is an unassigned display category in these charts, not a rock type. '
                   'The source CSV lithology cells remain empty.')

    # Colour maps are built from the data, so unseen labels get colours too.
    lithology_colors = build_color_map(df[COL_LITHOLOGY].astype(str).unique())
    cluster_colors = build_color_map(df[COL_CLUSTER_ID].astype(str).unique(), variant=3)

    metrics = overview_metrics(df)
    render_metric_row(overview_cards(metrics))

    st.markdown("---")

    # ---- Filters + profile -------------------------------------------------
    filter_col, profile_col = st.columns([1, 3.2], gap="large")

    with filter_col:
        state = render_filter_controls(df, key_prefix="single")
        if st.button("Reset to full profile", key="single_reset"):
            for key in ("single_lith_filter", "single_cluster_filter", "single_depth_range"):
                st.session_state.pop(key, None)
            st.session_state.pop("selected_interval", None)
            st.rerun()

        st.markdown("---")
        # Read the chart's selection state before rendering the panel so the
        # detail reflects the click that triggered this run.
        selected_row = get_selected_interval("single_profile")
        render_interval_details(
            df,
            selected_row=selected_row,
            color_map=lithology_colors,
        )

    filtered = apply_filter_state(df, state)
    color_map = cluster_colors if state.color_by == "cluster" else lithology_colors
    highlight_labels = state.lithologies if state.color_by == "lithology" else state.clusters

    with profile_col:
        st.markdown('<div class="gl-panel-title">Geological profile</div>', unsafe_allow_html=True)
        st.caption(
            f"{len(filtered)} of {len(df)} intervals shown · layer heights are proportional to "
            "base_depth − top_depth · scroll to zoom, drag to pan"
        )
        render_profile(
            filtered,
            color_by=state.color_by,
            color_map=color_map,
            highlight_labels=highlight_labels,
            highlight_mode=state.highlight_mode,
            selected_row=selected_row,
            height=640,
            key="single_profile",
        )

    # ---- Tabs --------------------------------------------------------------
    tabs = st.tabs(
        ["Lithology statistics", "Cluster explorer", "Data quality", "Interval table"]
    )

    with tabs[0]:
        _render_lithology_statistics(df, lithology_colors)

    with tabs[1]:
        _render_cluster_explorer(df, cluster_colors)

    with tabs[2]:
        _render_quality(record)

    with tabs[3]:
        render_interval_table(filtered)
        st.caption("Filters applied: " + ", ".join(f"{k}={v}" for k, v in describe_filters(state).items()))


def _render_lithology_statistics(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    stats = lithology_statistics(df)
    if stats.empty:
        st.info("No lithology statistics available.")
        return

    left, right = st.columns(2)
    with left:
        render_chart(
            lithology_bar_chart(
                stats,
                value_column="interval_count",
                color_map=color_map,
                title="Intervals per lithology",
            ),
            key="single_lith_count",
        )
    with right:
        render_chart(
            lithology_donut(
                stats,
                value_column="total_thickness",
                color_map=color_map,
                title="Thickness distribution",
            ),
            key="single_lith_donut",
        )

    render_chart(
        lithology_bar_chart(
            stats,
            value_column="total_thickness",
            color_map=color_map,
            title="Total thickness per lithology",
            height=320,
        ),
        key="single_lith_thickness",
    )

    display = stats.copy()
    display["total_thickness"] = display["total_thickness"].map(lambda v: format_number(v))
    display["avg_thickness"] = display["avg_thickness"].map(lambda v: format_number(v))
    display["percent_of_depth"] = display["percent_of_depth"].map(lambda v: f"{v:.1f}%")
    st.dataframe(display, width='stretch', hide_index=True)
    st.caption(
        "Percentages are of cumulative interval thickness within this document, not of total depth coverage."
    )


def _render_cluster_explorer(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    stats = cluster_statistics(df)
    if stats.empty:
        st.info("No clusters detected.")
        return

    selected = render_cluster_explorer(df, cluster_stats=stats, color_map=color_map)

    if selected:
        subset = df[df[COL_CLUSTER_ID].astype(str) == str(selected)]
        st.markdown(f"**Intervals in cluster {selected}**")
        render_interval_table(subset, max_rows=100)

    st.markdown("---")
    st.caption(
        "Cluster IDs are scoped to this document. The same ID in another document does not "
        "imply the same geological pattern."
    )
    st.dataframe(stats, width='stretch', hide_index=True)


def _render_quality(record) -> None:
    findings = record.findings
    summary = quality_summary(findings)

    columns = st.columns(3)
    columns[0].metric("Errors", summary.get("error", 0))
    columns[1].metric("Warnings", summary.get("warning", 0))
    columns[2].metric("Observations", summary.get("info", 0))

    st.caption(
        "These checks describe structural consistency of the uploaded CSV only. "
        "They are not an assessment of extraction accuracy."
    )

    frame = findings_to_frame(findings)
    if frame.empty:
        st.success("No structural inconsistencies detected in this document.")
    else:
        st.dataframe(frame, width='stretch', hide_index=True)
