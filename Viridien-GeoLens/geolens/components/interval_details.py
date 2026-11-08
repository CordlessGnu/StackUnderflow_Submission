"""Interval detail panel and cluster explorer components."""

from __future__ import annotations

from typing import Mapping
from html import escape

import pandas as pd
import streamlit as st

from geolens.models.schemas import (
    COL_BASE_DEPTH,
    COL_BOTTOM_Y,
    COL_CLUSTER_ID,
    COL_LITHOLOGY,
    COL_SOURCE_ROW,
    COL_THICKNESS,
    COL_TOP_DEPTH,
    COL_TOP_Y,
)
from geolens.services.data_processor import interval_thickness
from geolens.utils.colors import color_for
from geolens.utils.formatting import format_label, format_number


def _kv(key: str, value: str) -> str:
    return f'<div class="gl-kv"><span class="gl-kv-key">{key}</span><span class="gl-kv-val">{value}</span></div>'


def render_interval_details(
    df: pd.DataFrame,
    *,
    selected_row: int | None,
    color_map: Mapping[str, str] | None = None,
) -> None:
    """Render the detail panel for the selected interval.

    Shows all six original CSV fields plus the calculated thickness.
    """
    st.markdown('<div class="gl-panel-title">Interval detail</div>', unsafe_allow_html=True)

    if df.empty:
        st.markdown('<div class="gl-note">No intervals available.</div>', unsafe_allow_html=True)
        return

    row = None
    if selected_row is not None and COL_SOURCE_ROW in df.columns:
        match = df[df[COL_SOURCE_ROW] == selected_row]
        if not match.empty:
            row = match.iloc[0]

    if row is None:
        st.markdown(
            '<div class="gl-note">Select an interval in the profile to inspect its source values.</div>',
            unsafe_allow_html=True,
        )
        return

    thickness = row.get(COL_THICKNESS)
    if pd.isna(thickness):
        thickness = interval_thickness(pd.DataFrame([row])).iloc[0]

    lithology = format_label(row.get(COL_LITHOLOGY))
    cluster = format_label(row.get(COL_CLUSTER_ID))
    colour = color_for(str(row.get(COL_LITHOLOGY)), color_map)

    html = [
        '<div class="gl-panel">',
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{colour};display:inline-block;"></span>'
        f'<strong style="font-size:0.95rem;">{lithology}</strong></div>',
        _kv("Lithology", lithology),
        _kv("Cluster ID", cluster),
        _kv("Top depth", format_number(row.get(COL_TOP_DEPTH))),
        _kv("Base depth", format_number(row.get(COL_BASE_DEPTH))),
        _kv("Thickness (calculated)", format_number(thickness)),
        _kv("Top pixel (top_y)", format_number(row.get(COL_TOP_Y), 0)),
        _kv("Bottom pixel (bottom_y)", format_number(row.get(COL_BOTTOM_Y), 0)),
        _kv("Source row", format_number(row.get(COL_SOURCE_ROW), 0)),
        *([_kv('Match status', escape(str(row.get('match_status', ''))))] if 'match_status' in row.index else []),
        *([_kv('Assignment', 'Runner-up fallback — not a verified match')]
          if bool(row.get('is_runner_up_fallback', False)) else []),
        *([_kv('Assignment', 'Intentional blank interval (source lithology cell is empty)')]
          if bool(row.get('is_blank_interval', False)) else []),
        "</div>",
    ]
    st.markdown("".join(html), unsafe_allow_html=True)


def render_cluster_explorer(
    df: pd.DataFrame,
    *,
    cluster_stats: pd.DataFrame,
    color_map: Mapping[str, str] | None = None,
    key_prefix: str = "cluster",
) -> str | None:
    """Render the cluster explorer and return the selected cluster id."""
    st.markdown('<div class="gl-panel-title">Cluster explorer</div>', unsafe_allow_html=True)

    if cluster_stats.empty:
        st.markdown('<div class="gl-note">No clusters detected in this document.</div>', unsafe_allow_html=True)
        return None

    options = cluster_stats[COL_CLUSTER_ID].astype(str).tolist()
    selected = st.selectbox("Select a cluster", options=options, key=f"{key_prefix}_select")

    if selected is None:
        return None

    match = cluster_stats[cluster_stats[COL_CLUSTER_ID].astype(str) == str(selected)]
    if match.empty:
        return selected

    row = match.iloc[0]
    colour = color_for(str(selected), color_map)
    html = [
        '<div class="gl-panel">',
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{colour};display:inline-block;"></span>'
        f'<strong style="font-size:0.95rem;">Cluster {selected}</strong></div>',
        _kv("Intervals", format_number(row["interval_count"], 0)),
        _kv("Lithology labels", format_label(row["lithologies"])),
        _kv("Distinct lithologies", format_number(row["lithology_count"], 0)),
        _kv("Total thickness", format_number(row["total_thickness"])),
        _kv("Depth span", f'{format_number(row["min_top"])} – {format_number(row["max_base"])}'),
        "</div>",
    ]
    st.markdown("".join(html), unsafe_allow_html=True)

    with st.expander("Depth ranges where this cluster appears"):
        ranges = str(row.get("depth_ranges", "")).split("; ")
        for item in ranges:
            if item.strip():
                st.markdown(f"- {item}")

    return str(selected)


def render_interval_table(df: pd.DataFrame, *, max_rows: int = 200) -> None:
    """Render a compact interval table for inspection."""
    if df.empty:
        st.markdown('<div class="gl-note">No intervals to list.</div>', unsafe_allow_html=True)
        return
    columns = [
        c
        for c in (
            COL_TOP_DEPTH,
            COL_BASE_DEPTH,
            COL_THICKNESS,
            COL_LITHOLOGY,
            COL_CLUSTER_ID,
            COL_TOP_Y,
            COL_BOTTOM_Y,
            'match_status',
            'lithology_fallback',
            'match_seam_score',
            'fallback_distance',
            'depth_extrapolated',
            'depth_unit',
        )
        if c in df.columns
    ]
    view = df[columns].head(max_rows).copy()
    st.dataframe(view, width='stretch', hide_index=True, height=320)
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows} of {len(df)} intervals.")
