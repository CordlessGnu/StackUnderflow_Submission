"""Filter controls for the geological profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
import streamlit as st

from geolens.models.schemas import COL_CLUSTER_ID, COL_LITHOLOGY, COL_TOP_DEPTH
from geolens.services.data_processor import apply_filters
from geolens.services.statistics import depth_range_for


@dataclass
class FilterState:
    """Resolved filter selections for a single document."""

    lithologies: list[str] = field(default_factory=list)
    clusters: list[str] = field(default_factory=list)
    depth_range: tuple[float, float] | None = None
    highlight_mode: bool = False
    color_by: str = "lithology"

    def as_dict(self) -> dict[str, object]:
        return {
            "lithologies": self.lithologies,
            "clusters": self.clusters,
            "depth_range": self.depth_range,
            "highlight_mode": self.highlight_mode,
            "color_by": self.color_by,
        }


def discover_options(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sorted unique lithologies and clusters from a dataframe."""
    lithologies: list[str] = []
    clusters: list[str] = []
    if not df.empty:
        if COL_LITHOLOGY in df.columns:
            lithologies = sorted({str(v) for v in df[COL_LITHOLOGY].dropna().unique()})
        if COL_CLUSTER_ID in df.columns:
            clusters = sorted({str(v) for v in df[COL_CLUSTER_ID].dropna().unique()})
    return lithologies, clusters


def render_filter_controls(
    df: pd.DataFrame,
    *,
    key_prefix: str = "single",
    show_cluster_filter: bool = True,
) -> FilterState:
    """Render lithology, cluster, depth and display-mode controls.

    Options are discovered from the data, so new lithologies and clusters appear
    automatically without code changes.
    """
    lithologies, clusters = discover_options(df)
    state = FilterState()

    st.markdown('<div class="gl-panel-title">Filters</div>', unsafe_allow_html=True)

    state.color_by = st.radio(
        "Colour by",
        options=["lithology", "cluster"],
        format_func=lambda v: "Lithology" if v == "lithology" else "Cluster ID",
        horizontal=True,
        key=f"{key_prefix}_color_by",
    )

    state.lithologies = st.multiselect(
        "Lithology",
        options=lithologies,
        default=[],
        placeholder="All lithologies",
        key=f"{key_prefix}_lith_filter",
    )

    if show_cluster_filter:
        state.clusters = st.multiselect(
            "Cluster ID",
            options=clusters,
            default=[],
            placeholder="All clusters",
            key=f"{key_prefix}_cluster_filter",
        )

    state.highlight_mode = st.checkbox(
        "Highlight selection (keep surrounding sequence)",
        value=False,
        key=f"{key_prefix}_highlight_mode",
        help="When enabled, non-selected intervals stay visible but are muted.",
    )

    low, high = depth_range_for(df)
    if high > low:
        step = max((high - low) / 200.0, 0.01)
        selected = st.slider(
            "Depth range",
            min_value=float(low),
            max_value=float(high),
            value=(float(low), float(high)),
            step=float(step),
            key=f"{key_prefix}_depth_range",
        )
        if selected[0] > low or selected[1] < high:
            state.depth_range = (float(selected[0]), float(selected[1]))

    return state


def apply_filter_state(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Apply a :class:`FilterState` to a dataframe."""
    return apply_filters(
        df,
        lithologies=state.lithologies,
        clusters=state.clusters,
        depth_range=state.depth_range,
    )


def describe_filters(state: FilterState) -> dict[str, object]:
    """Return a human-readable description of active filters for exports."""
    return {
        "Lithology": state.lithologies or "All",
        "Cluster": state.clusters or "All",
        "Depth range": (
            f"{state.depth_range[0]:g} – {state.depth_range[1]:g}"
            if state.depth_range
            else "Full profile"
        ),
        "Colour mode": state.color_by,
        "Highlight mode": "On" if state.highlight_mode else "Off",
    }


def reset_filters(key_prefix: str = "single") -> None:
    """Clear filter-related session state keys."""
    for suffix in ("lith_filter", "cluster_filter", "depth_range", "highlight_mode"):
        st.session_state.pop(f"{key_prefix}_{suffix}", None)


def depth_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Expose depth bounds for external widgets."""
    return depth_range_for(df)


def top_depth_series(df: pd.DataFrame) -> pd.Series:
    """Return the top depth series for ordering checks."""
    return pd.to_numeric(df.get(COL_TOP_DEPTH), errors="coerce")


def selected_labels(state: FilterState) -> Sequence[str]:
    """Return the labels currently selected for highlighting."""
    return state.lithologies if state.color_by == "lithology" else state.clusters
