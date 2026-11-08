"""Interactive vertical geological profile built with Plotly.

Intervals are drawn as bars whose heights are proportional to their real
thickness (``base_depth - top_depth``), never as equal-height blocks.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geolens.components.theme import (
    CHART_MARGIN,
    ON_COLOR_TEXT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    axis_style,
    plotly_layout,
)
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
from geolens.utils.colors import (
    HIGHLIGHT_ACCENT,
    INTERVAL_BORDER_COLOR,
    MUTED_COLOR,
    color_for,
)
from geolens.utils.formatting import format_number, truncate

PROFILE_KEY = "profile"

# Non-breaking-space gutter that shifts hover text to the right of the cursor.
_HOVER_GUTTER = "&nbsp;" * 14


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort intervals by depth and ensure a usable thickness column."""
    work = df.copy()
    work[COL_THICKNESS] = interval_thickness(work)
    work = work.dropna(subset=[COL_TOP_DEPTH, COL_BASE_DEPTH])
    work = work[work[COL_THICKNESS] > 0]
    return work.sort_values(COL_TOP_DEPTH, kind="mergesort").reset_index(drop=True)


def build_profile_figure(
    df: pd.DataFrame,
    *,
    color_by: str = "lithology",
    color_map: Mapping[str, str] | None = None,
    highlight_labels: Sequence[str] | None = None,
    highlight_mode: bool = False,
    selected_row: int | None = None,
    title: str | None = None,
    height: int = 620,
    show_labels: bool = True,
) -> go.Figure:
    """Build the geological column figure.

    Args:
        df: processed intervals for one document.
        color_by: ``"lithology"`` or ``"cluster"``.
        color_map: label -> colour mapping (generated dynamically if absent).
        highlight_labels: labels to emphasise when ``highlight_mode`` is on.
        highlight_mode: mute non-selected intervals instead of hiding them.
        selected_row: ``source_row_index`` of the selected interval.
        title: optional chart title.
        height: figure height in pixels.
        show_labels: whether to draw lithology labels inside intervals.
    """
    work = _prepare(df)
    fig = go.Figure()

    if work.empty:
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(
            text="No intervals to display for the current filters.",
            showarrow=False,
            font={"size": 13, "color": TEXT_MUTED},
        )
        return fig

    label_column = COL_CLUSTER_ID if color_by == "cluster" else COL_LITHOLOGY
    highlight_set = set(highlight_labels or [])

    colors: list[str] = []
    for _, row in work.iterrows():
        label = str(row[label_column])
        base_color = color_for(label, color_map)
        if highlight_mode and highlight_set:
            colors.append(base_color if label in highlight_set else MUTED_COLOR)
        else:
            colors.append(base_color)

    # White separators make adjacent intervals clearly distinct on the dark
    # canvas; the selected interval gets a thicker, brighter outline.
    line_colors = [
        HIGHLIGHT_ACCENT if row[COL_SOURCE_ROW] == selected_row else INTERVAL_BORDER_COLOR
        for _, row in work.iterrows()
    ]
    line_widths = [2.5 if row[COL_SOURCE_ROW] == selected_row else 1.2 for _, row in work.iterrows()]

    customdata = []
    for _, row in work.iterrows():
        customdata.append(
            [
                str(row[COL_LITHOLOGY]),
                str(row[COL_CLUSTER_ID]),
                format_number(row[COL_TOP_DEPTH]),
                format_number(row[COL_BASE_DEPTH]),
                format_number(row[COL_THICKNESS]),
                format_number(row.get(COL_TOP_Y)),
                format_number(row.get(COL_BOTTOM_Y)),
                int(row[COL_SOURCE_ROW]),
            ]
        )

    text = [truncate(str(v), 22) for v in work[COL_LITHOLOGY]] if show_labels else None

    fig.add_trace(
        go.Bar(
            x=[1] * len(work),
            y=work[COL_THICKNESS],
            base=work[COL_TOP_DEPTH],
            orientation="v",
            marker={
                "color": colors,
                "line": {"color": line_colors, "width": line_widths},
            },
            text=text,
            textposition="inside",
            insidetextanchor="middle",
            textfont={"size": 10, "color": ON_COLOR_TEXT},
            customdata=customdata,
            # Plotly anchors the hover box at the cursor and offers no offset
            # property, so the tooltip is pushed clear of the interval with a
            # leading blank line (moves the box below the cursor) plus a wide
            # non-breaking-space gutter (moves the text right of the cursor).
            # The result is a tooltip that sits beside the coloured box instead
            # of covering it.
            hovertemplate=(
                "<br>"
                f"{_HOVER_GUTTER}<b>%{{customdata[0]}}</b><br>"
                f"{_HOVER_GUTTER}Cluster: %{{customdata[1]}}<br>"
                f"{_HOVER_GUTTER}Top depth: %{{customdata[2]}}<br>"
                f"{_HOVER_GUTTER}Base depth: %{{customdata[3]}}<br>"
                f"{_HOVER_GUTTER}Thickness: %{{customdata[4]}}<br>"
                f"{_HOVER_GUTTER}Top pixel: %{{customdata[5]}}<br>"
                f"{_HOVER_GUTTER}Bottom pixel: %{{customdata[6]}}"
                "<extra></extra>"
            ),
            width=0.72,
            showlegend=False,
        )
    )

    # Legend entries for the active colour dimension.
    seen: set[str] = set()
    for _, row in work.iterrows():
        label = str(row[label_column])
        if label in seen:
            continue
        seen.add(label)
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                marker={"color": color_for(label, color_map)},
                name=truncate(label, 28),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Reserve space above the plot for the colour guide so it never overlaps
    # the geological column.
    legend_rows = max(1, -(-len(seen) // 5))
    legend_space = legend_rows * 24 + 16
    layout = plotly_layout(
        height=height + legend_space,
        title={"text": title or "", "font": {"size": 13, "color": TEXT_PRIMARY}},
        barmode="overlay",
        bargap=0.02,
        showlegend=True,
        dragmode="pan",
        clickmode="event+select",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 10, "color": TEXT_SECONDARY},
            "itemsizing": "constant",
        },
        margin={
            "l": CHART_MARGIN["l"],
            "r": CHART_MARGIN["r"],
            "t": CHART_MARGIN["t"] + legend_space,
            "b": CHART_MARGIN["b"],
        },
    )
    fig.update_layout(**layout)
    fig.update_xaxes(
        **axis_style(
            "",
            showticklabels=False,
            showgrid=False,
            range=[0.35, 1.65],
            fixedrange=False,
        )
    )
    fig.update_yaxes(**axis_style("Depth", autorange="reversed"))
    return fig


def _extract_selected_row(state: object) -> int | None:
    """Pull the clicked interval's ``source_row_index`` out of a widget state.

    Accepts either the raw ``st.session_state[key]`` mapping or an event object
    with a ``selection`` attribute, since Streamlit has used both shapes.
    """
    if state is None:
        return None

    selection = None
    if isinstance(state, dict):
        selection = state.get("selection")
    else:
        selection = getattr(state, "selection", None)
        if selection is None:
            try:
                selection = state["selection"]  # type: ignore[index]
            except (KeyError, TypeError):
                return None

    if selection is None:
        return None

    points = selection.get("points") if isinstance(selection, dict) else getattr(selection, "points", None)
    if not points:
        return None

    customdata = points[0].get("customdata")
    if not customdata or len(customdata) < 8:
        return None
    try:
        return int(customdata[7])
    except (TypeError, ValueError):
        return None


def get_selected_interval(key: str = PROFILE_KEY) -> int | None:
    """Return the currently selected interval's ``source_row_index``.

    Reads the chart's widget state from ``st.session_state``. Call this *before*
    rendering panels that depend on the selection, so they reflect the latest
    click on the same run.
    """
    row = _extract_selected_row(st.session_state.get(key))
    if row is not None:
        st.session_state["selected_interval"] = row
        return row
    return st.session_state.get("selected_interval")


def render_profile(
    df: pd.DataFrame,
    *,
    color_by: str = "lithology",
    color_map: Mapping[str, str] | None = None,
    highlight_labels: Sequence[str] | None = None,
    highlight_mode: bool = False,
    selected_row: int | None = None,
    title: str | None = None,
    height: int = 620,
    key: str = PROFILE_KEY,
) -> None:
    """Render the profile with selection support.

    Selection is stored in ``st.session_state["selected_interval"]`` as the
    ``source_row_index`` of the clicked interval.
    """
    fig = build_profile_figure(
        df,
        color_by=color_by,
        color_map=color_map,
        highlight_labels=highlight_labels,
        highlight_mode=highlight_mode,
        selected_row=selected_row,
        title=title,
        height=height,
    )
    st.plotly_chart(
        fig,
        width='stretch',
        key=key,
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
    )


def build_highlight_figure(
    df: pd.DataFrame,
    *,
    label: str,
    color_by: str = "lithology",
    color_map: Mapping[str, str] | None = None,
    height: int = 420,
) -> go.Figure:
    """Build a compact profile highlighting one label across a document."""
    return build_profile_figure(
        df,
        color_by=color_by,
        color_map=color_map,
        highlight_labels=[label],
        highlight_mode=True,
        height=height,
        show_labels=False,
    )
