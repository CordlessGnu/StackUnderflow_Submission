"""Reusable Plotly chart components."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from geolens.components.theme import (
    ACCENT,
    CHART_MARGIN,
    SURFACE,
    SURFACE_ALT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    axis_style,
    chart_title,
    plotly_layout,
)
from geolens.models.schemas import COL_DOCUMENT_NAME, COL_LITHOLOGY
from geolens.utils.colors import color_for
from geolens.utils.formatting import truncate


def _colors_for(labels: Sequence[str], color_map: Mapping[str, str] | None) -> list[str]:
    return [color_for(str(label), color_map) for label in labels]


def lithology_bar_chart(
    stats: pd.DataFrame,
    *,
    value_column: str = "interval_count",
    color_map: Mapping[str, str] | None = None,
    title: str = "",
    height: int = 340,
    horizontal: bool = True,
) -> go.Figure:
    """Bar chart of a lithology statistic, coloured by lithology."""
    if stats.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(text="No data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    data = stats.sort_values(value_column, ascending=horizontal)
    labels = data[COL_LITHOLOGY].astype(str).tolist()
    values = data[value_column].tolist()
    colors = _colors_for(labels, color_map)

    if horizontal:
        trace = go.Bar(
            x=values,
            y=[truncate(lbl, 26) for lbl in labels],
            orientation="h",
            marker={"color": colors},
            hovertemplate="<b>%{y}</b><br>%{x:,.2f}<extra></extra>",
        )
    else:
        trace = go.Bar(
            x=[truncate(lbl, 18) for lbl in labels],
            y=values,
            marker={"color": colors},
            hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
        )

    fig = go.Figure(trace)
    fig.update_layout(**plotly_layout(height=height, title=chart_title(title), showlegend=False))
    if horizontal:
        fig.update_xaxes(**axis_style(""))
        fig.update_yaxes(**axis_style("", showgrid=False, autorange="reversed"))
    else:
        fig.update_xaxes(**axis_style("", showgrid=False))
        fig.update_yaxes(**axis_style(""))
    return fig


def lithology_donut(
    stats: pd.DataFrame,
    *,
    value_column: str = "total_thickness",
    color_map: Mapping[str, str] | None = None,
    title: str = "",
    height: int = 340,
) -> go.Figure:
    """Donut chart showing lithology proportions."""
    if stats.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(text="No data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    data = stats[stats[value_column] > 0]
    if data.empty:
        data = stats

    labels = data[COL_LITHOLOGY].astype(str).tolist()
    fig = go.Figure(
        go.Pie(
            labels=[truncate(lbl, 24) for lbl in labels],
            values=data[value_column].tolist(),
            hole=0.55,
            marker={"colors": _colors_for(labels, color_map), "line": {"color": SURFACE, "width": 1.5}},
            textinfo="percent",
            textfont={"size": 10},
            hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )
    # Colour guide sits *below* the donut with reserved space, so it can never
    # overlay the ring regardless of how many lithologies there are.
    legend_rows = max(1, -(-len(labels) // 4))
    legend_space = legend_rows * 22 + 14
    fig.update_layout(
        **plotly_layout(
            height=height + legend_space,
            title=chart_title(title),
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.02,
                "x": 0,
                "xanchor": "left",
                "font": {"size": 10, "color": TEXT_SECONDARY},
                "itemsizing": "constant",
            },
            margin={
                "l": CHART_MARGIN["l"],
                "r": CHART_MARGIN["r"],
                "t": CHART_MARGIN["t"],
                "b": CHART_MARGIN["b"] + legend_space,
            },
        )
    )
    return fig


def comparison_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "",
    value_suffix: str = "",
    height: int = 420,
    colorscale: str = "Blues",
) -> go.Figure:
    """Heatmap for a lithology x document comparison matrix."""
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(text="No comparison data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=[str(c) for c in matrix.columns],
            y=[truncate(str(i), 26) for i in matrix.index],
            colorscale=colorscale,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.2f}" + value_suffix + "<extra></extra>",
            colorbar={
                "thickness": 10,
                "len": 0.7,
                "outlinewidth": 0,
                "x": 1.015,
                "xpad": 8,
                "tickfont": {"size": 10, "color": TEXT_MUTED},
            },
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        **plotly_layout(
            height=height,
            title=chart_title(title),
            # Reserve room on the right for the colour bar.
            margin={
                "l": CHART_MARGIN["l"],
                "r": CHART_MARGIN["r"] + 34,
                "t": CHART_MARGIN["t"],
                "b": CHART_MARGIN["b"],
            },
        )
    )
    fig.update_xaxes(**axis_style("", showgrid=False, tickangle=-25))
    fig.update_yaxes(**axis_style("", showgrid=False, autorange="reversed"))
    return fig


def grouped_bar_chart(
    matrix: pd.DataFrame,
    *,
    title: str = "",
    y_title: str = "",
    height: int = 420,
) -> go.Figure:
    """Grouped bar chart of lithology counts per document."""
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(text="No data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    fig = go.Figure()
    for column in matrix.columns:
        fig.add_trace(
            go.Bar(
                name=str(column),
                x=[truncate(str(i), 20) for i in matrix.index],
                y=matrix[column].tolist(),
                hovertemplate="<b>%{x}</b><br>" + str(column) + ": %{y:,.2f}<extra></extra>",
            )
        )
    # Reserve room above the plot for the colour guide.
    legend_rows = max(1, -(-len(matrix.columns) // 5))
    legend_space = legend_rows * 24 + 16
    fig.update_layout(
        **plotly_layout(
            height=height + legend_space,
            title=chart_title(title),
            barmode="group",
            bargap=0.18,
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
    )
    fig.update_xaxes(**axis_style("", showgrid=False, tickangle=-30))
    fig.update_yaxes(**axis_style(y_title))
    return fig


def presence_matrix_chart(
    matrix: pd.DataFrame,
    *,
    title: str = "Lithology presence",
    height: int = 420,
) -> go.Figure:
    """Binary presence heatmap (present / absent) for lithologies."""
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height, title=title))
        fig.add_annotation(text="No data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    binary = (matrix > 0).astype(int)
    fig = go.Figure(
        go.Heatmap(
            z=binary.values,
            x=[str(c) for c in binary.columns],
            y=[truncate(str(i), 26) for i in binary.index],
            colorscale=[[0, SURFACE_ALT], [1, ACCENT]],
            showscale=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(**plotly_layout(height=height, title=chart_title(title)))
    fig.update_xaxes(**axis_style("", showgrid=False, tickangle=-25))
    fig.update_yaxes(**axis_style("", showgrid=False, autorange="reversed"))
    return fig


def similarity_heatmap(matrix: pd.DataFrame, *, height: int = 420) -> go.Figure:
    """Heatmap of pairwise lithology sequence similarity."""
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height))
        fig.add_annotation(text="No comparison data", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="Teal",
            zmin=0,
            zmax=100,
            hovertemplate="%{y} ↔ %{x}<br>%{z:.1f}%<extra></extra>",
            colorbar={
                "thickness": 10,
                "len": 0.7,
                "outlinewidth": 0,
                "x": 1.015,
                "xpad": 8,
                "tickfont": {"size": 10, "color": TEXT_MUTED},
            },
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        **plotly_layout(
            height=height,
            title=chart_title("Sequence similarity (%)"),
            margin={
                "l": CHART_MARGIN["l"],
                "r": CHART_MARGIN["r"] + 34,
                "t": CHART_MARGIN["t"],
                "b": CHART_MARGIN["b"],
            },
        )
    )
    fig.update_xaxes(**axis_style("", showgrid=False, tickangle=-25))
    fig.update_yaxes(**axis_style("", showgrid=False, autorange="reversed"))
    return fig


def sequence_strip(
    sequences: Mapping[str, Sequence[str]],
    *,
    color_map: Mapping[str, str] | None = None,
    height: int = 260,
    show_legend: bool = True,
) -> go.Figure:
    """Render lithology sequences as horizontal coloured strips.

    When ``show_legend`` is true, a colour guide is added so each block can be
    identified without hovering. Legend entries are built from the labels that
    actually appear in the sequences, in first-seen order.
    """
    fig = go.Figure()
    if not sequences:
        fig.update_layout(**plotly_layout(height=height))
        fig.add_annotation(text="No sequences available", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    names = list(sequences.keys())
    seen_labels: list[str] = []

    for name in names:
        sequence = list(sequences[name])
        for position, label in enumerate(sequence):
            text_label = str(label)
            if text_label not in seen_labels:
                seen_labels.append(text_label)
            fig.add_trace(
                go.Bar(
                    x=[1],
                    y=[name],
                    base=[position],
                    orientation="h",
                    marker={
                        "color": color_for(text_label, color_map),
                        "line": {"color": SURFACE, "width": 1},
                    },
                    hovertemplate=(
                        f"<b>{name}</b><br>{text_label}<br>position {position + 1}<extra></extra>"
                    ),
                    showlegend=False,
                    width=0.6,
                )
            )

    # Legend entries on top of the plot, matching the single-document profile.
    if show_legend:
        for label in seen_labels:
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

    # Reserve vertical space for the legend so it never overlaps the plot.
    # Roughly 5 entries fit per row at this width; each row is ~26px tall and
    # an extra 18px keeps clear air between the legend and the first sequence row.
    legend_rows = max(1, -(-len(seen_labels) // 5)) if show_legend else 0
    legend_space = legend_rows * 26 + 18 if show_legend else 0
    top_margin = CHART_MARGIN["t"] + legend_space

    fig.update_layout(
        **plotly_layout(
            height=max(height, 90 + 46 * len(names)) + legend_space,
            barmode="overlay",
            bargap=0.25,
            showlegend=show_legend,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                # Sit the legend above the plot with a clear gap so it never
                # touches the first sequence row.
                "y": 1.0,
                "x": 0,
                "xanchor": "left",
                "font": {"size": 11, "color": TEXT_SECONDARY},
                "itemsizing": "constant",
            },
            margin={
                "l": CHART_MARGIN["l"],
                "r": CHART_MARGIN["r"],
                "t": top_margin,
                "b": CHART_MARGIN["b"],
            },
        )
    )
    fig.update_xaxes(**axis_style("Sequence position", showgrid=False))
    fig.update_yaxes(**axis_style("", showgrid=False, autorange="reversed"))
    return fig


def render_chart(fig: go.Figure, *, key: str | None = None) -> None:
    """Render a Plotly figure with consistent config."""
    st.plotly_chart(
        fig,
        width='stretch',
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def render_scrollable_chart(
    fig: go.Figure, *, key: str | None = None, max_height: int = 620
) -> None:
    """Render a chart inside a bounded, scrollable container.

    Large datasets produce very tall figures; constraining the container keeps
    the page navigable while still allowing the full chart to be scrolled.
    """
    with st.container(height=max_height, border=False):
        render_chart(fig, key=key)


def render_scrollable_table(
    df: pd.DataFrame, *, key: str | None = None, max_height: int = 420
) -> None:
    """Render a dataframe inside a bounded, scrollable container."""
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=max_height,
        key=key,
    )


def document_comparison_bars(
    df: pd.DataFrame,
    *,
    metric: str = "interval_count",
    height: int = 340,
) -> go.Figure:
    """Bar chart comparing documents on a chosen metric."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_layout(height=height))
        fig.add_annotation(text="No documents", showarrow=False, font={"color": TEXT_MUTED})
        return fig

    fig = px.bar(
        df,
        x=COL_DOCUMENT_NAME,
        y=metric,
        color=COL_DOCUMENT_NAME,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(**plotly_layout(height=height, showlegend=False))
    fig.update_xaxes(**axis_style("", showgrid=False, tickangle=-25))
    fig.update_yaxes(**axis_style(metric.replace("_", " ").title()))
    return fig
