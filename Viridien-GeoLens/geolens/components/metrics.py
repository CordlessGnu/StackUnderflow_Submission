"""Metric card components."""

from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st

from geolens.utils.formatting import format_number


def metric_card(label: str, value: str, *, unit: str = "", note: str = "") -> str:
    """Return the HTML for a single metric card."""
    unit_html = f'<span class="gl-metric-unit">{unit}</span>' if unit else ""
    note_html = f'<div class="gl-metric-note">{note}</div>' if note else ""
    return (
        f'<div class="gl-metric">'
        f'<div class="gl-metric-label">{label}</div>'
        f'<div class="gl-metric-value">{value}{unit_html}</div>'
        f"{note_html}</div>"
    )


def render_metric_row(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """Render a row of metric cards.

    Args:
        cards: sequence of ``(label, value, unit, note)`` tuples.
    """
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, (label, value, unit, note) in zip(columns, cards):
        with column:
            st.markdown(metric_card(label, value, unit=unit, note=note), unsafe_allow_html=True)


def overview_cards(metrics: dict) -> list[tuple[str, str, str, str]]:
    """Build the standard single-document overview cards."""
    thickest = metrics.get("thickest_interval")
    thickest_note = ""
    if thickest:
        thickest_note = f"{thickest['lithology']} · {format_number(thickest['thickness'])}"

    return [
        ("Intervals", format_number(metrics["interval_count"], 0), "", "detected lithological intervals"),
        ("Lithologies", format_number(metrics["lithology_count"], 0), "", "unique labels in this document"),
        ("Clusters", format_number(metrics["cluster_count"], 0), "", "unique cluster ids"),
        ("Depth coverage", format_number(metrics["depth_coverage"]), "", "min top → max base"),
        ("Avg thickness", format_number(metrics["average_thickness"]), "", "mean interval thickness"),
        ("Thickest", format_number(thickest["thickness"]) if thickest else "—", "", thickest_note),
    ]


def collection_cards(stats: dict) -> list[tuple[str, str, str, str]]:
    """Build the organisation-wide overview cards."""
    largest = stats.get("largest_document")
    largest_note = f"{largest['document']} · {largest['intervals']} intervals" if largest else ""
    return [
        ("Intervals", format_number(stats["interval_count"], 0), "", "across all documents"),
        ("Lithologies", format_number(stats["lithology_count"], 0), "", "unique labels discovered"),
        ("Clusters", format_number(stats["cluster_count"], 0), "", "document-scoped cluster ids"),
        ("Avg thickness", format_number(stats["average_thickness"]), "", "mean interval thickness"),
        ("Largest document", format_number(largest["intervals"], 0) if largest else "—", "", largest_note),
    ]


def render_chips(items: Iterable[tuple[str, str]]) -> None:
    """Render coloured chips for ``(label, colour)`` pairs.

    Chips flow in a single full-width row and wrap onto the next line when the
    available width is exhausted.
    """
    html = "".join(
        f'<span class="gl-chip"><span class="gl-chip-dot" style="background:{colour}"></span>{label}</span>'
        for label, colour in items
    )
    st.markdown(f'<div class="gl-chip-row">{html}</div>', unsafe_allow_html=True)
