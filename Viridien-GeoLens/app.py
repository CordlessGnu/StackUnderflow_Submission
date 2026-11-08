"""GeoLens — Geological Data Intelligence & Comparison Platform.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from geolens.components.sidebar import render_sidebar
from geolens.components.theme import apply_theme
from geolens.components.uploader import get_collection
from geolens.pages import batch_analysis, comparison, export_centre, single_analysis

PAGE_RENDERERS = {
    "Single Analysis": single_analysis.render,
    "Batch Analysis": batch_analysis.render,
    "Comparison Studio": comparison.render,
    "Export Centre": export_centre.render,
}


def configure_page() -> None:
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="GeoLens · Geological Data Intelligence",
        page_icon="◧",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def main() -> None:
    """Application entry point."""
    configure_page()
    apply_theme()

    collection = get_collection()
    page = render_sidebar(default_page="Single Analysis", collection=collection)

    renderer = PAGE_RENDERERS.get(page, single_analysis.render)
    renderer()


if __name__ == "__main__":
    main()
