"""Consistent application sidebar."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from geolens.models.data_models import DocumentCollection

NAV_ITEMS = (
    "Single Analysis",
    "Batch Analysis",
    "Comparison Studio",
    "Export Centre",
)


def render_branding() -> None:
    """Render the GeoLens wordmark."""
    st.markdown(
        '<div class="gl-brand">'
        '<span class="gl-brand-name">GeoLens</span>'
        '<span class="gl-brand-tag">Geological Intelligence</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_navigation(default: str = "Single Analysis") -> str:
    """Render the main navigation and return the selected section."""
    render_branding()
    st.markdown('<div class="gl-panel-title">Workspace</div>', unsafe_allow_html=True)
    selection = st.radio(
        "Navigation",
        options=list(NAV_ITEMS),
        index=list(NAV_ITEMS).index(default) if default in NAV_ITEMS else 0,
        label_visibility="collapsed",
        key="nav_section",
    )
    return selection


def render_document_info(collection: DocumentCollection | None) -> None:
    """Render uploaded document information and analysis status.

    The list is collapsed by default so a large collection never pushes the
    navigation out of reach, and a single action clears everything.
    """
    st.markdown("---")

    if collection is None or len(collection) == 0:
        st.markdown('<div class="gl-panel-title">Documents</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="gl-note">Upload a CSV or run TIFF + JSON extraction in Single Analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    documents = list(collection)
    ready = sum(1 for d in documents if d.status == "Ready")

    st.markdown(
        f'<div class="gl-status"><span class="gl-status-dot"></span>'
        f"{len(documents)} document(s) · {ready} ready</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"Loaded documents ({len(documents)})", expanded=False):
        for doc in documents:
            icon = "●" if doc.status == "Ready" else "○"
            st.markdown(
                f'<div class="gl-kv"><span class="gl-kv-key">{icon} {doc.document_name}</span>'
                f'<span class="gl-kv-val">{doc.interval_count}</span></div>',
                unsafe_allow_html=True,
            )

    if st.button("Remove all documents", key="sidebar_clear_all", width="stretch"):
        from geolens.components.uploader import clear_collection, set_single_document

        clear_collection()
        set_single_document(None)
        st.session_state.pop("selected_interval", None)
        st.rerun()


def render_footer_note() -> None:
    """Render the sidebar disclaimer."""
    st.markdown("---")
    st.caption(
        "GeoLens runs the local extraction pipeline and analyses its CSV outputs. "
        "It does not verify geological accuracy."
    )


def render_sidebar(
    *,
    default_page: str = "Single Analysis",
    collection: DocumentCollection | None = None,
) -> str:
    """Render the full sidebar and return the selected page."""
    with st.sidebar:
        page = render_navigation(default_page)
        render_document_info(collection)
        render_footer_note()
    return page


def render_page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Render a consistent page header."""
    st.markdown(f'<div class="gl-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f'<div class="gl-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_empty_state(title: str, text: str) -> None:
    """Render a meaningful empty state."""
    st.markdown(
        f'<div class="gl-empty"><div class="gl-empty-title">{title}</div>'
        f'<div class="gl-empty-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_status_line(items: Sequence[tuple[str, str]]) -> None:
    """Render a row of key/value status items."""
    html = "".join(
        f'<div class="gl-kv"><span class="gl-kv-key">{key}</span>'
        f'<span class="gl-kv-val">{value}</span></div>'
        for key, value in items
    )
    st.markdown(f'<div class="gl-panel">{html}</div>', unsafe_allow_html=True)
