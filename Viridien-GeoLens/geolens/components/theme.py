"""Application theme: dark-only CSS and Plotly layout defaults.

GeoLens is intentionally dark-mode only. The palette below is the single source
of truth for UI colours; ``.streamlit/config.toml`` mirrors it so Streamlit's
native widgets match.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BACKGROUND = "#0B1220"      # app canvas
SURFACE = "#131C2B"         # panels, cards, sidebar
SURFACE_ALT = "#1A2436"     # raised surfaces, hover labels
BORDER = "#243044"          # hairlines and dividers
GRID = "#1E2A3C"            # chart gridlines

TEXT_PRIMARY = "#F1F5F9"    # headings
TEXT_BODY = "#DCE5EF"       # body copy
TEXT_SECONDARY = "#A9B8C9"  # subtitles, tab labels
TEXT_MUTED = "#7D8FA5"      # captions, labels, axis ticks

ACCENT = "#4FA3C7"
ACCENT_SOFT = "#16283A"

#: Near-black used for text drawn on top of coloured lithology bars.
ON_COLOR_TEXT = "#0B1220"

_CSS = f"""
<style>
    /* ---- Base canvas ------------------------------------------------- */
    .stApp, [data-testid="stAppViewContainer"] {{ background: {BACKGROUND}; }}
    html, body, [class*="css"] {{
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
        color: {TEXT_BODY};
    }}

    /* ---- Header: transparent, no decoration bar, no overlay ----------- */
    header[data-testid="stHeader"] {{
        background: transparent;
        border-bottom: none;
        box-shadow: none;
        height: 0;
    }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stToolbar"] {{ right: 0.75rem; }}

    /* Remove the main menu (which hosts "Record a screencast"). */
    #MainMenu,
    [data-testid="stMainMenu"],
    [data-testid="stStatusWidget"] {{ display: none !important; visibility: hidden !important; }}

    /* Push page content clear of the fixed header. Streamlit's own rule is
       more specific, so this needs !important to win. */
    .stMainBlockContainer,
    .block-container {{
        padding-top: 2.4rem !important;
        padding-bottom: 2rem !important;
        max-width: 1500px;
    }}
    /* Tighten the gap Streamlit inserts above the first element. */
    [data-testid="stMainBlockContainer"] > div:first-child {{ margin-top: 0 !important; }}
    h1 {{ margin-top: 0.1rem !important; }}

    /* ---- Typography --------------------------------------------------- */
    h1, h2, h3, h4 {{ color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}
    h1 {{ font-size: 1.85rem; font-weight: 650; }}
    h2 {{ font-size: 1.3rem; font-weight: 600; margin-top: 1.4rem; }}
    h3 {{ font-size: 1.05rem; font-weight: 600; }}
    p, li, span, label {{ color: {TEXT_BODY}; }}
    a {{ color: {ACCENT}; }}

    /* ---- Brand -------------------------------------------------------- */
    .gl-brand {{
        display: flex; align-items: baseline; gap: 0.5rem;
        padding-bottom: 0.6rem; border-bottom: 1px solid {BORDER};
        margin-bottom: 0.9rem;
    }}
    .gl-brand-name {{ font-size: 1.35rem; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.02em; }}
    .gl-brand-tag {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: {TEXT_MUTED}; }}

    .gl-eyebrow {{
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.14em;
        color: {TEXT_MUTED}; font-weight: 600; margin-bottom: 0.15rem;
    }}
    .gl-subtitle {{ color: {TEXT_SECONDARY}; font-size: 0.92rem; margin-top: -0.4rem; margin-bottom: 1.2rem; }}

    /* ---- Metric cards ------------------------------------------------- */
    .gl-metric {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 0.85rem 1rem; height: 100%;
    }}
    .gl-metric-label {{
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: {TEXT_MUTED}; font-weight: 600; margin-bottom: 0.3rem;
    }}
    .gl-metric-value {{ font-size: 1.5rem; font-weight: 650; color: {TEXT_PRIMARY}; line-height: 1.15; }}
    .gl-metric-unit {{ font-size: 0.75rem; color: {TEXT_MUTED}; font-weight: 500; margin-left: 0.2rem; }}
    .gl-metric-note {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 0.25rem; }}

    /* ---- Panels and key/value rows ------------------------------------ */
    .gl-panel {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 1rem 1.15rem;
    }}
    .gl-panel-title {{
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: {TEXT_MUTED}; font-weight: 600; margin-bottom: 0.6rem;
    }}
    .gl-kv {{ display: flex; justify-content: space-between; gap: 1rem; padding: 0.28rem 0;
        border-bottom: 1px dashed {BORDER}; font-size: 0.85rem; }}
    .gl-kv:last-child {{ border-bottom: none; }}
    .gl-kv-key {{ color: {TEXT_MUTED}; }}
    .gl-kv-val {{ color: {TEXT_PRIMARY}; font-weight: 550; text-align: right; }}

    /* ---- Chips -------------------------------------------------------- */
    .gl-chip {{
        display: inline-block; padding: 0.12rem 0.5rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; border: 1px solid {BORDER};
        background: {SURFACE_ALT}; color: {TEXT_BODY}; margin: 0 0.25rem 0.25rem 0;
    }}
    .gl-chip-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.35rem; }}
    .gl-chip-row {{
        display: flex; flex-wrap: wrap; gap: 0.35rem;
        width: 100%; margin: 0;
    }}
    .gl-chip-row .gl-chip {{ margin: 0; }}
    .gl-legend-label {{
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: {TEXT_MUTED}; font-weight: 600;
        margin: 0.9rem 0 0.45rem 0;
        padding-top: 0.75rem; border-top: 1px solid {BORDER};
    }}

    /* ---- Scrollable chart / table containers --------------------------- */
    .gl-scroll {{
        max-height: 620px; overflow: auto;
        border: 1px solid {BORDER}; border-radius: 6px;
        background: {SURFACE}; padding: 0.75rem;
    }}
    /* Horizontal scroller for wide side-by-side profile rows. */
    .gl-hscroll {{
        overflow-x: auto; overflow-y: hidden;
        padding-bottom: 0.5rem;
    }}
    .gl-hscroll::-webkit-scrollbar {{ height: 10px; }}
    .gl-hscroll::-webkit-scrollbar-track {{ background: {BACKGROUND}; border-radius: 6px; }}
    .gl-hscroll::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 6px; }}
    .gl-hscroll::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}
    .gl-scroll::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    .gl-scroll::-webkit-scrollbar-track {{ background: {BACKGROUND}; border-radius: 6px; }}
    .gl-scroll::-webkit-scrollbar-thumb {{
        background: {BORDER}; border-radius: 6px;
    }}
    .gl-scroll::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}

    /* ---- Consistent card padding for bordered containers --------------- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        padding: 0.9rem 1rem;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding-left: 0; padding-right: 0;
    }}
    /* Plotly titles sit inside the SVG; give them room from the card edge. */
    .js-plotly-plot .gtitle {{ dominant-baseline: hanging; }}

    /* ---- Empty states ------------------------------------------------- */
    .gl-empty {{
        border: 1px dashed {BORDER}; border-radius: 8px; padding: 2.4rem 1.5rem;
        text-align: center; background: {SURFACE};
    }}
    .gl-empty-title {{ font-size: 1rem; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 0.35rem; }}
    .gl-empty-text {{ font-size: 0.86rem; color: {TEXT_MUTED}; max-width: 520px; margin: 0 auto; }}

    /* ---- Status pill -------------------------------------------------- */
    .gl-status {{
        display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem;
        color: {TEXT_SECONDARY}; padding: 0.2rem 0.55rem; border-radius: 4px;
        background: {ACCENT_SOFT}; border: 1px solid {BORDER};
    }}
    .gl-status-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {ACCENT}; }}

    .gl-note {{
        font-size: 0.78rem; color: {TEXT_MUTED}; border-left: 2px solid {BORDER};
        padding-left: 0.7rem; margin: 0.5rem 0 1rem 0;
    }}

    /* ---- Sidebar ------------------------------------------------------ */
    section[data-testid="stSidebar"] {{
        background: {SURFACE}; border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

    /* ---- Native Streamlit widgets ------------------------------------- */
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}
    [data-testid="stCaptionContainer"], .stCaption {{ color: {TEXT_MUTED} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.86rem; font-weight: 550; color: {TEXT_SECONDARY}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT}; }}
    [data-testid="stExpander"] details {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE}; border: 1px dashed {BORDER};
    }}
    [data-testid="stCode"] pre, .stCode pre {{
        background: {SURFACE_ALT} !important; color: {TEXT_BODY} !important;
        border: 1px solid {BORDER}; border-radius: 6px;
    }}
    hr {{ border-color: {BORDER}; }}
</style>
"""


def apply_theme() -> None:
    """Inject the GeoLens stylesheet once per session."""
    st.markdown(_CSS, unsafe_allow_html=True)


#: Consistent inner padding for every chart, so titles and axes never touch the
#: card edge. Left margin is generous to fit rotated category labels.
CHART_MARGIN = {"l": 72, "r": 28, "t": 56, "b": 56}


def chart_title(text: str, *, size: int = 13) -> dict:
    """Return a left-aligned chart title with consistent padding.

    Plotly centres titles by default, which makes them sit flush against the
    card edge on wide charts. Anchoring to the left with a small inset keeps
    every chart title aligned with the plot area.
    """
    return {
        "text": text,
        "x": 0.01,
        "xanchor": "left",
        "y": 0.98,
        "yanchor": "top",
        "font": {"size": size, "color": TEXT_PRIMARY},
    }


def plotly_layout(height: int = 520, **overrides) -> dict:
    """Return a consistent Plotly layout dictionary for the dark theme."""
    layout = {
        "height": height,
        "margin": dict(CHART_MARGIN),
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": TEXT_BODY},
        "hoverlabel": {
            "bgcolor": SURFACE_ALT,
            "font": {"color": TEXT_PRIMARY, "size": 12},
            "bordercolor": BORDER,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 11, "color": TEXT_SECONDARY},
        },
    }
    layout.update(overrides)
    return layout


def axis_style(title_text: str | None = None, **overrides) -> dict:
    """Return a consistent Plotly axis dictionary.

    ``title_text`` is the axis title string; any other keyword is merged into
    the axis dictionary (e.g. ``autorange="reversed"``).
    """
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "zeroline": False,
        "linecolor": BORDER,
        "ticks": "outside",
        "tickcolor": BORDER,
        "tickfont": {"size": 11, "color": TEXT_MUTED},
        "title": {"text": title_text or "", "font": {"size": 12, "color": TEXT_SECONDARY}},
    }
    axis.update(overrides)
    return axis
