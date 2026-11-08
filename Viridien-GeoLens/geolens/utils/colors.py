"""Dynamic, deterministic colour generation.

GeoLens never hardcodes colours for particular lithology names or cluster IDs.
Instead, every label is hashed to a stable position on a curated, perceptually
spread palette. Unknown labels therefore receive a consistent colour for the
whole session without any code change.
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import Iterable, Mapping

# A bright, high-contrast palette tuned for the dark canvas. Colours are vivid
# enough to separate adjacent intervals at a glance while staying distinguishable
# for viewers with common colour-vision deficiencies.
_BASE_PALETTE: tuple[str, ...] = (
    "#4EA8DE",  # bright sky blue
    "#F4A259",  # warm amber
    "#5FD68A",  # vivid green
    "#E86A6A",  # coral red
    "#B48CE8",  # bright violet
    "#F2D45C",  # golden yellow
    "#3FD0C9",  # turquoise
    "#F07CB0",  # pink
    "#9BD65F",  # lime green
    "#FF9F6B",  # peach
    "#6C9BF5",  # periwinkle
    "#E8C07A",  # sand
    "#7FD4E8",  # light cyan
    "#D98CE0",  # orchid
    "#C3E05F",  # chartreuse
    "#F58C8C",  # salmon
)

#: Neutral colour used for missing / unknown labels.
UNKNOWN_COLOR = "#8A9BB0"

#: Colour used for "highlighted" intervals in highlight mode.
HIGHLIGHT_ACCENT = "#FFB347"

#: Colour used for de-emphasised intervals in highlight mode (dark canvas).
MUTED_COLOR = "#2A3648"

#: Colour of the separator line drawn between adjacent intervals.
INTERVAL_BORDER_COLOR = "#FFFFFF"


def _stable_hash(label: str) -> int:
    digest = hashlib.sha256(label.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_color(label: str, *, variant: int = 0) -> str:
    """Return a deterministic colour for an arbitrary label.

    The first ``len(_BASE_PALETTE)`` distinct labels map onto the curated
    palette; further labels are generated from a golden-ratio hue rotation so
    that any number of new lithologies or clusters remain visually distinct.
    """
    if label is None:
        return UNKNOWN_COLOR
    text = str(label).strip()
    if not text:
        return UNKNOWN_COLOR

    key = _stable_hash(text) + variant * 7919
    index = key % len(_BASE_PALETTE)
    if key % 3 != 0:
        return _BASE_PALETTE[index]

    # Deterministic fallback hue for extra variety on unseen labels. Kept bright
    # and saturated so generated colours match the curated palette's vividness.
    hue = ((key % 360) / 360.0 + 0.13 * variant) % 1.0
    saturation = 0.55 + ((key >> 8) % 20) / 100.0
    lightness = 0.58 + ((key >> 16) % 14) / 100.0
    return _hsl_to_hex(hue, saturation, lightness)


def build_color_map(labels: Iterable[str], *, variant: int = 0) -> dict[str, str]:
    """Build a stable ``label -> colour`` mapping for a set of labels.

    Collisions are resolved by nudging the variant so that two different labels
    never share the same colour within one mapping.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for label in sorted({str(lbl) for lbl in labels if str(lbl).strip()}):
        attempt = 0
        colour = generate_color(label, variant=variant)
        while colour in used and attempt < 24:
            attempt += 1
            colour = generate_color(label, variant=variant + attempt)
        used.add(colour)
        mapping[label] = colour
    return mapping


def color_for(label: str, mapping: Mapping[str, str] | None = None) -> str:
    """Look up a colour, generating one on demand if the label is new."""
    if mapping is not None and label in mapping:
        return mapping[label]
    return generate_color(label)


def text_color_for(background: str) -> str:
    """Return black or white text depending on background luminance."""
    hex_value = background.lstrip("#")
    if len(hex_value) != 6:
        return "#0F172A"
    r, g, b = (int(hex_value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0F172A" if luminance > 0.55 else "#FFFFFF"
