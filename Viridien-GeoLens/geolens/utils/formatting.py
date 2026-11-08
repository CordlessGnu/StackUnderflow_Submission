"""Formatting helpers for consistent numeric and label presentation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

MISSING_DISPLAY = "—"


def is_missing(value: Any) -> bool:
    """Return True for None, NaN, NaT and empty strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NaT:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_number(value: Any, decimals: int = 2) -> str:
    """Format a number with thousands separators, or a dash when missing."""
    if is_missing(value):
        return MISSING_DISPLAY
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return MISSING_DISPLAY
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"


def format_depth(value: Any, decimals: int = 2) -> str:
    """Format a depth value without assuming a unit."""
    return format_number(value, decimals)


def format_percent(value: Any, decimals: int = 1) -> str:
    """Format a 0-100 percentage value."""
    if is_missing(value):
        return MISSING_DISPLAY
    return f"{float(value):.{decimals}f}%"


def format_label(value: Any) -> str:
    """Format a categorical label, preserving the original text."""
    if is_missing(value):
        return MISSING_DISPLAY
    return str(value)


def truncate(text: str, max_length: int = 40) -> str:
    """Shorten long labels for chart axes."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide while guarding against zero / missing denominators."""
    try:
        if denominator in (0, None) or is_missing(denominator):
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
