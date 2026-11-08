"""Small, dependency-free helpers shared across GeoLens."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str, fallback: str = "document") -> str:
    """Create a filesystem-safe identifier from a document name."""
    cleaned = _SAFE_NAME.sub("_", str(value).strip()).strip("._-")
    return cleaned or fallback


def stable_id(*parts: str) -> str:
    """Create a short, stable identifier from arbitrary string parts."""
    joined = "::".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    """Return unique values while keeping first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def human_file_size(num_bytes: int | None) -> str:
    """Format a byte count for display."""
    if not num_bytes:
        return "—"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
