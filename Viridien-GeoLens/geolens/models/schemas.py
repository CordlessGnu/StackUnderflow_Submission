"""Internal data schema definitions for GeoLens.

This module is the single source of truth for column names, required fields and
the calculated (derived) columns that GeoLens adds on top of the raw extraction
output. Source columns are never renamed or overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Source schema (exactly as produced by the AI extraction pipeline)
# ---------------------------------------------------------------------------

COL_TOP_Y: Final[str] = "top_y"
COL_BOTTOM_Y: Final[str] = "bottom_y"
COL_CLUSTER_ID: Final[str] = "cluster_id"
COL_LITHOLOGY: Final[str] = "lithology"
COL_TOP_DEPTH: Final[str] = "top_depth"
COL_BASE_DEPTH: Final[str] = "base_depth"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    COL_TOP_Y,
    COL_BOTTOM_Y,
    COL_CLUSTER_ID,
    COL_LITHOLOGY,
    COL_TOP_DEPTH,
    COL_BASE_DEPTH,
)

NUMERIC_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    COL_TOP_Y,
    COL_BOTTOM_Y,
    COL_TOP_DEPTH,
    COL_BASE_DEPTH,
)

#: Accepted alternative spellings for each canonical column.
#:
#: Real extraction outputs are not always named identically — for example the
#: company's own export uses ``bottom_depth`` rather than ``base_depth``.
#: Matching is case-insensitive and ignores surrounding whitespace. The first
#: spelling found in a file is mapped onto the canonical column name, and the
#: mapping is reported back to the user so nothing is silently renamed.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    COL_TOP_DEPTH: (
        "top",
        "from_depth",
        "start_depth",
        "depth_top",
        "topdepth",
        "top_depth_m",
    ),
    COL_BASE_DEPTH: (
        "bottom_depth",
        "base",
        "bottom",
        "to_depth",
        "end_depth",
        "depth_base",
        "basedepth",
        "bottomdepth",
        "base_depth_m",
    ),
    COL_LITHOLOGY: (
        "lith",
        "lithology_name",
        "lith_type",
        "lithtype",
        "label",
    ),
    COL_CLUSTER_ID: (
        "cluster",
        "cluster_label",
        "clusterid",
        "cluster_no",
    ),
    COL_TOP_Y: (
        "top_pixel",
        "y_top",
        "topy",
        "top_px",
    ),
    COL_BOTTOM_Y: (
        "bottom_pixel",
        "y_bottom",
        "bottomy",
        "bottom_px",
    ),
}

# ---------------------------------------------------------------------------
# Calculated schema (added by GeoLens, never replaces source columns)
# ---------------------------------------------------------------------------

COL_THICKNESS: Final[str] = "thickness"
COL_DOCUMENT_ID: Final[str] = "document_id"
COL_DOCUMENT_NAME: Final[str] = "document_name"
COL_SOURCE_ROW: Final[str] = "source_row_index"
COL_IS_VALID: Final[str] = "is_valid"
COL_ISSUES: Final[str] = "issues"
COL_LITHOLOGY_MISSING: Final[str] = "lithology_missing"
COL_CLUSTER_MISSING: Final[str] = "cluster_missing"

CALCULATED_COLUMNS: Final[tuple[str, ...]] = (
    COL_THICKNESS,
    COL_DOCUMENT_ID,
    COL_DOCUMENT_NAME,
    COL_SOURCE_ROW,
    COL_IS_VALID,
    COL_ISSUES,
    COL_LITHOLOGY_MISSING,
    COL_CLUSTER_MISSING,
)

#: Column order used for processed / exported interval tables.
PROCESSED_COLUMN_ORDER: Final[tuple[str, ...]] = (
    COL_DOCUMENT_ID,
    COL_DOCUMENT_NAME,
    COL_SOURCE_ROW,
    COL_TOP_DEPTH,
    COL_BASE_DEPTH,
    COL_THICKNESS,
    COL_LITHOLOGY,
    COL_CLUSTER_ID,
    COL_TOP_Y,
    COL_BOTTOM_Y,
    COL_IS_VALID,
    COL_ISSUES,
)

#: Placeholder used when a lithology label is missing in the source CSV.
UNKNOWN_LITHOLOGY: Final[str] = "Unknown"

#: Placeholder used when a cluster identifier is missing in the source CSV.
UNKNOWN_CLUSTER: Final[str] = "Unassigned"


# ---------------------------------------------------------------------------
# Validation / quality result models
# ---------------------------------------------------------------------------

SEVERITY_INFO: Final[str] = "info"
SEVERITY_WARNING: Final[str] = "warning"
SEVERITY_ERROR: Final[str] = "error"


@dataclass(frozen=True)
class SchemaCheck:
    """Result of validating a CSV against the required GeoLens schema."""

    is_valid: bool
    missing_columns: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    message: str = ""
    #: ``(original_name, canonical_name)`` pairs for columns that were matched
    #: through an alias, e.g. ``("bottom_depth", "base_depth")``.
    renamed_columns: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "missing_columns": list(self.missing_columns),
            "extra_columns": list(self.extra_columns),
            "message": self.message,
            "renamed_columns": [list(pair) for pair in self.renamed_columns],
        }


@dataclass(frozen=True)
class QualityFinding:
    """A single data-quality observation for one document.

    Findings describe structural consistency only. They are *not* an accuracy
    assessment of the underlying AI extraction.
    """

    document_id: str
    check: str
    severity: str
    count: int
    message: str
    row_indices: tuple[int, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "check": self.check,
            "severity": self.severity,
            "count": self.count,
            "message": self.message,
            "row_indices": list(self.row_indices),
        }
