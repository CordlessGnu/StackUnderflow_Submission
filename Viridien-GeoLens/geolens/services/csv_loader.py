"""CSV loading and schema validation for GeoLens extraction outputs."""

from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd

from geolens.models.schemas import (
    COLUMN_ALIASES,
    NUMERIC_SOURCE_COLUMNS,
    REQUIRED_COLUMNS,
    SchemaCheck,
)


class CSVLoadError(Exception):
    """Raised when a CSV cannot be parsed at all."""


def _normalise(name: str) -> str:
    """Normalise a column name for alias comparison."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_column_aliases(
    columns: list[str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Map alternative column spellings onto canonical GeoLens names.

    Args:
        columns: the column names as they appear in the uploaded file.

    Returns:
        ``(rename_map, renamed_pairs)`` where ``rename_map`` maps the original
        column name to its canonical name, and ``renamed_pairs`` is a list of
        ``(original, canonical)`` tuples for reporting.

    Canonical names always win: if a file already contains ``base_depth``, an
    alias such as ``bottom_depth`` is left untouched rather than overwriting it.
    """
    present = {_normalise(c): c for c in columns}
    rename_map: dict[str, str] = {}
    renamed_pairs: list[tuple[str, str]] = []

    for canonical, aliases in COLUMN_ALIASES.items():
        if _normalise(canonical) in present:
            continue  # canonical column already present
        for alias in aliases:
            key = _normalise(alias)
            if key in present:
                original = present[key]
                rename_map[original] = canonical
                renamed_pairs.append((original, canonical))
                break

    return rename_map, renamed_pairs


def _read_bytes(source: bytes | BinaryIO | str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        return source.encode("utf-8")
    data = source.read()
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def validate_schema(
    df: pd.DataFrame,
    *,
    renamed_columns: tuple[tuple[str, str], ...] = (),
) -> SchemaCheck:
    """Check that a dataframe contains the six required extraction columns.

    Args:
        df: the loaded dataframe (after alias normalisation).
        renamed_columns: alias mappings applied during loading, reported back so
            the user can see that e.g. ``bottom_depth`` was read as
            ``base_depth``.
    """
    present = {str(c).strip() for c in df.columns}
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in present)
    extra = tuple(sorted(present - set(REQUIRED_COLUMNS)))

    if missing:
        message = (
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Expected: "
            + ", ".join(REQUIRED_COLUMNS)
            + "."
        )
        if renamed_columns:
            message += " Recognised alternative names: " + ", ".join(
                f"{orig} → {canon}" for orig, canon in renamed_columns
            ) + "."
        return SchemaCheck(
            is_valid=False,
            missing_columns=missing,
            extra_columns=extra,
            message=message,
            renamed_columns=renamed_columns,
        )

    message = "All six required columns are present."
    if renamed_columns:
        message += " Column names were matched to the GeoLens schema: " + ", ".join(
            f"{orig} → {canon}" for orig, canon in renamed_columns
        ) + "."
    return SchemaCheck(
        is_valid=True,
        missing_columns=(),
        extra_columns=extra,
        message=message,
        renamed_columns=renamed_columns,
    )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric source columns without dropping unparseable values.

    Unparseable values become NaN so that validation can report them instead of
    silently discarding the row.
    """
    out = df.copy()
    for column in NUMERIC_SOURCE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_csv_detailed(
    source: bytes | BinaryIO | str, *, filename: str = "upload.csv"
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load a CSV and normalise alternative column names.

    Values are preserved exactly; only column *names* are mapped onto the
    canonical GeoLens schema (e.g. ``bottom_depth`` → ``base_depth``).

    Returns:
        ``(dataframe, renamed_pairs)``

    Raises:
        CSVLoadError: if the file is empty or cannot be parsed.
    """
    raw = _read_bytes(source)
    if not raw.strip():
        raise CSVLoadError(f"'{filename}' is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:  # pragma: no cover - pandas raises many types
        raise CSVLoadError(f"Could not parse '{filename}': {exc}") from exc

    if df.empty:
        raise CSVLoadError(f"'{filename}' contains no data rows.")

    df.columns = [str(c).strip() for c in df.columns]

    rename_map, renamed_pairs = resolve_column_aliases(list(df.columns))
    if rename_map:
        df = df.rename(columns=rename_map)

    # Preserve the original row position for traceability in exports.
    df.insert(0, "source_row_index", range(len(df)))
    return _coerce_numeric(df), renamed_pairs


def load_csv(source: bytes | BinaryIO | str, *, filename: str = "upload.csv") -> pd.DataFrame:
    """Load a CSV into a dataframe with canonical column names.

    Alternative spellings such as ``bottom_depth`` are mapped onto the canonical
    schema so downstream services always see the same columns.

    Raises:
        CSVLoadError: if the file is empty or cannot be parsed.
    """
    df, _ = load_csv_detailed(source, filename=filename)
    return df


def load_csv_with_check(
    source: bytes | BinaryIO | str, *, filename: str = "upload.csv"
) -> tuple[pd.DataFrame | None, SchemaCheck, str | None]:
    """Load a CSV and validate its schema.

    Returns:
        (dataframe or None, schema check, error message or None)
    """
    try:
        df, renamed_pairs = load_csv_detailed(source, filename=filename)
    except CSVLoadError as exc:
        return None, SchemaCheck(is_valid=False, message=str(exc)), str(exc)

    check = validate_schema(df, renamed_columns=tuple(renamed_pairs))
    return df, check, None


def restore_source_columns(
    df: pd.DataFrame, renamed_columns: tuple[tuple[str, str], ...]
) -> pd.DataFrame:
    """Rename canonical columns back to their original names.

    Used so the "Original CSV" export reproduces the uploaded file's headers
    exactly, even when aliases were normalised for processing. GeoLens's
    internal ``source_row_index`` column is dropped because it was not part of
    the uploaded file.
    """
    out = df.drop(columns=["source_row_index"], errors="ignore").copy()
    if not renamed_columns:
        return out
    reverse = {canonical: original for original, canonical in renamed_columns}
    return out.rename(columns=reverse)


def discover_lithologies(df: pd.DataFrame) -> list[str]:
    """Return all unique lithology labels found in a dataframe."""
    from geolens.models.schemas import COL_LITHOLOGY

    if COL_LITHOLOGY not in df.columns:
        return []
    values = df[COL_LITHOLOGY].dropna().astype(str).str.strip()
    return sorted({v for v in values if v})


def discover_clusters(df: pd.DataFrame) -> list[str]:
    """Return all unique cluster identifiers found in a dataframe."""
    from geolens.models.schemas import COL_CLUSTER_ID

    if COL_CLUSTER_ID not in df.columns:
        return []
    values = df[COL_CLUSTER_ID].dropna().astype(str).str.strip()
    return sorted({v for v in values if v})
