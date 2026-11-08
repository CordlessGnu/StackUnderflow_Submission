"""Domain models shared across GeoLens services and UI components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from geolens.models.schemas import QualityFinding, SchemaCheck


@dataclass
class DocumentRecord:
    """A single uploaded geological document (one extraction CSV).

    ``source_df`` holds the untouched CSV content. ``processed_df`` holds the
    same rows plus calculated columns (thickness, document id, validity flags).
    """

    document_id: str
    document_name: str
    source_df: pd.DataFrame
    processed_df: pd.DataFrame
    schema_check: SchemaCheck
    findings: list[QualityFinding] = field(default_factory=list)
    load_error: str | None = None

    # -- convenience -------------------------------------------------------
    @property
    def interval_count(self) -> int:
        return int(len(self.processed_df))

    @property
    def renamed_columns(self) -> tuple[tuple[str, str], ...]:
        """Alias mappings applied on load, e.g. ``("bottom_depth", "base_depth")``."""
        return self.schema_check.renamed_columns

    def original_source_df(self) -> pd.DataFrame:
        """Return the source data with the uploaded file's original headers."""
        from geolens.services.csv_loader import restore_source_columns

        return restore_source_columns(self.source_df, self.renamed_columns)

    @property
    def status(self) -> str:
        if self.load_error:
            return "Error"
        if not self.schema_check.is_valid:
            return "Invalid schema"
        if any(f.severity == "error" for f in self.findings):
            return "Needs review"
        return "Ready"

    def summary(self) -> dict[str, Any]:
        """Return a compact summary row used by overview tables."""
        from geolens.services.statistics import document_summary

        return document_summary(self)


@dataclass
class DocumentCollection:
    """An ordered, de-duplicated collection of uploaded documents."""

    documents: dict[str, DocumentRecord] = field(default_factory=dict)

    def add(self, record: DocumentRecord) -> None:
        """Add or replace a document by its unique id."""
        self.documents[record.document_id] = record

    def remove(self, document_id: str) -> None:
        self.documents.pop(document_id, None)

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents.values())

    @property
    def names(self) -> list[str]:
        return [d.document_name for d in self.documents.values()]

    def valid_documents(self) -> list[DocumentRecord]:
        return [d for d in self.documents.values() if d.schema_check.is_valid and not d.load_error]

    def consolidated(self) -> pd.DataFrame:
        """Concatenate processed frames from all valid documents."""
        from geolens.services.data_processor import consolidate_documents

        return consolidate_documents(self.valid_documents())
