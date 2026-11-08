"""Folder upload support via ZIP archives.

Streamlit's file uploader cannot open a directory picker, so GeoLens accepts a
ZIP of a folder instead. This module extracts the CSVs from that archive while
ignoring junk entries and path traversal attempts.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

CSV_EXTENSIONS = (".csv",)

#: Directory / file names that should never be treated as documents.
_IGNORED_PARTS = {
    "__macosx",
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".git",
    "__pycache__",
}


class ArchiveError(Exception):
    """Raised when a ZIP archive cannot be read."""


@dataclass
class ExtractedFile:
    """A single file recovered from an uploaded archive."""

    name: str
    data: bytes
    path: str

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class ArchiveContents:
    """CSV files recovered from an archive."""

    csv_files: list[ExtractedFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.csv_files)

    @property
    def is_empty(self) -> bool:
        return self.total == 0


def _is_ignored(path: PurePosixPath) -> bool:
    """Return True for hidden files and known OS/junk entries."""
    for part in path.parts:
        lowered = part.lower()
        if lowered in _IGNORED_PARTS or lowered.startswith("."):
            return True
    return False


def _is_safe(path: PurePosixPath) -> bool:
    """Reject absolute paths and parent-directory traversal."""
    if path.is_absolute():
        return False
    return ".." not in path.parts


def extract_archive(data: bytes, *, filename: str = "upload.zip") -> ArchiveContents:
    """Extract CSVs from a ZIP archive.

    Args:
        data: raw ZIP bytes.
        filename: original upload name, used in error messages.

    Returns:
        The recovered CSV files, plus a list of skipped entries.

    Raises:
        ArchiveError: if the archive is not a readable ZIP.
    """
    if not data:
        raise ArchiveError(f"'{filename}' is empty.")

    contents = ArchiveContents()

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ArchiveError(
            f"'{filename}' is not a valid ZIP archive. "
            "Compress the folder first, then upload the .zip file."
        ) from exc

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            path = PurePosixPath(info.filename.replace("\\", "/"))
            if _is_ignored(path) or not _is_safe(path):
                contents.skipped.append(info.filename)
                continue

            suffix = path.suffix.lower()
            if suffix not in CSV_EXTENSIONS:
                contents.skipped.append(info.filename)
                continue

            try:
                payload = archive.read(info)
            except (RuntimeError, zipfile.BadZipFile):
                # Encrypted or corrupt entry — report rather than crash.
                contents.skipped.append(info.filename)
                continue

            if not payload:
                contents.skipped.append(info.filename)
                continue

            contents.csv_files.append(
                ExtractedFile(name=path.name, data=payload, path=str(path))
            )

    contents.csv_files.sort(key=lambda f: f.path.lower())
    return contents


def is_zip(filename: str) -> bool:
    """Return True when a filename looks like a ZIP archive."""
    return filename.lower().endswith(".zip")


def describe_contents(contents: ArchiveContents) -> str:
    """Build a short human-readable summary of an archive's contents."""
    parts = [f"{len(contents.csv_files)} CSV file(s)"]
    if contents.skipped:
        parts.append(f"{len(contents.skipped)} entry/entries skipped")
    return ", ".join(parts)
