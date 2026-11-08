"""Tests for folder (ZIP) upload support."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from geolens.services.archive_loader import (
    ArchiveError,
    describe_contents,
    extract_archive,
    is_zip,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def folder_zip() -> bytes:
    """A ZIP mimicking a folder of CSVs, including sub-folders."""
    return _make_zip(
        {
            "geology/well_a.csv": (SAMPLE_DIR / "well_a.csv").read_bytes(),
            "geology/well_b.csv": (SAMPLE_DIR / "well_b.csv").read_bytes(),
            "geology/nested/well_c.csv": (SAMPLE_DIR / "well_c.csv").read_bytes(),
            "geology/notes.txt": b"not a document",
        }
    )


def test_extract_finds_csvs(folder_zip: bytes) -> None:
    contents = extract_archive(folder_zip, filename="folder.zip")
    assert len(contents.csv_files) == 3
    assert {f.name for f in contents.csv_files} == {"well_a.csv", "well_b.csv", "well_c.csv"}


def test_extract_searches_subfolders(folder_zip: bytes) -> None:
    contents = extract_archive(folder_zip, filename="folder.zip")
    paths = {f.path for f in contents.csv_files}
    assert "geology/nested/well_c.csv" in paths


def test_extract_skips_unsupported_files(folder_zip: bytes) -> None:
    contents = extract_archive(folder_zip, filename="folder.zip")
    assert "geology/notes.txt" in contents.skipped


def test_extract_ignores_macos_junk() -> None:
    data = _make_zip(
        {
            "__MACOSX/._well_a.csv": b"junk",
            "well_a.csv": (SAMPLE_DIR / "well_a.csv").read_bytes(),
            ".DS_Store": b"junk",
        }
    )
    contents = extract_archive(data, filename="mac.zip")
    assert len(contents.csv_files) == 1
    assert contents.csv_files[0].name == "well_a.csv"


def test_extract_rejects_path_traversal() -> None:
    data = _make_zip({"../../evil.csv": b"top_depth\n1\n"})
    contents = extract_archive(data, filename="evil.zip")
    assert contents.csv_files == []
    assert "../../evil.csv" in contents.skipped


def test_extract_ignores_directories() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("folder/", b"")
        archive.writestr("folder/well_a.csv", (SAMPLE_DIR / "well_a.csv").read_bytes())
    contents = extract_archive(buffer.getvalue(), filename="dirs.zip")
    assert len(contents.csv_files) == 1


def test_extract_raises_on_invalid_zip() -> None:
    with pytest.raises(ArchiveError):
        extract_archive(b"not a zip", filename="bad.zip")


def test_extract_raises_on_empty_bytes() -> None:
    with pytest.raises(ArchiveError):
        extract_archive(b"", filename="empty.zip")


def test_empty_archive_reports_empty() -> None:
    data = _make_zip({"readme.txt": b"hello"})
    contents = extract_archive(data, filename="empty.zip")
    assert contents.is_empty


def test_is_zip_detects_extension() -> None:
    assert is_zip("folder.zip")
    assert is_zip("FOLDER.ZIP")
    assert not is_zip("folder.csv")


def test_describe_contents(folder_zip: bytes) -> None:
    contents = extract_archive(folder_zip, filename="folder.zip")
    description = describe_contents(contents)
    assert "3 CSV" in description
    assert "image" not in description


def test_extracted_csvs_are_loadable(folder_zip: bytes) -> None:
    """Files recovered from a ZIP must feed straight into the document pipeline."""
    from geolens.services.data_processor import build_document

    contents = extract_archive(folder_zip, filename="folder.zip")
    for item in contents.csv_files:
        doc = build_document(csv_bytes=item.data, csv_name=item.name)
        assert doc.status == "Ready"
        assert doc.interval_count > 0
