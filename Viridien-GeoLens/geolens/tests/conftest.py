"""Shared pytest fixtures for GeoLens tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geolens.services.data_processor import add_calculated_fields, build_document

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture
def well_a_bytes() -> bytes:
    return (SAMPLE_DIR / "well_a.csv").read_bytes()


@pytest.fixture
def well_b_bytes() -> bytes:
    return (SAMPLE_DIR / "well_b.csv").read_bytes()


@pytest.fixture
def well_c_bytes() -> bytes:
    return (SAMPLE_DIR / "well_c.csv").read_bytes()


@pytest.fixture
def messy_bytes() -> bytes:
    return (SAMPLE_DIR / "well_d_messy.csv").read_bytes()


@pytest.fixture
def simple_frame() -> pd.DataFrame:
    """A small, clean processed dataframe."""
    raw = pd.DataFrame(
        {
            "top_y": [0, 100, 200],
            "bottom_y": [100, 200, 300],
            "cluster_id": ["C1", "C2", "C1"],
            "lithology": ["Shale", "Sandstone", "Shale"],
            "top_depth": [0.0, 10.0, 20.0],
            "base_depth": [10.0, 20.0, 30.0],
            "source_row_index": [0, 1, 2],
        }
    )
    return add_calculated_fields(raw, document_id="doc1", document_name="Doc 1")


@pytest.fixture
def document_a(well_a_bytes: bytes):
    return build_document(csv_bytes=well_a_bytes, csv_name="well_a.csv")


@pytest.fixture
def document_b(well_b_bytes: bytes):
    return build_document(csv_bytes=well_b_bytes, csv_name="well_b.csv")


@pytest.fixture
def document_c(well_c_bytes: bytes):
    return build_document(csv_bytes=well_c_bytes, csv_name="well_c.csv")
