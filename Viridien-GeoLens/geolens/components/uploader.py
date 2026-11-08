"""Upload widgets and session-state helpers."""

from __future__ import annotations

from typing import Sequence
import hashlib

import pandas as pd
import streamlit as st

from geolens.models.data_models import DocumentCollection, DocumentRecord
from geolens.services.archive_loader import (
    ArchiveError,
    extract_archive,
)
from geolens.services.data_processor import CSV_PROCESSING_VERSION, build_document
from geolens.utils.helpers import stable_id

SINGLE_KEY = "single_document"
COLLECTION_KEY = "document_collection"


def get_collection() -> DocumentCollection:
    """Return the session document collection, creating it if needed."""
    if COLLECTION_KEY not in st.session_state:
        st.session_state[COLLECTION_KEY] = DocumentCollection()
    collection = st.session_state[COLLECTION_KEY]
    if st.session_state.get('collection_processing_version') != CSV_PROCESSING_VERSION:
        from geolens.services.cluster_ids import align_pipeline_cluster_ids
        from geolens.services.validation import check_document
        for record in collection:
            record.processed_df = align_pipeline_cluster_ids(record.processed_df)
            if record.schema_check.is_valid and not record.load_error:
                record.findings = check_document(record.processed_df, document_id=record.document_id)
        st.session_state['collection_processing_version'] = CSV_PROCESSING_VERSION
    return collection


def get_single_document() -> DocumentRecord | None:
    """Return the single-analysis document from session state."""
    return st.session_state.get(SINGLE_KEY)


def set_single_document(record: DocumentRecord | None) -> None:
    """Store the active document; clear stale selections when its identity changes."""
    previous = st.session_state.get(SINGLE_KEY)
    if record is not None and (previous is None or previous.document_id != record.document_id):
        from geolens.components.filters import reset_filters
        reset_filters('single')
        st.session_state.pop('selected_interval', None)
    st.session_state[SINGLE_KEY] = record


def add_to_collection(record: DocumentRecord) -> None:
    """Add a document to the session collection."""
    get_collection().add(record)


def clear_collection() -> None:
    """Remove all documents from the session collection."""
    st.session_state[COLLECTION_KEY] = DocumentCollection()


def render_single_upload() -> DocumentRecord | None:
    """Choose an existing CSV or run extraction and reuse the same analytics."""
    mode = st.radio('Input source', ['Extraction CSV', 'TIFF + JSON'], horizontal=True,
                    key='single_input_mode')
    if mode == 'TIFF + JSON':
        from geolens.components.extraction import render_tiff_upload
        record = render_tiff_upload()
        set_single_document(record)
        return record
    return _render_single_csv_upload()


def _render_single_csv_upload() -> DocumentRecord | None:
    st.markdown('<div class="gl-panel-title">Upload extraction output</div>', unsafe_allow_html=True)

    csv_file = st.file_uploader(
        "Extraction CSV",
        type=["csv"],
        key="single_csv_upload",
        help="The structured CSV produced by the extraction pipeline.",
    )

    if csv_file is None:
        set_single_document(None)
        return None

    csv_bytes = csv_file.getvalue()
    signature = stable_id(CSV_PROCESSING_VERSION, csv_file.name, hashlib.sha256(csv_bytes).hexdigest())
    existing = st.session_state.get('single_csv_document')
    if existing is not None and st.session_state.get("single_signature") == signature:
        set_single_document(existing)
        return existing

    record = build_document(
        csv_bytes=csv_bytes,
        csv_name=csv_file.name,
        document_id=signature,
    )
    st.session_state["single_signature"] = signature
    st.session_state['single_csv_document'] = record
    set_single_document(record)
    st.session_state.pop("selected_interval", None)
    return record


def _collect_uploaded_files(
    csv_files: Sequence, zip_files: Sequence
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Merge direct uploads and ZIP contents into a flat CSV list.

    Returns:
        ``(csvs, notes)`` where each file is ``(name, bytes)`` and ``notes``
        describes anything skipped or renamed.
    """
    csvs: list[tuple[str, bytes]] = [(f.name, f.getvalue()) for f in csv_files or []]
    notes: list[str] = []

    for archive in zip_files or []:
        try:
            contents = extract_archive(archive.getvalue(), filename=archive.name)
        except ArchiveError as exc:
            notes.append(str(exc))
            continue

        if contents.is_empty:
            notes.append(f"'{archive.name}' contained no CSV files.")
            continue

        csvs.extend((item.name, item.data) for item in contents.csv_files)

    # De-duplicate by filename, keeping the first occurrence.
    seen: set[str] = set()
    unique: list[tuple[str, bytes]] = []
    for name, data in csvs:
        if name in seen:
            notes.append(f"Duplicate filename '{name}' ignored.")
            continue
        seen.add(name)
        unique.append((name, data))

    return unique, notes


def render_batch_upload() -> list[DocumentRecord]:
    """Render the multi-document upload interface.

    Supports two input routes:
      * a ZIP of a folder (the closest Streamlit allows to a folder picker),
      * multiple CSVs selected at once.
    """
    st.markdown('<div class="gl-panel-title">Upload document collection</div>', unsafe_allow_html=True)

    upload_mode = st.radio(
        "Upload method",
        options=["Folder (ZIP)", "Individual files"],
        horizontal=True,
        key="batch_upload_mode",
        help=(
            "Streamlit cannot open a folder picker, so compress the folder first "
            "and upload the .zip. Individual files can also be selected together."
        ),
    )

    zip_files: list = []
    csv_files: list = []

    if upload_mode == "Folder (ZIP)":
        zip_files = st.file_uploader(
            "Folder archive (.zip)",
            type=["zip"],
            accept_multiple_files=True,
            key="batch_zip_upload",
            help=(
                "Zip the folder containing your extraction CSVs, then upload it. "
                "Sub-folders are searched automatically."
            ),
        )
    else:
        csv_files = st.file_uploader(
            "Extraction CSVs",
            type=["csv"],
            accept_multiple_files=True,
            key="batch_csv_upload",
        )

    csvs, notes = _collect_uploaded_files(csv_files, zip_files)

    if not csvs:
        for note in notes:
            st.caption(note)
        return []

    # ---- Compact summary ---------------------------------------------------
    summary_cols = st.columns([1, 1.4])
    summary_cols[0].metric("CSV files", len(csvs))
    summary_cols[1].metric("Ready to add", len(csvs))

    for note in notes:
        st.caption(note)

    if st.button("Add documents to collection", type="primary", key="batch_add"):
        added = 0
        for csv_name, csv_bytes in csvs:
            record = build_document(csv_bytes=csv_bytes, csv_name=csv_name)
            add_to_collection(record)
            added += 1
        st.success(f"Added {added} document(s) to the collection.")
        st.rerun()

    return []


def render_collection_manager(collection: DocumentCollection) -> None:
    """Render the document collection header with remove controls.

    The per-document table lives in the page's own "Collection overview
    (detailed)" expander; this header only hosts the remove actions.
    """
    if len(collection) == 0:
        return

    documents = list(collection)
    ready = sum(1 for d in documents if d.status == "Ready")

    header = st.columns([3, 1.6])
    with header[0]:
        st.markdown('<div class="gl-panel-title">Document collection</div>', unsafe_allow_html=True)
    with header[1]:
        st.markdown(
            f'<div class="gl-status"><span class="gl-status-dot"></span>'
            f"{len(documents)} documents · {ready} ready</div>",
            unsafe_allow_html=True,
        )

    actions = st.columns([1, 1, 3])
    with actions[0]:
        if st.button("Remove all", key="collection_clear_all", width="stretch"):
            clear_collection()
            st.rerun()
    with actions[1]:
        remove_one = st.selectbox(
            "Remove one",
            options=["—"] + [d.document_name for d in documents],
            key="collection_remove_one",
            label_visibility="collapsed",
        )
    with actions[2]:
        if remove_one != "—":
            target = next((d for d in documents if d.document_name == remove_one), None)
            if target and st.button(f"Remove {remove_one}", key="collection_remove_btn"):
                collection.remove(target.document_id)
                st.rerun()


def collection_dataframe(collection: DocumentCollection):
    """Return the collection overview as a dataframe."""
    rows = [doc.summary() for doc in collection]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
