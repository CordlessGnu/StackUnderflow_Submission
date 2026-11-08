"""TIFF + bbox JSON controls, live process progress and CSV analytics handoff."""
from __future__ import annotations

import atexit
from pathlib import Path

import streamlit as st

from geolens.models.data_models import DocumentRecord
from geolens.services.data_processor import CSV_PROCESSING_VERSION, build_document
from geolens.services.extraction_service import (
    DEFAULT_OUTPUT_ROOT, ExtractionError, ExtractionJobManager, ExtractionOptions,
)
from geolens.services.local_picker import PickerError, pick_local_path

JOB_KEY = 'extraction_job_id'
RECORD_KEY = 'extraction_document'
LOADED_KEY = 'extraction_loaded_job_id'


@st.cache_resource
def get_job_manager() -> ExtractionJobManager:
    manager = ExtractionJobManager()
    atexit.register(manager.shutdown)
    return manager


def _browse(kind: str, key: str) -> None:
    try:
        selected = pick_local_path(kind, str(st.session_state.get(key, '')))
        if selected:
            st.session_state[key] = selected
        st.session_state.pop('extraction_picker_error', None)
    except PickerError as exc:
        st.session_state['extraction_picker_error'] = str(exc)


def _path_control(label: str, key: str, kind: str) -> str:
    field, button = st.columns([5, 1], vertical_alignment='bottom')
    with field:
        value = st.text_input(label, key=key)
    with button:
        st.button('Browse…', key=f'{key}_browse', on_click=_browse, args=(kind, key), width='stretch')
    return value


@st.cache_data(show_spinner=False, max_entries=12)
def _artifact_bytes(path: str, modification_time: int) -> bytes:
    return Path(path).read_bytes()


@st.fragment(run_every='1s')
def _render_job(job_id: str) -> None:
    manager = get_job_manager()
    try:
        job = manager.snapshot(job_id)
    except ExtractionError as exc:
        st.error(str(exc))
        return
    st.caption(f'Extraction: {job.source_name} · elapsed {job.elapsed_seconds:.1f}s')
    st.progress(job.progress, text=job.stage if job.state == 'running' else job.state.title())
    st.caption('Run output directory')
    st.code(str(job.output_dir), language=None)
    if job.state == 'running':
        if st.button('Cancel extraction', key=f'extraction_cancel_{job_id}'):
            manager.cancel(job_id)
            st.rerun()
    elif job.state == 'succeeded':
        if (st.session_state.get(LOADED_KEY) != job_id
                or st.session_state.get('extraction_processing_version') != CSV_PROCESSING_VERSION):
            try:
                name = f'{Path(job.source_name).stem}_{job_id[:6]}.csv'
                record = build_document(csv_bytes=job.csv_path.read_bytes(), csv_name=name,
                                        document_id=f'extraction_{job_id}')
                st.session_state[RECORD_KEY] = record
                st.session_state[LOADED_KEY] = job_id
                st.session_state['extraction_processing_version'] = CSV_PROCESSING_VERSION
                st.rerun()  # refresh the normal Single Analysis page, not a separate analysis implementation
            except OSError as exc:
                st.error(f'Cannot load the generated CSV: {exc}')
                return
        st.success('Extraction complete. The generated CSV is loaded into the analytics below.')
        manifest = job.manifest
        columns = st.columns(3)
        columns[0].metric('Verified labels', manifest.get('verified_intervals', 0))
        columns[1].metric('Runner-up fallbacks', manifest.get('fallback_intervals', 0))
        columns[2].metric('Blank intervals', manifest.get('blank_intervals', 0))
        st.caption('Fallback labels are not verified matches. Blank intervals remain unassigned in the source CSV.')
        from geolens.components.uploader import add_to_collection, get_collection
        record = st.session_state.get(RECORD_KEY)
        already_added = record is not None and get_collection().get(record.document_id) is not None
        if st.button('Add this extraction to collection', key=f'extraction_add_{job_id}',
                     disabled=record is None or already_added):
            add_to_collection(record)
            st.rerun()
        downloads = st.columns(3)
        for index, (filename, label, mime) in enumerate([
            ('intervals.csv', 'Extraction CSV', 'text/csv'),
            ('intervals.json', 'Results JSON', 'application/json'),
            ('intervals.png', 'Annotated column', 'image/png'),
        ]):
            path = job.output_dir / filename
            if path.is_file():
                with downloads[index]:
                    try:
                        st.download_button(label, data=_artifact_bytes(str(path), path.stat().st_mtime_ns),
                                           file_name=f'{Path(job.source_name).stem}_{filename}', mime=mime,
                                           key=f'extraction_download_{job_id}_{index}', on_click='ignore')
                    except OSError:
                        st.caption(f'{filename} is no longer available.')
    elif job.state == 'cancelled':
        st.warning(job.message)
    else:
        st.error(job.message)
        st.caption('No CSV from this failed run has been loaded. Partial outputs and logs are retained.')
    notes = job.manifest.get('warnings', [])
    if notes:
        with st.expander(f'Pipeline warnings ({len(notes)})'):
            for note in notes:
                st.warning(str(note))
    if job.validation:
        with st.expander('Input geometry'):
            st.caption(f"TIFF: {job.validation['width']} × {job.validation['height']} pixels. Crops are not resized.")
            st.json(job.validation['bounding_boxes'])
    with st.expander('Worker log', expanded=job.state == 'failed'):
        st.code(job.log_tail or 'Waiting for pipeline output…', language=None)


def render_tiff_upload() -> DocumentRecord | None:
    """Select two files and an output parent, then reuse the existing CSV display."""
    st.markdown('<div class="gl-panel-title">Extract from a local document</div>', unsafe_allow_html=True)
    st.caption('Run the connected pipeline, then analyse its CSV here. Only one extraction runs at a time.')
    method = st.radio('TIFF/JSON selection', ['Local file paths', 'Upload files'], horizontal=True,
                      key='extraction_file_method')
    tiff_upload = annotation_upload = None
    tiff_path = annotation_path = ''
    if method == 'Local file paths':
        tiff_path = _path_control('TIFF document path', 'extraction_tiff_path', 'tiff')
        annotation_path = _path_control('Bounding-box JSON path', 'extraction_annotation_path', 'json')
        st.caption('Recommended for large TIFFs: no browser copy. Relative paths are resolved from the repository root.')
    else:
        left, right = st.columns(2)
        with left:
            tiff_upload = st.file_uploader('TIFF document', type=['tif', 'tiff'], key='extraction_tiff_upload')
        with right:
            annotation_upload = st.file_uploader('Bounding-box JSON', type=['json'], key='extraction_annotation_upload')
        st.caption('JSON must contain interpreted_lithology_column, depth_column and lithology_legend bounding boxes.')
    st.session_state.setdefault('extraction_output_root',
                                st.session_state.get('extraction_output_preference', str(DEFAULT_OUTPUT_ROOT)))
    output = _path_control('Output directory', 'extraction_output_root', 'directory')
    st.session_state['extraction_output_preference'] = output
    st.caption('Each run gets a new document-specific subfolder here; existing outputs are never overwritten. '
               'Browse opens a dialog on this local machine. You can also paste a directory path.')
    picker_error = st.session_state.pop('extraction_picker_error', None)
    if picker_error:
        st.warning(picker_error)
    with st.expander('Extraction settings'):
        left, right = st.columns(2)
        with left:
            matcher = st.selectbox('Lithology matcher', ['hybrid', 'descriptor'], key='extraction_matcher')
            unit = st.selectbox('Depth unit', ['unspecified', 'ft', 'm'], key='extraction_depth_unit')
            rounding = st.number_input('Depth rounding (units; 0 = full precision)', min_value=0.0,
                                       value=10.0, step=1.0, key='extraction_depth_rounding')
        with right:
            minimum = st.number_input('Minimum interval height (px)', min_value=1, value=2,
                                      key='extraction_min_interval')
            automatic = st.checkbox('Automatic texture context', value=True, key='extraction_auto_window')
            window = st.number_input('Base texture window (px)', min_value=4, value=48, step=4,
                                     disabled=automatic, key='extraction_window')
        st.caption('Rounding follows the pipeline setting; choose 0 to avoid rounding thin intervals to zero thickness.')
        legend = st.text_input('Reviewed legend manifest path (optional)', key='extraction_legend_manifest')
        calibration = st.text_input('Reviewed depth calibration path (optional)', key='extraction_calibration')
    manager = get_job_manager()
    active = manager.active_job()
    if active and active != st.session_state.get(JOB_KEY):
        st.info('An extraction is already running on this local server.')
        if st.button('View active extraction', key='extraction_attach_active'):
            st.session_state[JOB_KEY] = active
            st.session_state.pop(LOADED_KEY, None)
            st.session_state.pop(RECORD_KEY, None)
            st.rerun()
    if st.button('Run extraction', type='primary', key='extraction_start', disabled=active is not None):
        options = ExtractionOptions(matcher=matcher, depth_unit=unit, depth_rounding=float(rounding),
                                    min_interval=int(minimum), window=None if automatic else int(window),
                                    legend_manifest=legend, calibration=calibration)
        try:
            if method == 'Upload files':
                if tiff_upload is None or annotation_upload is None:
                    raise ExtractionError('Select both the TIFF and bounding-box JSON files.')
                job_id = manager.start_uploads(tiff_upload.name, tiff_upload, annotation_upload.name,
                                               annotation_upload, output, options)
            else:
                job_id = manager.start_paths(tiff_path, annotation_path, output, options)
            st.session_state[JOB_KEY] = job_id
            st.session_state.pop(LOADED_KEY, None)
            st.session_state.pop(RECORD_KEY, None)
            st.session_state.pop('selected_interval', None)
        except (ExtractionError, OSError) as exc:
            st.error(str(exc))
    job_id = st.session_state.get(JOB_KEY)
    if job_id:
        _render_job(job_id)
    return st.session_state.get(RECORD_KEY)
