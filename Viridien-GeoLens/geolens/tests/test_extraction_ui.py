"""Streamlit handoff tests reuse the real CSV analytics with a completed job."""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from geolens.models.schemas import COL_LITHOLOGY_MISSING
from geolens.services.data_processor import build_document
from geolens.services.extraction_service import JobSnapshot
from geolens.services.local_picker import PickerError, pick_local_path

APP = Path(__file__).resolve().parents[2] / 'app.py'
CSV = (b'top_y,bottom_y,cluster_id,lithology,top_depth,base_depth,match_status,lithology_fallback\n'
       b'0,10,C001,,0,10,blank,False\n'
       b'10,20,C002,Sandstone,10,20,ambiguous,True\n'
       b'20,30,C003,Shale,20,30,matched,False\n')


class CompleteManager:
    def __init__(self, output):
        self.output = output
        self.calls = []
    def active_job(self): return None
    def start_paths(self, tiff, annotation, output, options):
        self.calls.append((tiff, annotation, output, options))
        return 'demo-job'
    def snapshot(self, job_id):
        return JobSnapshot(job_id, 'Example.tiff', self.output, 'succeeded', 'Completed', 1,
                           'Complete', {'verified_intervals': 1, 'fallback_intervals': 1,
                                        'blank_intervals': 1, 'warnings': ['Example warning']},
                           'Completed local pipeline', 1, {})


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from geolens.components import extraction
    output = tmp_path / 'outputs'; output.mkdir()
    (output / 'intervals.csv').write_bytes(CSV)
    (output / 'intervals.json').write_text('{}')
    manager = CompleteManager(output)
    monkeypatch.setattr(extraction, 'get_job_manager', lambda: manager)
    return manager


def test_source_blanks_and_fallback_flags_survive_analytics():
    record = build_document(csv_bytes=CSV, csv_name='pipeline.csv')
    assert record.schema_check.is_valid
    assert pd.isna(record.source_df.loc[0, 'lithology'])
    assert record.processed_df.loc[0, 'lithology'] == 'Blank'
    assert not bool(record.processed_df.loc[0, COL_LITHOLOGY_MISSING])
    assert bool(record.processed_df.loc[1, 'is_runner_up_fallback'])
    assert not bool(record.processed_df.loc[2, 'is_runner_up_fallback'])
    assert not any(f.check == 'missing_lithology' for f in record.findings)
    assert any(f.check == 'runner_up_fallbacks' for f in record.findings)
    # Original CSV export still has an empty source cell, not the display label.
    source = pd.read_csv(io.StringIO(record.original_source_df().to_csv(index=False)), keep_default_na=False)
    assert source.loc[0, 'lithology'] == ''


def test_tiff_mode_exposes_inputs_output_and_upload_alternative(completed):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    app.radio(key='single_input_mode').set_value('TIFF + JSON').run()
    assert not app.exception
    assert app.text_input(key='extraction_output_root')
    assert app.text_input(key='extraction_tiff_path')
    app.radio(key='extraction_file_method').set_value('Upload files').run()
    assert not app.exception
    app.button(key='extraction_start').click().run()
    assert any('Select both' in e.value for e in app.error)
    assert completed.calls == []


def test_successful_run_automatically_enters_existing_analytics(completed, tmp_path):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key='single_input_mode').set_value('TIFF + JSON').run()
    app.text_input(key='extraction_tiff_path').set_value('/local/document.tiff')
    app.text_input(key='extraction_annotation_path').set_value('/local/annotation.json')
    app.text_input(key='extraction_output_root').set_value(str(tmp_path / 'chosen directory'))
    app.number_input(key='extraction_depth_rounding').set_value(0.0)
    app.button(key='extraction_start').click().run()
    assert not app.exception
    record = app.session_state['single_document']
    assert record.interval_count == 3
    assert record.document_id == 'extraction_demo-job'
    assert app.session_state['extraction_loaded_job_id'] == 'demo-job'
    assert completed.calls[0][2] == str(tmp_path / 'chosen directory')
    assert completed.calls[0][3].depth_rounding == 0
    assert any('Extraction complete' in msg.value for msg in app.success)
    assert len(app.get('plotly_chart')) > 0
    app.button(key='extraction_add_demo-job').click().run()
    assert not app.exception
    assert len(app.session_state['document_collection']) == 1


def test_native_directory_picker_updates_the_path_field(completed, monkeypatch, tmp_path):
    from geolens.components import extraction
    chosen = str(tmp_path / 'picked folder')
    monkeypatch.setattr(extraction, 'pick_local_path', lambda *args: chosen)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key='single_input_mode').set_value('TIFF + JSON').run()
    app.button(key='extraction_output_root_browse').click().run()
    assert not app.exception
    assert app.text_input(key='extraction_output_root').value == chosen
    app.radio(key='single_input_mode').set_value('Extraction CSV').run()
    app.radio(key='single_input_mode').set_value('TIFF + JSON').run()
    assert not app.exception
    assert app.text_input(key='extraction_output_root').value == chosen


def test_picker_failure_keeps_manual_path_available(completed, monkeypatch):
    from geolens.components import extraction
    def unavailable(*args): raise PickerError('Paste the path instead')
    monkeypatch.setattr(extraction, 'pick_local_path', unavailable)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key='single_input_mode').set_value('TIFF + JSON').run()
    app.button(key='extraction_output_root_browse').click().run()
    assert not app.exception
    assert any('Paste the path' in w.value for w in app.warning)
    app.text_input(key='extraction_output_root').set_value('/manual/folder').run()
    assert app.text_input(key='extraction_output_root').value == '/manual/folder'


def test_csv_upload_still_works_and_same_size_updates_are_not_stale(monkeypatch):
    from geolens.components import uploader
    class Upload:
        name = 'example.csv'
        def getvalue(self): return self.data
    upload = Upload(); upload.data = CSV
    monkeypatch.setattr(uploader.st, 'file_uploader', lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    first = app.session_state['single_document'].document_id
    upload.data = CSV.replace(b'Shale', b'Chalk')
    app.run()
    assert not app.exception
    record = app.session_state['single_document']
    assert record.document_id != first
    assert record.source_df.loc[2, 'lithology'] == 'Chalk'


def test_picker_subprocess_is_cancel_safe_and_needs_no_streamlit_thread(monkeypatch):
    import geolens.services.local_picker as picker
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"path":"/chosen/folder"}')
    monkeypatch.setattr(picker.subprocess, 'run', run)
    assert pick_local_path('directory') == '/chosen/folder'
    assert calls[0][1] == '-c'
    monkeypatch.setattr(picker.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"path":""}'))
    assert pick_local_path('directory') is None
