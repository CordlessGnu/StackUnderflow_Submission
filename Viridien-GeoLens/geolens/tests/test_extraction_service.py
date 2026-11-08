"""Real subprocess tests of the local bridge, using a cheap fake pipeline."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from geolens.services.extraction_service import (
    ExtractionError, ExtractionJobManager, ExtractionOptions, PIPELINE_SCRIPT, REPO_ROOT,
    UNET_PIPELINE_SCRIPT, UNET_STAGE, resolve_local_path, validate_inputs,
)

WORKER = r'''
import argparse, csv, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('tiff'); p.add_argument('annotation'); p.add_argument('--output', required=True)
a, options = p.parse_known_args()
mode = json.loads(Path(a.annotation).read_text()).get('test_behavior', 'success')
out = Path(a.output); out.mkdir(parents=True)
manifest = {'status':'running', 'current_stage':'1/6 crop TIFF', 'stages':{}}
(out/'run.json').write_text(json.dumps(manifest)); print('worker started', flush=True)
if mode == 'wait': time.sleep(60)
if mode == 'fail':
    manifest.update(status='failed', error='Synthetic depth calibration failed')
    (out/'run.json').write_text(json.dumps(manifest)); raise SystemExit(2)
with (out/'intervals.csv').open('w', newline='') as f:
    w = csv.writer(f)
    if mode == 'bad_csv': w.writerow(['not','the','schema'])
    else:
        w.writerow(['top_y','bottom_y','cluster_id','lithology','top_depth','base_depth','match_status','lithology_fallback'])
        w.writerow([0,10,'C001','',0,10,'blank',False])
        w.writerow([10,20,'C002','Sandstone',10,20,'ambiguous',True])
        w.writerow([20,30,'C003','Shale',20,30,'matched',False])
manifest.update(status='completed_with_warnings', stages={str(i):{'status':'complete'} for i in range(6)},
                verified_intervals=1, fallback_intervals=1, blank_intervals=1, warnings=['Synthetic warning'])
manifest.pop('current_stage', None)
(out/'run.json').write_text(json.dumps(manifest))
print('worker completed', flush=True)
'''


@pytest.fixture
def inputs(tmp_path):
    image = tmp_path / 'well with spaces.tiff'
    annotation = tmp_path / 'boxes.json'
    Image.new('RGB', (60, 300), 'white').save(image, compression='tiff_lzw')
    data = {'width': 60, 'height': 300, 'bounding_boxes': {
        'interpreted_lithology_column': [0, 0, 20, 300],
        'depth_column': [20, 0, 40, 300], 'lithology_legend': [40, 0, 60, 100]}}
    annotation.write_text(json.dumps(data))
    return image, annotation, data


@pytest.fixture
def manager(tmp_path):
    worker = tmp_path / 'fake pipeline.py'
    worker.write_text(WORKER)
    value = ExtractionJobManager(job_root=tmp_path / 'jobs', pipeline_script=worker)
    yield value
    value.shutdown()


def finish(manager, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.snapshot(job_id)
        if job.state != 'running':
            return job
        time.sleep(.03)
    pytest.fail('Worker did not finish')


def test_header_validation_does_not_change_pillow_guard(inputs):
    image, annotation, _ = inputs
    old = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 1
        assert validate_inputs(image, annotation)['height'] == 300
        assert Image.MAX_IMAGE_PIXELS == 1
    finally:
        Image.MAX_IMAGE_PIXELS = old


@pytest.mark.parametrize('change', ['missing', 'reversed', 'outside', 'dimensions', 'nan'])
def test_invalid_annotations_are_rejected(inputs, change):
    image, annotation, data = inputs
    if change == 'missing': del data['bounding_boxes']['depth_column']
    if change == 'reversed': data['bounding_boxes']['depth_column'] = [20, 0, 10, 100]
    if change == 'outside': data['bounding_boxes']['depth_column'] = [100, 0, 110, 100]
    if change == 'dimensions': data['height'] = 999
    if change == 'nan': data['bounding_boxes']['depth_column'][0] = float('nan')
    annotation.write_text(json.dumps(data))
    with pytest.raises(ExtractionError): validate_inputs(image, annotation)


def test_bad_json_and_non_tiff_fail_cleanly(inputs, tmp_path):
    image, annotation, _ = inputs
    annotation.write_text('not json')
    with pytest.raises(ExtractionError): validate_inputs(image, annotation)
    wrong = tmp_path / 'not.tiff'; wrong.write_text('not TIFF content')
    with pytest.raises(ExtractionError): validate_inputs(wrong, annotation)


def test_process_runs_asynchronously_and_produces_csv(manager, inputs, tmp_path):
    image, annotation, _ = inputs
    job_id = manager.start_paths(image, annotation, tmp_path / 'chosen output')
    assert manager.snapshot(job_id).state in ('running', 'succeeded')
    job = finish(manager, job_id)
    assert job.state == 'succeeded'
    assert job.progress == 1
    assert job.csv_path.is_file()
    assert job.output_dir.parent == tmp_path / 'chosen output'
    assert job.manifest['fallback_intervals'] == 1
    assert 'worker completed' in job.log_tail
    elapsed = job.elapsed_seconds
    time.sleep(.12)
    assert manager.snapshot(job_id).elapsed_seconds == elapsed


def test_repeated_runs_never_overwrite(manager, inputs, tmp_path):
    image, annotation, _ = inputs
    first = finish(manager, manager.start_paths(image, annotation, tmp_path / 'out'))
    content = first.csv_path.read_bytes()
    second = finish(manager, manager.start_paths(image, annotation, tmp_path / 'out'))
    assert first.output_dir != second.output_dir
    assert first.csv_path.read_bytes() == content


def test_worker_failure_and_incompatible_csv_do_not_succeed(manager, inputs, tmp_path):
    image, annotation, data = inputs
    for behavior in ('fail', 'bad_csv'):
        data['test_behavior'] = behavior; annotation.write_text(json.dumps(data))
        job = finish(manager, manager.start_paths(image, annotation, tmp_path / 'out'))
        assert job.state == 'failed'
        assert 'failed' in job.message.lower() or 'compatible CSV' in job.message


def test_one_job_limit_and_cancellation(manager, inputs, tmp_path):
    image, annotation, data = inputs
    data['test_behavior'] = 'wait'; annotation.write_text(json.dumps(data))
    job_id = manager.start_paths(image, annotation, tmp_path / 'out')
    with pytest.raises(ExtractionError, match='already running'):
        manager.start_paths(image, annotation, tmp_path / 'other')
    deadline = time.monotonic() + 5
    while not manager.snapshot(job_id).manifest and time.monotonic() < deadline: time.sleep(.02)
    manager.cancel(job_id)
    job = manager.snapshot(job_id)
    assert job.state == 'cancelled'
    assert job.manifest['status'] == 'cancelled'
    assert manager.active_job() is None
    manager.cancel(job_id)  # idempotent


def test_uploaded_inputs_are_staged_safely_and_cursors_restored(manager, inputs, tmp_path):
    image, annotation, _ = inputs
    tiff, boxes = io.BytesIO(image.read_bytes()), io.BytesIO(annotation.read_bytes())
    tiff.seek(4); boxes.seek(3)
    job_id = manager.start_uploads('../../well.tiff', tiff, 'C:\\fakepath\\boxes.json', boxes, tmp_path / 'out')
    assert tiff.tell() == 4 and boxes.tell() == 3
    job = finish(manager, job_id)
    assert job.state == 'succeeded'
    request = json.loads((manager.job_root / job_id / 'request.json').read_text())
    assert Path(request['tiff']).name == 'well.tiff'
    assert Path(request['annotation']).name == 'boxes.json'
    assert manager.job_root in Path(request['tiff']).parents


def test_failed_upload_validation_cleans_only_its_own_staging(manager, tmp_path):
    with pytest.raises(ExtractionError):
        manager.start_uploads('bad.tiff', io.BytesIO(b'no'), 'bad.json', io.BytesIO(b'{}'), tmp_path / 'out')
    assert list(manager.job_root.glob('upload_*')) == []


def test_options_are_typed_and_do_not_use_shell_interpolation(inputs):
    args = ExtractionOptions(depth_rounding=0, matcher='descriptor', window=48).arguments()
    assert args[args.index('--depth-rounding') + 1] == '0'
    assert args[args.index('--window') + 1] == '48'
    for options in [ExtractionOptions(matcher='bad'), ExtractionOptions(min_interval=0),
                    ExtractionOptions(depth_rounding=-1), ExtractionOptions(window=2)]:
        with pytest.raises(ExtractionError): options.arguments()


def test_relative_paths_resolve_from_repo_not_streamlit_directory():
    assert resolve_local_path('Dataset/example.tif') == REPO_ROOT / 'Dataset/example.tif'


# ---- TIFF only (U-Net boxes) -------------------------------------------------
UNET_WORKER = r'''
import argparse, csv, json, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('tiff'); p.add_argument('--output', required=True)
a, options = p.parse_known_args()
out = Path(a.output)
print('U-Net worker started', flush=True)
if 'wait' in Path(a.tiff).name: time.sleep(60)
if 'unetfail' in Path(a.tiff).name:
    out.mkdir(parents=True)
    (out/'run.json').write_text(json.dumps({'status':'failed', 'error':'U-Net region detection failed: synthetic'}))
    raise SystemExit(1)
out.mkdir(parents=True)
boxes = {'depth_column':[20,0,40,300], 'interpreted_lithology_column':[0,0,20,300], 'lithology_legend':[40,0,60,100]}
with (out/'intervals.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['top_y','bottom_y','cluster_id','lithology','top_depth','base_depth'])
    w.writerow([0,10,'C001','Shale',0,10])
manifest = {'status':'completed', 'stages':{'0/6 U-Net region detection':{'status':'complete'}, **{str(i):{'status':'complete'} for i in range(6)}},
            'box_source':{'method':'unet', 'bounding_boxes':boxes}, 'argv':sys.argv[1:]}
(out/'run.json').write_text(json.dumps(manifest))
'''


@pytest.fixture
def unet_manager(tmp_path):
    pipeline = tmp_path / 'fake pipeline.py'; pipeline.write_text(WORKER)
    worker = tmp_path / 'fake unet pipeline.py'; worker.write_text(UNET_WORKER)
    value = ExtractionJobManager(job_root=tmp_path / 'jobs', pipeline_script=pipeline, unet_script=worker)
    yield value
    value.shutdown()


def tiff_named(tmp_path, name):
    path = tmp_path / name
    Image.new('RGB', (60, 300), 'white').save(path, compression='tiff_lzw')
    return path


def test_tiff_only_runs_unet_worker_without_annotation(unet_manager, tmp_path):
    image = tiff_named(tmp_path, 'well only.tif')
    options = ExtractionOptions(depth_rounding=0, matcher='descriptor')
    job_id = unet_manager.start_tiff_path(image, tmp_path / 'out', options)
    job = finish(unet_manager, job_id)
    assert job.state == 'succeeded', job.message
    assert job.validation == {'width': 60, 'height': 300, 'bounding_boxes': None, 'box_source': 'unet'}
    assert job.manifest['box_source']['method'] == 'unet'
    request = json.loads((unet_manager.job_root / job_id / 'request.json').read_text())
    assert request['annotation'] is None and request['box_source'] == 'unet'
    assert Path(request['command'][3]).name == 'fake unet pipeline.py'
    assert not any(part.endswith('.json') for part in request['command'])
    argv = job.manifest['argv']  # the pipeline options reach the worker unchanged
    assert argv[argv.index('--depth-rounding') + 1] == '0' and argv[argv.index('--matcher') + 1] == 'descriptor'
    assert job.progress == 1


def test_tiff_only_shows_unet_stage_while_waiting_and_blocks_other_jobs(unet_manager, inputs, tmp_path):
    image = tiff_named(tmp_path, 'wait.tif')
    job_id = unet_manager.start_tiff_path(image, tmp_path / 'out')
    job = unet_manager.snapshot(job_id)
    assert job.state == 'running' and job.stage == UNET_STAGE and job.progress == 0
    with pytest.raises(ExtractionError, match='already running'):
        unet_manager.start_paths(inputs[0], inputs[1], tmp_path / 'other')  # one job across both modes
    unet_manager.cancel(job_id)
    assert unet_manager.snapshot(job_id).state == 'cancelled'


def test_tiff_only_failure_is_not_loaded(unet_manager, tmp_path):
    job = finish(unet_manager, unet_manager.start_tiff_path(tiff_named(tmp_path, 'unetfail.tif'), tmp_path / 'out'))
    assert job.state == 'failed'
    assert 'U-Net region detection failed' in job.message


def test_tiff_only_upload_is_staged_safely(unet_manager, tmp_path):
    image = tiff_named(tmp_path, 'upload.tif')
    stream = io.BytesIO(image.read_bytes()); stream.seek(5)
    job_id = unet_manager.start_tiff_upload('C:\\fakepath\\..\\well.tif', stream, tmp_path / 'out')
    assert stream.tell() == 5
    assert finish(unet_manager, job_id).state == 'succeeded'
    request = json.loads((unet_manager.job_root / job_id / 'request.json').read_text())
    assert Path(request['tiff']).name == 'well.tif'
    assert unet_manager.job_root in Path(request['tiff']).parents


def test_tiff_only_rejects_non_tiff_and_cleans_staging(unet_manager, tmp_path):
    with pytest.raises(ExtractionError):
        unet_manager.start_tiff_upload('bad.tif', io.BytesIO(b'not a tiff'), tmp_path / 'out')
    assert list(unet_manager.job_root.glob('upload_*')) == []
    png = tmp_path / 'page.tif'; Image.new('RGB', (5, 5)).save(png, format='PNG')
    with pytest.raises(ExtractionError, match='not TIFF'):
        unet_manager.start_tiff_path(png, tmp_path / 'out')
    with pytest.raises(ExtractionError):
        unet_manager.start_tiff_path(tmp_path / 'missing.tif', tmp_path / 'out')
    assert unet_manager.active_job() is None


def test_tiff_and_json_mode_still_uses_the_normal_pipeline(unet_manager, inputs, tmp_path):
    image, annotation, _ = inputs
    job_id = unet_manager.start_paths(image, annotation, tmp_path / 'out')
    assert finish(unet_manager, job_id).state == 'succeeded'
    request = json.loads((unet_manager.job_root / job_id / 'request.json').read_text())
    assert Path(request['command'][3]).name == 'fake pipeline.py'
    assert request['annotation'] == str(annotation) and request['box_source'] == 'supplied'


def test_real_unet_worker_script_exists():
    assert UNET_PIPELINE_SCRIPT.is_file()
    assert UNET_PIPELINE_SCRIPT.parent == PIPELINE_SCRIPT.parent


def test_output_file_is_not_treated_as_directory(manager, inputs, tmp_path):
    image, annotation, _ = inputs
    target = tmp_path / 'a file'; target.write_text('keep me')
    with pytest.raises(ExtractionError): manager.start_paths(image, annotation, target)
    assert target.read_text() == 'keep me'
