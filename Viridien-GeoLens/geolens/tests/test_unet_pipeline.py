"""TIFF-only worker (info extraction/unet_pipeline.py): U-Net boxes -> unchanged pipeline.main.

Most tests replace the U-Net and pipeline.main with fakes so they run in milliseconds;
one test runs the real exported U-Net on a small synthetic page.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from geolens.services.extraction_service import REPO_ROOT, UNET_PIPELINE_SCRIPT, UNET_STAGE

BOXES = {'depth_column': [10, 40, 30, 290], 'interpreted_lithology_column': [40, 40, 70, 290],
         'lithology_legend': [0, 0, 80, 30]}


def load_worker():
    spec = importlib.util.spec_from_file_location('unet_pipeline_under_test', UNET_PIPELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(monkeypatch):
    module = load_worker()
    calls = SimpleNamespace(detect=[], pipeline=[])

    def fake_detect(tiff, weights, snap=True):
        calls.detect.append((tiff, weights, snap))
        return dict(boxes=dict(BOXES), raw_boxes=dict(BOXES), probs=None, width=80, height=300,
                    page=np.full((288, 768, 3), 255, np.uint8), meta={'version': 'v0.2', 'snap_mode': 'xyL'},
                    device='cpu')
    monkeypatch.setattr(module, 'detect_regions', fake_detect)
    return module, calls


@pytest.fixture
def fake_pipeline(monkeypatch, worker):
    """Stands in for pipeline.main: records argv, checks the bbox JSON, writes run.json like run_pipeline."""
    _, calls = worker
    behaviour = {'fail': False}

    def main():
        argv = list(sys.argv)
        calls.pipeline.append(argv)
        tiff, annotation = Path(argv[1]), Path(argv[2])
        output = Path(argv[argv.index('--output') + 1])
        data = json.loads(annotation.read_text())
        assert data['bounding_boxes'] == BOXES and data['width'] == 80 and data['height'] == 300
        assert data['image_name'] == tiff.name and data['source'] == 'unet'
        output.mkdir(parents=True)
        manifest = dict(status='completed', inputs=dict(tiff=str(tiff.resolve()), annotation=str(annotation.resolve())),
                        stages={'1/6 crop TIFF': {'status': 'complete'}}, warnings=[])
        if behaviour['fail']:
            manifest.update(status='failed', error='ValueError: synthetic')
        (output / 'run.json').write_text(json.dumps(manifest))
        if behaviour['fail']:
            raise SystemExit(1)
    monkeypatch.setitem(sys.modules, 'pipeline', SimpleNamespace(main=main))
    return behaviour


@pytest.fixture
def tiff(tmp_path):
    path = tmp_path / 'page one.tif'
    Image.new('RGB', (80, 300), 'white').save(path, compression='tiff_lzw')
    return path


def test_predicted_boxes_feed_the_unchanged_pipeline(worker, fake_pipeline, tiff, tmp_path):
    module, calls = worker
    output = tmp_path / 'run'
    module.main([str(tiff), '--output', str(output), '--depth-rounding', '0', '--matcher', 'descriptor'])
    argv = calls.pipeline[0]
    assert Path(argv[0]).name == 'pipeline.py'
    assert argv[1] == str(tiff.resolve())
    assert argv[argv.index('--depth-rounding') + 1] == '0'  # pipeline options pass straight through
    assert argv[argv.index('--matcher') + 1] == 'descriptor'
    assert not Path(argv[2]).exists()  # the temporary bbox JSON is cleaned up afterwards
    assert calls.detect[0][2] is True  # edge snapping on by default
    manifest = json.loads((output / 'run.json').read_text())
    source = manifest['box_source']
    assert source['method'] == 'unet' and source['version'] == 'v0.2' and source['snap_mode'] == 'xyL'
    assert source['bounding_boxes'] == BOXES
    assert list(manifest['stages'])[0] == UNET_STAGE
    with Image.open(output / 'predicted_boxes.png') as preview:
        assert preview.height <= 1600


def test_no_snap_flag_and_custom_model(worker, fake_pipeline, tiff, tmp_path):
    module, calls = worker
    model = tmp_path / 'other.pt'; model.write_bytes(b'x')
    module.main([str(tiff), '--output', str(tmp_path / 'run'), '--no-snap', '--unet-model', str(model)])
    assert calls.detect[0][1] == model.resolve() and calls.detect[0][2] is False
    manifest = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert manifest['box_source']['snap_mode'] is None


def test_pipeline_failure_still_records_boxes_and_exits_nonzero(worker, fake_pipeline, tiff, tmp_path):
    module, _ = worker
    fake_pipeline['fail'] = True
    with pytest.raises(SystemExit) as exit_info:
        module.main([str(tiff), '--output', str(tmp_path / 'run')])
    assert exit_info.value.code == 1
    manifest = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['box_source']['bounding_boxes'] == BOXES


def test_unet_failure_is_reported_in_run_json(worker, fake_pipeline, tiff, tmp_path, monkeypatch):
    module, calls = worker
    def broken(*args, **kwargs): raise RuntimeError('out of memory')
    monkeypatch.setattr(module, 'detect_regions', broken)
    with pytest.raises(SystemExit) as exit_info:
        module.main([str(tiff), '--output', str(tmp_path / 'run')])
    assert exit_info.value.code == 1
    manifest = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert manifest['status'] == 'failed' and 'out of memory' in manifest['error']
    assert calls.pipeline == []


def test_non_empty_output_is_refused_before_running_the_unet(worker, fake_pipeline, tiff, tmp_path):
    module, calls = worker
    output = tmp_path / 'run'; output.mkdir(); (output / 'keep.txt').write_text('keep me')
    with pytest.raises(SystemExit):
        module.main([str(tiff), '--output', str(output)])
    assert calls.detect == [] and calls.pipeline == []
    assert sorted(p.name for p in output.iterdir()) == ['keep.txt']


@pytest.mark.parametrize('name', ['page.png', 'missing.tif'])
def test_bad_tiff_paths_fail_cleanly(worker, fake_pipeline, tmp_path, name):
    module, calls = worker
    path = tmp_path / name
    if name.endswith('.png'):
        Image.new('RGB', (10, 10)).save(path)
    with pytest.raises(SystemExit):
        module.main([str(path), '--output', str(tmp_path / 'run')])
    assert calls.detect == []


def test_record_ignores_a_run_json_from_another_run(worker, tmp_path):
    module, _ = worker
    output = tmp_path / 'run'; output.mkdir()
    other = dict(status='completed', inputs=dict(annotation=str(tmp_path / 'someone else.json')))
    (output / 'run.json').write_text(json.dumps(other))
    detection = dict(boxes=BOXES, raw_boxes=BOXES, width=80, height=300, meta={}, device='cpu', snapped=True)
    module._record(output, tmp_path / 'ours.json', detection, Path('m.pt'), 1.0, None)
    assert json.loads((output / 'run.json').read_text()) == other


def test_real_unet_returns_valid_boxes_on_a_synthetic_page(tmp_path):
    """Runs the exported U-Net (CPU is fine). Checks shape/validity only, not accuracy."""
    pytest.importorskip('torch'); pytest.importorskip('cv2'); pytest.importorskip('tifffile')
    weights = REPO_ROOT / 'work/export/boxfinder_v0.2.pt'
    if not weights.is_file():
        pytest.skip('exported U-Net not present')
    page = np.full((3000, 600, 3), 255, np.uint8)
    page[:400, 20:580] = (230, 230, 230)                       # header block
    for x in (100, 160, 300, 380):                             # ruled column lines
        page[450:, x:x + 2] = 0
    page[450:, 302:380][::40] = (90, 170, 90)                  # coloured stripes in one column
    path = tmp_path / 'synthetic.tif'
    Image.fromarray(page).save(path, compression='tiff_lzw')
    module = load_worker()
    detection = module.detect_regions(path, weights)
    assert (detection['width'], detection['height']) == (600, 3000)
    for key, (x1, y1, x2, y2) in detection['boxes'].items():
        assert 0 <= x1 < x2 <= 600 and 0 <= y1 < y2 <= 3000, key
    assert not (REPO_ROOT / 'work/cache/synthetic.png').exists()  # TIFF-only runs never touch the cache
    assert module.preview_image(detection).shape[2] == 3
