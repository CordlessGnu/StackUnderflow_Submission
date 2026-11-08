"""Local asynchronous bridge to the repository's TIFF + JSON pipeline.

No ML modules are imported into Streamlit. A single isolated Python process
writes the normal pipeline artifacts; GeoLens later ingests intervals.csv using
its existing analytics services. Paths and uploads are supported without a shell.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_SCRIPT = REPO_ROOT / 'info extraction/pipeline.py'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'crops/geolens_outputs'
DEFAULT_JOB_ROOT = REPO_ROOT / 'crops/geolens_jobs'
REQUIRED_REGIONS = ('interpreted_lithology_column', 'depth_column', 'lithology_legend')
REQUIRED_CSV = {'top_y', 'bottom_y', 'cluster_id', 'lithology', 'top_depth', 'base_depth'}
TERMINAL_STATES = {'succeeded', 'failed', 'cancelled'}


class ExtractionError(ValueError):
    """A user-facing extraction input/job error."""


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ExtractionOptions:
    matcher: str = 'hybrid'
    depth_unit: str = 'unspecified'
    depth_rounding: float = 10.0
    min_interval: int = 2
    window: int | None = None
    legend_manifest: str = ''
    calibration: str = ''

    def arguments(self) -> list[str]:
        if self.matcher not in ('hybrid', 'descriptor'):
            raise ExtractionError('Choose hybrid or descriptor matching.')
        if self.depth_unit not in ('unspecified', 'ft', 'm'):
            raise ExtractionError('Choose unspecified, ft, or m for the depth unit.')
        if not math.isfinite(self.depth_rounding) or self.depth_rounding < 0:
            raise ExtractionError('Depth rounding must be nonnegative (0 preserves full precision).')
        if not isinstance(self.min_interval, int) or self.min_interval < 1:
            raise ExtractionError('Minimum interval height must be a positive integer.')
        if self.window is not None and (not isinstance(self.window, int) or self.window < 4):
            raise ExtractionError('Texture window must be at least 4 pixels.')
        args = ['--matcher', self.matcher, '--depth-unit', self.depth_unit,
                '--depth-rounding', str(self.depth_rounding), '--min-interval', str(self.min_interval)]
        if self.window is not None:
            args += ['--window', str(self.window)]
        for flag, value in [('--legend-manifest', self.legend_manifest), ('--calibration', self.calibration)]:
            if value.strip():
                path = resolve_local_path(value)
                if not path.is_file():
                    raise ExtractionError(f'Optional input does not exist: {path}')
                args += [flag, str(path)]
        return args


def validate_inputs(tiff: Path, annotation: Path) -> dict:
    """Validate headers/boxes without decoding the full large TIFF raster."""
    if tiff.suffix.lower() not in ('.tif', '.tiff') or not tiff.is_file():
        raise ExtractionError('Select an existing .tif or .tiff document.')
    if annotation.suffix.lower() != '.json' or not annotation.is_file():
        raise ExtractionError('Select an existing bounding-box .json file.')
    if annotation.stat().st_size > 2 * 1024 * 1024:
        raise ExtractionError('The bounding-box JSON must be smaller than 2 MB.')
    try:
        data = json.loads(annotation.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f'Cannot read bounding-box JSON: {exc}') from exc
    boxes = data.get('bounding_boxes') if isinstance(data, dict) else None
    if not isinstance(boxes, dict) or any(key not in boxes for key in REQUIRED_REGIONS):
        raise ExtractionError('JSON needs bounding_boxes for: ' + ', '.join(REQUIRED_REGIONS))
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(tiff) as image:
            if image.format != 'TIFF':
                raise ExtractionError('The selected document is not TIFF content.')
            width, height = image.size
            if width * height > 1_000_000_000:
                raise ExtractionError('Document exceeds the pipeline limit of one billion pixels.')
            if image.getexif().get(274, 1) != 1:
                raise ExtractionError('Normalize the TIFF orientation and update its boxes first.')
    except (OSError, Image.UnidentifiedImageError) as exc:
        raise ExtractionError(f'Cannot open TIFF: {exc}') from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    for key, actual in [('width', width), ('height', height)]:
        if key in data and data[key] != actual:
            raise ExtractionError(f'JSON {key}={data[key]} does not match TIFF {key}={actual}.')
    for key in REQUIRED_REGIONS:
        box = boxes[key]
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            raise ExtractionError(f'{key} must be [left, top, right, bottom] with finite pixel coordinates.')
        left, top, right, bottom = map(round, box)
        if (right <= left or bottom <= top or min(width, right) <= max(0, left)
                or min(height, bottom) <= max(0, top)):
            raise ExtractionError(f'{key} has an empty, reversed, or out-of-image box.')
    return dict(width=width, height=height, bounding_boxes=boxes)


@dataclass
class JobSnapshot:
    job_id: str
    source_name: str
    output_dir: Path
    state: str
    stage: str
    progress: float
    message: str
    manifest: dict
    log_tail: str
    elapsed_seconds: float
    validation: dict

    @property
    def csv_path(self) -> Path:
        return self.output_dir / 'intervals.csv'


@dataclass
class _Job:
    job_id: str
    source_name: str
    output_dir: Path
    work_dir: Path
    validation: dict
    process: subprocess.Popen
    started: float = field(default_factory=time.monotonic)
    state: str = 'running'
    message: str = ''
    finished: float | None = None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}  # a concurrently written run.json may be incomplete for one poll


def _tail(path: Path) -> str:
    try:
        with path.open('rb') as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 12000))
            text = file.read().decode('utf-8', errors='replace')
        return re.sub(r'\x1b\[[0-9;]*m', '', text)
    except OSError:
        return ''


class ExtractionJobManager:
    """One active process at a time; safe across Streamlit sessions/reruns."""
    def __init__(self, *, job_root: Path = DEFAULT_JOB_ROOT, pipeline_script: Path = PIPELINE_SCRIPT,
                 python: str = sys.executable):
        self.job_root = Path(job_root).resolve()
        self.pipeline_script = Path(pipeline_script).resolve()
        self.python = python
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def _refresh(self, job: _Job) -> None:
        if job.state in TERMINAL_STATES:
            return
        code = job.process.poll()
        if code is None:
            return
        manifest = _read_json(job.output_dir / 'run.json')
        if code == 0 and manifest.get('status') in ('completed', 'completed_with_warnings'):
            try:
                with (job.output_dir / 'intervals.csv').open(newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader, [])
                    first_row = next(reader, None)
                if not REQUIRED_CSV.issubset(header):
                    raise ValueError('Required CSV columns are missing.')
                if first_row is None:
                    raise ValueError('CSV contains no intervals.')
                job.state = 'succeeded'
                job.message = 'Extraction finished; the CSV is ready for GeoLens analytics.'
            except (OSError, ValueError) as exc:
                job.state, job.message = 'failed', f'Pipeline did not produce a compatible CSV: {exc}'
        else:
            job.state = 'failed'
            job.message = str(manifest.get('error') or f'Pipeline exited with code {code}; inspect the log below.')
        job.finished = time.monotonic()
        self._save_job(job)

    def _save_job(self, job: _Job) -> None:
        path = job.work_dir / 'job.json'
        path.write_text(json.dumps(dict(job_id=job.job_id, source_name=job.source_name,
                                       output_dir=str(job.output_dir), state=job.state, message=job.message), indent=2))

    def active_job(self) -> str | None:
        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
                if job.state == 'running':
                    return job.job_id
        return None

    def _ensure_idle(self) -> None:
        if self.active_job() is not None:
            raise ExtractionError('An extraction is already running. Wait for it or cancel it first.')

    def start_paths(self, tiff: str | Path, annotation: str | Path, output_root: str | Path,
                    options: ExtractionOptions | None = None) -> str:
        with self._lock:
            self._ensure_idle()
            tiff, annotation = resolve_local_path(tiff), resolve_local_path(annotation)
            validation = validate_inputs(tiff, annotation)
            return self._launch(tiff, annotation, output_root, options or ExtractionOptions(), validation)

    def start_uploads(self, tiff_name: str, tiff_stream: BinaryIO,
                      annotation_name: str, annotation_stream: BinaryIO,
                      output_root: str | Path, options: ExtractionOptions | None = None) -> str:
        with self._lock:
            self._ensure_idle()
            self.job_root.mkdir(parents=True, exist_ok=True)
            staged = Path(tempfile.mkdtemp(prefix='upload_', dir=self.job_root))
            def stage(name, stream, destination):
                # Do not trust browser filenames as filesystem paths.
                basename = Path(name.replace('\\', '/')).name
                if not basename or basename in ('.', '..'):
                    raise ExtractionError('Invalid upload filename.')
                path = destination / basename
                destination.mkdir()
                position = stream.tell()
                stream.seek(0)
                try:
                    with path.open('wb') as file:
                        while chunk := stream.read(1024 * 1024):
                            file.write(chunk)
                finally:
                    stream.seek(position)
                return path
            try:
                tiff = stage(tiff_name, tiff_stream, staged / 'document')
                annotation = stage(annotation_name, annotation_stream, staged / 'annotation')
                validation = validate_inputs(tiff, annotation)
                return self._launch(tiff, annotation, output_root, options or ExtractionOptions(), validation)
            except Exception:
                # Only this method's own unsuccessful staging folder is removed.
                import shutil
                shutil.rmtree(staged)
                raise

    def _launch(self, tiff: Path, annotation: Path, output_root, options, validation) -> str:
        arguments = options.arguments()
        if not self.pipeline_script.is_file():
            raise ExtractionError(f'Pipeline entry point is missing: {self.pipeline_script}')
        if not str(output_root).strip():
            raise ExtractionError('Select an output directory.')
        output_root = resolve_local_path(output_root)
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=output_root):
                pass
        except OSError as exc:
            raise ExtractionError(f'Output directory is not writable: {exc}') from exc
        job_id = uuid.uuid4().hex
        safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', tiff.stem).strip('._')[:70] or 'document'
        output = output_root / f'{safe_stem}_{job_id[:8]}'
        work = self.job_root / job_id
        work.mkdir(parents=True)
        command = [self.python, '-u', '-B', str(self.pipeline_script), str(tiff), str(annotation),
                   '--output', str(output), *arguments]
        (work / 'request.json').write_text(json.dumps(dict(command=command, options=asdict(options),
                                                          tiff=str(tiff), annotation=str(annotation)), indent=2))
        with (work / 'worker.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != 'nt'),
                                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        job = _Job(job_id, tiff.name, output, work, validation, process)
        self._jobs[job_id] = job
        self._save_job(job)
        return job_id

    def snapshot(self, job_id: str) -> JobSnapshot:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionError('This extraction job is no longer available; load its CSV from the output folder.')
            job = self._jobs[job_id]
            self._refresh(job)
            manifest = _read_json(job.output_dir / 'run.json')
            stage = str(manifest.get('current_stage') or ('Finalising outputs' if manifest else 'Starting pipeline'))
            stages = manifest.get('stages', {})
            completed = sum(s.get('status') == 'complete' for s in stages.values() if isinstance(s, dict)) if isinstance(stages, dict) else 0
            progress = 1.0 if job.state == 'succeeded' else min(0.99, completed / 6)
            return JobSnapshot(job_id, job.source_name, job.output_dir, job.state, stage, progress,
                               job.message, manifest, _tail(job.work_dir / 'worker.log'),
                               round((job.finished or time.monotonic()) - job.started, 1), job.validation)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ExtractionError('Unknown extraction job.')
            self._refresh(job)
            if job.state != 'running':
                return
            try:
                if os.name == 'nt':
                    job.process.terminate()
                else:
                    os.killpg(job.process.pid, signal.SIGTERM)
                try:
                    job.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        job.process.kill()
                    else:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    job.process.wait(timeout=3)
            except ProcessLookupError:
                job.process.wait(timeout=3)
            job.state, job.message = 'cancelled', 'Cancelled. Partial outputs were retained; analytics were not loaded.'
            job.finished = time.monotonic()
            manifest = _read_json(job.output_dir / 'run.json')
            if manifest:
                manifest.update(status='cancelled', error='Cancelled from GeoLens.')
                (job.output_dir / 'run.json').write_text(json.dumps(manifest, indent=2))
            self._save_job(job)

    def shutdown(self) -> None:
        for job_id in list(self._jobs):
            self.cancel(job_id)
