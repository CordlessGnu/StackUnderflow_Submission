"""End-to-end lithology extraction from TWO inputs: a TIFF and its bbox JSON.

Example:
  python 'info extraction/pipeline.py' document.tif annotation.json --output crops/run

Native-resolution crops -> RGB boundaries -> visual IDs/legend names -> OCR depth
calibration -> final CSV/JSON and annotated column. No pre-generated crops needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from depth_distance import calibrate_depth_column, load_calibration, save_calibration
from final_output import CropAlignment, attach_depths, write_final
from interval_lithology import (_open_rgb, extract_legend, label_intervals,
                                load_swatch_manifest, save_swatch_manifest, write_results)
from legend_utils import LegendExtractionError
from texture_boundaries import detect_boundaries

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('interpreted_lithology_column', 'depth_column', 'lithology_legend')


@dataclass
class PipelineConfig:
    model: Path = ROOT / 'models/swatch_detection.pt'
    legend_manifest: Path | None = None
    calibration: Path | None = None
    window: int | None = None
    min_interval: int | None = None
    boundary_threshold: float = 0.16
    color_threshold: float = 0.10
    refine_radius: int | None = None
    margin: float = 0.12
    cluster_threshold: float = 0.20
    max_match_distance: float = 0.32
    min_match_margin: float = 0.06
    matcher: str = 'hybrid'
    shortlist_size: int = 3
    max_representatives: int = 3
    seam_threshold: float = 0.20
    depth_samples: int = 12
    depth_rotation: str = 'auto'
    depth_label_mode: str = 'auto'
    depth_unit: str = 'unspecified'
    depth_rounding: float = 10
    coordinate_system: str = 'document'
    strict_legend: bool = False
    page: int = 0
    max_pixels: int = 1_000_000_000
    overwrite: bool = False

    def __post_init__(self):
        if self.coordinate_system not in ('document', 'crop'):
            raise ValueError('coordinate_system must be document or crop')
        if self.page < 0 or self.max_pixels <= 0 or self.depth_samples < 2:
            raise ValueError('Need page >= 0, max_pixels > 0 and depth_samples >= 2')
        if self.depth_rotation not in ('auto', '0', '90', '270'):
            raise ValueError('Invalid depth rotation')
        if self.depth_label_mode not in ('auto', 'integer', 'decimal', 'all'):
            raise ValueError('Invalid depth label mode')
        if not math.isfinite(self.depth_rounding) or self.depth_rounding < 0:
            raise ValueError('depth_rounding must be finite and nonnegative')
        if not 0 <= self.margin < 0.45:
            raise ValueError('margin must be in [0, 0.45)')
        if self.matcher not in ('hybrid', 'descriptor'):
            raise ValueError('matcher must be hybrid or descriptor')
        if not isinstance(self.shortlist_size, int) or not 1 <= self.shortlist_size <= 10:
            raise ValueError('shortlist_size must be in [1,10]')
        if not isinstance(self.max_representatives, int) or not 1 <= self.max_representatives <= 5:
            raise ValueError('max_representatives must be in [1,5]')
        for value in (self.boundary_threshold, self.color_threshold, self.cluster_threshold,
                      self.max_match_distance, self.seam_threshold):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Detection/matching thresholds must be finite and positive')
        if not math.isfinite(self.min_match_margin) or self.min_match_margin < 0:
            raise ValueError('min_match_margin must be finite and nonnegative')
        if self.window is not None and (not isinstance(self.window, int) or self.window < 4):
            raise ValueError('window must be an integer >= 4')
        if self.min_interval is not None and (not isinstance(self.min_interval, int) or self.min_interval < 1):
            raise ValueError('min_interval must be a positive integer')
        if self.refine_radius is not None and (not isinstance(self.refine_radius, int) or self.refine_radius < 0):
            raise ValueError('refine_radius must be a nonnegative integer')


@dataclass
class DocumentCrops:
    paths: dict[str, Path]
    boxes: dict[str, tuple[int, int, int, int]]
    image_size: tuple[int, int]
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    output: Path
    interval_count: int
    cluster_count: int
    matched_intervals: int
    warnings: list[str]
    fallback_intervals: int = 0
    blank_intervals: int = 0

    @property
    def csv_path(self) -> Path:
        return self.output / 'intervals.csv'


def crop_document(image_path: str | Path, annotation_path: str | Path, output: str | Path,
                  *, page: int = 0, max_pixels: int = 1_000_000_000) -> DocumentCrops:
    """Read this TIFF once, save the three crops, then release the full raster.

    Pillow's global pixel guard is relaxed ONLY while handling this explicitly
    supplied TIFF and is restored even on failure. Validate a separate size limit
    before decoding. Native pixels are retained; no downsampling or resizing.
    """
    image_path, annotation_path, output = map(Path, (image_path, annotation_path, output))
    if image_path.suffix.lower() not in ('.tif', '.tiff'):
        raise ValueError('The document must be a .tif or .tiff file')
    if page < 0 or max_pixels <= 0:
        raise ValueError('page and max_pixels are invalid')
    data = json.loads(annotation_path.read_text())
    boxes = data.get('bounding_boxes')
    if not isinstance(boxes, dict) or any(key not in boxes for key in CLASSES):
        raise ValueError('Annotation needs bounding_boxes for ' + ', '.join(CLASSES))
    warnings = []
    if data.get('image_name') and Path(data['image_name']).stem != image_path.stem:
        warnings.append('Annotation image_name differs from the supplied TIFF name; dimensions are checked.')
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(image_path) as image:
            if image.format != 'TIFF':
                raise ValueError('The document contents are not TIFF')
            if page >= getattr(image, 'n_frames', 1):
                raise ValueError(f'TIFF has no page {page}')
            image.seek(page)
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError(f'TIFF has {width * height:,} pixels, exceeding max_pixels={max_pixels:,}')
            for key, actual in [('width', width), ('height', height)]:
                if key in data and data[key] != actual:
                    raise ValueError(f'Annotation {key}={data[key]} does not match TIFF {key}={actual}')
            if image.getexif().get(274, 1) != 1:
                raise ValueError('Normalize TIFF orientation and update bounding boxes before extraction')
            if getattr(image, 'n_frames', 1) > 1:
                warnings.append(f'Multi-page TIFF: processing only explicitly selected page {page}.')
            used = {}
            for key in CLASSES:
                box = boxes[key]
                if (not isinstance(box, (list, tuple)) or len(box) != 4
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
                    raise ValueError(f'{key} needs four finite pixel coordinates [x1,y1,x2,y2]')
                left, top, right, bottom = map(round, box)
                if right <= left or bottom <= top:
                    raise ValueError(f'{key} has an empty or reversed bounding box')
                clipped = (max(0, left), max(0, top), min(width, right), min(height, bottom))
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    raise ValueError(f'{key} lies outside the TIFF')
                if clipped != (left, top, right, bottom):
                    warnings.append(f'{key} bounding box clipped to image bounds: {clipped}')
                used[key] = clipped
            output.mkdir(parents=True, exist_ok=True)
            # This can be a large decode. Do it once, before loading YOLO/OCR,
            # and keep no reference to the full image after returning.
            image.load()
            paths = {}
            for key, box in used.items():
                path = output / f'{key}.png'
                with image.crop(box) as region:
                    with region.convert('RGB') as rgb:
                        rgb.save(path)
                paths[key] = path
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    return DocumentCrops(paths, used, (width, height), warnings)


def _prepare_output(output: Path, protected: list[Path], overwrite: bool) -> Path | None:
    for source in protected:
        if source == output or output in source.parents:
            raise ValueError('Output folder must not contain input/model/cache files')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not overwrite:
            raise FileExistsError(f'Output is not empty: {output}; choose another folder or use --overwrite')
        index = 1
        archive = output.with_name(f'{output.name}.previous-{index}')
        while archive.exists():
            index += 1
            archive = output.with_name(f'{output.name}.previous-{index}')
        output.rename(archive)  # preserve the previous run; never delete it
    else:
        archive = None
    output.mkdir(parents=True, exist_ok=True)
    return archive


def run_pipeline(image_path: str | Path, annotation_path: str | Path,
                 output: str | Path | None = None, *, config: PipelineConfig | None = None,
                 progress: Callable[[str], None] | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    image_path, annotation_path = Path(image_path).resolve(), Path(annotation_path).resolve()
    if not image_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError('Both the TIFF and bounding-box JSON must exist')
    output = (Path(output) if output is not None else ROOT / 'crops/pipeline_output' / image_path.stem).resolve()
    protected = [image_path, annotation_path, Path(config.model).resolve()]
    protected += [Path(p).resolve() for p in (config.legend_manifest, config.calibration) if p is not None]
    previous = _prepare_output(output, protected, config.overwrite)
    started = time.perf_counter()
    manifest = dict(status='running', inputs=dict(tiff=str(image_path), annotation=str(annotation_path),
                    annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest()),
                    config=asdict(config), stages={}, warnings=[], previous_run=str(previous) if previous else None)
    manifest_path = output / 'run.json'
    warnings = manifest['warnings']

    def save_manifest():
        manifest['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def stage(name, action):
        manifest['current_stage'] = name
        save_manifest()
        if progress:
            progress(name)
        stage_start = time.perf_counter()
        try:
            result = action()
        except Exception:
            manifest['stages'][name] = dict(status='failed', seconds=round(time.perf_counter() - stage_start, 3))
            raise
        manifest['stages'][name] = dict(status='complete', seconds=round(time.perf_counter() - stage_start, 3))
        return result

    save_manifest()
    try:
        shutil.copy2(annotation_path, output / 'annotation.json')
        crops = stage('1/6 crop TIFF', lambda: crop_document(image_path, annotation_path, output / 'crops',
                                                            page=config.page, max_pixels=config.max_pixels))
        warnings.extend(crops.warnings)
        manifest['image_size'] = crops.image_size
        manifest['crops'] = {key: dict(path=str(path.relative_to(output)), bbox=crops.boxes[key])
                             for key, path in crops.paths.items()}
        column = _open_rgb(crops.paths['interpreted_lithology_column'])
        alignment = CropAlignment(column_top=crops.boxes['interpreted_lithology_column'][1],
                                  depth_top=crops.boxes['depth_column'][1], source=str(annotation_path))
        boundaries = stage('2/6 RGB interval boundaries', lambda: detect_boundaries(
            column, window=config.window, min_interval=config.min_interval,
            threshold=config.boundary_threshold, margin=config.margin,
            color_threshold=config.color_threshold, refine_radius=config.refine_radius))
        np.save(output / 'boundary_scores.npy', boundaries.scores)
        np.save(output / 'boundary_color_scores.npy', boundaries.color_scores)
        manifest['boundary_scales'] = boundaries.scales
        manifest['effective_min_interval'] = boundaries.min_interval
        manifest['effective_refine_radius'] = boundaries.refine_radius

        def legend_stage():
            if config.legend_manifest is not None:
                swatches = load_swatch_manifest(config.legend_manifest)
                details = json.loads(Path(config.legend_manifest).read_text())
                warnings.extend(details.get('warnings', []))
                return swatches
            try:
                swatches, notes = extract_legend(crops.paths['lithology_legend'], config.model)
            except LegendExtractionError as exc:
                if config.strict_legend:
                    raise
                warnings.append(f'Legend unavailable: {exc}; all lithology names will remain unknown.')
                return []
            warnings.extend(notes)
            return swatches

        swatches = stage('3/6 RGB legend swatches and OCR', legend_stage)
        if swatches:
            save_swatch_manifest(swatches, output / 'legend', warnings)
        manifest['usable_legend_swatches'] = len(swatches)
        labelled = stage('4/6 visual clusters and lithology', lambda: label_intervals(
            column, boundaries, swatches, cluster_threshold=config.cluster_threshold,
            max_match_distance=config.max_match_distance, min_match_margin=config.min_match_margin,
            margin=config.margin, matcher=config.matcher, shortlist_size=config.shortlist_size,
            max_representatives=config.max_representatives, seam_threshold=config.seam_threshold))
        warnings.extend(labelled.warnings)
        manifest['matching_diagnostics'] = labelled.diagnostics
        write_results(labelled, column, output / 'classification', 'intervals', swatches)

        depth_path = crops.paths['depth_column']
        calibration = stage('5/6 depth OCR and calibration', lambda:
                            load_calibration(config.calibration) if config.calibration else
                            calibrate_depth_column(depth_path, samples=config.depth_samples,
                                                   rotation=config.depth_rotation, label_mode=config.depth_label_mode))
        with Image.open(depth_path) as depth:
            if depth.height != calibration.image_height:
                raise ValueError('Cached calibration resolution does not match the new native-resolution depth crop')
        warnings.extend(calibration.warnings)
        rows = attach_depths([asdict(row) for row in labelled.intervals], calibration, alignment,
                             column.height, depth_unit=config.depth_unit,
                             depth_rounding=config.depth_rounding)
        extrapolated = sum(row['depth_extrapolated'] for row in rows)
        if extrapolated:
            warnings.append(f'{extrapolated} intervals require flagged depth extrapolation.')

        def export():
            write_final(rows, column, calibration, alignment, output, 'intervals', warnings=warnings,
                        clusters=labelled.clusters, coordinate_system=config.coordinate_system,
                        matching_diagnostics=labelled.diagnostics, depth_rounding=config.depth_rounding,
                        visual_clusters=labelled.visual_clusters)
                        
            save_calibration(calibration, depth_path, output / 'depth_calibration')

        stage('6/6 final CSV, JSON and annotated column', export)
        matched = sum(row['match_status'] == 'matched' for row in rows)
        fallback_count = sum(row['lithology_fallback'] for row in rows)
        unknown_count = sum(row['lithology'].strip().casefold() == 'unknown' for row in rows)
        blank_count = sum(str(row['match_status']).strip().casefold() == 'blank' for row in rows)
        manifest.update(status='completed_with_warnings' if warnings else 'completed',
                        interval_count=len(rows), cluster_count=len(labelled.clusters),
                        visual_cluster_count=len(labelled.visual_clusters), cluster_id_policy='final_lithology',
                        named_intervals=len(rows) - unknown_count - blank_count, verified_intervals=matched,
                        fallback_intervals=fallback_count, blank_intervals=blank_count, unknown_intervals=unknown_count,
                        depth_anchor_count=len(calibration.anchors), depth_fit_rmse_px=calibration.rmse_px,
                        coordinate_system=config.coordinate_system,
                        artifacts=dict(csv='intervals.csv', json='intervals.json', annotated_column='intervals.png',
                                       calibration='depth_calibration.json', classification='classification/intervals.json'))
        manifest.pop('current_stage', None)
        save_manifest()
        return PipelineResult(output, len(rows), len(labelled.clusters), matched, list(warnings), fallback_count, blank_count)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_manifest()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tiff', type=Path)
    parser.add_argument('annotation', type=Path)
    parser.add_argument('--output', type=Path, help='run directory; default crops/pipeline_output/<TIFF stem>')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/swatch_detection.pt')
    parser.add_argument('--legend-manifest', type=Path, help='optional reviewed swatches instead of YOLO/OCR')
    parser.add_argument('--calibration', type=Path, help='optional inspected native-resolution depth calibration')
    parser.add_argument('--coordinate-system', choices=['document', 'crop'], default='document')
    parser.add_argument('--depth-unit', default='unspecified')
    parser.add_argument('--depth-rounding', type=float, default=10,
                        help='round exported depths to the nearest N units; 0 keeps full precision')
    parser.add_argument('--depth-samples', type=int, default=12)
    parser.add_argument('--depth-rotation', choices=['auto', '0', '90', '270'], default='auto')
    parser.add_argument('--depth-label-mode', choices=['auto', 'integer', 'decimal', 'all'], default='auto')
    parser.add_argument('--window', type=int)
    parser.add_argument('--min-interval', type=int)
    parser.add_argument('--boundary-threshold', type=float, default=0.16)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--color-threshold', type=float, default=0.10)
    parser.add_argument('--refine-radius', type=int)
    parser.add_argument('--cluster-threshold', type=float, default=0.20)
    parser.add_argument('--max-match-distance', type=float, default=0.32)
    parser.add_argument('--min-match-margin', type=float, default=0.06)
    parser.add_argument('--matcher', choices=['hybrid', 'descriptor'], default='hybrid')
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--max-representatives', type=int, default=3)
    parser.add_argument('--seam-threshold', type=float, default=0.20)
    parser.add_argument('--strict-legend', action='store_true', help='fail instead of outputting unknown if no usable legend is detected')
    parser.add_argument('--page', type=int, default=0, help='zero-based TIFF page, default 0')
    parser.add_argument('--max-pixels', type=int, default=1_000_000_000)
    parser.add_argument('--overwrite', action='store_true', help='archive the previous output folder and start a new run')
    args = parser.parse_args()
    values = vars(args).copy()
    image, annotation, output = values.pop('tiff'), values.pop('annotation'), values.pop('output')
    try:
        result = run_pipeline(image, annotation, output, config=PipelineConfig(**values), progress=print)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Pipeline failed: {exc}\nInspect run.json in the output folder if it was created.\n')
    print(f'{result.interval_count} intervals -> {result.cluster_count} clusters; '
          f'{result.matched_intervals} verified; {result.fallback_intervals} runner-up fallbacks; '
          f'{result.blank_intervals} blank; '
          f'{result.interval_count - result.matched_intervals - result.fallback_intervals - result.blank_intervals} unknown; '
          f'output: {result.output}')
    for warning in result.warnings:
        print(f'Warning: {warning}')


if __name__ == '__main__':
    main()
