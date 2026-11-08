"""Join labelled lithology intervals with OCR-calibrated depth values.

Reuses interval_lithology.py and depth_distance.py. Accepts existing labelled CSV
results or runs the RGB boundary/legend pipeline. Crop origin/scale alignment is
read from the document annotation, explicitly supplied, or explicitly assumed.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from depth_distance import (DepthCalibration, calibrate_depth_column,
                             load_calibration, save_calibration)
from interval_lithology import (apply_runner_up_fallback, _open_rgb, extract_legend, label_intervals,
                                load_swatch_manifest, save_swatch_manifest)
from texture_boundaries import detect_boundaries

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['top_y', 'bottom_y', 'cluster_id', 'lithology', 'top_depth', 'base_depth']


@dataclass
class CropAlignment:
    column_top: float = 0
    depth_top: float = 0
    column_scale: float = 1
    depth_scale: float = 1
    source: str = 'explicit'
    document_coordinates_known: bool = True

    def __post_init__(self):
        values = (self.column_top, self.depth_top, self.column_scale, self.depth_scale)
        if not all(math.isfinite(v) for v in values) or min(self.column_scale, self.depth_scale) <= 0:
            raise ValueError('Crop origins must be finite and scales must be finite and positive')

    def document_y(self, column_y: float) -> float:
        return self.column_top + column_y / self.column_scale

    def depth_y(self, column_y: float) -> float:
        return (self.document_y(column_y) - self.depth_top) * self.depth_scale


def find_annotation(column: str | Path, directory: str | Path) -> Path | None:
    """Resolve original versus sanitized/truncated crop filenames, uniquely."""
    column, directory = Path(column), Path(directory)
    exact = directory / f'{column.stem}.json'
    if exact.is_file():
        return exact
    normalize = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    wanted = normalize(column.stem)
    matches = []
    for candidate in directory.glob('*.json'):
        data = json.loads(candidate.read_text())
        names = [candidate.stem, Path(data.get('image_name', candidate.stem)).stem]
        signatures = set()
        for name in names:
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')[:80]
            signatures.update((normalize(name), normalize(safe)))
        if wanted in signatures or (len(column.stem) >= 79 and len(wanted) >= 48
                                    and any(s.startswith(wanted) for s in signatures)):
            matches.append(candidate)
    if len(matches) > 1:
        raise ValueError('Ambiguous annotation filename match; supply --annotation explicitly')
    return matches[0] if matches else None


def alignment_from_annotation(path: str | Path, column_size: tuple[int, int],
                              depth_size: tuple[int, int]) -> CropAlignment:
    """Infer source document origins and uniform resize scales from bboxes."""
    path = Path(path)
    data = json.loads(path.read_text())
    boxes = data['bounding_boxes']
    geometries = []
    for key, size in [('interpreted_lithology_column', column_size), ('depth_column', depth_size)]:
        box = boxes[key]
        left, right = sorted((round(box[0]), round(box[2])))
        top, bottom = sorted((round(box[1]), round(box[3])))
        left, right = max(0, left), min(data['width'], right)
        top, bottom = max(0, top), min(data['height'], bottom)
        if right <= left or bottom <= top:
            raise ValueError(f'Invalid {key} bounding box')
        scale = size[1] / (bottom - top)
        if abs(size[0] - (right - left) * scale) > max(2, size[0] * 0.02):
            raise ValueError(f'{key} crop dimensions disagree with annotation; '
                             'supply explicit origins/scales for padded or stretched crops')
        geometries.append((top, scale))
    return CropAlignment(geometries[0][0], geometries[1][0],
                         geometries[0][1], geometries[1][1], source=str(path))


def read_labelled_csv(path: str | Path) -> list[dict]:
    """Keep extra confidence/diagnostic columns instead of discarding them."""
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'top_y', 'bottom_y', 'cluster_id', 'lithology'}.issubset(reader.fieldnames or []):
            raise ValueError('Labelled CSV needs top_y,bottom_y,cluster_id,lithology')
        rows = list(reader)
    for row in rows:
        row['top_y'], row['bottom_y'] = int(row['top_y']), int(row['bottom_y'])
        blank = str(row.get('match_status', '')).strip().casefold() == 'blank'
        if not row['cluster_id'] or (not row['lithology'] and not blank):
            raise ValueError('Every interval needs a cluster_id and lithology; empty labels are allowed for blank status')
    return rows


def round_depth(value: float, step: float) -> float:
    """Half-up rounding to a multiple of step; step <= 0 disables rounding."""
    if not step or step <= 0:
        return value
    rounded = math.floor(value / step + 0.5) * step
    return int(rounded) if float(rounded).is_integer() else rounded


def format_depth(value: float) -> str:
    """Render a depth without trailing zeros, e.g. 2350 or 11879.5."""
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def attach_depths(rows: Sequence[dict], calibration: DepthCalibration,
                   alignment: CropAlignment, column_height: int, *,
                   depth_unit: str = 'unspecified', depth_rounding: float = 10) -> list[dict]:
    """Preserve crop-local interval edges; compute depths in depth-crop space.

    top_depth/base_depth are rounded to the nearest depth_rounding units for the
    exported table. The unrounded values are kept in top_depth_exact/base_depth_exact
    so intervals.json stays auditable against the fit residual.
    """
    if not math.isfinite(depth_rounding) or depth_rounding < 0:
        raise ValueError('depth_rounding must be finite and nonnegative')
    output = []
    previous_end = 0
    for original in rows:
        row = apply_runner_up_fallback(original)
        top, bottom = row['top_y'], row['bottom_y']
        if (not isinstance(top, (int, np.integer)) or not isinstance(bottom, (int, np.integer))
                or not 0 <= top < bottom <= column_height or top < previous_end):
            raise ValueError('Intervals must be sorted, non-overlapping integer ranges within the column')
        previous_end = bottom
        depth_top_y, depth_bottom_y = alignment.depth_y(top), alignment.depth_y(bottom)
        exact_top, exact_bottom = calibration.depth_at(depth_top_y), calibration.depth_at(depth_bottom_y)
        row.update(top_depth=round_depth(exact_top, depth_rounding),
                   base_depth=round_depth(exact_bottom, depth_rounding),
                   top_depth_exact=round(exact_top, 6),
                   base_depth_exact=round(exact_bottom, 6),
                   depth_unit=depth_unit,
                   depth_extrapolated=bool(depth_top_y < calibration.min_y or depth_bottom_y > calibration.max_y),
                   document_top_y=(round(alignment.document_y(top), 6)
                                   if alignment.document_coordinates_known else None),
                   document_bottom_y=(round(alignment.document_y(bottom), 6)
                                      if alignment.document_coordinates_known else None))
        output.append(row)
    return output


def write_final(rows: Sequence[dict], image: Image.Image, calibration: DepthCalibration,
                alignment: CropAlignment, output: str | Path, stem: str, *,
                warnings: Sequence[str] = (), clusters: Sequence[dict] = (),
                coordinate_system: str = 'crop', matching_diagnostics: dict | None = None, depth_rounding: float = 10) -> None:
    if coordinate_system not in ('crop', 'document'):
        raise ValueError('coordinate_system must be crop or document')
    if coordinate_system == 'document' and not alignment.document_coordinates_known:
        raise ValueError('Document coordinates require known crop geometry')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    local_rows = [apply_runner_up_fallback(row) for row in rows]
    records = [dict(row) for row in local_rows]
    if coordinate_system == 'document':
        for row in records:
            row['crop_top_y'], row['crop_bottom_y'] = row['top_y'], row['bottom_y']
            for key in ('top_y', 'bottom_y'):
                value = alignment.document_y(row[key])
                row[key] = int(value) if float(value).is_integer() else value
    # The CSV carries the rounded table; the JSON keeps the unrounded fit values.
    json_records = [dict(row, top_depth=row.get('top_depth_exact', row['top_depth']),
                         base_depth=row.get('base_depth_exact', row['base_depth']))
                    for row in records]
    extra = list(dict.fromkeys(key for row in records for key in row if key not in REQUIRED))
    with (output / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED + extra)
        writer.writeheader()
        writer.writerows(records)
    diagnostics = dict(matching_diagnostics or {})
    diagnostics['runner_up_fallback_intervals'] = sum(r['lithology_fallback'] for r in local_rows)
    data = dict(intervals=records, clusters=[apply_runner_up_fallback(c) for c in clusters],
                calibration=calibration.to_dict(),
                alignment=asdict(alignment), warnings=list(warnings),
                matching_diagnostics=diagnostics,
                coordinate_system=('top_y/bottom_y are original-document half-open pixel edges; '
                                   'crop_*_y are local to the interpreted-lithology crop'
                                   if coordinate_system == 'document' else
                                   'top_y/bottom_y are crop-local half-open pixel edges; '
                                   'document_*_y are original-image edges when geometry is known'),
                depth_rounding=depth_rounding)
    (output / f'{stem}.json').write_text(json.dumps(data, indent=2))
    overlay = Image.new('RGB', (image.width + 400, image.height), 'white')
    overlay.paste(image, (0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in local_rows:
        cluster_id = str(row['cluster_id'])
        index = (int(cluster_id[1:]) if cluster_id.startswith('C') and cluster_id[1:].isdigit()
                 else int(hashlib.sha256(cluster_id.encode()).hexdigest()[:8], 16))
        color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.7, 0.75))
        top, bottom = row['top_y'], row['bottom_y']
        draw.line((0, top, overlay.width, top), fill=color, width=2)
        draw.rectangle((image.width, top, image.width + 8, bottom - 1), fill=color)
        if bottom - top >= 14:
            suffix = ' [runner-up]' if row['lithology_fallback'] else ''
            draw.text((image.width + 15, top + 2), f"{cluster_id}  {row['lithology']}{suffix}", fill=color)
        if bottom - top >= 28:
            extrapolated = ' [extrapolated]' if row['depth_extrapolated'] else ''
            draw.text((image.width + 15, top + 15),
                      f"{format_depth(row['top_depth'])} - {format_depth(row['base_depth'])} "
                      f"{row['depth_unit']}{extrapolated}", fill=color)
    overlay.save(output / f'{stem}.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('column', type=Path, help='interpreted-lithology crop')
    parser.add_argument('--labelled-csv', type=Path, help='reuse interval_lithology output, including diagnostics')
    parser.add_argument('--depth-column', type=Path, help='defaults to crops/depth_column/<column filename>')
    parser.add_argument('--calibration', type=Path, help='reuse an inspected depth_distance calibration JSON')
    parser.add_argument('--annotation', type=Path, help='document bbox JSON; looked up by column stem by default')
    parser.add_argument('--column-top', type=float, help='explicit column crop origin in original-image pixels')
    parser.add_argument('--depth-top', type=float, help='explicit depth crop origin in original-image pixels')
    parser.add_argument('--column-scale', type=float, default=1, help='column pixels per original-image pixel')
    parser.add_argument('--depth-scale', type=float, default=1, help='depth pixels per original-image pixel')
    parser.add_argument('--assume-aligned', action='store_true', help='explicitly assume crops have identical vertical coordinates')
    parser.add_argument('--depth-unit', default='unspecified', help='e.g. ft or m; never inferred from the numeric labels')
    parser.add_argument('--depth-rounding', type=float, default=10,
                        help='round exported depths to the nearest N units; 0 keeps full precision')
    parser.add_argument('--depth-samples', type=int, default=12)
    parser.add_argument('--depth-rotation', choices=['auto', '0', '90', '270'], default='auto')
    parser.add_argument('--depth-label-mode', choices=['auto', 'integer', 'decimal', 'all'], default='auto')
    legend = parser.add_mutually_exclusive_group()
    legend.add_argument('--legend', type=Path)
    legend.add_argument('--legend-manifest', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/swatch_detection.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'crops/final_output')
    parser.add_argument('--window', type=int)
    parser.add_argument('--boundary-threshold', type=float, default=0.16)
    parser.add_argument('--min-interval', type=int)
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
    args = parser.parse_args()
    column = _open_rgb(args.column)
    depth_path = args.depth_column or args.column.parent.parent / 'depth_column' / args.column.name
    depth_image = _open_rgb(depth_path)
    warnings, clusters = [], []
    matching_diagnostics = {}
    explicit = args.column_top is not None or args.depth_top is not None
    if explicit:
        if args.column_top is None or args.depth_top is None:
            parser.error('Supply both --column-top and --depth-top')
        alignment = CropAlignment(args.column_top, args.depth_top, args.column_scale, args.depth_scale)
    elif args.column_scale != 1 or args.depth_scale != 1:
        parser.error('Explicit scales require --column-top and --depth-top')
    else:
        annotation = args.annotation or find_annotation(args.column, ROOT / 'Dataset/dataset/bbox_annotation')
        if annotation is not None:
            alignment = alignment_from_annotation(annotation, column.size, depth_image.size)
        elif args.assume_aligned:
            if column.height != depth_image.height:
                parser.error('--assume-aligned requires equal crop heights; use explicit origins/scales instead')
            alignment = CropAlignment(source='assumed-aligned', document_coordinates_known=False)
            warnings.append('Crop alignment explicitly assumed; original document coordinates are unavailable.')
        else:
            parser.error('No crop annotation: provide --annotation, explicit origins/scales, or --assume-aligned')
    calibration = (load_calibration(args.calibration) if args.calibration else
                   calibrate_depth_column(depth_image, samples=args.depth_samples,
                                           rotation=args.depth_rotation, label_mode=args.depth_label_mode))
    if calibration.image_height != depth_image.height:
        parser.error('Calibration height differs from the depth crop; recalibrate at this resolution')
    if args.labelled_csv:
        rows = read_labelled_csv(args.labelled_csv)
        sidecar = args.labelled_csv.with_suffix('.json')
        if sidecar.is_file():
            try:
                details = json.loads(sidecar.read_text())
                saved_rows = details.get('intervals', [])
                consistent = (len(rows) == len(saved_rows)
                              and all(all(row[k] == saved[k] for k in ('top_y', 'bottom_y', 'cluster_id', 'lithology'))
                                      for row, saved in zip(rows, saved_rows)))
                if consistent:
                    clusters = details.get('clusters', [])
                    matching_diagnostics = details.get('diagnostics', details.get('matching_diagnostics', {}))
                    warnings.extend(details.get('warnings', []))
                else:
                    warnings.append('Classification JSON differs from CSV; stale sidecar metadata ignored.')
            except (OSError, ValueError, KeyError, TypeError):
                warnings.append('Classification sidecar could not be read; CSV values retained.')
    else:
        boundaries = detect_boundaries(column, window=args.window, threshold=args.boundary_threshold,
                                        min_interval=args.min_interval, margin=args.margin,
                                        color_threshold=args.color_threshold, refine_radius=args.refine_radius)
        if args.legend_manifest:
            swatches = load_swatch_manifest(args.legend_manifest)
        else:
            legend_path = args.legend or args.column.parent.parent / 'lithology_legend' / args.column.name
            swatches, legend_warnings = extract_legend(legend_path, args.model)
            warnings.extend(legend_warnings)
            save_swatch_manifest(swatches, args.output / f'{args.column.stem}_legend', legend_warnings)
        result = label_intervals(column, boundaries, swatches, cluster_threshold=args.cluster_threshold,
                                 max_match_distance=args.max_match_distance,
                                 min_match_margin=args.min_match_margin, margin=args.margin,
                                 matcher=args.matcher, shortlist_size=args.shortlist_size,
                                 max_representatives=args.max_representatives, seam_threshold=args.seam_threshold)
        rows, clusters = [asdict(row) for row in result.intervals], result.clusters
        matching_diagnostics = result.diagnostics
        warnings.extend(result.warnings)
    final = attach_depths(rows, calibration, alignment, column.height, depth_unit=args.depth_unit,
                          depth_rounding=args.depth_rounding)
    warnings.extend(calibration.warnings)
    extrapolated = sum(row['depth_extrapolated'] for row in final)
    if extrapolated:
        warnings.append(f'{extrapolated} intervals extend beyond accepted depth anchors; linear extrapolation is flagged.')
    write_final(final, column, calibration, alignment, args.output, args.column.stem,
                warnings=warnings, clusters=clusters, matching_diagnostics=matching_diagnostics, depth_rounding=args.depth_rounding)
    save_calibration(calibration, depth_path, args.output / f'{args.column.stem}_depth')
    fallback_count = sum(row['lithology_fallback'] for row in final)
    print(f'{len(final)} final intervals; {len(set(r["cluster_id"] for r in final))} clusters; '
          f'{fallback_count} runner-up fallback labels; '
          f'{len(calibration.anchors)} depth anchors; depth/pixel={calibration.slope:.8g}; '
          f'fit RMSE={calibration.rmse_px:.2f}px; output={args.output}')
    for warning in warnings:
        print(f'Warning: {warning}')


if __name__ == '__main__':
    main()
