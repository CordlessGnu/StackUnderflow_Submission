"""Cluster RGB lithology intervals, then name each cluster from its own legend.

Uses texture_boundaries for segmentation/features. YOLO and RapidOCR are loaded
only for automatic legend extraction; a reviewed JSON swatch manifest bypasses
both. Run this module with --help for the command-line interface.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from texture_boundaries import BoundaryResult, _distance, _row_features, detect_boundaries

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LegendSwatch:
    name: str
    image: Image.Image
    source: str = ''
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class LabelledInterval:
    top_y: int
    bottom_y: int
    cluster_id: str
    lithology: str
    top_depth: float | None = None
    base_depth: float | None = None
    match_distance: float | None = None
    match_margin: float | None = None
    runner_up: str = ''
    match_status: str = 'unknown'


@dataclass
class LithologyResult:
    intervals: list[LabelledInterval]
    clusters: list[dict]
    warnings: list[str] = field(default_factory=list)


def _open_rgb(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert('RGB')
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError('array must be uint8 grayscale/RGB/RGBA')
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError('array must have 3 or 4 channels')
    return Image.fromarray(array).convert('RGB')


def describe_patch(image: Image.Image, *, y_scale: float = 1.0) -> np.ndarray:
    """Phase-pooled RGB/texture descriptor, never squash a whole bed to a square.

    Normalize WIDTH to 64 and preserve aspect ratio (with an optional legend
    y-scale). Bound sampling to seven native-resolution patches in long beds.
    The same representation is used for column intervals and legend swatches.
    """
    im = _open_rgb(image)
    width, height = im.size
    if min(width, height) < 1 or not math.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('patch dimensions and y_scale must be positive')
    native_height = min(height, max(16, round(width * 2 / y_scale)))
    starts = np.unique(np.linspace(0, height - native_height,
                                  min(7, max(1, math.ceil(height / native_height)))).astype(int))
    descriptors = []
    for start in starts:
        patch = im.crop((0, int(start), width, int(start) + native_height))
        target_height = max(9, round(native_height * 64 / width * y_scale))
        patch = patch.resize((64, target_height), Image.Resampling.BILINEAR)
        rgb = np.asarray(patch, dtype=np.float32) / 255
        descriptors.append(_row_features(rgb, ignore_rules=True, use_color=True).mean(axis=0))
    return np.mean(descriptors, axis=0)


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Broadcasting keeps distance definitions identical to boundary detection.
    return _distance(np.broadcast_to(a, b.shape), b, use_color=True)


def cluster_descriptors(descriptors: np.ndarray, heights: Sequence[int],
                        threshold: float = 0.20) -> list[list[int]]:
    """Deterministic bounded-diameter clustering, without scipy/sklearn.

    Long intervals seed prototypes first. An interval joins only if it is within
    threshold of EVERY member, avoiding single-linkage A~B~C chaining. IDs are
    assigned later by first occurrence, not by an arbitrary numeric ML label.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('cluster threshold must be finite and positive')
    n = len(descriptors)
    if len(heights) != n:
        raise ValueError('one height is required per descriptor')
    if not n:
        return []
    # Only O(N) working distance storage, rather than a potentially huge NxN
    # matrix. Each interval-to-cluster comparison still uses all members.
    groups: list[list[int]] = []
    for i in sorted(range(n), key=lambda k: (-heights[k], k)):
        distances = _distances(descriptors[i], descriptors)
        eligible = [(float(np.mean(distances[g])), j) for j, g in enumerate(groups)
                    if np.max(distances[g]) <= threshold]
        if eligible:
            groups[min(eligible)[1]].append(i)
        else:
            groups.append([i])
    return sorted((sorted(g) for g in groups), key=lambda g: g[0])


def load_swatch_manifest(path: str | Path) -> list[LegendSwatch]:
    """Load {swatches: [{name, image}]} with paths relative to the JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    entries = data.get('swatches') if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest needs a nonempty "swatches" list')
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('name'), str):
            raise ValueError('each swatch needs a text name and image path')
        name = ' '.join(entry['name'].split())
        if not name or not isinstance(entry.get('image'), str) or not entry['image']:
            raise ValueError('each swatch needs a name and image path')
        source = path.parent / entry['image']
        result.append(LegendSwatch(name, _open_rgb(source), str(source),
                                   tuple(entry['bbox']) if entry.get('bbox') else None))
    return result


def save_swatch_manifest(swatches: Sequence[LegendSwatch], directory: str | Path,
                         warnings: Sequence[str] = ()) -> Path:
    """Save reviewable RGB swatches, raw OCR names, and extraction warnings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, swatch in enumerate(swatches):
        filename = f'swatch_{i:03d}.png'
        swatch.image.save(directory / filename)
        entries.append(dict(name=swatch.name, image=filename, source=swatch.source, bbox=swatch.bbox))
    path = directory / 'legend.json'
    path.write_text(json.dumps(dict(swatches=entries, warnings=list(warnings)), indent=2))
    return path


def extract_legend(legend: str | Path, model_path: str | Path = ROOT / 'models/swatch_detection.pt',
                   confidence: float = 0.25, image_size: int = 960
                   ) -> tuple[list[LegendSwatch], list[str]]:
    """Use the existing YOLO detector/pairing and OCR, keeping swatches in RGB.

    Full-legend OCR supplies missing material_name boxes. Unreadable labels,
    geometrically implausible matches and nearest-neighbour pairing fallbacks
    are reported, never silently converted into names. OCR spelling is retained
    so users can review/correct the exported manifest.
    """
    from legend_utils import (merge_overlapping_boxes as _merge_overlapping_boxes,
                              pair_boxes as _pair_boxes, ocr_name as _ocr_name,
                              is_lithology_name, LegendExtractionError)
    from rapidocr import RapidOCR
    from ultralytics import YOLO

    legend = Path(legend)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Swatch model not found: {model_path}')
    im = _open_rgb(legend)
    model = YOLO(str(model_path))
    prediction = model.predict(str(legend), verbose=False, conf=confidence, imgsz=image_size)[0]
    textures, name_boxes = [], []
    for box in prediction.boxes:
        label = model.names[int(box.cls)]
        xyxy = tuple(float(v) for v in box.xyxy[0])
        if label == 'material_texture':
            textures.append(xyxy)
        elif label == 'material_name':
            name_boxes.append(xyxy)
    textures, merged = _merge_overlapping_boxes(textures)
    warnings = []
    if merged:
        warnings.append(f'Merged {merged} overlapping texture detections.')
    if not textures:
        raise LegendExtractionError(f'No material_texture detections in {legend}; use a reviewed swatch manifest.')

    # Model name boxes help combine multiline labels. OCR adds labels the model
    # missed (notably some coloured legends). Avoid duplicating the same text.
    ocr = RapidOCR()(np.asarray(im))
    recognized = []
    boxes = getattr(ocr, 'boxes', None)
    texts = getattr(ocr, 'txts', None)
    if boxes is not None and texts is not None:
        for polygon, text in zip(boxes, texts):
            text = ' '.join(text.split())
            if not is_lithology_name(text) or any(k in text.lower() for k in ('legend', 'lithology symbols')):
                continue
            polygon = np.asarray(polygon)
            bbox = (float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                    float(polygon[:, 0].max()), float(polygon[:, 1].max()))
            recognized.append((bbox, text))
    named: dict[tuple, str] = {}
    used = set()
    name_boxes, _ = _merge_overlapping_boxes(name_boxes)
    for box in name_boxes:
        included = []
        for i, (ocr_box, text) in enumerate(recognized):
            cx, cy = (ocr_box[0] + ocr_box[2]) / 2, (ocr_box[1] + ocr_box[3]) / 2
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                included.append((ocr_box[1], ocr_box[0], text))
                used.add(i)
        text = ' '.join(t[2] for t in sorted(included)) if included else _ocr_name(im.crop(box))
        if text and is_lithology_name(text):
            named[box] = text
    added = 0
    for i, (box, text) in enumerate(recognized):
        if i not in used:
            named[box] = text
            added += 1
    if added:
        warnings.append(f'Full-legend OCR added {added} label boxes missed by the detector.')
    pairs, unmatched, fallbacks, layout = _pair_boxes(textures, list(named))
    if unmatched or fallbacks:
        warnings.append(f'{len(unmatched)} unmatched and {len(fallbacks)} guessed swatch/name pairs dropped.')
    swatches = []
    for texture, name_box in pairs:
        tw, th = texture[2] - texture[0], texture[3] - texture[1]
        right = (name_box[0] >= texture[2] - tw * 0.1
                 and abs((name_box[1] + name_box[3]) / 2 - (texture[1] + texture[3]) / 2) < th
                 and name_box[0] - texture[2] <= tw * 2)
        below = (name_box[1] >= texture[3] - th * 0.1
                 and abs((name_box[0] + name_box[2]) / 2 - (texture[0] + texture[2]) / 2) < tw
                 and name_box[1] - texture[3] <= th)
        if not (right or below):
            warnings.append(f'Implausible swatch/name geometry for {named[name_box]!r}; dropped.')
            continue
        trim = max(1, round(min(tw, th) * 0.06))
        bbox = (max(0, int(texture[0]) + trim), max(0, int(texture[1]) + trim),
                min(im.width, int(texture[2]) - trim), min(im.height, int(texture[3]) - trim))
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            warnings.append(f'Tiny swatch for {named[name_box]!r}; dropped.')
            continue
        swatches.append(LegendSwatch(named[name_box], im.crop(bbox), str(legend), bbox))
    warnings.append(f'Legend layout: {layout}; {len(swatches)}/{len(textures)} texture detections usable. Review OCR names.')
    if not swatches:
        raise LegendExtractionError(f'No usable named swatches in {legend}. Use --legend-manifest with reviewed RGB crops.')
    return swatches, warnings


def label_intervals(
    image: str | Path | Image.Image | np.ndarray,
    intervals: Sequence[tuple[int, int]] | BoundaryResult,
    swatches: Sequence[LegendSwatch],
    *,
    cluster_threshold: float = 0.20,
    max_match_distance: float = 0.32,
    min_match_margin: float = 0.06,
    margin: float = 0.12,
) -> LithologyResult:
    """Assign a common cluster_id and consistent lithology to similar beds.

    IDs are document-local, ordered by first appearance. Different visual
    clusters can map to the SAME lithology; identical names do not merge clusters.
    A cluster is unknown if blank, too far from its legend, or ambiguous between
    distinct names. Distances/margins are diagnostic, NOT probabilities.
    Boundary coordinates are left unchanged. Depth fields remain unset.
    """
    if not 0 <= margin < 0.45:
        raise ValueError('margin must be in [0, 0.45)')
    if not math.isfinite(max_match_distance) or max_match_distance <= 0:
        raise ValueError('max_match_distance must be finite and positive')
    if not math.isfinite(min_match_margin) or min_match_margin < 0:
        raise ValueError('min_match_margin must be finite and nonnegative')
    if not math.isfinite(cluster_threshold) or cluster_threshold <= 0:
        raise ValueError('cluster_threshold must be finite and positive')
    im = _open_rgb(image)
    if min(im.size) < 3:
        raise ValueError('column must be at least 3 by 3 pixels')
    intervals = list(intervals.intervals if isinstance(intervals, BoundaryResult) else intervals)
    previous_end = 0
    for top, bottom in intervals:
        if (not isinstance(top, (int, np.integer)) or not isinstance(bottom, (int, np.integer))
                or not 0 <= top < bottom <= im.height or top < previous_end):
            raise ValueError('intervals must be sorted, non-overlapping integer pixel ranges within the image')
        previous_end = bottom
    if not intervals:
        return LithologyResult([], [])
    for swatch in swatches:
        if not swatch.name.strip():
            raise ValueError('legend swatches must have nonempty names')
    trim = round(im.width * margin)
    descriptors = np.stack([describe_patch(im.crop((trim, t, im.width - trim, b)))
                            for t, b in intervals])
    heights = [b - t for t, b in intervals]
    groups = cluster_descriptors(descriptors, heights, cluster_threshold)
    # Different renderers stretch the legend patterns vertically. Search a small
    # explicit scale bank instead of forcing every swatch/bed into one square.
    bank = [(swatch, describe_patch(swatch.image, y_scale=scale))
            for swatch in swatches for scale in (0.5, 0.75, 1.0, 1.5, 2.0)]
    bank_descriptors = np.stack([d for _, d in bank]) if bank else np.empty((0, 47))
    rows: list[LabelledInterval | None] = [None] * len(intervals)
    summaries = []
    for cluster_index, members in enumerate(groups, 1):
        cluster_id = f'C{cluster_index:03d}'
        # Capped height weights avoid letting a huge interval monopolize a group.
        weights = np.minimum(np.asarray(heights)[members], 256)
        descriptor = np.average(descriptors[members], axis=0, weights=weights)
        representative = members[int(np.argmin(_distances(descriptor, descriptors[members])))]
        blank = descriptor[:8].sum() < 0.005 and np.min(descriptor[17:20]) > 0.97
        lithology, status, runner, distance, match_margin = 'unknown', 'no_legend', '', None, None
        ranked = []
        if bank:
            scores = _distances(descriptor, bank_descriptors)
            # A runner-up must be a DISTINCT name, not another scale/crop of the
            # same name; otherwise repeated swatches make every match look tied.
            distinct = {}
            for (swatch, _), score in zip(bank, scores):
                key = swatch.name.casefold().strip()
                if key not in distinct or score < distinct[key][0]:
                    distinct[key] = (float(score), swatch.name)
            ranked = sorted(distinct.values())
            distance, best_name = ranked[0]
            if len(ranked) > 1:
                runner = ranked[1][1]
                match_margin = ranked[1][0] - distance
            if distance > max_match_distance:
                status = 'poor_match'
            elif match_margin is not None and match_margin < min_match_margin:
                status = 'ambiguous'
            else:
                lithology, status = best_name, 'matched'
        if blank:
            lithology, status = 'unknown', 'blank'
        for i in members:
            top, bottom = intervals[i]
            rows[i] = LabelledInterval(int(top), int(bottom), cluster_id, lithology,
                                      match_distance=distance, match_margin=match_margin,
                                      runner_up=runner, match_status=status)
        summaries.append(dict(cluster_id=cluster_id, lithology=lithology, match_status=status,
                              interval_count=len(members), member_indices=members,
                              representative_index=representative, match_distance=distance,
                              match_margin=match_margin, runner_up=runner,
                              candidates=[dict(lithology=n, distance=d) for d, n in ranked[:3]]))
    warnings = [] if bank else ['No legend supplied: cluster IDs are available but lithologies are unknown.']
    return LithologyResult(rows, summaries, warnings)


def read_intervals(path: str | Path) -> list[tuple[int, int]]:
    with Path(path).open(newline='') as f:
        return [(int(row['top_y']), int(row['bottom_y'])) for row in csv.DictReader(f)]


def write_results(result: LithologyResult, image: Image.Image, output: str | Path, stem: str,
                  swatches: Sequence[LegendSwatch] = ()) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in result.intervals]
    with (output / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(LabelledInterval.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(records)
    (output / f'{stem}.json').write_text(json.dumps(dict(intervals=records, clusters=result.clusters,
                                                        warnings=result.warnings), indent=2))
    overlay = Image.new('RGB', (image.width + 300, image.height), 'white')
    overlay.paste(image, (0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in result.intervals:
        hue = (int(row.cluster_id[1:]) * 0.61803398875) % 1
        color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, 0.7, 0.8))
        draw.line((0, row.top_y, overlay.width, row.top_y), fill=color, width=2)
        draw.rectangle((image.width, row.top_y, image.width + 8, row.bottom_y - 1), fill=color)
        if row.bottom_y - row.top_y >= 14:
            draw.text((image.width + 14, row.top_y + 2),
                      f'{row.cluster_id}  {row.lithology} [{row.match_status}]', fill=color)
    overlay.save(output / f'{stem}.png')
    # Compact review sheet: a representative interval beside its matched swatch.
    # Unknown clusters show the nearest CANDIDATE, explicitly not an assignment.
    if result.clusters:
        sheet = Image.new('RGB', (1200, math.ceil(len(result.clusters) / 4) * 260), 'white')
        draw = ImageDraw.Draw(sheet)
        by_name = {s.name: s.image for s in swatches}
        for index, cluster in enumerate(result.clusters):
            x, y = (index % 4) * 300, (index // 4) * 260
            row = result.intervals[cluster['representative_index']]
            title = f"{row.cluster_id}: {row.lithology} ({cluster['interval_count']} intervals)"
            draw.text((x + 5, y + 5), '\n'.join(textwrap.wrap(title, 38)), fill='black')
            draw.text((x + 5, y + 35), row.match_status, fill='black')
            center = (row.top_y + row.bottom_y) // 2
            patch = image.crop((0, max(row.top_y, center - 90), image.width,
                                min(row.bottom_y, center + 90)))
            patch.thumbnail((100, 180))
            sheet.paste(patch, (x + 5, y + 70))
            name = row.lithology
            if row.match_status != 'matched' and cluster['candidates']:
                name = cluster['candidates'][0]['lithology']
                draw.text((x + 115, y + 50), 'candidate only', fill='gray')
            if name in by_name:
                swatch = by_name[name].copy()
                swatch.thumbnail((165, 150))
                sheet.paste(swatch, (x + 115, y + 85))
                draw.text((x + 115, y + 65), name[:25], fill='black')
        sheet.save(output / f'{stem}.clusters.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('column', type=Path)
    parser.add_argument('--intervals', type=Path, help='existing top_y,bottom_y CSV; otherwise detect boundaries')
    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument('--legend', type=Path, help='legend image; YOLO+OCR extracts named RGB swatches')
    legend_group.add_argument('--legend-manifest', type=Path, help='reviewed {swatches: [{name,image}]} JSON')
    parser.add_argument('--legend-dir', type=Path, default=ROOT / 'crops/lithology_legend')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/swatch_detection.pt')
    parser.add_argument('--swatch-conf', type=float, default=0.25)
    parser.add_argument('--output', type=Path, default=ROOT / 'crops/labelled_intervals')
    parser.add_argument('--window', type=int)
    parser.add_argument('--boundary-threshold', type=float, default=0.16)
    parser.add_argument('--min-interval', type=int)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--color-threshold', type=float, default=0.10)
    parser.add_argument('--refine-radius', type=int)
    parser.add_argument('--cluster-threshold', type=float, default=0.20)
    parser.add_argument('--max-match-distance', type=float, default=0.32)
    parser.add_argument('--min-match-margin', type=float, default=0.06)
    args = parser.parse_args()
    im = _open_rgb(args.column)
    intervals = (read_intervals(args.intervals) if args.intervals else
                 detect_boundaries(im, window=args.window, threshold=args.boundary_threshold,
                                   min_interval=args.min_interval, margin=args.margin,
                                   color_threshold=args.color_threshold, refine_radius=args.refine_radius).intervals)
    warnings = []
    if args.legend_manifest:
        swatches = load_swatch_manifest(args.legend_manifest)
        warnings = json.loads(args.legend_manifest.read_text()).get('warnings', [])
    else:
        legend = args.legend or args.legend_dir / args.column.name
        if not legend.is_file():
            parser.error(f'Legend not found: {legend}; provide --legend or --legend-manifest')
        swatches, warnings = extract_legend(legend, args.model, args.swatch_conf)
        manifest = save_swatch_manifest(swatches, args.output / f'{args.column.stem}_legend', warnings)
        print(f'Reviewable legend manifest: {manifest}')
    result = label_intervals(im, intervals, swatches, cluster_threshold=args.cluster_threshold,
                             max_match_distance=args.max_match_distance,
                             min_match_margin=args.min_match_margin, margin=args.margin)
    result.warnings.extend(warnings)
    write_results(result, im, args.output, args.column.stem, swatches)
    matched = sum(row.match_status == 'matched' for row in result.intervals)
    print(f'{len(result.intervals)} intervals -> {len(result.clusters)} visual clusters; '
          f'{matched} named intervals, {len(result.intervals) - matched} unknown. Output: {args.output}')
    for warning in result.warnings:
        print(f'Warning: {warning}')


if __name__ == '__main__':
    main()
