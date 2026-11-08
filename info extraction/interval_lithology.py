"""Cluster RGB lithology intervals, then name each cluster from its own legend.

Uses texture_boundaries for segmentation/features. YOLO and RapidOCR are loaded
only for automatic legend extraction; a reviewed JSON swatch manifest bypasses
both. Run this module with --help for the command-line interface.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from texture_boundaries import BoundaryResult, _distance, _row_features, detect_boundaries
from texture_seam import PreparedTexture, SeamVerifier, prepare_texture
from lithology_groups import consolidate_lithology_clusters

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
    grouping_status: str = 'descriptor_only'
    grouping_seam_score: float | None = None
    match_descriptor_distance: float | None = None
    match_seam_score: float | None = None
    match_seam_status: str = 'not_checked'
    match_method: str = 'descriptor'
    lithology_fallback: bool = False
    fallback_distance: float | None = None
    visual_cluster_id: str = ''


@dataclass
class LithologyResult:
    intervals: list[LabelledInterval]
    clusters: list[dict]
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    visual_clusters: list[dict] = field(default_factory=list)


def apply_runner_up_fallback(original: dict) -> dict:
    """Replace unknown with the SECOND candidate, retaining original evidence.

    Pure/idempotent; also accepts CSV string fields. An unavailable/unknown
    runner-up cannot supply a name. Blank-status intervals always have an empty
    lithology cell, even if an older export assigned a fallback. match_* retain
    the original evidence; fallback_distance describes nonblank replacements.
    """
    row = dict(original)
    flag = row.get('lithology_fallback', False)
    row['lithology_fallback'] = (flag.strip().lower() in ('true', '1', 'yes')
                                  if isinstance(flag, str) else bool(flag))
    row.setdefault('fallback_distance', None)
    if str(row.get('match_status', '')).strip().casefold() == 'blank':
        row['lithology'] = ''
        row['lithology_fallback'] = False
        row['fallback_distance'] = None
        return row
    runner = row.get('runner_up')
    if (str(row.get('lithology', '')).strip().casefold() == 'unknown'
            and isinstance(runner, str) and runner.strip()
            and runner.strip().casefold() != 'unknown'):
        row['lithology'] = runner.strip()
        row['lithology_fallback'] = True
        try:
            distance = float(row['match_distance']) + float(row['match_margin'])
            row['fallback_distance'] = distance if math.isfinite(distance) else None
        except (KeyError, TypeError, ValueError):
            row['fallback_distance'] = None
    return row


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


def _representatives(members: Sequence[int], descriptors: np.ndarray,
                     heights: Sequence[int], maximum: int) -> list[int]:
    """Centroid-nearest exemplar plus bounded, diverse long-bed exemplars."""
    center = np.average(descriptors[list(members)], axis=0,
                        weights=np.minimum(np.asarray(heights)[list(members)], 256))
    first = min(members, key=lambda i: (float(_distances(center, descriptors[i:i + 1])[0]), -heights[i], i))
    selected = [first]
    while len(selected) < min(maximum, len(members)):
        remaining = [i for i in members if i not in selected]
        # Prefer intervals with actual texture context over almost-empty slivers.
        reliable = [i for i in remaining if heights[i] >= 8]
        candidates = reliable or remaining
        def diversity(i):
            distances = _distances(descriptors[i], descriptors[selected])
            return (float(distances.min()), min(heights[i], 256), -i)
        selected.append(max(candidates, key=diversity))
    return selected


def _hybrid_groups(prepared: Sequence[PreparedTexture], heights: Sequence[int],
                   threshold: float, verifier: SeamVerifier, shortlist_size: int,
                   max_representatives: int, seam_threshold: float
                   ) -> tuple[list[list[int]], dict[int, dict]]:
    descriptors = np.stack([p.descriptor for p in prepared])
    groups: list[list[int]] = []
    states = {}
    gate = threshold * 1.5
    for i in sorted(range(len(prepared)), key=lambda k: (-heights[k], k)):
        centers = np.stack([np.average(descriptors[g], axis=0,
                           weights=np.minimum(np.asarray(heights)[g], 256)) for g in groups]) if groups else None
        distances = _distances(descriptors[i], centers) if centers is not None else np.empty(0)
        shortlist = sorted(range(len(groups)), key=lambda g: (distances[g], g))[:shortlist_size]
        accepted, uncertain = [], False
        for group in shortlist:
            if distances[group] > gate:
                continue
            members = groups[group]
            # Cheap complete-link descriptor guard still prevents A~B~C drift.
            if np.max(_distances(descriptors[i], descriptors[members])) > gate:
                continue
            refs = _representatives(members, descriptors, heights, max_representatives)
            evidence = [verifier.compare(prepared[i], prepared[j], threshold=seam_threshold) for j in refs]
            if all(e.status == 'same' for e in evidence):
                score = max(e.score for e in evidence)
                accepted.append((float(score), float(distances[group]), group))
            elif not any(e.status == 'different' for e in evidence):
                uncertain = True
        accepted.sort()
        if len(accepted) > 1 and accepted[1][0] - accepted[0][0] < 0.02:
            # Two confirmed candidates may be duplicate visual groups. Merge
            # only after DIRECT cross-group checks, never through the new
            # interval alone (which would reintroduce A~B~C chaining).
            a, b = accepted[0][2], accepted[1][2]
            compatible = all(np.max(_distances(descriptors[j], descriptors[groups[b]])) <= gate
                             for j in groups[a])
            if compatible:
                refs_a = _representatives(groups[a], descriptors, heights, max_representatives)
                refs_b = _representatives(groups[b], descriptors, heights, max_representatives)
                cross = [verifier.compare(prepared[j], prepared[k], threshold=seam_threshold)
                         for j in refs_a for k in refs_b]
                compatible = all(e.status == 'same' for e in cross)
            if compatible:
                for j in groups[a] + groups[b]:
                    if states[j]['status'] == 'uncertain':
                        states[j] = dict(status='verified', score=max(accepted[0][0], accepted[1][0]))
                keep, drop = min(a, b), max(a, b)
                groups[keep].extend(groups[drop])
                groups.pop(drop)
                accepted = [(max(accepted[0][0], accepted[1][0]), 0.0, keep)]
            else:
                uncertain = True
                accepted = []
        if accepted:
            score, _, group = accepted[0]
            if len(groups[group]) == 1 and states[groups[group][0]]['status'] == 'uncertain':
                states[groups[group][0]] = dict(status='verified', score=score)
            groups[group].append(i)
            states[i] = dict(status='verified', score=score)
        else:
            groups.append([i])
            thin = min(prepared[i].native_height, prepared[i].normalized_height) < 8
            states[i] = dict(status='uncertain' if uncertain or thin else 'new', score=None)
    return sorted((sorted(g) for g in groups), key=lambda g: g[0]), states


def _hybrid_legend_ranking(descriptor: np.ndarray, refs: Sequence[PreparedTexture],
                           bank: Sequence[tuple[LegendSwatch, PreparedTexture]],
                           verifier: SeamVerifier, shortlist_size: int,
                           max_distance: float) -> list[dict]:
    bank_descriptors = np.stack([p.descriptor for _, p in bank])
    scores = _distances(descriptor, bank_descriptors)
    by_name: dict[str, list[int]] = {}
    for index, (swatch, _) in enumerate(bank):
        by_name.setdefault(swatch.name.casefold().strip(), []).append(index)
    ranked = []
    names = sorted(by_name, key=lambda n: (min(scores[i] for i in by_name[n]), n))
    shortlisted = set(names[:max(2, shortlist_size)])  # retain a distinct-name runner-up
    for name in names:
        indices = sorted(by_name[name], key=lambda i: (scores[i], i))
        best = dict(lithology=bank[indices[0]][0].name, distance=float(scores[indices[0]]),
                    descriptor_distance=float(scores[indices[0]]), seam_score=None,
                    seam_status='not_checked', seam_reason='outside_shortlist', seam_scales=[])
        variants = []
        if name in shortlisted:
            # At most TWO variants per name. Do not spend both checks on
            # undersized renderings if a real-context variant is available.
            chosen = indices[:2]
            def reliable(index):
                p = bank[index][1]
                return min(p.native_height, p.normalized_height) >= 8 and len(p.heads) >= 2
            if not any(reliable(index) for index in chosen):
                usable = next((index for index in indices if reliable(index)), None)
                if usable is not None:
                    chosen[-1] = usable
            for index in dict.fromkeys(chosen):
                if scores[index] > max_distance * 1.75:
                    continue
                evidence = [verifier.compare(p, bank[index][1], threshold=max_distance) for p in refs]
                available = [e.score for e in evidence if e.score is not None]
                seam = float(max(available)) if available else None
                status = ('same' if all(e.status == 'same' for e in evidence) else
                          'different' if any(e.status == 'different' for e in evidence) else 'uncertain')
                reason = next((e.reason for e in evidence if e.status != 'same'), 'multiscale_seam_agreement')
                fused = float(0.45 * scores[index] + 0.55 * seam) if seam is not None else float(scores[index])
                item = dict(lithology=bank[index][0].name, distance=fused,
                            descriptor_distance=float(scores[index]), seam_score=seam,
                            seam_status=status, seam_reason=reason,
                            seam_scales=sorted(set.intersection(*(set(e.scales) for e in evidence))))
                variants.append(item)
            if variants:
                best = min(variants, key=lambda v: ({'same': 0, 'uncertain': 1, 'different': 2}[v['seam_status']], v['distance']))
        ranked.append(best)
    return sorted(ranked, key=lambda item: (item['distance'], item['lithology'].casefold()))


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
    matcher: str = 'hybrid',
    shortlist_size: int = 3,
    max_representatives: int = 3,
    seam_threshold: float = 0.20,
) -> LithologyResult:
    """Assign document-local IDs and names with descriptor + seam verification.

    Hybrid mode caches bounded local patches, shortlists groups/names cheaply,
    and verifies multiscale seams against up to three representatives. Thin or
    ambiguous evidence is not forced into a group. Rejected names use the
    runner-up fallback when available, without changing their original match
    status. Descriptor mode retains the earlier comparison method.

    IDs are document-local, ordered by first appearance. Different visual
    clusters can map to the SAME lithology; identical names do not merge clusters.
    Blank rows have an empty lithology cell. Other poor, ambiguous or unverified
    decisions use the runner-up when available, with the original status retained.
    Distances/margins are diagnostic, NOT probabilities.
    Boundary coordinates are left unchanged. Depth fields remain unset.
    """
    if matcher not in ('hybrid', 'descriptor'):
        raise ValueError('matcher must be hybrid or descriptor')
    if not isinstance(shortlist_size, int) or not 1 <= shortlist_size <= 10:
        raise ValueError('shortlist_size must be an integer in [1,10]')
    if not isinstance(max_representatives, int) or not 1 <= max_representatives <= 5:
        raise ValueError('max_representatives must be an integer in [1,5]')
    if not math.isfinite(seam_threshold) or seam_threshold <= 0:
        raise ValueError('seam_threshold must be finite and positive')
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
    heights = [b - t for t, b in intervals]
    verifier = SeamVerifier()
    if matcher == 'hybrid':
        prepared = [prepare_texture(im.crop((trim, t, im.width - trim, b))) for t, b in intervals]
        descriptors = np.stack([p.descriptor for p in prepared])
        groups, group_states = _hybrid_groups(prepared, heights, cluster_threshold, verifier,
                                              shortlist_size, max_representatives, seam_threshold)
        bank = [(swatch, prepare_texture(swatch.image, y_scale=scale))
                for swatch in swatches for scale in (0.5, 0.75, 1.0, 1.5, 2.0)]
        bank_descriptors = None
    else:
        descriptors = np.stack([describe_patch(im.crop((trim, t, im.width - trim, b))) for t, b in intervals])
        groups = cluster_descriptors(descriptors, heights, cluster_threshold)
        group_states = {i: dict(status='descriptor_only', score=None) for i in range(len(intervals))}
        bank = [(swatch, describe_patch(swatch.image, y_scale=scale))
                for swatch in swatches for scale in (0.5, 0.75, 1.0, 1.5, 2.0)]
        bank_descriptors = np.stack([d for _, d in bank]) if bank else np.empty((0, 47))
    group_evaluations = verifier.evaluations
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
        ranked, evidence_ranking = [], []
        reference_indices = (_representatives(members, descriptors, heights, max_representatives)
                             if matcher == 'hybrid' else [representative])
        seam_info = None
        if bank:
            if matcher == 'hybrid':
                evidence_ranking = _hybrid_legend_ranking(descriptor, [prepared[i] for i in reference_indices],
                                                         bank, verifier, shortlist_size, max_match_distance)
                ranked = [(item['distance'], item['lithology']) for item in evidence_ranking]
                seam_info = evidence_ranking[0]
            else:
                scores = _distances(descriptor, bank_descriptors)
                # Distinct names, not repeated crops/scales of one name.
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
            elif seam_info is not None and seam_info['seam_status'] != 'same':
                status = ('insufficient_context' if seam_info['seam_reason'] == 'insufficient_texture_context'
                          else 'seam_rejected' if seam_info['seam_status'] == 'different' else 'unverified_match')
            else:
                lithology, status = best_name, 'matched'
        if blank:
            lithology, status = 'unknown', 'blank'
        assignment = apply_runner_up_fallback(dict(lithology=lithology, runner_up=runner,
                                                    match_status=status, match_distance=distance,
                                                    match_margin=match_margin))
        lithology = assignment['lithology']
        for i in members:
            top, bottom = intervals[i]
            rows[i] = LabelledInterval(int(top), int(bottom), cluster_id, lithology,
                                      match_distance=distance, match_margin=match_margin,
                                      runner_up=runner, match_status=status,
                                      grouping_status=group_states[i]['status'],
                                      grouping_seam_score=group_states[i]['score'],
                                      match_descriptor_distance=(seam_info['descriptor_distance'] if seam_info else distance),
                                      match_seam_score=(seam_info['seam_score'] if seam_info else None),
                                      match_seam_status=(seam_info['seam_status'] if seam_info else 'not_checked'),
                                      match_method=matcher,
                                      lithology_fallback=assignment['lithology_fallback'],
                                      fallback_distance=assignment['fallback_distance'])
        summaries.append(dict(cluster_id=cluster_id, lithology=lithology, match_status=status,
                              lithology_fallback=assignment['lithology_fallback'],
                              fallback_distance=assignment['fallback_distance'],
                              interval_count=len(members), member_indices=members,
                              representative_index=representative, representative_indices=reference_indices,
                              match_distance=distance, match_margin=match_margin, runner_up=runner,
                              grouping_status=('descriptor_only' if matcher == 'descriptor' else
                                               'uncertain' if any(group_states[i]['status'] == 'uncertain' for i in members)
                                               else 'singleton' if len(members) == 1 else 'verified'),
                              candidates=(evidence_ranking[:3] if matcher == 'hybrid' else
                                          [dict(lithology=n, distance=d) for d, n in ranked[:3]])))
    warnings = [] if bank else ['No legend supplied: cluster IDs are available but lithologies are unknown.']
    uncertain = sum(state['status'] == 'uncertain' for state in group_states.values())
    if uncertain:
        warnings.append(f'{uncertain} intervals have uncertain grouping evidence; no forced seam-based merge was made.')
    fallback_count = sum(row.lithology_fallback for row in rows)
    if fallback_count:
        warnings.append(f'{fallback_count} intervals use runner-up fallback labels; original match statuses are retained.')
    diagnostics = dict(matcher=matcher, shortlist_size=shortlist_size, max_representatives=max_representatives,
                       runner_up_fallback_intervals=fallback_count,
                       seam_threshold=seam_threshold,
                       descriptor_diameter_limit=cluster_threshold * (1.5 if matcher == 'hybrid' else 1),
                       group_seam_evaluations=group_evaluations,
                       legend_seam_evaluations=verifier.evaluations - group_evaluations,
                       uncertain_group_intervals=uncertain, **verifier.statistics())
    records, named_groups = consolidate_lithology_clusters([asdict(row) for row in rows], summaries)
    diagnostics.update(cluster_id_policy='final_lithology', visual_cluster_count=len(summaries),
                       lithology_cluster_count=len(named_groups))
    return LithologyResult([LabelledInterval(**row) for row in records], named_groups,
                           warnings, diagnostics, summaries)


def read_intervals(path: str | Path) -> list[tuple[int, int]]:
    with Path(path).open(newline='') as f:
        return [(int(row['top_y']), int(row['bottom_y'])) for row in csv.DictReader(f)]


def write_results(result: LithologyResult, image: Image.Image, output: str | Path, stem: str,
                  swatches: Sequence[LegendSwatch] = ()) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, named_groups = consolidate_lithology_clusters(
        [apply_runner_up_fallback(asdict(row)) for row in result.intervals], result.clusters)
    with (output / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(LabelledInterval.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(records)
    diagnostics = dict(result.diagnostics)
    diagnostics['runner_up_fallback_intervals'] = sum(r['lithology_fallback'] for r in records)
    diagnostics.update(cluster_id_policy='final_lithology', lithology_cluster_count=len(named_groups),
                       visual_cluster_count=len({r['visual_cluster_id'] for r in records if r['visual_cluster_id']}))
    (output / f'{stem}.json').write_text(json.dumps(dict(intervals=records, clusters=named_groups,
                                                        visual_clusters=result.visual_clusters,
                                                        warnings=result.warnings, diagnostics=diagnostics), indent=2))
    overlay = Image.new('RGB', (image.width + 300, image.height), 'white')
    overlay.paste(image, (0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, row in enumerate(result.intervals):
        assigned = records[index]
        cluster_id = assigned['cluster_id']
        number = (int(cluster_id[1:]) if cluster_id.startswith('C') and cluster_id[1:].isdigit()
                  else int(hashlib.sha256(cluster_id.encode()).hexdigest()[:8], 16))
        hue = (number * 0.61803398875) % 1
        color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(hue, 0.7, 0.8))
        draw.line((0, row.top_y, overlay.width, row.top_y), fill=color, width=2)
        draw.rectangle((image.width, row.top_y, image.width + 8, row.bottom_y - 1), fill=color)
        if row.bottom_y - row.top_y >= 14:
            draw.text((image.width + 14, row.top_y + 2),
                      f"{cluster_id}  {assigned['lithology']} [{row.match_status}]"
                      + (' [runner-up]' if assigned['lithology_fallback'] else ''), fill=color)
    overlay.save(output / f'{stem}.png')
    # Compact review sheet: a representative interval beside its matched swatch.
    # Unknown clusters show the nearest CANDIDATE, explicitly not an assignment.
    if named_groups:
        sheet = Image.new('RGB', (1200, math.ceil(len(named_groups) / 4) * 260), 'white')
        draw = ImageDraw.Draw(sheet)
        by_name = {s.name: s.image for s in swatches}
        for index, cluster in enumerate(named_groups):
            x, y = (index % 4) * 300, (index // 4) * 260
            row = result.intervals[cluster['representative_index']]
            assigned = records[cluster['representative_index']]
            title = f"{cluster['cluster_id']}: {assigned['lithology']} ({cluster['interval_count']} intervals)"
            draw.text((x + 5, y + 5), '\n'.join(textwrap.wrap(title, 38)), fill='black')
            detail = f"{cluster['verified_intervals']} verified, {cluster['fallback_intervals']} fallback"
            if row.match_seam_score is not None:
                detail += f'  rep seam={row.match_seam_score:.3f}'
            draw.text((x + 5, y + 35), detail, fill='black')
            center = (row.top_y + row.bottom_y) // 2
            patch = image.crop((0, max(row.top_y, center - 90), image.width,
                                min(row.bottom_y, center + 90)))
            patch.thumbnail((100, 180))
            sheet.paste(patch, (x + 5, y + 70))
            name = assigned['lithology']
            if assigned['lithology_fallback']:
                draw.text((x + 115, y + 50), 'runner-up fallback', fill='gray')
            elif row.match_status not in ('matched', 'blank') and cluster['candidates']:
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
    parser.add_argument('--matcher', choices=['hybrid', 'descriptor'], default='hybrid')
    parser.add_argument('--shortlist-size', type=int, default=3)
    parser.add_argument('--max-representatives', type=int, default=3)
    parser.add_argument('--seam-threshold', type=float, default=0.20)
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
                             min_match_margin=args.min_match_margin, margin=args.margin,
                             matcher=args.matcher, shortlist_size=args.shortlist_size,
                             max_representatives=args.max_representatives, seam_threshold=args.seam_threshold)
    result.warnings.extend(warnings)
    write_results(result, im, args.output, args.column.stem, swatches)
    matched = sum(row.match_status == 'matched' for row in result.intervals)
    fallback_count = sum(row.lithology_fallback for row in result.intervals)
    unknown = sum(row.lithology.strip().casefold() == 'unknown' for row in result.intervals)
    blank_count = sum(row.match_status == 'blank' for row in result.intervals)
    print(f'{len(result.intervals)} intervals -> {len(result.clusters)} visual clusters; '
          f'{matched} verified, {fallback_count} runner-up fallbacks, {blank_count} blank, '
          f'{unknown} unknown. Output: {args.output}')
    for warning in result.warnings:
        print(f'Warning: {warning}')


if __name__ == '__main__':
    main()
