"""Read depth labels with RapidOCR and fit depth = slope * crop_y + intercept.

The original two-label ratio approach is retained as an affine calibration, but
validated against multiple labels across the column. Supports rotated labels,
thousands separators, 5+ digit depths, OCR outliers, and very tall crops. Uses
the already installed `rapidocr` package, not `rapidocr_onnxruntime`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

_ENGINE = None


@dataclass
class DepthAnchor:
    depth: float
    y: float
    confidence: float = 1.0
    text: str = ''
    is_decimal: bool = False


@dataclass
class DepthCalibration:
    slope: float
    intercept: float
    image_height: int
    anchors: list[DepthAnchor]
    rejected: list[DepthAnchor]
    rmse_px: float
    max_error_px: float
    rotation: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (not math.isfinite(self.slope) or self.slope <= 0
                or not math.isfinite(self.intercept) or self.image_height <= 0
                or not math.isfinite(self.rmse_px) or self.rmse_px < 0
                or not math.isfinite(self.max_error_px) or self.max_error_px < 0
                or len(self.anchors) < 2
                or max(a.y for a in self.anchors) <= min(a.y for a in self.anchors)
                or any(not math.isfinite(a.y) or not math.isfinite(a.depth)
                       or not 0 <= a.y <= self.image_height for a in self.anchors)):
            raise ValueError('Invalid depth calibration')

    @property
    def min_y(self) -> float:
        return min(a.y for a in self.anchors)

    @property
    def max_y(self) -> float:
        return max(a.y for a in self.anchors)

    def depth_at(self, y: float) -> float:
        return self.slope * y + self.intercept

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(method='robust_affine', depth_per_pixel=self.slope,
                    anchor_y_range=[self.min_y, self.max_y],
                    coverage=(self.max_y - self.min_y) / self.image_height)
        return data


def parse_depth(text: str) -> tuple[float, bool] | None:
    """Strict numeric labels only; no guessed letter-to-digit substitutions."""
    text = text.strip().replace('\u00a0', ' ')
    # Commas/spaces are accepted only as proper thousands separators.
    if not re.fullmatch(r'(?:\d+|\d{1,3}(?:[, ]\d{3})+)(?:\.\d+)?', text):
        return None
    value = float(text.replace(',', '').replace(' ', ''))
    if not math.isfinite(value) or not 0 <= value <= 10_000_000:
        return None
    return value, '.' in text


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _read_tile(tile: Image.Image, start: int, rotation: int, engine,
               min_confidence: float) -> list[DepthAnchor]:
    # Side borders hinder OCR. Preserve most of these very narrow strips.
    trim = min(2, max(0, (tile.width - 8) // 2))
    tile = tile.crop((trim, 0, tile.width - trim, tile.height))
    height = tile.height
    padding, scale = 12, 2
    prepared = ImageOps.expand(tile.rotate(rotation, expand=True), border=padding, fill='white')
    prepared = prepared.resize((prepared.width * scale, prepared.height * scale), Image.Resampling.BICUBIC)
    result = engine(np.asarray(prepared), use_cls=True)
    texts, boxes, scores = (getattr(result, k, None) for k in ('txts', 'boxes', 'scores'))
    if texts is None or boxes is None:
        return []
    anchors = []
    for index, (text, polygon) in enumerate(zip(texts, boxes)):
        parsed = parse_depth(text)
        confidence = float(scores[index]) if scores is not None else 0.0
        if parsed is None or not math.isfinite(confidence) or confidence < min_confidence:
            continue
        polygon = np.asarray(polygon, dtype=float) / scale - padding
        if rotation == 0:
            y = float(polygon[:, 1].mean())
        elif rotation == 90:
            y = float(polygon[:, 0].mean())
        else:  # 270 degrees counter-clockwise
            y = height - float(polygon[:, 0].mean())
        if 0 <= y <= height:
            anchors.append(DepthAnchor(parsed[0], y + start, confidence, text, parsed[1]))
    return anchors


def extract_depth_anchors(image: str | Path | Image.Image, *, samples: int = 12,
                          tile_height: int = 1024, rotation: str | int = 'auto',
                          min_confidence: float = 0.8, engine=None
                          ) -> tuple[list[DepthAnchor], int, int]:
    """OCR tiles spanning the full height; never shrink an entire long log.

    Returns (anchors in original crop coordinates, crop height, rotation).
    Automatic orientation tests 0/90/270 on a pilot tile, then reuses the best
    orientation. The engine argument is injectable for deterministic tests.
    """
    if samples < 2 or tile_height < 64 or not 0 <= min_confidence <= 1:
        raise ValueError('need samples >= 2, tile_height >= 64, and confidence in [0,1]')
    if str(rotation) not in ('auto', '0', '90', '270'):
        raise ValueError('rotation must be auto, 0, 90, or 270')
    if isinstance(image, Image.Image):
        im = image.convert('RGB')
    else:
        with Image.open(image) as loaded:
            im = loaded.convert('RGB')
    engine = _get_engine() if engine is None else engine
    tile_height = min(tile_height, im.height)
    starts = np.unique(np.linspace(0, im.height - tile_height, samples).astype(int)).tolist()
    pilot_cache = {}
    if str(rotation) == 'auto':
        best = None
        # If the top of a crop is blank, try a middle tile as the orientation pilot.
        for pilot in dict.fromkeys((starts[0], starts[len(starts) // 2])):
            tile = im.crop((0, pilot, im.width, pilot + tile_height))
            for angle in (0, 90, 270):
                points = _read_tile(tile, pilot, angle, engine, min_confidence)
                pilot_cache[(pilot, angle)] = points
                integer = [a for a in points if not a.is_decimal]
                selected = sorted(integer if len(integer) >= 2 else points, key=lambda a: a.y)
                if not selected:
                    continue
                increasing = np.mean(np.diff([a.depth for a in selected]) > 0) if len(selected) > 1 else 0.5
                score = sum(a.confidence for a in selected) * increasing
                candidate = (score, -angle, angle)
                if best is None or candidate > best:
                    best = candidate
            if best is not None and best[0] >= 2:
                break
        if best is None:
            raise ValueError('No numeric depth labels found in orientation pilot tiles')
        angle = best[2]
    else:
        angle = int(rotation)
    points = []
    for start in starts:
        if (start, angle) in pilot_cache:
            points.extend(pilot_cache[(start, angle)])
        else:
            points.extend(_read_tile(im.crop((0, start, im.width, start + tile_height)),
                                     start, angle, engine, min_confidence))
    # Overlapping tiles can recognize the same physical label more than once.
    unique = []
    for point in sorted(points, key=lambda a: -a.confidence):
        if not any(p.depth == point.depth and abs(p.y - point.y) < 8
                   and p.is_decimal == point.is_decimal for p in unique):
            unique.append(point)
    return sorted(unique, key=lambda a: a.y), im.height, angle


def fit_depth_calibration(anchors: Sequence[DepthAnchor], image_height: int, *,
                          label_mode: str = 'auto', max_residual_px: float = 8,
                          min_anchors: int = 3, min_coverage: float = 0.25
                          ) -> DepthCalibration:
    """Deterministic robust positive affine fit; reject bad OCR and weak fits.

    Auto mode prefers integer labels when enough exist. This avoids mixing a
    main MD track with a smaller decimal/TVD track. It does NOT infer depth units
    or semantic MD/TVD meaning. Decimal-only scales remain supported.
    """
    if (image_height <= 0 or max_residual_px <= 0 or not math.isfinite(max_residual_px)
            or min_anchors < 2 or not 0 < min_coverage <= 1):
        raise ValueError('invalid calibration settings')
    if label_mode not in ('auto', 'integer', 'decimal', 'all'):
        raise ValueError('label_mode must be auto, integer, decimal, or all')
    valid = [a for a in anchors if math.isfinite(a.depth) and math.isfinite(a.y)
             and math.isfinite(a.confidence) and a.depth >= 0 and 0 <= a.y <= image_height]
    unique = []
    for point in sorted(valid, key=lambda a: -a.confidence):
        if not any(point.depth == other.depth and abs(point.y - other.y) < 8
                   and point.is_decimal == other.is_decimal for other in unique):
            unique.append(point)
    valid = sorted(unique, key=lambda a: a.y)
    integers = [a for a in valid if not a.is_decimal]
    if label_mode == 'integer' or (label_mode == 'auto' and len(integers) >= min_anchors):
        selected = integers
    elif label_mode == 'decimal':
        selected = [a for a in valid if a.is_decimal]
    else:
        selected = valid
    if len(selected) < min_anchors:
        raise ValueError(f'Only {len(selected)} usable depth labels; need at least {min_anchors}')
    y = np.array([a.y for a in selected])
    d = np.array([a.depth for a in selected])
    weights = np.clip([a.confidence for a in selected], 0.01, 1)
    n = len(y)
    # Sample widely separated pairs so character-centre jitter cannot dominate
    # the slope. For small sets evaluate every pair; otherwise use a fixed seed.
    if n <= 40:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = np.random.default_rng(7).integers(0, n, (1500, 2))
    best = None
    for i, j in pairs:
        if abs(y[j] - y[i]) < image_height * min_coverage:
            continue
        slope = (d[j] - d[i]) / (y[j] - y[i])
        if slope <= 0:
            continue
        intercept = d[i] - slope * y[i]
        residual = np.abs((d - intercept) / slope - y)
        keep = residual <= max_residual_px
        if keep.sum() < min_anchors:
            continue
        coverage = np.ptp(y[keep]) / image_height
        score = float(weights[keep].sum() + math.sqrt(n) * coverage)
        if best is None or score > best[0]:
            best = (score, keep)
    if best is None:
        raise ValueError('Cannot find a consistent increasing depth scale spanning enough of the image')
    keep = best[1]
    for _ in range(10):
        center_y = np.average(y[keep], weights=weights[keep])
        center_d = np.average(d[keep], weights=weights[keep])
        dy = y[keep] - center_y
        variance = np.sum(weights[keep] * dy * dy)
        if variance <= 0:
            raise ValueError('Depth anchors have no vertical separation')
        slope = float(np.sum(weights[keep] * dy * (d[keep] - center_d)) / variance)
        if slope <= 0:
            raise ValueError('Depth must increase down the image')
        intercept = float(center_d - slope * center_y)
        new_keep = np.abs((d - intercept) / slope - y) <= max_residual_px
        if np.array_equal(new_keep, keep):
            break
        if new_keep.sum() < min_anchors:
            raise ValueError('Too few labels remain after calibration refinement')
        keep = new_keep
    else:
        raise ValueError('Depth calibration did not converge; inspect OCR anchors')
    coverage = float(np.ptp(y[keep]) / image_height)
    if coverage < min_coverage or keep.mean() < 0.6:
        raise ValueError('Depth fit has insufficient coverage/inlier agreement; inspect OCR or a changing depth scale')
    residuals = (d[keep] - intercept) / slope - y[keep]
    accepted = [a for a, good in zip(selected, keep) if good]
    accepted_ids = {id(a) for a in accepted}
    rejected = [a for a in anchors if id(a) not in accepted_ids]
    warnings = []
    if label_mode == 'auto' and len(integers) >= min_anchors and any(a.is_decimal for a in valid):
        warnings.append('Preferred integer labels; decimal labels may belong to a secondary depth track.')
    if rejected:
        warnings.append(f'{len(rejected)} OCR labels excluded as outliers, duplicates, or another label format.')
    if len(accepted) == 2:
        warnings.append('Only two calibration anchors: no independent outlier validation is possible.')
    if coverage < 0.8:
        warnings.append(f'Anchors cover only {coverage:.0%} of crop height; review extrapolated depths.')
    return DepthCalibration(slope, intercept, image_height, accepted, rejected,
                            float(np.sqrt(np.mean(residuals ** 2))),
                            float(np.max(np.abs(residuals))), warnings=warnings)


def calibrate_depth_column(image: str | Path | Image.Image, *, samples: int = 12,
                           tile_height: int = 1024, rotation: str | int = 'auto',
                           min_confidence: float = 0.8, label_mode: str = 'auto',
                           max_residual_px: float = 8) -> DepthCalibration:
    anchors, height, angle = extract_depth_anchors(image, samples=samples, tile_height=tile_height,
                                                  rotation=rotation, min_confidence=min_confidence)
    calibration = fit_depth_calibration(anchors, height, label_mode=label_mode,
                                         max_residual_px=max_residual_px)
    calibration.rotation = angle
    return calibration


def save_calibration(calibration: DepthCalibration, image: str | Path | Image.Image,
                     output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = calibration.to_dict()
    data['source_image'] = str(image) if isinstance(image, (str, Path)) else None
    Path(f'{output}.json').write_text(json.dumps(data, indent=2))
    if isinstance(image, Image.Image):
        original = image.convert('RGB')
    else:
        with Image.open(image) as loaded:
            original = loaded.convert('RGB')
    annotated = Image.new('RGB', (original.width + 180, original.height), 'white')
    annotated.paste(original, (0, 0))
    draw = ImageDraw.Draw(annotated)
    for points, color in ((calibration.rejected, 'red'), (calibration.anchors, 'green')):
        for anchor in points:
            draw.line((0, anchor.y, original.width, anchor.y), fill=color)
            draw.text((original.width + 5, anchor.y), f'{anchor.depth:g} ({anchor.confidence:.2f})', fill=color)
    annotated.save(Path(f'{output}.png'))


def load_calibration(path: str | Path) -> DepthCalibration:
    data = json.loads(Path(path).read_text())
    fields = ('slope', 'intercept', 'image_height', 'rmse_px', 'max_error_px')
    return DepthCalibration(**{key: data[key] for key in fields},
                            anchors=[DepthAnchor(**a) for a in data['anchors']],
                            rejected=[DepthAnchor(**a) for a in data.get('rejected', [])],
                            rotation=data.get('rotation', 0), warnings=data.get('warnings', []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='cropped depth column')
    parser.add_argument('--output', type=Path, default=Path('crops/depth_calibration/calibration'))
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--rotation', choices=['auto', '0', '90', '270'], default='auto')
    parser.add_argument('--label-mode', choices=['auto', 'integer', 'decimal', 'all'], default='auto')
    args = parser.parse_args()
    calibration = calibrate_depth_column(args.image, samples=args.samples, rotation=args.rotation,
                                         label_mode=args.label_mode)
    save_calibration(calibration, args.image, args.output)
    print(f'{len(calibration.anchors)} anchors; rotation={calibration.rotation}; '
          f'1 pixel = {calibration.slope:.8g} depth units; '
          f'depth = {calibration.slope:.8g} * crop_y + {calibration.intercept:.8g}; '
          f'fit RMSE={calibration.rmse_px:.2f}px')
    for warning in calibration.warnings:
        print(f'Warning: {warning}')


if __name__ == '__main__':
    main()
