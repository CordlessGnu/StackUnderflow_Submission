"""Training-free interval boundaries from a cropped lithology column.

Only NumPy and Pillow are required. Coordinates are crop-local pixel edges;
intervals are half-open [top_y, bottom_y). No lithology naming is attempted.
Run ``python 'info extraction/texture_boundaries.py' --help`` for the CLI.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class BoundaryResult:
    boundaries: list[int]  # internal cuts, excluding 0 and image height
    intervals: list[tuple[int, int]]
    scores: np.ndarray  # one uncalibrated change score per pixel edge (H + 1)
    threshold: float
    window: int


def _mean_windows(rows: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Means immediately above/below every edge; no vertical downsampling."""
    height = len(rows)
    sums = np.vstack((np.zeros((1, rows.shape[1])), np.cumsum(rows, axis=0)))
    y = np.arange(height + 1)
    lo, hi = np.maximum(0, y - window), np.minimum(height, y + window)
    above = (sums[y] - sums[lo]) / np.maximum(y - lo, 1)[:, None]
    below = (sums[hi] - sums[y]) / np.maximum(hi - y, 1)[:, None]
    return above, below


def _row_features(rgb: np.ndarray, ignore_rules: bool, use_color: bool) -> np.ndarray:
    # Maximum RGB channel distinguishes black marks from saturated backgrounds:
    # yellow and cyan are not mistakenly treated as black ink. This intentionally
    # discards colour unless use_color=True. It assumes dark strokes on light fill.
    ink = np.clip((0.8 - rgb.max(axis=2)) / 0.5, 0, 1)
    height, width = ink.shape
    if ignore_rules:
        # Remove only SHORT, almost full-width rules, not long solid/coal beds.
        rules = np.mean(ink > 0.6, axis=1) > 0.90
        edges = np.flatnonzero(np.diff(np.r_[False, rules, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start <= max(2, width // 20) and start > 0 and end < height:
                alpha = np.arange(1, end - start + 1)[:, None] / (end - start + 1)
                ink[start:end] = (1 - alpha) * ink[start - 1] + alpha * ink[end]

    gy, gx = np.gradient(ink)
    magnitude = np.hypot(gx, gy)
    # Soft voting avoids jumps at orientation-bin edges. Do NOT make this
    # rotation invariant: horizontal and vertical dashes are different textures.
    angle = np.mod(np.arctan2(gy, gx), np.pi) * (8 / np.pi)
    lower = np.floor(angle).astype(np.int32)
    fraction = angle - lower
    hist = np.zeros((height, 8), dtype=np.float64)
    row_ids = np.broadcast_to(np.arange(height)[:, None], ink.shape)
    np.add.at(hist, (row_ids, lower % 8), magnitude * (1 - fraction) / width)
    np.add.at(hist, (row_ids, (lower + 1) % 8), magnitude * fraction / width)

    # Short-range spatial differences capture spacing and dot/brick structure,
    # rather than just the mean ink density. All features are pooled over x.
    spatial = []
    for lag in (1, 2, 4, 8):
        if lag < width:
            spatial.append(np.abs(ink[:, lag:] - ink[:, :-lag]).mean(axis=1))
        if lag < height:
            delta = np.abs(ink[lag:] - ink[:-lag]).mean(axis=1)
            spatial.append(np.pad(delta, (lag // 2, lag - lag // 2), mode='edge'))
    features = [hist, ink.mean(axis=1)[:, None], np.stack(spatial, axis=1)]
    if use_color:
        features.append(rgb.mean(axis=1))
    return np.column_stack(features)


def _distance(a: np.ndarray, b: np.ndarray, use_color: bool) -> np.ndarray:
    ha, hb = a[:, :8], b[:, :8]
    ea, eb = ha.sum(axis=1), hb.sum(axis=1)
    pa = ha / np.maximum(ea[:, None], 1e-9)
    pb = hb / np.maximum(eb[:, None], 1e-9)
    hellinger = np.sqrt(np.square(np.sqrt(pa) - np.sqrt(pb)).sum(axis=1) / 2)
    # Near-blank windows have unreliable orientation distributions.
    reliability = np.minimum(1, np.minimum(ea, eb) / 0.015)
    end = -3 if use_color else a.shape[1]
    spatial = np.sqrt(np.square(a[:, 9:end] - b[:, 9:end]).mean(axis=1))
    score = (0.65 * hellinger * reliability
             + 0.20 * np.minimum(1, np.abs(ea - eb) / 0.12)
             + 0.25 * np.abs(a[:, 8] - b[:, 8])
             + 0.30 * spatial)
    if use_color:
        score += 0.35 * np.sqrt(np.square(a[:, -3:] - b[:, -3:]).mean(axis=1))
    return score


def detect_boundaries(
    image: str | Path | Image.Image | np.ndarray,
    *,
    window: int | None = None,
    min_interval: int | None = None,
    threshold: float = 0.16,
    margin: float = 0.12,
    ignore_rules: bool = True,
    use_color: bool = False,
) -> BoundaryResult:
    """Find texture-change boundaries in a *cropped* lithology column.

    Args:
        image: Image path, PIL image, or uint8 grayscale/RGB/RGBA array.
        window: Rows on EACH side of a candidate. Choose at least 2-3 texture
            repetitions; default is a width-based heuristic, not pitch estimation.
            A second comparison uses twice this window for phase robustness.
        min_interval: Minimum output interval height; default window // 2.
            Cuts are separated by at least max(min_interval, window), and must
            be at least that far from image edges. Thinner beds can be missed.
        threshold: Minimum texture-change score, NOT a probability. Lower it
            for faint changes; raise it to reduce oversegmentation.
        margin: Fraction removed from each side to avoid column borders/gutters.
        ignore_rules: Suppress short near-solid horizontal rules. Disable for
            very thin solid beds or almost continuous horizontal hatch patterns.
        use_color: Also treat fill-colour changes as evidence. Off by default
            because a colour change need not imply a different pattern.

    Returns:
        Internal boundary rows, complete contiguous intervals (including blank
        parts), and the score curve. No legend, OCR, or ML model is needed.
        Headers/page breaks are NOT automatically removed. Input should contain
        one upright column, not an entire document or multiple side-by-side tracks.
    """
    if not 0 <= margin < 0.45:
        raise ValueError('margin must be in [0, 0.45)')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be finite and positive')
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            rgb = np.asarray(loaded.convert('RGB'), dtype=np.float32) / 255
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError('arrays must be uint8 grayscale, RGB, or RGBA')
        if array.ndim == 3 and array.shape[2] not in (3, 4):
            raise ValueError('arrays must have 3 or 4 channels')
        rgb = np.asarray(Image.fromarray(array).convert('RGB'), dtype=np.float32) / 255
    height, width = rgb.shape[:2]
    if height < 3 or width < 3:
        raise ValueError('column must be at least 3 by 3 pixels')
    window = max(24, min(96, round(width * 0.6))) if window is None else window
    if not isinstance(window, (int, np.integer)) or window < 4:
        raise ValueError('window must be an integer >= 4')
    min_interval = max(4, window // 2) if min_interval is None else min_interval
    if not isinstance(min_interval, (int, np.integer)) or min_interval < 1:
        raise ValueError('min_interval must be a positive integer')
    trim = round(width * margin)
    interior = rgb[:, trim:width - trim]
    rows = _row_features(interior, ignore_rules, use_color)
    fine = _distance(*_mean_windows(rows, window), use_color)
    coarse = _distance(*_mean_windows(rows, window * 2), use_color)
    # A periodic stroke can make a narrow-window peak; demand support from the
    # larger context too. Geometric pooling retains the narrower peak location.
    scores = np.sqrt(fine * coarse)
    smoothing = max(1, window // 12)
    kernel = np.ones(2 * smoothing + 1) / (2 * smoothing + 1)
    scores = np.convolve(np.pad(scores, smoothing, mode='edge'), kernel, mode='valid')
    # Never infer changes at the image edges, where one side has no evidence.
    guard = max(min_interval, window)
    scores[:min(guard, height + 1)] = 0
    scores[max(0, height - guard + 1):] = 0
    candidates = np.flatnonzero((scores[1:-1] >= scores[:-2])
                               & (scores[1:-1] > scores[2:])
                               & (scores[1:-1] >= threshold)) + 1
    # A single broad peak may contain small scan-noise ripples. Suppress these
    # over a context-sized radius as well as honouring the minimum bed size.
    separation = max(min_interval, window)
    kept: list[int] = []
    for idx in candidates[np.argsort(-scores[candidates], kind='stable')]:
        y = int(idx)
        if all(abs(y - other) >= separation for other in kept):
            kept.append(y)
    kept.sort()
    edges = [0, *kept, height]
    return BoundaryResult(kept, list(zip(edges[:-1], edges[1:])), scores,
                          threshold, int(window))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='cropped lithology column')
    parser.add_argument('--output', type=Path, default=Path('boundary_output'))
    parser.add_argument('--window', type=int)
    parser.add_argument('--min-interval', type=int)
    parser.add_argument('--threshold', type=float, default=0.16)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--use-color', action='store_true')
    parser.add_argument('--keep-rules', action='store_true')
    args = parser.parse_args()
    result = detect_boundaries(args.image, window=args.window,
                               min_interval=args.min_interval, threshold=args.threshold,
                               margin=args.margin, use_color=args.use_color,
                               ignore_rules=not args.keep_rules)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.image.stem
    with stem.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['top_y', 'bottom_y'])
        writer.writerows(result.intervals)
    with Image.open(args.image) as original:
        overlay = original.convert('RGB')
    draw = ImageDraw.Draw(overlay)
    for y in result.boundaries:
        draw.line((0, y, overlay.width - 1, y), fill='red', width=2)
    overlay.save(stem.with_suffix('.png'))
    np.save(stem.with_suffix('.scores.npy'), result.scores)
    print(f'{len(result.intervals)} intervals; window={result.window}; '
          f'CSV, overlay, and scores in {args.output}')


if __name__ == '__main__':
    main()
