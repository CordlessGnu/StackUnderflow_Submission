"""Training-free interval boundaries from a cropped lithology column.

Only NumPy and Pillow are required. Coordinates are crop-local pixel edges;
intervals are half-open [top_y, bottom_y). No lithology naming is attempted.
Run ``python 'info extraction/texture_boundaries.py' --help`` for the CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    min_interval: int = 2
    scales: tuple[int, ...] = ()
    color_scores: np.ndarray | None = None
    refine_radius: int = 0


def _mean_windows(rows: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Means immediately above/below every edge; no vertical downsampling."""
    height = len(rows)
    sums = np.vstack((np.zeros((1, rows.shape[1])), np.cumsum(rows, axis=0)))
    y = np.arange(height + 1)
    lo, hi = np.maximum(0, y - window), np.minimum(height, y + window)
    above = (sums[y] - sums[lo]) / np.maximum(y - lo, 1)[:, None]
    below = (sums[hi] - sums[y]) / np.maximum(hi - y, 1)[:, None]
    return above, below


def _rgb_histogram(rgb: np.ndarray) -> np.ndarray:
    """Soft joint RGB histogram (3 x 3 x 3 bins) for each row.

    Unlike channel means, this distinguishes mixtures with the same average
    RGB. Soft binning avoids abrupt changes at quantization boundaries. Process
    small row blocks to keep temporary arrays bounded on very tall columns.
    """
    height, width = rgb.shape[:2]
    hist = np.empty((height, 27), dtype=np.float32)
    for start in range(0, height, 1024):
        block = rgb[start:start + 1024] * 2
        lower = np.minimum(block.astype(np.int32), 1)
        fraction = block - lower
        n = len(block)
        row_offset = np.arange(n)[:, None] * 27
        pooled = np.zeros(n * 27, dtype=np.float64)
        for r in (0, 1):
            for g in (0, 1):
                for b in (0, 1):
                    offsets = np.array([r, g, b])
                    bins = lower + offsets
                    index = row_offset + bins[:, :, 0] * 9 + bins[:, :, 1] * 3 + bins[:, :, 2]
                    weight = np.prod(np.where(offsets, fraction, 1 - fraction), axis=2)
                    pooled += np.bincount(index.ravel(), weights=weight.ravel(), minlength=n * 27)
        hist[start:start + n] = pooled.reshape(n, 27) / width
    return hist


def _row_features(rgb: np.ndarray, ignore_rules: bool, use_color: bool) -> np.ndarray:
    # Ink density still helps distinguish blank regions and solid beds, but RGB
    # gradients and distributions below also retain coloured strokes and fills.
    ink = np.clip((0.8 - rgb.max(axis=2)) / 0.5, 0, 1)
    height, width = ink.shape
    if ignore_rules:
        rgb = rgb.copy()  # the interior can be a view; never mutate the caller
        # Remove only SHORT, almost full-width rules, not long solid/coal beds.
        rules = np.mean(ink > 0.6, axis=1) > 0.90
        edges = np.flatnonzero(np.diff(np.r_[False, rules, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start <= max(2, width // 20) and start > 0 and end < height:
                alpha = np.arange(1, end - start + 1)[:, None] / (end - start + 1)
                ink[start:end] = (1 - alpha) * ink[start - 1] + alpha * ink[end]
                rgb[start:end] = ((1 - alpha[:, :, None]) * rgb[start - 1]
                                  + alpha[:, :, None] * rgb[end])

    if use_color:
        # RGB structure tensor: edges remain visible even when two colours have
        # the same grayscale intensity or the same maximum channel value.
        jxx = np.zeros((height, width), dtype=np.float32)
        jyy = np.zeros_like(jxx)
        jxy = np.zeros_like(jxx)
        for channel in range(3):
            gy, gx = np.gradient(rgb[:, :, channel])
            jxx += gx * gx / 3
            jyy += gy * gy / 3
            jxy += gx * gy / 3
        direction = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        magnitude = np.sqrt(np.maximum(0, (jxx + jyy + np.hypot(jxx - jyy, 2 * jxy)) / 2))
    else:
        gy, gx = np.gradient(ink)
        magnitude = np.hypot(gx, gy)
        direction = np.arctan2(gy, gx)
    # Soft voting avoids jumps at orientation-bin edges. Do NOT make this
    # rotation invariant: horizontal and vertical dashes are different textures.
    angle = np.mod(direction, np.pi) * (8 / np.pi)
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
        features.extend([rgb.mean(axis=1), _rgb_histogram(rgb)])
    return np.column_stack(features)


def _distance(a: np.ndarray, b: np.ndarray, use_color: bool) -> np.ndarray:
    ha, hb = a[:, :8], b[:, :8]
    ea, eb = ha.sum(axis=1), hb.sum(axis=1)
    pa = ha / np.maximum(ea[:, None], 1e-9)
    pb = hb / np.maximum(eb[:, None], 1e-9)
    hellinger = np.sqrt(np.square(np.sqrt(pa) - np.sqrt(pb)).sum(axis=1) / 2)
    # Near-blank windows have unreliable orientation distributions.
    reliability = np.minimum(1, np.minimum(ea, eb) / 0.015)
    end = -30 if use_color else a.shape[1]
    spatial = np.sqrt(np.square(a[:, 9:end] - b[:, 9:end]).mean(axis=1))
    score = (0.65 * hellinger * reliability
             + 0.20 * np.minimum(1, np.abs(ea - eb) / 0.12)
             + 0.25 * np.abs(a[:, 8] - b[:, 8])
             + 0.30 * spatial)
    if use_color:
        color_hist = np.sqrt(np.square(np.sqrt(np.maximum(a[:, -27:], 0))
                                       - np.sqrt(np.maximum(b[:, -27:], 0))).sum(axis=1) / 2)
        color_mean = np.sqrt(np.square(a[:, -30:-27] - b[:, -30:-27]).mean(axis=1))
        score += 0.50 * color_hist + 0.20 * color_mean
    return score


def _fill_profile(rgb: np.ndarray, min_interval: int) -> tuple[np.ndarray, np.ndarray]:
    """Dominant bright fill per row, ignoring dark strokes and mixed symbols.

    A maximum-channel floor of 0.5 deliberately treats darker marks as ink;
    very dark, thin fills are left to the texture detector rather than guessed.
    Missing/ink-only rows are interpolated, but long unsupported runs are NOT
    allowed to propose colour boundaries (e.g. an actual solid-black bed).
    """
    height, width = rgb.shape[:2]
    fill = np.full((height, 3), np.nan, dtype=np.float64)
    valid = np.zeros(height, dtype=bool)
    for start in range(0, height, 2048):
        block = rgb[start:start + 2048]
        value = block.max(axis=2)
        luminance = block @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        cutoff = np.quantile(luminance, 0.85, axis=1)
        bright = (luminance >= cutoff[:, None]) & (value >= 0.50)
        usable = bright.sum(axis=1) >= max(1, round(width * 0.1))
        if not usable.any():
            continue
        selected = block[usable]
        mask = bright[usable]
        centers = np.nanmedian(np.where(mask[:, :, None], selected, np.nan), axis=1)
        non_dark = value[usable] >= 0.50
        agreement = ((np.sqrt(np.square(selected - centers[:, None]).mean(axis=2)) < 0.12)
                     & non_dark).sum(axis=1) / non_dark.sum(axis=1)
        ids = start + np.flatnonzero(usable)
        coherent = agreement >= 0.60
        fill[ids[coherent]] = centers[coherent]
        valid[ids[coherent]] = True
    if valid.any():
        value = np.nan_to_num(fill, nan=0.0).max(axis=1)
        envelope = np.lib.stride_tricks.sliding_window_view(np.pad(value, 4, mode='edge'), 9).max(axis=1)
        dips = valid & (value < envelope * 0.75)
        edges = np.flatnonzero(np.diff(np.r_[False, dips, False]))
        for lo, hi in zip(edges[::2], edges[1::2]):
            if hi - lo <= 3:
                valid[lo:hi] = False
    if not valid.any():
        return np.zeros((height, 3)), valid
    y = np.arange(height)
    for channel in range(3):
        fill[:, channel] = np.interp(y, y[valid], fill[valid, channel])
    supported = valid.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, ~valid, False]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if lo > 0 and hi < height and hi - lo <= 2:
            supported[lo:hi] = True
    if min_interval >= 3:
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(fill, ((1, 1), (0, 0)), mode='edge'), 3, axis=0)
        fill = np.median(windows, axis=-1)
    return fill, supported


def _peaks(scores: np.ndarray, threshold: float, guard: int,
           context: int, prominence: float) -> list[int]:
    height = len(scores) - 1
    candidates = np.flatnonzero((scores[1:-1] >= scores[:-2])
                               & (scores[1:-1] > scores[2:])
                               & (scores[1:-1] >= threshold)) + 1
    result = []
    for index in candidates:
        y = int(index)
        if y < guard or y > height - guard:
            continue
        left = scores[max(guard, y - context):y + 1].min()
        right = scores[y:min(height - guard, y + context) + 1].min()
        if scores[y] - max(left, right) >= prominence:
            result.append(y)
    return result


def _periodic_color_cut(y: int, signatures: np.ndarray, maximum_lag: int,
                        contrast: float) -> bool:
    """Veto a colour edge that is itself a repeated stroke in one texture.

    Demand the same signed before/after colours over multiple cycles, not just
    similar flat rows. Short isolated beds therefore remain eligible. Near image
    edges, require three repeats on the available side instead of inventing data.
    """
    height = len(signatures) - 1
    current = signatures[y]
    edge_contrast = np.sqrt(np.square(current[:3] - current[3:]).mean())
    if edge_contrast < contrast * 0.8:
        return False
    tolerance = max(0.035, min(0.08, edge_contrast * 0.15))
    for lag in range(3, maximum_lag + 1):
        if y - 2 * lag >= 2 and y + 2 * lag <= height - 2:
            offsets = (-2, -1, 1, 2)
        elif y - 2 * lag < 2 and y + 3 * lag <= height - 2:
            offsets = (1, 2, 3)
        elif y + 2 * lag > height - 2 and y - 3 * lag >= 2:
            offsets = (-1, -2, -3)
        else:
            continue
        matched = True
        for multiple in offsets:
            center = y + multiple * lag
            neighborhood = signatures[max(2, center - 1):min(height - 1, center + 2)]
            if np.sqrt(np.square(neighborhood - current).mean(axis=1)).min() > tolerance:
                matched = False
                break
        if matched:
            return True
    return False


def _refine_fill_cut(y: int, fill: np.ndarray, previous: int, following: int,
                     radius: int, minimum: int, contrast: float) -> int:
    """Locate a contact by two-region SSE, bounded by neighbouring contacts.

    Use colour only when it supplies a real contrast; otherwise retain the
    texture estimate instead of snapping to a random dash/dot edge.
    """
    if radius == 0:
        return y
    start, stop = max(previous, y - radius), min(following, y + radius)
    lo, hi = max(previous + minimum, start + 1), min(following - minimum, stop - 1)
    if lo > hi:
        return y
    patch = fill[start:stop]
    sums = np.vstack((np.zeros((1, 3)), np.cumsum(patch, axis=0)))
    squares = np.r_[0.0, np.cumsum(np.square(patch).sum(axis=1))]
    positions = np.arange(lo, hi + 1)
    left_n, right_n = positions - start, stop - positions
    left, right = sums[left_n], sums[-1] - sums[left_n]
    costs = (squares[left_n] - np.square(left).sum(axis=1) / left_n
             + squares[-1] - squares[left_n] - np.square(right).sum(axis=1) / right_n)
    # Prefer the original proposal when several splits have indistinguishable costs.
    best = int(np.argmin(costs + 1e-8 * np.square(positions - y)))
    difference = np.sqrt(np.square(left[best] / left_n[best]
                                    - right[best] / right_n[best]).mean())
    return int(positions[best]) if difference >= contrast else y


def _verify_cuts(cuts: list[int], rows: np.ndarray, fill: np.ndarray,
                 threshold: float, color_threshold: float, use_color: bool) -> list[int]:
    """Merge adjacent fragments only when their interiors agree in appearance."""
    height = len(rows)
    sums = np.vstack((np.zeros((1, rows.shape[1])), np.cumsum(rows, axis=0)))
    colors = np.vstack((np.zeros((1, 3)), np.cumsum(fill, axis=0)))
    for _ in range(32):
        if not cuts:
            break
        edges = np.array([0, *cuts, height])
        trim = np.minimum(2, (np.diff(edges) - 1) // 4)
        lo, hi = edges[:-1] + trim, edges[1:] - trim
        means = (sums[hi] - sums[lo]) / (hi - lo)[:, None]
        fills = (colors[hi] - colors[lo]) / (hi - lo)[:, None]
        distances = _distance(means[:-1], means[1:], use_color)
        fill_delta = np.sqrt(np.square(fills[:-1] - fills[1:]).mean(axis=1))
        weak = np.flatnonzero((distances < threshold * 0.8)
                              & ((fill_delta < color_threshold * 0.8) | (not use_color)))
        if not len(weak):
            break
        # Do not simultaneously erase both sides of a potentially distinct bed.
        removed = set()
        for index in weak[np.argsort(distances[weak], kind='stable')]:
            i = int(index)
            if i - 1 not in removed and i + 1 not in removed:
                removed.add(i)
        cuts = [y for i, y in enumerate(cuts) if i not in removed]
    return cuts


def detect_boundaries(
    image: str | Path | Image.Image | np.ndarray,
    *,
    window: int | None = None,
    min_interval: int | None = None,
    threshold: float = 0.16,
    margin: float = 0.12,
    ignore_rules: bool = True,
    use_color: bool = True,
    color_threshold: float = 0.10,
    refine_radius: int | None = None,
) -> BoundaryResult:
    """Find texture-change boundaries in a *cropped* lithology column.

    Args:
        image: Image path, PIL image, or uint8 grayscale/RGB/RGBA array.
        window: Base rows on EACH side of texture candidates. Comparisons use
            half/base/double/quadruple contexts; adjacent scales support each
            other. The default base is width-derived, not pitch estimation.
        min_interval: Minimum interval height in pixels, default 2. Independent
            of the texture window, so a wide context does not delete thin beds.
        threshold: Minimum texture-change score, NOT a probability. Lower it
            for faint changes; raise it to reduce oversegmentation.
        margin: Fraction removed from each side to avoid column borders/gutters.
        ignore_rules: Suppress short near-solid horizontal rules. Disable for
            very thin solid beds or almost continuous horizontal hatch patterns.
        use_color: Use RGB gradients, soft joint colour histograms, and RGB
            means alongside texture features. On by default. False restores the
            previous ink-only features when fill-colour changes should be ignored.
        color_threshold: Minimum RMS RGB fill change in [0,1] units, default 0.10.
            Short-context colour candidates can preserve both edges of thin beds.
        refine_radius: Search radius for contact localization, default up to 24
            pixels. Colour-free contacts retain their texture estimate. 0 disables.

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
    if not np.isfinite(color_threshold) or color_threshold <= 0:
        raise ValueError('color_threshold must be finite and positive')
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
    min_interval = 2 if min_interval is None else min_interval
    if not isinstance(min_interval, (int, np.integer)) or min_interval < 1:
        raise ValueError('min_interval must be a positive integer')
    refine_radius = min(24, max(6, window)) if refine_radius is None else refine_radius
    if not isinstance(refine_radius, (int, np.integer)) or refine_radius < 0:
        raise ValueError('refine_radius must be a nonnegative integer')
    trim = round(width * margin)
    interior = rgb[:, trim:width - trim]
    rows = _row_features(interior, ignore_rules, use_color)
    scales = tuple(sorted(set((max(4, window // 2), window, window * 2, window * 4))))
    curves = [_distance(*_mean_windows(rows, w), use_color) for w in scales]
    scores = np.zeros(height + 1)
    texture_candidates = set()
    primary_candidates = set()
    primary_scores = np.zeros(height + 1)
    secondary_support = np.zeros(height + 1, dtype=bool)
    for index, context in enumerate(scales[:-1]):
        curve = np.sqrt(curves[index] * curves[index + 1])
        if index + 2 < len(curves):
            secondary_support |= ((curves[index] >= threshold) & (curves[index + 1] >= threshold)
                                  & (curves[index + 2] >= threshold))
        smoothing = max(1, context // 12)
        kernel = np.ones(2 * smoothing + 1) / (2 * smoothing + 1)
        curve = np.convolve(np.pad(curve, smoothing, mode='edge'), kernel, mode='valid')
        guard = max(min_interval, context)
        curve[:min(guard, height + 1)] = 0
        curve[max(0, height - guard + 1):] = 0
        peaks = _peaks(curve, threshold, guard, context, threshold * 0.25)
        texture_candidates.update(peaks)
        if context == window:
            primary_candidates.update(peaks)
            primary_scores = curve
        scores = np.maximum(scores, curve)
    # Collapse overlapping TEXTURE hypotheses for one broad contact, while
    # leaving short-context colour contacts outside this suppression entirely.
    # Prefer the established base-scale localization when that scale works.
    texture_kept = []
    for y in sorted(primary_candidates, key=lambda p: -primary_scores[p]):
        if all(abs(y - other) >= window for other in texture_kept):
            texture_kept.append(y)
    for y in sorted(texture_candidates - primary_candidates, key=lambda p: -scores[p]):
        occupied = primary_candidates | set(texture_kept)
        if (secondary_support[y] and scores[y] >= threshold * 1.25
                and all(abs(y - other) >= window for other in occupied)):
            texture_kept.append(y)
    texture_candidates = set(texture_kept)

    fill, supported = _fill_profile(interior, min_interval) if use_color else (np.zeros((height, 3)), np.zeros(height, bool))
    color_scores = np.zeros(height + 1)
    color_cuts: list[int] = []
    if use_color:
        signature_a, signature_b = _mean_windows(fill, 2)
        signatures = np.column_stack((signature_a, signature_b))
        normalized_a = signature_a / np.maximum(signature_a.max(axis=1, keepdims=True), 1e-6)
        normalized_b = signature_b / np.maximum(signature_b.max(axis=1, keepdims=True), 1e-6)
        chromaticity_change = np.sqrt(np.square(normalized_a - normalized_b).mean(axis=1))
        wide_a, wide_b = _mean_windows(fill, window)
        wide_change = np.sqrt(np.square(wide_a - wide_b).mean(axis=1))
        for context in sorted(set((max(2, min_interval), max(4, min_interval * 2), max(8, min_interval * 4)))):
            above, below = _mean_windows(fill, context)
            support_a, support_b = _mean_windows(supported[:, None], context)
            contrast = np.sqrt(np.square(above - below).mean(axis=1))
            chroma = np.maximum(np.ptp(above, axis=1), np.ptp(below, axis=1))
            reliable = ((support_a[:, 0] >= 0.7) & (support_b[:, 0] >= 0.7)
                        & (chroma >= 0.08))
            contrast[~reliable] = 0
            contrast[:min_interval] = 0
            contrast[max(0, height - min_interval + 1):] = 0
            color_scores = np.maximum(color_scores, contrast)
            proposed = _peaks(contrast, color_threshold, min_interval, context, color_threshold * 0.15)
            # Let fine-scale contacts win. A wide-window duplicate may otherwise
            # land halfway across a thin bed and steal one of its two edges.
            for y in sorted(proposed, key=lambda p: -contrast[p]):
                if (all(abs(y - other) >= context for other in color_cuts)
                        and not _periodic_color_cut(y, signatures, min(64, max(4, window // 2)), color_threshold)):
                    color_cuts.append(y)
        color_cuts.sort()
        initial = [0, *color_cuts, height]
        color_cuts = [_refine_fill_cut(y, fill, initial[i], initial[i + 2],
                                       refine_radius, min_interval, color_threshold * 0.7)
                      for i, y in enumerate(color_cuts)]
        color_cuts = [y for y in color_cuts
                      if (np.sqrt(np.square(signatures[y, :3] - signatures[y, 3:]).mean()) >= color_threshold * 0.7
                          and (chromaticity_change[y] >= color_threshold * 1.5 or wide_change[y] >= color_threshold)
                          and not _periodic_color_cut(y, signatures, min(64, max(4, window // 2)), color_threshold))]

    # Strong colour contacts locate a broad texture peak precisely. Keep both
    # ends of a nearby colour interval instead of window-sized suppression.
    proposals = set(color_cuts)
    for y in texture_candidates:
        if color_cuts and min(abs(y - other) for other in color_cuts) <= refine_radius:
            continue
        proposals.add(y)
    priority = lambda y: (y in color_cuts, float(max(scores[y], color_scores[y])), -y)
    kept = []
    for y in sorted(proposals, key=priority, reverse=True):
        if min_interval <= y <= height - min_interval and all(abs(y - other) >= min_interval for other in kept):
            kept.append(y)
    kept.sort()
    kept = _verify_cuts(kept, rows, fill, threshold, color_threshold, use_color)
    scores = np.maximum(scores, color_scores * threshold / color_threshold)
    edges = [0, *kept, height]
    return BoundaryResult(kept, list(zip(edges[:-1], edges[1:])), scores,
                          threshold, int(window), int(min_interval), scales, color_scores, int(refine_radius))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='cropped lithology column')
    parser.add_argument('--output', type=Path, default=Path('boundary_output'))
    parser.add_argument('--window', type=int)
    parser.add_argument('--min-interval', type=int)
    parser.add_argument('--threshold', type=float, default=0.16)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--color-threshold', type=float, default=0.10)
    parser.add_argument('--refine-radius', type=int)
    color = parser.add_mutually_exclusive_group()
    color.add_argument('--use-color', dest='use_color', action='store_true',
                       help='use RGB and texture evidence (default)')
    color.add_argument('--no-color', dest='use_color', action='store_false',
                       help='disable RGB features and short-context colour contacts')
    parser.set_defaults(use_color=True)
    parser.add_argument('--keep-rules', action='store_true')
    args = parser.parse_args()
    result = detect_boundaries(args.image, window=args.window,
                               min_interval=args.min_interval, threshold=args.threshold,
                               margin=args.margin, use_color=args.use_color,
                               ignore_rules=not args.keep_rules, color_threshold=args.color_threshold,
                               refine_radius=args.refine_radius)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.image.stem
    with Path(f'{stem}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['top_y', 'bottom_y'])
        writer.writerows(result.intervals)
    with Image.open(args.image) as original:
        overlay = original.convert('RGB')
    draw = ImageDraw.Draw(overlay)
    for y in result.boundaries:
        draw.line((0, y, overlay.width - 1, y), fill='red', width=2)
    overlay.save(Path(f'{stem}.png'))
    np.save(Path(f'{stem}.scores.npy'), result.scores)
    np.save(Path(f'{stem}.color_scores.npy'), result.color_scores)
    Path(f'{stem}.parameters.json').write_text(json.dumps(dict(
        window=result.window, scales=result.scales, min_interval=result.min_interval,
        threshold=result.threshold, color_threshold=args.color_threshold,
        refine_radius=result.refine_radius, use_color=args.use_color), indent=2))
    print(f'{len(result.intervals)} intervals; scales={result.scales}; '
          f'min_interval={result.min_interval}; CSV, overlay, and diagnostics in {args.output}')


if __name__ == '__main__':
    main()
