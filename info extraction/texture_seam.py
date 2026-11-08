"""Bounded artificial-seam comparisons using the boundary detector's features.

No full stacked image is built or re-segmented. Features are extracted separately
on each side, avoiding an artificial derivative at the join. Cached multiscale
head/tail summaries allow small phase shifts and both stacking orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from texture_boundaries import _distance, _row_features

CONTEXTS = (4, 8, 16, 32, 64)


@dataclass(eq=False)
class PreparedTexture:
    descriptor: np.ndarray
    heads: dict[int, np.ndarray]
    tails: dict[int, np.ndarray]
    native_height: int
    normalized_height: int
    samples: int


@dataclass
class SeamComparison:
    score: float | None
    descriptor_distance: float
    status: str  # same / different / uncertain; never a probability
    reason: str
    scales: tuple[int, ...]
    scale_scores: tuple[float, ...]


def prepare_texture(image: str | Path | Image.Image | np.ndarray, *,
                    y_scale: float = 1.0, max_samples: int = 3) -> PreparedTexture:
    """Cache at most three 64x128 local patches; do not squash a whole bed.

    Very short inputs are edge-padded ONLY for fixed-size feature extraction;
    the real evidence height is retained, so padding cannot validate a match.
    Legend y_scale changes rendering geometry, not the native evidence count.
    """
    if not math.isfinite(y_scale) or y_scale <= 0 or not 1 <= max_samples <= 7:
        raise ValueError('y_scale must be positive; max_samples must be in [1,7]')
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            im = loaded.convert('RGB')
    elif isinstance(image, Image.Image):
        im = image.convert('RGB')
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError('image array must be uint8 grayscale/RGB/RGBA')
        if array.ndim == 3 and array.shape[2] not in (3, 4):
            raise ValueError('image array must have 3 or 4 channels')
        im = Image.fromarray(array).convert('RGB')
    width, height = im.size
    if min(width, height) < 1:
        raise ValueError('patch must be nonempty')
    patch_height = min(height, max(1, int(width * 2 / y_scale)))
    count = min(max_samples, max(1, math.ceil(height / patch_height)))
    starts = np.unique(np.linspace(0, height - patch_height, count).astype(int))
    target_height = max(1, min(128, round(patch_height * 64 / width * y_scale)))
    heads: dict[int, list] = {context: [] for context in CONTEXTS}
    tails: dict[int, list] = {context: [] for context in CONTEXTS}
    pooled = []
    for start in starts:
        patch = im.crop((0, int(start), width, int(start) + patch_height))
        patch = patch.resize((64, target_height), Image.Resampling.BILINEAR)
        rgb = np.asarray(patch, dtype=np.float32) / 255
        padded = np.pad(rgb, ((0, max(0, 9 - target_height)), (0, 0), (0, 0)), mode='edge')
        rows = _row_features(padded, ignore_rules=True, use_color=True)[:target_height]
        # Discard derivative pollution at patch boundaries when there is room.
        if len(rows) >= 24:
            rows = rows[2:-2]
        pooled.append(rows.mean(axis=0))
        for context in CONTEXTS:
            if context > len(rows):
                continue
            for shift in (0, 2, 4, 8):
                if context + shift > len(rows):
                    continue
                heads[context].append(rows[shift:shift + context].mean(axis=0))
                end = len(rows) - shift
                tails[context].append(rows[end - context:end].mean(axis=0))
    return PreparedTexture(np.mean(pooled, axis=0),
                           {c: np.asarray(v, dtype=np.float32) for c, v in heads.items() if v},
                           {c: np.asarray(v, dtype=np.float32) for c, v in tails.items() if v},
                           patch_height, target_height, len(starts))


def _coverage_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric phase-tolerant coverage, not one lucky best patch pair."""
    shape = (len(a), len(b), a.shape[1])
    left = np.broadcast_to(a[:, None, :], shape).reshape(-1, a.shape[1])
    right = np.broadcast_to(b[None, :, :], shape).reshape(-1, a.shape[1])
    distances = _distance(left, right, use_color=True).reshape(len(a), len(b))
    return float(max(np.median(distances.min(axis=0)), np.median(distances.min(axis=1))))


class SeamVerifier:
    """Memoized local seam evidence; cached scores can use different thresholds."""
    def __init__(self):
        self._cache = {}
        self.evaluations = 0
        self.cache_hits = 0

    def compare(self, a: PreparedTexture, b: PreparedTexture, *, threshold: float = 0.20) -> SeamComparison:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('seam threshold must be finite and positive')
        key = frozenset((a, b))
        if key in self._cache:
            self.cache_hits += 1
            descriptor_distance, scales, scores, score = self._cache[key]
        else:
            self.evaluations += 1
            descriptor_distance = float(_distance(a.descriptor[None], b.descriptor[None], True)[0])
            scales = tuple(c for c in CONTEXTS if c in a.heads and c in b.heads)
            scores = tuple(max(_coverage_distance(a.tails[c], b.heads[c]),
                               _coverage_distance(a.heads[c], b.tails[c])) for c in scales)
            # The larger supported contexts are less sensitive to individual
            # strokes; narrow contexts still serve short but usable swatches.
            score = float(np.median(scores[-2:])) if scores else None
            self._cache[key] = (descriptor_distance, scales, scores, score)
        enough = min(a.native_height, b.native_height, a.normalized_height, b.normalized_height) >= 8 and len(scales) >= 2
        blank_a = a.descriptor[:8].sum() < 0.005 and np.min(a.descriptor[17:20]) > 0.97
        blank_b = b.descriptor[:8].sum() < 0.005 and np.min(b.descriptor[17:20]) > 0.97
        if enough and blank_a != blank_b and max(a.descriptor[:8].sum(), b.descriptor[:8].sum()) > 0.01:
            status, reason = 'different', 'blank_vs_texture'
        elif descriptor_distance > max(0.35, threshold * 2):
            status, reason = 'different', 'strong_appearance_difference'
        elif not enough:
            status, reason = 'uncertain', 'insufficient_texture_context'
        elif score <= threshold and descriptor_distance <= threshold * 1.6:
            status, reason = 'same', 'multiscale_seam_agreement'
        elif score >= threshold * 1.5:
            status, reason = 'different', 'multiscale_seam_change'
        else:
            status, reason = 'uncertain', 'borderline_seam_evidence'
        return SeamComparison(score, descriptor_distance, status, reason, scales, scores)

    def statistics(self) -> dict:
        return dict(seam_evaluations=self.evaluations, seam_cache_hits=self.cache_hits)


def compare_interval_patches(a, b, *, threshold: float = 0.20) -> SeamComparison:
    """Public convenience API; repeated matching should reuse prepared caches."""
    return SeamVerifier().compare(prepare_texture(a), prepare_texture(b), threshold=threshold)
