#!/usr/bin/env python3
"""
Find the lithology intervals inside an interpreted lithology column.

The interpreted lithology column is a vertical strip in which every depth
interval is filled with the lithology pattern (texture) of that interval. This
module turns that strip into intervals:

    [(top=1032, bottom=1846, lithology='Sandstone'),
     (top=1847, bottom=2122, lithology='Granite'), ...]

Pipeline
--------
1. **Straighten** - the scanned column is slightly skewed, so the left/right
   rules are located per band instead of once for the whole image.
2. **Describe** - every band is reduced to a small texture descriptor
   (edge-orientation histogram, run lengths, ink density). This is scale and
   phase independent, which matters because the legend swatch and the same
   pattern inside the column are rendered at different sizes.
3. **Detect boundaries** - a change point is where the mean descriptor of the
   bands above differs most from the mean of the bands below. Working on
   change points instead of per-band labels keeps the boundaries stable.
4. **Classify** - each finished segment is matched against the pattern library
   using the mean descriptor of the whole segment.

Pattern library
---------------
The library is the pair of folders produced by ``crop_dataset_classes.py``:

    crops/lithology/patterns/*.png   texture swatch
    crops/lithology/name/*.png       matching name crop (same file stem)

Names come from ``--manifest``, from OCR (``--ocr``), or the file stem.

Usage
-----
    # one column, print the intervals
    python "boundary_detection.py" --column crops/interpreted_lithology_column/foo.png

    # save JSON + CSV + an annotated overlay
    python "boundary_detection.py" --column foo.png --json out.json --csv out.csv --overlay out.png

    # every column in a folder
    python "boundary_detection.py" --column-dir crops/interpreted_lithology_column --out-dir results

    # tune the segmentation
    python "boundary_detection.py" --column foo.png --sensitivity 1.2 --min-interval 40
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency ({exc.name}):  pip install -r requirements.txt")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATTERNS = REPO_ROOT / "crops" / "lithology" / "patterns"
DEFAULT_NAMES = REPO_ROOT / "crops" / "lithology" / "name"
DEFAULT_COLUMNS = REPO_ROOT / "crops" / "interpreted_lithology_column"

# Feature weights: orientation dominates, run lengths refine, density only
# separates solid fills (coal) from line art.
FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.3])

# Edge-orientation bins in degrees (0 = horizontal edge, 90 = vertical edge).
ORIENTATION_BINS = [(0.0, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5)]

# A column rule is a run of columns that are almost entirely ink.
RULE_DENSITY = 0.80

# Patches are block-averaged onto this grid before describing them, which makes
# the descriptor independent of the swatch/column scale difference.
DESCRIPTOR_GRID = 64

# Legend entries that are not lithologies. They are blank or generic swatches
# that otherwise win matches against sparse textures, so --filter-names drops
# them. Anything containing a real lithology word is always kept.
NON_LITHOLOGY = {
    "not", "available", "casing", "shoe", "void", "survey", "core", "cutting",
    "cuttings", "sample", "loss", "show", "others", "other", "test", "fit",
    "deviation", "sidewall", "mud", "gas", "oil", "water", "total", "depth",
    "date", "company", "well", "field", "hole", "size", "diameter", "bit",
}

# Words that mark a genuine lithology / texture entry.
LITHOLOGY_WORDS = {
    "sand", "sandstone", "sandy", "shale", "shaly", "clay", "claystone",
    "silt", "siltstone", "mud", "mudstone", "lime", "limestone", "dolomite",
    "dolomitic", "dolostone", "chert", "cherty", "halite", "salt", "anhydrite",
    "gypsum", "coal", "lignite", "marl", "chalk", "tuff", "till", "gravel",
    "conglomerate", "breccia", "granite", "basalt", "gneiss", "schist", "slate",
    "quartz", "feldspar", "mica", "pyrite", "glauconite", "fossil", "fossils",
    "shell", "wackestone", "packstone", "grainstone", "boundstone", "calcareous",
    "siliceous", "argillaceous", "ferruginous", "carbonate", "evaporite",
    "igneous", "metamorphic", "sedimentary", "andesite", "gabbro", "diorite",
    "rhyolite", "dacite", "syenite", "tonalite", "trachyte", "peridotite",
    "serpentine", "hornfels", "marble", "quartzite", "amphibolite", "phyllite",
    "pegmatite", "felsic", "mafic", "dike", "dyke", "sill", "vein", "fault",
    "altered", "alteration", "sericitization", "biotitization", "chloritization",
    "silicification", "dolomitization", "ash", "pumice", "scoria", "lapilli",
    "agglomerate", "ignimbrite", "welded", "tephra", "diatomite", "bitumen",
    "bituminous", "tar", "pebble", "pebbles", "extrusive", "intrusive",
    "foraminifera", "obsidian", "paleosol", "phosphate", "glassy", "wash",
    "porcelanite", "porcelaneous", "calcilutite", "calcarenite", "calcirudite",
    "coquina", "pyroclastic", "pyroclastics", "volcaniclastic", "volcaniclastics",
    "clast", "clasts", "clastic", "sediment", "sediments", "seds", "hornblende",
    "sylvite", "bentonite", "trona", "potash", "zeolite", "barite", "celestite",
    "fluorite", "phosphatic", "gypsiferous", "siderite", "limonite", "hematite",
    "magnetite", "goethite", "kaolinite", "illite", "smectite", "chlorite",
    "cement", "volcanic", "volcanics", "tuffaceous", "micrite", "sparite",
    "loess", "diamictite", "aragonite", "calcite", "peat", "rubble", "cavings",
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Interval:
    """One lithology interval inside the column."""

    top: int
    bottom: int
    lithology: str
    confidence: float = 0.0
    runner_up: str = ""

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_dict(self) -> dict:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "height": self.height,
            "lithology": self.lithology,
            "confidence": round(self.confidence, 3),
            "runner_up": self.runner_up,
        }

    def __repr__(self) -> str:  # the requested output style
        return (
            f"(top={self.top}, bottom={self.bottom}, "
            f"lithology={self.lithology!r})"
        )


@dataclass
class Pattern:
    """One entry of the pattern library."""

    name: str
    path: Path
    descriptor: np.ndarray


@dataclass
class ColumnResult:
    """Segmentation result for one column image."""

    image: str
    width: int
    height: int
    strip_width: int
    intervals: list[Interval] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "strip_width": self.strip_width,
            "interval_count": len(self.intervals),
            "intervals": [i.to_dict() for i in self.intervals],
        }


# --------------------------------------------------------------------------- #
# Texture descriptor
# --------------------------------------------------------------------------- #
def binarize(image: Image.Image) -> np.ndarray:
    """Return a float array with 1.0 for ink and 0.0 for background."""
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return (array < 128).astype(np.float32)


def _run_lengths(values: np.ndarray) -> list[int]:
    """Lengths of the runs of 1s in a 1-D array."""
    lengths: list[int] = []
    count = 0
    for value in values:
        if value:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def block_mean(binary: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Downsample a binary patch to `rows` x `cols` by block averaging."""
    height, width = binary.shape
    if height < rows or width < cols:
        return binary
    y_edges = np.arange(rows) * height // rows
    x_edges = np.arange(cols) * width // cols
    sums = np.add.reduceat(
        np.add.reduceat(binary, y_edges, axis=0), x_edges, axis=1
    )
    counts = np.outer(
        np.diff(np.append(y_edges, height)), np.diff(np.append(x_edges, width))
    )
    return (sums / counts).astype(np.float32)


def texture_descriptor(
    binary: np.ndarray, grid: int = DESCRIPTOR_GRID
) -> np.ndarray:
    """
    Reduce a binary patch to a scale- and phase-independent descriptor.

    The patch is first block-averaged onto a fixed `grid` x `grid` raster, so a
    small legend swatch and a wide column band produce comparable numbers.

    Returns 7 values: 4 edge-orientation bins, mean horizontal run length,
    mean vertical run length, and ink density.
    """
    if grid and binary.shape[0] > grid and binary.shape[1] > grid:
        binary = block_mean(binary, grid, grid)

    height, width = binary.shape
    if height < 3 or width < 3:
        return np.zeros(7, dtype=np.float32)

    # The gradient of a binary stroke points across it, so a horizontal dash
    # produces a vertical gradient. Fold the angle into 0-180 degrees.
    gy = np.zeros_like(binary)
    gx = np.zeros_like(binary)
    gy[1:-1, :] = binary[2:, :] - binary[:-2, :]
    gx[:, 1:-1] = binary[:, 2:] - binary[:, :-2]
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    mask = magnitude > 0.5

    histogram = np.zeros(4, dtype=np.float32)
    if mask.any():
        for i, (low, high) in enumerate(ORIENTATION_BINS):
            histogram[i] = np.count_nonzero(mask & (angle >= low) & (angle < high))
        total = histogram.sum()
        if total:
            histogram /= total

    horizontal = [length for row in binary for length in _run_lengths(row)]
    vertical = [length for column in binary.T for length in _run_lengths(column)]
    mean_h = (float(np.mean(horizontal)) / width) if horizontal else 0.0
    mean_v = (float(np.mean(vertical)) / height) if vertical else 0.0

    return np.array(
        [*histogram, mean_h, mean_v, float(binary.mean())], dtype=np.float32
    )


def descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted Euclidean distance between two descriptors."""
    return float(np.linalg.norm((a - b) * FEATURE_WEIGHTS))


# --------------------------------------------------------------------------- #
# Pattern library
# --------------------------------------------------------------------------- #
_OCR_ENGINE = None


def _ocr_name(image: Image.Image) -> str:
    """Read a name crop with RapidOCR (recognition stage only)."""
    global _OCR_ENGINE
    try:
        from rapidocr import RapidOCR
    except ImportError:  # pragma: no cover
        return ""
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    try:
        result = _OCR_ENGINE(
            np.asarray(image.convert("RGB")),
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
    except Exception:  # noqa: BLE001 - OCR must never break segmentation
        return ""
    texts = getattr(result, "txts", None) or ()
    return " ".join(" ".join(texts).split())


def load_library(
    patterns_dir: Path,
    names_dir: Path | None = None,
    manifest: Path | None = None,
    use_ocr: bool = False,
) -> list[Pattern]:
    """Build the pattern library from a folder of swatches."""
    if not patterns_dir.is_dir():
        raise SystemExit(f"pattern folder not found: {patterns_dir}")

    name_by_stem: dict[str, str] = {}
    if manifest and manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in data.get("crops", data.get("pairs", [])):
            path = entry.get("path") or entry.get("name_crop") or ""
            text = entry.get("name_text") or entry.get("lithology") or ""
            if path and text:
                name_by_stem[Path(path).stem] = text

    library: list[Pattern] = []
    for path in sorted(patterns_dir.glob("*.png")):
        stem = path.stem
        name = name_by_stem.get(stem, "")

        if not name and names_dir is not None:
            name_path = names_dir / f"{stem}.png"
            if name_path.is_file():
                if use_ocr:
                    with Image.open(name_path) as name_image:
                        name = _ocr_name(name_image)
                if not name:
                    name = stem

        if not name:
            name = stem

        with Image.open(path) as image:
            descriptor = texture_descriptor(binarize(image))
        library.append(Pattern(name=name, path=path, descriptor=descriptor))

    if not library:
        raise SystemExit(f"no .png swatches found in {patterns_dir}")
    return library


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end indices of the runs of 1s in a 1-D array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_swatches(
    binary: np.ndarray,
    min_width: int = 50,
    min_height: int = 30,
    max_width: int = 400,
    max_height: int = 300,
    tolerance: int = 4,
) -> list[tuple[int, int, int, int]]:
    """
    Find the bordered swatch rectangles in a legend image.

    The left and right sides of a swatch are continuous vertical rules, while
    the top and bottom rules are often broken where the pattern touches them.
    So we pair vertical rules first and then confirm with the horizontal ones.
    """
    height, width = binary.shape

    verticals: list[tuple[int, int, int]] = []
    for x in range(width):
        for y1, y2 in _runs_1d(binary[:, x]):
            if y2 - y1 >= min_height:
                verticals.append((x, int(y1), int(y2)))

    horizontals: dict[int, list[tuple[int, int]]] = {}
    for y in range(height):
        for x1, x2 in _runs_1d(binary[y]):
            if x2 - x1 >= min_width:
                horizontals.setdefault(y, []).append((int(x1), int(x2)))

    def has_horizontal(y: int, x1: int, x2: int) -> bool:
        span = x2 - x1
        for dy in range(-tolerance, tolerance + 1):
            for left, right in horizontals.get(y + dy, ()):
                # The rule is often clipped where the pattern touches it, so
                # require substantial overlap rather than full coverage.
                overlap = min(right, x2) - max(left, x1)
                if overlap >= 0.85 * span:
                    return True
        return False

    found: list[tuple[int, int, int, int]] = []
    for i, (x1, top1, bottom1) in enumerate(verticals):
        for x2, top2, bottom2 in verticals[i + 1 :]:
            if not (min_width <= x2 - x1 <= max_width):
                continue
            if abs(top1 - top2) > tolerance or abs(bottom1 - bottom2) > tolerance:
                continue
            top, bottom = min(top1, top2), max(bottom1, bottom2)
            if not (min_height <= bottom - top <= max_height):
                continue
            if has_horizontal(top, x1, x2) and has_horizontal(bottom, x1, x2):
                found.append((x1, top, x2, bottom))

    # A solid swatch (coal) makes every interior column look like a rule, so
    # keep the largest rectangles and drop the slices nested inside them.
    def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    found.sort(key=lambda r: -((r[2] - r[0]) * (r[3] - r[1])))
    unique: list[tuple[int, int, int, int]] = []
    for rect in found:
        if not any(overlaps(rect, other) for other in unique):
            unique.append(rect)
    return sorted(unique, key=lambda r: (r[1], r[0]))


def load_library_from_legend(
    legend_path: Path, use_ocr: bool = True
) -> list[Pattern]:
    """
    Build a pattern library from a legend image of the same document.

    This is the accurate mode: the swatches and the column come from the same
    scan, so their rendering and scale match exactly.
    """
    with Image.open(legend_path) as image:
        image = image.convert("L")
        binary = binarize(image)
        swatches = detect_swatches(binary)
        if not swatches:
            raise SystemExit(f"no swatches found in legend: {legend_path}")
        if len(swatches) < 3:
            print(
                f"warning: only {len(swatches)} swatch(es) detected in "
                f"{legend_path.name}; the legend borders may be broken. "
                "Consider --patterns instead.",
                file=sys.stderr,
            )

        library: list[Pattern] = []
        for x1, y1, x2, y2 in swatches:
            # The label sits to the right of the swatch, up to the next swatch.
            label_right = min(image.width, x2 + 220)
            for other_x1, other_y1, _, _ in swatches:
                if other_x1 > x2 and abs(other_y1 - y1) < 20:
                    label_right = min(label_right, other_x1 - 4)
            name = ""
            if use_ocr and label_right > x2 + 4:
                name = _ocr_name(image.crop((x2 + 4, y1, label_right, y2)))
            if not name:
                name = f"swatch_{x1}_{y1}"

            swatch = image.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2))
            library.append(
                Pattern(
                    name=name,
                    path=legend_path,
                    descriptor=texture_descriptor(binarize(swatch)),
                )
            )
    return library


def is_lithology_name(name: str) -> bool:
    """True when a legend name looks like a lithology rather than a tool entry."""
    words = set(re.findall(r"[a-z]+", name.lower()))
    if words & LITHOLOGY_WORDS:
        return True
    return not (words & NON_LITHOLOGY)


def collapse_library(library: Sequence[Pattern]) -> list[Pattern]:
    """Average the descriptors of patterns that share a name."""
    grouped: dict[str, list[np.ndarray]] = {}
    first_path: dict[str, Path] = {}
    for pattern in library:
        grouped.setdefault(pattern.name, []).append(pattern.descriptor)
        first_path.setdefault(pattern.name, pattern.path)
    return [
        Pattern(
            name=name,
            path=first_path[name],
            descriptor=np.mean(descriptors, axis=0),
        )
        for name, descriptors in grouped.items()
    ]


# --------------------------------------------------------------------------- #
# Column geometry
# --------------------------------------------------------------------------- #
def detect_borders(band: np.ndarray, threshold: float = RULE_DENSITY):
    """
    Find the left and right rules of the pattern strip inside a band.

    Returns (left, right) as the first and last patterned columns, or None when
    the band has no clear rules (for example a solid coal interval).
    """
    density = band.mean(axis=0)
    solid = np.flatnonzero(density > threshold)
    if len(solid) < 2:
        return None

    left = int(solid[0])
    while left + 1 < len(density) and density[left + 1] > threshold:
        left += 1
    right = int(solid[-1])
    while right - 1 >= 0 and density[right - 1] > threshold:
        right -= 1

    if right - left < 8:
        return None
    return left + 1, right


def extract_patch(
    binary: np.ndarray, top: int, height: int, fallback: tuple[int, int] | None
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Cut one band out of the column, inside the detected rules."""
    band = binary[top : top + height]
    borders = detect_borders(band) or fallback
    if borders is None:
        return band, None
    left, right = borders
    return band[:, left:right], borders


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def _smooth(features: np.ndarray, width: int) -> np.ndarray:
    """Moving average over the feature sequence."""
    if width <= 1 or len(features) < width:
        return features
    kernel = np.ones(width, dtype=np.float32) / width
    padded = np.pad(features, ((width // 2, width // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(features.shape[1])],
        axis=1,
    )[: len(features)]


def detect_change_points(
    features: np.ndarray, window: int = 8, sensitivity: float = 1.0
) -> list[int]:
    """
    Return the indices where the texture changes.

    For every position the mean descriptor of the `window` bands above is
    compared with the mean of the `window` bands below; a boundary is a local
    maximum of that difference above `sensitivity` standard deviations.
    """
    count = len(features)
    if count < 2 * window + 2:
        return []

    difference = np.zeros(count, dtype=np.float32)
    for i in range(window, count - window):
        above = features[i - window : i].mean(axis=0)
        below = features[i : i + window].mean(axis=0)
        difference[i] = np.linalg.norm((above - below) * FEATURE_WEIGHTS)

    spread = float(difference.std())
    if spread <= 1e-6:
        return []
    threshold = float(difference.mean()) + sensitivity * spread

    peaks: list[int] = []
    for i in range(1, count - 1):
        if (
            difference[i] >= difference[i - 1]
            and difference[i] >= difference[i + 1]
            and difference[i] > threshold
        ):
            peaks.append(i)

    # Non-maximum suppression: keep the strongest peak in each window.
    kept: list[int] = []
    for peak in sorted(peaks, key=lambda p: -difference[p]):
        if all(abs(peak - other) >= window for other in kept):
            kept.append(peak)
    return sorted(kept)


def classify(
    descriptor: np.ndarray, library: Sequence[Pattern]
) -> tuple[str, float, str]:
    """Return (name, confidence, runner_up) for a descriptor."""
    scored = sorted(
        ((descriptor_distance(descriptor, p.descriptor), p.name) for p in library),
        key=lambda item: item[0],
    )
    best_distance, best_name = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else ""
    second_distance = scored[1][0] if len(scored) > 1 else best_distance + 1.0

    margin = (second_distance - best_distance) / (second_distance + 1e-6)
    confidence = float(np.clip(margin, 0.0, 1.0) * np.exp(-best_distance / 2.0))
    return best_name, confidence, runner_up


def segment_column(
    image_path: Path,
    library: Sequence[Pattern],
    band: int = 100,
    step: int = 10,
    window: int = 8,
    sensitivity: float = 1.0,
    min_interval: int = 30,
    min_confidence: float = 0.0,
    unknown_label: str = "unknown",
) -> ColumnResult:
    """Segment one interpreted lithology column into lithology intervals."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size
        binary = binarize(image)

    if height < band + 2:
        band = max(8, height // 2)

    # 1. Describe every band, following the rules as the column drifts.
    positions: list[int] = []
    descriptors: list[np.ndarray] = []
    fallback: tuple[int, int] | None = None
    strip_width = 0
    for top in range(0, max(1, height - band + 1), step):
        patch, fallback = extract_patch(binary, top, band, fallback)
        if patch.size == 0:
            continue
        strip_width = max(strip_width, patch.shape[1])
        positions.append(top)
        descriptors.append(texture_descriptor(patch))

    if not descriptors:
        return ColumnResult(str(image_path), width, height, 0)

    features = _smooth(np.vstack(descriptors), width=3)

    # 2. Boundaries from texture change points.
    change_points = detect_change_points(features, window=window, sensitivity=sensitivity)

    # 3. Turn change points into pixel segments.
    edges = [0] + [positions[i] for i in change_points] + [height]
    edges = sorted(set(edges))

    intervals: list[Interval] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start < 1:
            continue
        # Mean descriptor of every band inside this segment.
        inside = [
            descriptors[i]
            for i, top in enumerate(positions)
            if start <= top < end
        ]
        if not inside:
            continue
        name, confidence, runner_up = classify(np.mean(inside, axis=0), library)
        if confidence < min_confidence:
            name = unknown_label
        intervals.append(
            Interval(
                top=int(start),
                bottom=int(end),
                lithology=name,
                confidence=confidence,
                runner_up=runner_up,
            )
        )

    intervals = _merge_short(intervals, min_interval)
    return ColumnResult(
        image=str(image_path),
        width=width,
        height=height,
        strip_width=strip_width,
        intervals=intervals,
    )


def _merge_short(intervals: list[Interval], min_interval: int) -> list[Interval]:
    """Absorb intervals shorter than `min_interval` into a neighbour."""
    if min_interval <= 0 or len(intervals) < 2:
        return intervals

    merged = list(intervals)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, interval in enumerate(merged):
            if interval.height >= min_interval:
                continue
            left = merged[i - 1] if i > 0 else None
            right = merged[i + 1] if i + 1 < len(merged) else None
            if left is None and right is None:
                continue
            if left is None:
                target = right
            elif right is None:
                target = left
            else:
                target = left if left.height >= right.height else right
            target.top = min(target.top, interval.top)
            target.bottom = max(target.bottom, interval.bottom)
            merged.pop(i)
            changed = True
            break

    joined: list[Interval] = []
    for interval in merged:
        if joined and joined[-1].lithology == interval.lithology:
            joined[-1].bottom = interval.bottom
            joined[-1].confidence = max(joined[-1].confidence, interval.confidence)
        else:
            joined.append(interval)
    return joined


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def draw_overlay(image_path: Path, result: ColumnResult, dest: Path) -> Path:
    """Write the column with interval boundaries and labels drawn on it."""
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    palette = [
        (220, 50, 50), (50, 120, 220), (40, 160, 80), (200, 140, 30),
        (150, 60, 200), (30, 170, 170), (200, 60, 140), (110, 110, 40),
    ]
    for i, interval in enumerate(result.intervals):
        colour = palette[i % len(palette)]
        draw.line([(0, interval.top), (canvas.width, interval.top)], fill=colour, width=2)
        draw.line([(0, interval.bottom), (canvas.width, interval.bottom)], fill=colour, width=2)
        draw.text(
            (2, interval.top + 2),
            f"{interval.lithology} [{interval.top}-{interval.bottom}]",
            fill=colour,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


def write_json(results: Sequence[ColumnResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": len(results),
        "intervals": sum(len(r.intervals) for r in results),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(results: Sequence[ColumnResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image", "top", "bottom", "height", "lithology", "confidence", "runner_up",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for interval in result.intervals:
                row = interval.to_dict()
                row["image"] = result.image
                writer.writerow({k: row[k] for k in fields})


def print_intervals(result: ColumnResult, show_confidence: bool) -> None:
    print(f"\n{Path(result.image).name}")
    print(
        f"  size {result.width}x{result.height}  strip={result.strip_width}px  "
        f"intervals={len(result.intervals)}"
    )
    if not result.intervals:
        print("  no intervals found")
        return
    print()
    for interval in result.intervals:
        line = (
            f"  top={interval.top:<7} bottom={interval.bottom:<7} "
            f"height={interval.height:<6} {interval.lithology}"
        )
        if show_confidence:
            line += f"  (conf {interval.confidence:.2f}"
            if interval.runner_up:
                line += f", vs {interval.runner_up}"
            line += ")"
        print(line)


def validate(results: Sequence[ColumnResult]) -> dict:
    """Structural checks over the produced intervals."""
    checks: dict[str, dict] = {}
    total = sum(len(r.intervals) for r in results)

    gaps = overlaps = out_of_bounds = 0
    for result in results:
        previous_bottom = None
        for interval in result.intervals:
            if previous_bottom is not None:
                if interval.top > previous_bottom:
                    gaps += 1
                elif interval.top < previous_bottom:
                    overlaps += 1
            previous_bottom = interval.bottom
            if interval.top < 0 or interval.bottom > result.height:
                out_of_bounds += 1

    checks["intervals_found"] = {
        "ok": total > 0,
        "detail": f"{total} interval(s) across {len(results)} column(s)",
    }
    checks["contiguous"] = {
        "ok": gaps == 0 and overlaps == 0,
        "detail": f"{gaps} gap(s), {overlaps} overlap(s) between intervals",
    }
    checks["in_bounds"] = {
        "ok": out_of_bounds == 0,
        "detail": f"{out_of_bounds} interval(s) outside the image",
    }
    confidences = [i.confidence for r in results for i in r.intervals]
    if confidences:
        checks["confidence"] = {
            "ok": float(np.mean(confidences)) >= 0.15,
            "soft": True,
            "detail": f"mean confidence {float(np.mean(confidences)):.2f}",
        }
    return {
        "ok": all(c["ok"] for c in checks.values() if not c.get("soft")),
        "checks": checks,
    }


def print_report(report: dict) -> None:
    print("\nValidation")
    print("-" * 60)
    for name, check in report["checks"].items():
        status = "PASS" if check["ok"] else ("WARN" if check.get("soft") else "FAIL")
        print(f"  [{status}] {name:<16} {check['detail']}")
    print("-" * 60)
    print(f"  result: {'PASS' if report['ok'] else 'FAIL'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment an interpreted lithology column into lithology intervals "
            "by matching its textures against a pattern library."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--column", type=Path, action="append", help="Column image (repeatable)"
    )
    source.add_argument(
        "--column-dir", type=Path, default=None, help="Folder of column images"
    )
    parser.add_argument(
        "--legend", type=Path, default=None,
        help="Legend image of the same document; swatches are detected and "
        "used instead of --patterns (most accurate)",
    )
    parser.add_argument(
        "--patterns", type=Path, default=DEFAULT_PATTERNS,
        help="Folder of pattern swatches",
    )
    parser.add_argument(
        "--names", type=Path, default=DEFAULT_NAMES,
        help="Folder of name crops matching the swatch file stems",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="JSON manifest with name_text per crop (skips OCR)",
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Read pattern names with RapidOCR when no manifest is given",
    )
    parser.add_argument(
        "--band", type=int, default=100,
        help="Height of the texture window in pixels",
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Vertical step between windows in pixels",
    )
    parser.add_argument(
        "--window", type=int, default=8,
        help="Bands compared on each side of a candidate boundary",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="Boundary threshold in standard deviations (lower = more boundaries)",
    )
    parser.add_argument(
        "--min-interval", type=int, default=30,
        help="Intervals shorter than this are absorbed into a neighbour",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Label intervals below this confidence as 'unknown'",
    )
    parser.add_argument(
        "--filter-names", action="store_true",
        help="Drop non-lithology legend entries (casing shoe, not available, ...)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write intervals to JSON")
    parser.add_argument("--csv", type=Path, default=None, help="Write intervals to CSV")
    parser.add_argument(
        "--overlay", type=Path, default=None,
        help="Write an annotated image (single column only)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Per-column JSON output folder when using --column-dir",
    )
    parser.add_argument("--validate", action="store_true", help="Run structural checks")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 when --validate reports a failure",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.column:
        columns = list(args.column)
    elif args.column_dir:
        if not args.column_dir.is_dir():
            print(f"error: not a folder: {args.column_dir}", file=sys.stderr)
            return 2
        columns = sorted(
            p
            for p in args.column_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
    else:
        if not DEFAULT_COLUMNS.is_dir():
            print(
                "error: no --column or --column-dir given and the default "
                f"folder is missing: {DEFAULT_COLUMNS}",
                file=sys.stderr,
            )
            return 2
        columns = sorted(DEFAULT_COLUMNS.glob("*.png"))

    if not columns:
        print("error: no column images found", file=sys.stderr)
        return 1

    if args.legend:
        if not args.legend.is_file():
            print(f"error: legend not found: {args.legend}", file=sys.stderr)
            return 2
        library = collapse_library(
            load_library_from_legend(args.legend, use_ocr=args.ocr)
        )
    else:
        library = collapse_library(
            load_library(
                args.patterns,
                names_dir=args.names,
                manifest=args.manifest,
                use_ocr=args.ocr,
            )
        )
    if args.filter_names:
        kept = [p for p in library if is_lithology_name(p.name)]
        if kept:
            library = kept
    if not args.quiet:
        print(f"pattern library: {len(library)} distinct lithologies")

    results: list[ColumnResult] = []
    for column in columns:
        result = segment_column(
            column,
            library,
            band=args.band,
            step=args.step,
            window=args.window,
            sensitivity=args.sensitivity,
            min_interval=args.min_interval,
            min_confidence=args.min_confidence,
        )
        results.append(result)
        if not args.quiet:
            print_intervals(result, show_confidence=True)

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            write_json([result], args.out_dir / f"{column.stem}.json")

    if args.json:
        write_json(results, args.json)
    if args.csv:
        write_csv(results, args.csv)
    if args.overlay:
        if len(results) != 1:
            print("warning: --overlay needs exactly one column; skipping", file=sys.stderr)
        else:
            print(f"overlay -> {draw_overlay(columns[0], results[0], args.overlay)}")

    report = None
    if args.validate:
        report = validate(results)
        print_report(report)

    total = sum(len(r.intervals) for r in results)
    print(
        f"\n{total} interval(s) from {len(results)} column(s)"
        + (f" -> {args.json}" if args.json else "")
        + (f", {args.csv}" if args.csv else "")
    )

    if report is not None and not report["ok"] and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
