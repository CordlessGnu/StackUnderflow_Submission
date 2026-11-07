#!/usr/bin/env python3
"""
Find the lithology intervals inside an interpreted lithology column.

The interpreted lithology column is a vertical strip in which every depth
interval is filled with the lithology pattern (texture) of that interval. This
module turns that strip into intervals:

    [(top=1032, bottom=1846, lithology='Sandstone'),
     (top=1847, bottom=2122, lithology='Granite'), ...]

Two passes
----------
**Pass 1 - the centre line.** A narrow band (5% of the strip width) is read
down the centre of the pattern strip, one row at a time, and each row becomes
"ink" or "paper". Every flip is a candidate boundary, but striped patterns
flip constantly, so runs longer than `--stable` rows are treated as anchors
(blank -> `empty`, solid ink -> a solid interval) and bursts of short runs
merge into one texture region. Regions shorter than `--min-interval` are
absorbed into a neighbour and the loss is reported.

**Pass 2 - self-referential boundaries, then swatch naming.** Inside each
texture region the strip profile is compared with itself: for every row, how
well does the half below continue the half above (searching +/-`--bc-lags`
rows for the best continuation lag)? Inside one continuous texture the
continuation is strong everywhere; where the pattern stops continuing the
score spikes, and those spikes are the boundaries. Then every segment is named
by the legend swatch resized to the best scale. The median winning scale is
reported as the general swatch size for the column.

Blank regions are reported as `empty` and never given a lithology name. The
strip edges are located per band of rows because the crops are skewed and
sometimes include the depth-tick gutter or a neighbour's edge.

Legend / pattern library
------------------------
Every column is matched against the legend of its own document. The legend is
run through `models/swatch_detection.pt`, which returns the swatch boxes
(`material_texture`) and their labels (`material_name`); each swatch is paired
with the label beside or below it, the label is read with RapidOCR, and the
swatch becomes a library entry. Legends are paired with columns by file stem.
There is no global pattern pool: a swatch and the column it is matched against
must come from the same scan.

Errors are never swallowed silently: problems that would lose or corrupt
information (unreadable names, mispaired swatches, ambiguous ties, absorbed
regions) are collected in a Diagnostics report and fail the run (exit 1);
weaker observations are warnings (fail under --strict).

Usage
-----
    # one column, legend found automatically by file stem
    python "boundary_detection.py" --column crops/interpreted_lithology_column/foo.png

    # explicit legend + outputs
    python "boundary_detection.py" --column foo.png --legend bar.png --json out.json --overlay out.png

    # every column in a folder
    python "boundary_detection.py" --column-dir crops/interpreted_lithology_column --out-dir results
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
DEFAULT_COLUMNS = REPO_ROOT / "crops" / "interpreted_lithology_column"
DEFAULT_LEGENDS = REPO_ROOT / "crops" / "lithology_legend"
DEFAULT_SWATCH_MODEL = REPO_ROOT / "models" / "swatch_detection.pt"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
class OCRError(RuntimeError):
    """Raised when a name crop cannot be read."""


@dataclass
class Issue:
    """One problem found while building a library or segmenting a column."""

    severity: str  # "error" or "warning"
    where: str
    message: str


class Diagnostics:
    """
    Collects every problem instead of swallowing it.

    Errors mean information is missing or wrong and must be fixed. Warnings
    mean the result is usable but should be checked.
    """

    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, where: str, message: str) -> None:
        self.issues.append(Issue("error", where, message))

    def warning(self, where: str, message: str) -> None:
        self.issues.append(Issue("warning", where, message))

    def merge(self, other: "Diagnostics") -> None:
        self.issues.extend(other.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def report(self, limit: int = 25) -> None:
        if not self.issues:
            print("\nDiagnostics: no problems found")
            return
        print("\nDiagnostics")
        print("-" * 70)
        for severity, group in (("ERROR", self.errors), ("WARN", self.warnings)):
            if not group:
                continue
            print(f"  {severity} ({len(group)})")
            for issue in group[:limit]:
                print(f"    [{issue.where}] {issue.message}")
            if len(group) > limit:
                print(f"    ... and {len(group) - limit} more")
        print("-" * 70)
        print(
            f"  {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )

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

# --- two-pass segmentation parameters ---------------------------------------
# Pass 1 reads a narrow band at the centre of the pattern strip and records
# where it flips between ink and paper. Pass 2 matches legend swatches against
# the column at the swatch scale measured from the column itself.

CENTRE_BAND_FRAC = 0.05   # centre band width as a fraction of the strip
BAND_HEIGHT = 200         # rows per band when locating the strip edges
STABLE_RUN = 120          # runs longer than this are anchors, not stripes

# Pass-2 continuity test (boundary detection inside texture regions)
BC_HALF = 120             # half-window height of the continuity test
BC_LAGS = 40              # +/- rows searched for the best continuation lag
BC_STRIDE = 10            # rows between tested positions
BC_Z = 4.0                # peak threshold in robust deviations above the floor

# Swatch resize grid for pass 2, as multiples of the swatch's legend height.
SCALE_GRID = tuple(round(0.5 + 0.1 * i, 1) for i in range(26))  # 0.5 .. 3.0

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
    kind: str = "texture"   # texture | solid | empty
    ink: float = 0.0        # ink fraction of the full strip, for sanity checks
    scale: float = 0.0      # swatch scale that matched, 0 when not classified

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_dict(self) -> dict:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "height": self.height,
            "lithology": self.lithology,
            "kind": self.kind,
            "ink": round(self.ink, 3),
            "scale": round(self.scale, 2),
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
    image: np.ndarray | None = None   # binarized swatch, for pass-2 resizing


@dataclass
class ColumnResult:
    """Segmentation result for one column image."""

    image: str
    width: int
    height: int
    strip_width: int
    intervals: list[Interval] = field(default_factory=list)
    general_scale: float = 0.0   # median swatch scale pass 2 measured

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "strip_width": self.strip_width,
            "general_scale": round(self.general_scale, 2),
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
    """
    Read a name crop with RapidOCR (recognition stage only).

    Raises OCRError when OCR is unavailable or fails. An empty string is a
    legitimate result (a blank crop) and is reported by the caller, never
    silently turned into a placeholder name.
    """
    global _OCR_ENGINE
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover
        raise OCRError(
            "rapidocr is not installed:  pip install rapidocr onnxruntime"
        ) from exc
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    try:
        result = _OCR_ENGINE(
            np.asarray(image.convert("RGB")),
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as OCRError
        raise OCRError(f"OCR failed: {exc}") from exc
    texts = getattr(result, "txts", None) or ()
    return " ".join(" ".join(texts).split())


_SWATCH_MODEL = None


def _get_swatch_model(model_path: Path):
    """Load and cache the YOLO swatch detector."""
    global _SWATCH_MODEL
    if _SWATCH_MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "--legend needs ultralytics:  pip install ultralytics"
            ) from exc
        if not model_path.is_file():
            raise SystemExit(f"swatch model not found: {model_path}")
        _SWATCH_MODEL = YOLO(str(model_path))
    return _SWATCH_MODEL


def _merge_overlapping_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    overlap_threshold: float = 0.3,
) -> tuple[list[tuple[float, float, float, float]], int]:
    """
    Merge boxes that describe the same object into their union.

    The detector sometimes splits one swatch into several partial boxes (for
    example three slices of the same rectangle). Left alone, the extra boxes
    steal names from neighbouring swatches, so they are merged first.

    Returns ``(boxes, merged_count)``.
    """
    def area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def intersection(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0.0, min(a[3], b[3]) - max(a[1], b[1])
        )

    result = list(boxes)
    merged = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a, b = result[i], result[j]
                smaller = min(area(a), area(b))
                if smaller <= 0:
                    continue
                if intersection(a, b) / smaller > overlap_threshold:
                    result[i] = (
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    )
                    result.pop(j)
                    merged += 1
                    changed = True
                    break
            if changed:
                break
    return result, merged


def _pair_boxes(
    textures: Sequence[tuple[float, float, float, float]],
    names: Sequence[tuple[float, float, float, float]],
    min_v_overlap: float = 0.5,
    min_h_overlap: float = 0.5,
) -> tuple[
    list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]],
    list[tuple[float, float, float, float]],
    list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]],
    str,
]:
    """
    Match every texture box to its name box.

    Two legend layouts exist in this dataset and both are supported:

    * **right** - the name sits to the right of the swatch on the same row
      (most legends)
    * **below** - the name sits under the swatch in the same column
      (e.g. the d-582 legend)

    Returns ``(pairs, unmatched, fallbacks, layout)``:

    * ``pairs``     - confident matches
    * ``unmatched`` - swatches with no name in either position
    * ``fallbacks`` - swatches matched to the nearest leftover name, which is a
      guess and must be reported rather than trusted
    * ``layout``    - which layout produced most of the matches
    """
    if not names:
        return [], list(textures), [], "none"

    def height(box):
        return box[3] - box[1]

    def width(box):
        return box[2] - box[0]

    def centre_x(box):
        return (box[0] + box[2]) / 2.0

    def centre_y(box):
        return (box[1] + box[3]) / 2.0

    ordered = sorted(textures, key=lambda b: (round(centre_y(b) / 10), b[0]))
    used: set[int] = set()
    pairs = []
    fallbacks = []
    unmatched = []
    right_hits = 0
    below_hits = 0

    for texture in ordered:
        # Layout 1: name to the right, on the same row.
        best = None
        for i, name in enumerate(names):
            if i in used:
                continue
            overlap = min(texture[3], name[3]) - max(texture[1], name[1])
            reference = min(height(texture), height(name))
            if reference <= 0 or overlap / reference < min_v_overlap:
                continue
            gap = name[0] - texture[2]
            if gap < -0.5 * min(width(texture), width(name)):
                continue
            score = abs(gap) + 0.5 * abs(centre_y(texture) - centre_y(name))
            if gap < 0:
                score += 0.25 * abs(gap)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((texture, names[best[1]]))
            right_hits += 1
            continue

        # Layout 2: name below, in the same column.
        best = None
        for i, name in enumerate(names):
            if i in used:
                continue
            overlap = min(texture[2], name[2]) - max(texture[0], name[0])
            reference = min(width(texture), width(name))
            if reference <= 0 or overlap / reference < min_h_overlap:
                continue
            gap = name[1] - texture[3]
            if gap < -0.5 * min(height(texture), height(name)):
                continue
            score = abs(gap) + 0.5 * abs(centre_x(texture) - centre_x(name))
            if gap < 0:
                score += 0.25 * abs(gap)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((texture, names[best[1]]))
            below_hits += 1
            continue

        # Neither layout matched: fall back to the nearest unused name, but
        # record it because this pairing is a guess.
        nearest = None
        for i, name in enumerate(names):
            if i in used:
                continue
            distance = (
                (centre_x(texture) - centre_x(name)) ** 2
                + (centre_y(texture) - centre_y(name)) ** 2
            ) ** 0.5
            if nearest is None or distance < nearest[0]:
                nearest = (distance, i)

        if nearest is None:
            unmatched.append(texture)
        else:
            used.add(nearest[1])
            fallbacks.append((texture, names[nearest[1]]))

    layout = "right" if right_hits >= below_hits else "below"
    return pairs, unmatched, fallbacks, layout


class LegendError(RuntimeError):
    """Raised when a legend cannot be turned into a pattern library."""


def find_legend(
    column: Path, legend_dir: Path | None, explicit: Path | None
) -> Path | None:
    """
    Locate the legend that belongs to a column.

    An explicit ``--legend`` wins. Otherwise the legend is looked up in
    ``--legend-dir`` by file stem, so ``.../foo.png`` pairs with
    ``<legend-dir>/foo.png``.
    """
    if explicit is not None:
        return explicit
    if legend_dir is None:
        return None
    for extension in IMAGE_EXTS:
        candidate = legend_dir / f"{column.stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def load_library_from_legend(
    legend_path: Path,
    model_path: Path = DEFAULT_SWATCH_MODEL,
    confidence: float = 0.25,
    image_size: int = 960,
    diagnostics: Diagnostics | None = None,
) -> list[Pattern]:
    """
    Build a pattern library from a legend image using the YOLO swatch detector.

    The model returns two classes: `material_texture` (the swatch) and
    `material_name` (its label). Each swatch is paired with the label beside it,
    the label is read with RapidOCR, and the swatch becomes a library entry.

    A swatch whose name cannot be read is **dropped and reported**, never given
    a placeholder name: a fake name would silently become a lithology label in
    the output.
    """
    diag = diagnostics if diagnostics is not None else Diagnostics()
    where = legend_path.name

    model = _get_swatch_model(model_path)
    result = model.predict(
        str(legend_path), verbose=False, conf=confidence, imgsz=image_size
    )[0]

    textures: list[tuple[float, float, float, float]] = []
    names: list[tuple[float, float, float, float]] = []
    for box in result.boxes:
        label = model.names[int(box.cls)]
        xyxy = tuple(float(v) for v in box.xyxy[0])
        if label == "material_texture":
            textures.append(xyxy)
        elif label == "material_name":
            names.append(xyxy)

    if not textures:
        raise LegendError(
            f"no material_texture boxes in {where} "
            f"(confidence {confidence}); try a lower --swatch-conf"
        )

    textures, merged_textures = _merge_overlapping_boxes(textures)
    names, merged_names = _merge_overlapping_boxes(names)
    if merged_textures or merged_names:
        diag.warning(
            where,
            f"merged {merged_textures} overlapping swatch box(es) and "
            f"{merged_names} overlapping name box(es) that described the same "
            "object",
        )

    pairs, unmatched, fallbacks, layout = _pair_boxes(textures, names)

    if unmatched:
        diag.error(
            where,
            f"{len(unmatched)} swatch(es) have no name beside or below them "
            f"and were not used: {[tuple(round(v) for v in b) for b in unmatched[:5]]}",
        )
    if fallbacks:
        # A name on a different row is a guess, so these swatches are dropped
        # rather than given a possibly wrong lithology.
        diag.error(
            where,
            f"{len(fallbacks)} swatch(es) had no name beside or below them "
            "and were not used; the model is probably misreading this legend "
            f"(first: swatch {tuple(round(v) for v in fallbacks[0][0])})",
        )
    if len(names) > 2 * len(textures):
        diag.error(
            where,
            f"{len(names)} name boxes for only {len(textures)} swatches; the "
            "model is probably reading text fragments as names",
        )

    library: list[Pattern] = []
    dropped = 0
    with Image.open(legend_path) as image:
        image = image.convert("L")
        for texture, name_box in pairs:
            name_box_int = (
                int(name_box[0]),
                int(name_box[1]),
                int(name_box[2]),
                int(name_box[3]),
            )
            try:
                name = _ocr_name(image.crop(name_box_int))
            except OCRError as exc:
                diag.error(where, f"cannot read the name at {name_box_int}: {exc}")
                dropped += 1
                continue

            if not name:
                diag.error(
                    where,
                    f"OCR returned no text for the name at {name_box_int}; "
                    "swatch dropped",
                )
                dropped += 1
                continue

            # Crop 2 px inside the detected box so the border does not pollute
            # the texture statistics.
            swatch = image.crop(
                (
                    int(texture[0]) + 2,
                    int(texture[1]) + 2,
                    int(texture[2]) - 2,
                    int(texture[3]) - 2,
                )
            )
            if swatch.width < 4 or swatch.height < 4:
                diag.error(
                    where,
                    f"swatch at {tuple(round(v) for v in texture)} is only "
                    f"{swatch.width}x{swatch.height} px after trimming; dropped",
                )
                dropped += 1
                continue

            binary = binarize(swatch)
            library.append(
                Pattern(
                    name=name,
                    path=legend_path,
                    descriptor=texture_descriptor(binary),
                    image=binary,
                )
            )

    if dropped:
        diag.error(where, f"{dropped} swatch(es) dropped because their name "
                          "could not be read")
    if not library:
        raise LegendError(f"no usable swatches in {where}")
    if len(library) < 2:
        raise LegendError(
            f"only {len(library)} usable swatch in {where} "
            f"({library[0].name!r}); a legend needs at least 2 to be useful"
        )

    return library


def is_lithology_name(name: str) -> bool:
    """True when a legend name looks like a lithology rather than a tool entry."""
    words = set(re.findall(r"[a-z]+", name.lower()))
    # Reject OCR debris: no real word at all ("t", "1", "_", "11").
    if not any(len(word) >= 3 for word in words):
        return False
    if words & LITHOLOGY_WORDS:
        return True
    return not (words & NON_LITHOLOGY)


def collapse_library(library: Sequence[Pattern]) -> list[Pattern]:
    """
    Merge patterns that share a name into one entry.

    The descriptor is the mean of the group. The kept swatch image is the
    medoid - the member whose descriptor is closest to that mean - so pass 2
    resizes a representative swatch rather than an arbitrary one.
    """
    grouped: dict[str, list[Pattern]] = {}
    for pattern in library:
        grouped.setdefault(pattern.name, []).append(pattern)

    merged: list[Pattern] = []
    for name, members in grouped.items():
        descriptors = np.vstack([m.descriptor for m in members])
        mean = descriptors.mean(axis=0)
        medoid = min(members, key=lambda m: descriptor_distance(m.descriptor, mean))
        merged.append(
            Pattern(
                name=name,
                path=medoid.path,
                descriptor=mean,
                image=medoid.image,
            )
        )
    return merged


# --------------------------------------------------------------------------- #
# Pass 1 - pattern strip geometry and the centre-line signal
# --------------------------------------------------------------------------- #
def detect_strip_edges(
    binary: np.ndarray,
    band_height: int = BAND_HEIGHT,
    threshold: float = RULE_DENSITY,
    diagnostics: Diagnostics | None = None,
    where: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate the left and right edge of the pattern strip for every row.

    The crop is not always tight: a depth-tick gutter may be included on the
    left, bits of the neighbouring column on the right, and the scan is
    usually skewed. So the strip edges are located per band of rows and then
    interpolated linearly, which follows the skew.

    Bands with no rules of their own reuse the nearest band that has them;
    every such band is reported, because stale edges clip the pattern when the
    skew is real.
    """
    height, width = binary.shape
    diag = diagnostics if diagnostics is not None else Diagnostics()

    raw: list[tuple[float, int | None, int | None]] = []
    solid_bands = 0
    for top in range(0, height, band_height):
        band = binary[top : top + band_height]
        density = band.mean(axis=0)
        solid = np.flatnonzero(density > threshold)
        y_mid = top + min(band_height, height - top) / 2.0

        if len(solid) == 0:
            if density.max() < 0.02:
                raw.append((y_mid, None, None))       # blank band
            else:
                raw.append((y_mid, 0, width))         # pattern to the edges
            continue

        if len(solid) == 1:
            x = int(solid[0])
            if x < width / 2:
                raw.append((y_mid, x + 1, width))
            else:
                raw.append((y_mid, 0, x))
            continue

        left = int(solid[0])
        while left + 1 < width and density[left + 1] > threshold:
            left += 1
        right = int(solid[-1])
        while right - 1 >= 0 and density[right - 1] > threshold:
            right -= 1
        if right - left < 8:
            solid_bands += 1                           # ink everywhere
            raw.append((y_mid, None, None))
            continue
        raw.append((y_mid, left + 1, right))

    known = [r for r in raw if r[1] is not None]
    if not known:
        diag.error(
            where,
            "no strip edges found in any band; using the full image width",
        )
        return np.zeros(height), np.full(height, float(width))

    resolved: list[tuple[float, float, float]] = []
    reused = 0
    for y, left, right in raw:
        if left is None:
            nearest = min(known, key=lambda k: abs(k[0] - y))
            left, right = nearest[1], nearest[2]
            reused += 1
        resolved.append((y, float(left), float(right)))

    if reused:
        diag.warning(
            where,
            f"{reused} band(s) had no strip edges of their own (blank or "
            "solid ink) and reused the nearest band's",
        )

    ys = np.array([r[0] for r in resolved])
    lefts = np.array([r[1] for r in resolved])
    rights = np.array([r[2] for r in resolved])
    rows = np.arange(height)
    return np.interp(rows, ys, lefts), np.interp(rows, ys, rights)


def centre_line_signal(
    binary: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    frac: float = CENTRE_BAND_FRAC,
) -> np.ndarray:
    """
    Reduce every row to one bit: does the centre band contain ink?

    The band is `frac` of the strip width, centred on the strip. A one-pixel
    column would fall in the white gap of striped patterns; a band a few
    percent wide catches them while staying inside the pattern.
    """
    height, width = binary.shape
    half = (right - left) * frac / 2.0
    centre = (left + right) / 2.0
    x1 = np.clip((centre - half).astype(int), 0, width)
    x2 = np.clip((centre + half).astype(int) + 1, 0, width)

    # Prefix sums make the per-row "any ink in [x1, x2)" test vectorised.
    inclusive = binary.cumsum(axis=1)
    prefix = np.concatenate([np.zeros((height, 1), dtype=inclusive.dtype), inclusive], axis=1)
    rows = np.arange(height)
    return (prefix[rows, x2] - prefix[rows, x1]) > 0


def strip_ink(
    binary: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: int,
    bottom: int,
) -> float:
    """Ink fraction of the full strip between rows top and bottom."""
    if bottom <= top:
        return 0.0
    x1 = int(np.median(left[top:bottom]))
    x2 = int(np.median(right[top:bottom]))
    if x2 <= x1:
        return 0.0
    return float(binary[top:bottom, x1:x2].mean())


def group_regions(
    signal: np.ndarray, stable: int = STABLE_RUN
) -> list[list]:
    """
    Turn the centre-line signal into regions.

    Runs of one state longer than `stable` are anchors: blank when the state
    is paper, solid when it is ink. Everything else is stripe-level flipping
    inside a texture burst, so consecutive short runs merge into one texture
    region.

    Returns a list of [kind, top, bottom] with kind in
    {"empty", "solid", "texture"}.
    """
    height = len(signal)
    runs: list[tuple[int, int, int]] = []
    y = 0
    while y < height:
        value = signal[y]
        end = y
        while end < height and signal[end] == value:
            end += 1
        runs.append((int(value), y, end))
        y = end

    regions: list[list] = []
    for value, top, bottom in runs:
        if bottom - top >= stable:
            regions.append(["solid" if value else "empty", top, bottom])
        elif regions and regions[-1][0] == "texture":
            regions[-1][2] = bottom
        else:
            regions.append(["texture", top, bottom])
    return regions


def merge_tiny_regions(
    regions: list[list], min_interval: int,
) -> tuple[list[list], int]:
    """
    Absorb regions shorter than `min_interval` into the taller neighbour.

    Returns (regions, absorbed). The count is reported because every absorbed
    region is a boundary the output no longer shows.
    """
    if min_interval <= 0 or len(regions) < 2:
        return regions, 0

    absorbed = 0
    merged = [list(r) for r in regions]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, region in enumerate(merged):
            if region[2] - region[1] >= min_interval:
                continue
            before = merged[i - 1] if i > 0 else None
            after = merged[i + 1] if i + 1 < len(merged) else None
            if before is None and after is None:
                continue
            if before is None:
                target = after
            elif after is None:
                target = before
            else:
                target = (
                    before
                    if before[2] - before[1] >= after[2] - after[1]
                    else after
                )
            target[1] = min(target[1], region[1])
            target[2] = max(target[2], region[2])
            merged.pop(i)
            absorbed += 1
            changed = True
            break

    joined: list[list] = []
    for region in merged:
        if joined and joined[-1][0] == region[0]:
            joined[-1][2] = region[2]
        else:
            joined.append(region)
    return joined, absorbed


# --------------------------------------------------------------------------- #
# Pass 2 - match legend swatches to the column at the column's own scale
# --------------------------------------------------------------------------- #
def absorb_short_intervals(
    intervals: list[Interval], min_interval: int
) -> tuple[list[Interval], int]:
    """
    Absorb intervals shorter than `min_interval` into their taller neighbour.

    Returns (intervals, absorbed). Every absorbed interval is a boundary the
    output no longer shows, so the count is reported by the caller.
    """
    if min_interval <= 0 or len(intervals) < 2:
        return intervals, 0
    merged = list(intervals)
    absorbed = 0
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, interval in enumerate(merged):
            if interval.height >= min_interval:
                continue
            before = merged[i - 1] if i > 0 else None
            after = merged[i + 1] if i + 1 < len(merged) else None
            if before is None and after is None:
                continue
            if before is None:
                target = after
            elif after is None:
                target = before
            else:
                target = before if before.height >= after.height else after
            target.top = min(target.top, interval.top)
            target.bottom = max(target.bottom, interval.bottom)
            merged.pop(i)
            absorbed += 1
            changed = True
            break
    joined: list[Interval] = []
    for interval in merged:
        if joined and joined[-1].lithology == interval.lithology \
                and joined[-1].kind == interval.kind:
            joined[-1].bottom = interval.bottom
            joined[-1].confidence = max(joined[-1].confidence, interval.confidence)
        else:
            joined.append(interval)
    return joined, absorbed


def resize_binary(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a binary patch, returning floats in [0, 1]."""
    rendered = Image.fromarray((image * 255.0).astype(np.uint8)).resize(
        (max(1, width), max(1, height)), Image.LANCZOS
    )
    return np.asarray(rendered, dtype=np.float32) / 255.0


class ScaleLibrary:
    """
    Legend swatches pre-rendered at a grid of scales.

    Pass 2 resizes every swatch to the scale the column draws at, then
    compares texture descriptors. Resizing once per (swatch, scale) turns
    per-window matching into a 7-dimensional lookup instead of repeated image
    resizes.

    The strip width is baked in: the swatch is stretched to the strip's width
    because the column pattern spans the full strip.
    """

    def __init__(
        self,
        library: Sequence[Pattern],
        strip_width: int,
        scales: Sequence[float] = SCALE_GRID,
        diagnostics: Diagnostics | None = None,
        where: str = "",
    ):
        self.scales = tuple(scales)
        self.strip_width = int(strip_width)
        self.meta: list[tuple[str, float]] = []
        self.matrix: np.ndarray | None = None

        diag = diagnostics if diagnostics is not None else Diagnostics()
        rows: list[np.ndarray] = []
        for pattern in library:
            if pattern.image is None:
                diag.error(
                    where, f"swatch {pattern.name!r} has no image; skipped"
                )
                continue
            for scale in self.scales:
                resized = resize_binary(
                    pattern.image,
                    max(8, int(round(pattern.image.shape[0] * scale))),
                    self.strip_width,
                )
                rows.append(texture_descriptor((resized > 0.5).astype(np.float32)))
                self.meta.append((pattern.name, float(scale)))
        if rows:
            self.matrix = np.vstack(rows) * FEATURE_WEIGHTS

    def match(self, descriptor: np.ndarray) -> tuple[str, float, float, float]:
        """
        Return (name, distance, scale, runner_up_distance).

        The runner-up is the best distance achieved by a *different* name, so
        the margin reflects disagreement between lithologies rather than
        between two scales of the same one.
        """
        if self.matrix is None:
            raise ValueError("scale library is empty")
        weighted = descriptor * FEATURE_WEIGHTS
        distances = np.linalg.norm(self.matrix - weighted, axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        best_name, best_scale = self.meta[best]
        runner = float(distances[best])
        for index in order:
            if self.meta[int(index)][0] != best_name:
                runner = float(distances[int(index)])
                break
        return best_name, float(distances[best]), best_scale, runner


def classify_confidence(distance: float, runner_up: float) -> float:
    """Margin between the winner and the best other lithology, damped."""
    margin = (runner_up - distance) / (runner_up + 1e-6)
    return float(np.clip(margin, 0.0, 1.0) * np.exp(-distance / 2.0))


def continuity_boundaries(
    profile: np.ndarray,
    top: int,
    bottom: int,
    half: int = BC_HALF,
    lags: int = BC_LAGS,
    stride: int = BC_STRIDE,
    z: float = BC_Z,
) -> list[int]:
    """
    Find the rows inside [top, bottom) where the pattern stops continuing.

    The test is self-referential - the column is compared with itself, never
    with the legend, so the rendering mismatch between a legend swatch and the
    column fill cannot affect it.

    For every candidate row y the strip profile is cut into an upper half
    [y-half, y) and a lower half [y, y+half). The score is how well the lower
    half can be made to continue the upper half by shifting it up to +/-`lags`
    rows (the continuation lag of a period-P pattern is -half + k*P, which is
    usually positive, so the search must be symmetric). 1 - best correlation is
    the discontinuity d(y); inside one uniform texture d stays near the
    burst's own noise floor, at a real change it spikes.

    Peaks are kept when they clear `z` robust deviations above the burst's own
    median, are at least 80 rows apart, and sit at least `half` rows from the
    burst edges (edges are already boundaries - pass 1 put them there).

    Returns the sorted boundary rows.
    """
    height = bottom - top
    if height < 2 * half + 2 * lags:
        return []          # too short to test; the burst stays one interval

    lo = max(0, top - half - lags)
    hi = min(len(profile), bottom + half + lags)
    seg = profile[lo:hi].astype(np.float32)
    win_w = 2 * half + 2 * lags
    n_windows = (len(seg) - win_w) // stride + 1
    if n_windows < 3:
        return []

    windows = np.lib.stride_tricks.sliding_window_view(seg, win_w)[::stride][:n_windows]
    centers = lo + half + lags + np.arange(n_windows) * stride
    keep = (centers >= top) & (centers < bottom)
    windows = windows[keep]
    centers = centers[keep]
    n = windows.shape[0]
    if n < 3:
        return []

    upper = windows[:, lags: lags + half].astype(np.float64)
    d_upper = upper - upper.mean(axis=1, keepdims=True)
    u_norm = np.sqrt((d_upper * d_upper).sum(axis=1))

    best = np.full(n, -2.0)
    for lag in range(-lags, lags + 1):
        lo_b = lags + half + lag
        lower = windows[:, lo_b: lo_b + half].astype(np.float64)
        d_lower = lower - lower.mean(axis=1, keepdims=True)
        num = (d_upper * d_lower).sum(axis=1)
        den = u_norm * np.sqrt((d_lower * d_lower).sum(axis=1))
        r = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
        best = np.maximum(best, r)
    d = 1.0 - np.clip(best, -1.0, 1.0)
    d = np.convolve(d, np.ones(5) / 5.0, mode="same")

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    # A perfectly uniform burst has mad = 0, where any scan wobble would look
    # like a huge z-score; give the threshold a small absolute floor instead.
    thresh = med + z * max(mad, 0.02)
    idx = [
        i
        for i in range(1, n - 1)
        if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] > thresh
    ]
    idx.sort(key=lambda i: -d[i])
    kept: list[int] = []
    for i in idx:
        if all(abs(centers[i] - centers[j]) >= 80 for j in kept):
            kept.append(i)
    boundaries = sorted(int(centers[i]) for i in kept)
    boundaries = [
        y for y in boundaries if top + half <= y <= bottom - half
    ]
    return boundaries


def texture_intervals(
    binary: np.ndarray,
    profile: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: int,
    bottom: int,
    scale_library: ScaleLibrary,
    boundary_z: float = BC_Z,
    min_confidence: float = 0.0,
    diagnostics: Diagnostics | None = None,
    where: str = "",
) -> tuple[list[Interval], float]:
    """
    Turn one texture region into intervals.

    The region is first cut where the continuity test says the pattern stops
    continuing (see `continuity_boundaries`), then every segment is named by
    the scale library. Boundaries come from the column's own signal, so a
    stretch of continuous texture stays one interval even when two look-alike
    swatches tie - which is the correct reading per the dataset's own logic:
    same period and same waveform is the same texture.

    Returns (intervals, median_scale).
    """
    diag = diagnostics if diagnostics is not None else Diagnostics()
    cuts = continuity_boundaries(profile, top, bottom, z=boundary_z)
    edges = [top] + cuts + [bottom]

    intervals: list[Interval] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start < 4:
            diag.error(
                where, f"segment {start}-{end} is degenerate and was dropped"
            )
            continue
        x1 = int(np.median(left[start:end]))
        x2 = int(np.median(right[start:end]))
        patch = binary[start:end, x1:x2]
        name, distance, scale, runner = scale_library.match(
            texture_descriptor(patch)
        )
        confidence = classify_confidence(distance, runner)
        if confidence < min_confidence:
            name = "unknown"
        intervals.append(
            Interval(
                top=int(start),
                bottom=int(end),
                lithology=name,
                confidence=confidence,
                kind="texture",
                scale=scale,
            )
        )

    scales = [i.scale for i in intervals if i.scale]
    return intervals, (float(np.median(scales)) if scales else 0.0)


# --------------------------------------------------------------------------- #
# Column segmentation - the two passes together
# --------------------------------------------------------------------------- #
def segment_column(
    image_path: Path,
    library: Sequence[Pattern],
    band_height: int = BAND_HEIGHT,
    centre_frac: float = CENTRE_BAND_FRAC,
    stable: int = STABLE_RUN,
    boundary_z: float = BC_Z,
    min_interval: int = 30,
    min_confidence: float = 0.0,
    diagnostics: Diagnostics | None = None,
    scale_cache: dict | None = None,
    legend_key: str = "",
) -> ColumnResult:
    """
    Segment one interpreted lithology column into lithology intervals.

    Pass 1 reads the centre line of the pattern strip and turns its colour
    changes into regions: long blank runs become `empty`, long ink runs become
    `solid`, bursts of stripe-level flipping stay one `texture` region.

    Pass 2 asks, for every row inside a texture region, whether the pattern
    continues through it (the strip compared with itself at a tunable lag);
    where it stops continuing is a boundary. The segments are then named by
    the resized legend swatches. Blank regions are reported as `empty`, never
    given a lithology name.
    """
    diag = diagnostics if diagnostics is not None else Diagnostics()
    where = image_path.name

    if not library:
        raise ValueError(f"empty pattern library for {where}")

    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size
        binary = binarize(image)

    left, right = detect_strip_edges(
        binary, band_height, diagnostics=diag, where=where
    )
    signal = centre_line_signal(binary, left, right, centre_frac)

    # Row profile of the strip interior - the signal pass 2 tests.
    inclusive = binary.cumsum(axis=1)
    prefix = np.concatenate(
        [np.zeros((height, 1), dtype=inclusive.dtype), inclusive], axis=1
    )
    rows = np.arange(height)
    widths = np.maximum(right.astype(int) - left.astype(int), 1)
    profile = (
        prefix[rows, right.astype(int)] - prefix[rows, left.astype(int)]
    ) / widths

    regions = group_regions(signal, stable)
    regions, absorbed = merge_tiny_regions(regions, min_interval)
    if absorbed:
        diag.warning(
            where,
            f"{absorbed} region(s) shorter than {min_interval}px were "
            "absorbed into a neighbour, so those boundaries are gone",
        )
    if len(regions) <= 1:
        diag.warning(
            where,
            "pass 1 found a single region; either the column is one interval "
            "or the centre line never leaves one state",
        )

    # The scale library depends on the strip width, so it is cached per
    # (legend, width) rather than per legend alone.
    strip_width = int(np.median(right - left))
    key = (legend_key, strip_width)
    scale_library: ScaleLibrary | None = None
    if scale_cache is not None and key in scale_cache:
        scale_library = scale_cache[key]
    if scale_library is None:
        scale_library = ScaleLibrary(
            library, strip_width, diagnostics=diag, where=where
        )
        if scale_cache is not None:
            scale_cache[key] = scale_library

    intervals: list[Interval] = []
    scales_seen: list[float] = []
    for kind, top, bottom in regions:
        ink = strip_ink(binary, left, right, top, bottom)

        if kind == "empty":
            intervals.append(
                Interval(
                    top=top, bottom=bottom, lithology="empty",
                    confidence=1.0, kind="empty", ink=ink,
                )
            )
            continue

        if kind == "solid" or bottom - top < 2 * BC_HALF + 2 * BC_LAGS:
            x1 = int(np.median(left[top:bottom]))
            x2 = int(np.median(right[top:bottom]))
            patch = binary[top:bottom, x1:x2]
            name, distance, scale, runner = scale_library.match(
                texture_descriptor(patch)
            )
            confidence = classify_confidence(distance, runner)
            if confidence < min_confidence:
                name = "unknown"
            intervals.append(
                Interval(
                    top=top, bottom=bottom, lithology=name,
                    confidence=confidence, kind="solid",
                    ink=ink, scale=scale,
                )
            )
            scales_seen.append(scale)
            continue

        # Texture burst: cut where the pattern stops continuing, then name
        # the segments.
        sub_intervals, median_scale = texture_intervals(
            binary, profile, left, right, top, bottom, scale_library,
            boundary_z=boundary_z,
            min_confidence=min_confidence,
            diagnostics=diag, where=where,
        )
        for interval in sub_intervals:
            interval.ink = strip_ink(
                binary, left, right, interval.top, interval.bottom
            )
        intervals.extend(sub_intervals)
        if median_scale:
            scales_seen.append(median_scale)

    intervals, absorbed_after = absorb_short_intervals(intervals, min_interval)
    if absorbed_after:
        diag.warning(
            where,
            f"{absorbed_after} interval(s) shorter than {min_interval}px were "
            "absorbed into a neighbour after classification",
        )

    general_scale = float(np.median(scales_seen)) if scales_seen else 0.0

    return ColumnResult(
        image=str(image_path),
        width=width,
        height=height,
        strip_width=strip_width,
        intervals=intervals,
        general_scale=general_scale,
    )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def draw_overlay(image_path: Path, result: ColumnResult, dest: Path) -> Path:
    """
    Write the column with a red line at every interval boundary.

    Boundaries are shared between neighbouring intervals, so the positions are
    collected first and drawn once - otherwise every boundary would be drawn
    twice in two different colours. Image edges are not boundaries and are not
    drawn. Each interval tall enough gets its name written inside it.
    """
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    red = (230, 20, 20)
    dark = (120, 10, 10)

    boundaries = sorted(
        {interval.top for interval in result.intervals}
        | {interval.bottom for interval in result.intervals}
    )
    for y in boundaries:
        if y <= 0 or y >= result.height:
            continue  # the image edge is not a boundary
        draw.line([(0, y), (canvas.width, y)], fill=red, width=3)

    for interval in result.intervals:
        if interval.height < 40:
            continue  # too short for a readable label
        colour = dark if interval.kind == "empty" else red
        draw.text(
            (2, interval.top + 4),
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
        "image", "top", "bottom", "height", "lithology", "kind", "ink",
        "scale", "confidence", "runner_up",
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
        f"intervals={len(result.intervals)}  swatch scale={result.general_scale:.2f}"
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
            if interval.kind == "empty":
                line += "  [blank]"
            else:
                line += f"  (conf {interval.confidence:.2f}"
                if interval.runner_up:
                    line += f", vs {interval.runner_up}"
                if interval.scale:
                    line += f", scale {interval.scale:.1f}"
                line += f", ink {interval.ink:.2f})"
        print(line)


def validate(results: Sequence[ColumnResult]) -> dict:
    """Structural checks over the produced intervals."""
    checks: dict[str, dict] = {}
    total = sum(len(r.intervals) for r in results)

    gaps = overlaps = out_of_bounds = 0
    empty_on_ink = solid_on_blank = 0
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
            # An interval labelled empty must actually be blank over the full
            # strip, and a solid one must be mostly ink - otherwise the label
            # is wrong information, not just a weak guess.
            if interval.kind == "empty" and interval.ink > 0.05:
                empty_on_ink += 1
            if interval.kind == "solid" and interval.ink < 0.80:
                solid_on_blank += 1

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
    checks["empty_is_blank"] = {
        "ok": empty_on_ink == 0,
        "detail": f"{empty_on_ink} `empty` interval(s) with more than 5% ink",
    }
    checks["solid_is_ink"] = {
        "ok": solid_on_blank == 0,
        "detail": f"{solid_on_blank} `solid` interval(s) with less than 80% ink",
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
        help="Legend image to use for every column (default: pair each column "
        "with the legend of the same file stem in --legend-dir)",
    )
    parser.add_argument(
        "--legend-dir", type=Path, default=DEFAULT_LEGENDS,
        help="Folder of legend images, paired with columns by file stem",
    )
    parser.add_argument(
        "--swatch-model", type=Path, default=DEFAULT_SWATCH_MODEL,
        help="YOLO model used to detect swatches and names in the legend",
    )
    parser.add_argument(
        "--swatch-conf", type=float, default=0.25,
        help="Confidence threshold for the swatch detector",
    )
    parser.add_argument(
        "--swatch-imgsz", type=int, default=960,
        help="Inference size for the swatch detector (legends are wide, so "
        "960 finds more swatches than the 640 default)",
    )
    parser.add_argument(
        "--centre-band", type=float, default=CENTRE_BAND_FRAC,
        help="Centre band width as a fraction of the strip width (pass 1)",
    )
    parser.add_argument(
        "--band-height", type=int, default=BAND_HEIGHT,
        help="Rows per band when locating the strip edges",
    )
    parser.add_argument(
        "--stable", type=int, default=STABLE_RUN,
        help="Runs longer than this many rows are anchors, not stripes",
    )
    parser.add_argument(
        "--boundary-z", type=float, default=BC_Z,
        help="Continuity-test peak threshold in robust deviations above the "
        "burst's own noise floor (lower = more boundaries)",
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

    if args.legend is not None and not args.legend.is_file():
        print(f"error: legend not found: {args.legend}", file=sys.stderr)
        return 2
    if args.legend is None and not args.legend_dir.is_dir():
        print(
            f"error: legend folder not found: {args.legend_dir}", file=sys.stderr
        )
        return 2

    diag = Diagnostics()
    scale_cache: dict = {}

    # One library per legend; several columns may share the same legend.
    library_cache: dict[Path, list[Pattern]] = {}

    def library_for(legend: Path) -> list[Pattern]:
        if legend not in library_cache:
            library = collapse_library(
                load_library_from_legend(
                    legend,
                    model_path=args.swatch_model,
                    confidence=args.swatch_conf,
                    image_size=args.swatch_imgsz,
                    diagnostics=diag,
                )
            )
            if args.filter_names:
                kept = [p for p in library if is_lithology_name(p.name)]
                if not kept:
                    raise LegendError(
                        f"--filter-names removed all {len(library)} entries "
                        f"({', '.join(p.name for p in library[:4])}); the "
                        "legend names are not lithologies"
                    )
                if len(kept) < len(library):
                    diag.warning(
                        legend.name,
                        f"--filter-names dropped {len(library) - len(kept)} "
                        f"of {len(library)} entries",
                    )
                library = kept
            library_cache[legend] = library
        return library_cache[legend]

    results: list[ColumnResult] = []
    skipped: list[str] = []
    for column in columns:
        legend = find_legend(column, args.legend_dir, args.legend)
        if legend is None:
            diag.error(
                column.name,
                f"no legend with this file stem in {args.legend_dir}; "
                "column skipped (use --legend to force one)",
            )
            skipped.append(column.name)
            continue

        try:
            library = library_for(legend)
        except LegendError as exc:
            diag.error(column.name, f"legend unusable: {exc}; column skipped")
            skipped.append(column.name)
            continue
        if not args.quiet:
            print(
                f"\nlegend {legend.name}: {len(library)} distinct lithologies"
            )

        result = segment_column(
            column,
            library,
            band_height=args.band_height,
            centre_frac=args.centre_band,
            stable=args.stable,
            boundary_z=args.boundary_z,
            min_interval=args.min_interval,
            min_confidence=args.min_confidence,
            diagnostics=diag,
            scale_cache=scale_cache,
            legend_key=str(legend),
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
            diag.error(
                "--overlay",
                f"needs exactly one column, got {len(results)}; overlay not written",
            )
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
    if skipped:
        print(f"  {len(skipped)} column(s) skipped: {', '.join(skipped[:5])}")

    diag.report()

    # Errors mean information is missing or wrong, so they always fail the run.
    # Warnings only fail it under --strict.
    if diag.errors:
        return 1
    if args.strict and diag.warnings:
        return 1
    if report is not None and not report["ok"] and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
