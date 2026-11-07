#!/usr/bin/env python3
"""
Crop lithology legend bounding boxes from the pshhYolo dataset and pair every
pattern swatch with the name that belongs to it.

The dataset (data.yaml) has two classes:

    0 -> material_name      (the text label, e.g. "Sandstone")
    1 -> material_texture   (the lithology pattern swatch)

In a mudlog legend each swatch sits on the left and its name is written
immediately to the right on the same row, so a pattern is matched to the name
box that is horizontally closest while vertically overlapping.

Usage
-----
    # print the pairs for the train split
    python "crop_legend_patterns.py"

    # save crops + a JSON manifest
    python "crop_legend_patterns.py" --out crops --json pairs.json

    # also read the name text with tesseract (if installed)
    python "crop_legend_patterns.py" --out crops --ocr --csv pairs.csv

    # one image only
    python "crop_legend_patterns.py" --image path/to/legend.jpg --out crops

    # every split
    python "crop_legend_patterns.py" --split all --out crops --json pairs.json

    # validate the matches and render a contact sheet to eyeball them
    python "crop_legend_patterns.py" --split test --out crops --ocr \
        --validate --strict --contact-sheet crops/sheet.png

OCR
---
Name text is read with the `tesseract` command line tool (tesseract 5.x),
invoked through subprocess - no Python OCR package is needed. Each name crop is
greyscaled, autocontrasted, upscaled 4x and read with `--psm 7` (single text
line). If tesseract is not on PATH, `--ocr` silently yields empty names.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required:  pip install pillow")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ["material_name", "material_texture"]
DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent / "Dataset" / "pshhYolo.v1i.yolo26"
)

# Vocabulary used by --validate to sanity-check OCR output. A legend name that
# contains none of these tokens is reported as a warning, not a hard failure.
LITHOLOGY_TERMS = {
    # clastics
    "sand", "sandstone", "sandy", "shale", "shaly", "clay", "claystone",
    "silt", "siltstone", "mud", "mudstone", "gravel", "conglomerate",
    "breccia", "till", "loess", "diamictite",
    # carbonates
    "lime", "limestone", "dolomite", "dolomitic", "chalk", "marl",
    "marlstone", "calcite", "aragonite", "wackestone", "packstone",
    "grainstone", "boundstone", "micrite", "sparite",
    # siliceous / evaporites
    "chert", "cherty", "halite", "salt", "anhydrite", "gypsum", "sylvite",
    "bentonite", "trona", "potash", "zeolite", "barite", "celestite",
    "fluorite", "phosphatic", "gypsiferous",
    # ores / minerals
    "pyrite", "siderite", "limonite", "hematite", "magnetite", "goethite",
    "glauconite", "quartz", "feldspar", "mica", "kaolinite", "illite",
    "smectite", "chlorite", "calcareous", "siliceous", "argillaceous",
    "ferruginous", "carbonate", "evaporite", "cement",
    # igneous / metamorphic
    "granite", "basalt", "gneiss", "schist", "slate", "tuff", "tuffaceous",
    "volcanic", "volcanics", "igneous", "metamorphic", "sedimentary",
    "andesite", "gabbro", "granodiorite", "diorite", "rhyolite", "dacite",
    "syenite", "monzonite", "tonalite", "trachyte", "phonolite", "komatiite",
    "anorthosite", "dunite", "peridotite", "pyroxenite", "serpentine",
    "serpentinite", "hornfels", "marble", "quartzite", "amphibolite",
    "phyllite", "migmatite", "pegmatite", "aplite", "felsic", "mafic",
    "ultramafic", "dike", "dyke", "sill", "vein", "fault", "brecciated",
    "altered", "alteration", "sericitization", "biotitization",
    "chloritization", "silicification", "dolomitization", "ash", "pumice",
    "scoria", "lapilli", "agglomerate", "ignimbrite", "welded", "tephra",
    # organics / shows / misc legend entries
    "coal", "lignite", "peat", "fossil", "fossils", "shell", "oil", "gas",
    "water", "core", "cutting", "cuttings", "sample", "loss", "show",
    "unknown", "no", "not", "present", "absent",
    # other legend entries seen in this dataset
    "diatomite", "diatomaceous", "bitumen", "bituminous", "tar", "pebble",
    "pebbles", "extrusive", "intrusive", "foraminifera", "foram", "obsidian",
    "paleosol", "phosphate", "glassy", "wash", "rubble", "cavings", "shaker",
    "mudcake", "filter", "cake", "invasion", "resistivity", "porosity",
    "void", "casing", "shoe", "survey", "carbonaceous", "others", "other",
    "trace", "minor", "abundant", "common", "rare",
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    """A bounding box in absolute pixel coordinates."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_list(self) -> list[int]:
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]


@dataclass
class Pair:
    """One pattern swatch matched with its name box."""

    image: str
    split: str
    index: int
    pattern: Box
    name: Box
    gap: float
    score: float
    name_text: str = ""
    pattern_crop: str = ""
    name_crop: str = ""
    pair_crop: str = ""
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "split": self.split,
            "index": self.index,
            "name_text": self.name_text,
            "pattern_bbox": self.pattern.as_list(),
            "name_bbox": self.name.as_list(),
            "gap_px": round(self.gap, 1),
            "match_score": round(self.score, 1),
            "image_size": [self.image_width, self.image_height],
            "pattern_crop": self.pattern_crop,
            "name_crop": self.name_crop,
            "pair_crop": self.pair_crop,
        }


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def load_class_names(dataset_dir: Path) -> list[str]:
    """Read `names:` from data.yaml without requiring PyYAML."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        return list(DEFAULT_CLASSES)
    lines = yaml_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        if raw.startswith("["):  # inline list: names: ['a', 'b']
            raw = raw.strip("[]")
            return [n.strip().strip("'\"") for n in raw.split(",") if n.strip()]
        # block list on the following lines
        names: list[str] = []
        for follow in lines[i + 1 :]:
            follow = follow.strip()
            if not follow.startswith("-"):
                break
            names.append(follow.lstrip("-").strip().strip("'\""))
        return names or list(DEFAULT_CLASSES)
    return list(DEFAULT_CLASSES)


def find_label_file(image_path: Path) -> Path | None:
    """Locate the YOLO label file that belongs to an image."""
    # <split>/images/foo.jpg  ->  <split>/labels/foo.txt
    candidate = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if candidate.is_file():
        return candidate
    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.is_file() else None


def read_boxes(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    """Parse a YOLO label file into absolute-pixel boxes."""
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append(
            Box(
                cls=cls,
                x1=(cx - w / 2) * img_w,
                y1=(cy - h / 2) * img_h,
                x2=(cx + w / 2) * img_w,
                y2=(cy + h / 2) * img_h,
            )
        )
    return boxes


def iter_images(dataset_dir: Path, split: str) -> Iterable[tuple[str, Path]]:
    """Yield (split, image_path) for the requested split(s)."""
    splits = ["train", "valid", "test"] if split == "all" else [split]
    for name in splits:
        img_dir = dataset_dir / name / "images"
        if not img_dir.is_dir():
            continue
        for path in sorted(img_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                yield name, path


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def match_pairs(
    patterns: Sequence[Box],
    names: Sequence[Box],
    min_v_overlap: float = 0.5,
    allow_left: bool = True,
) -> list[tuple[Box, Box, float, float]]:
    """
    Match every pattern box to the name box on its right.

    Returns a list of (pattern, name, horizontal_gap, score) tuples, ordered
    top-to-bottom then left-to-right like the legend itself.
    """
    if not names:
        return []

    ordered = sorted(patterns, key=lambda b: (round(b.cy / 10), b.cx))
    used: set[int] = set()
    results: list[tuple[Box, Box, float, float]] = []

    for pattern in ordered:
        best: tuple[float, int, float] | None = None
        for i, name in enumerate(names):
            if i in used:
                continue
            overlap = _overlap(pattern.y1, pattern.y2, name.y1, name.y2)
            ref_h = min(pattern.height, name.height)
            if ref_h <= 0 or overlap / ref_h < min_v_overlap:
                continue
            gap = name.x1 - pattern.x2
            if gap < -0.5 * min(pattern.width, name.width):
                if not allow_left:
                    continue
            # Distance, not signed gap: a name far to the left must never beat
            # the name sitting right next to the swatch. A small penalty keeps
            # the usual "name on the right" layout preferred on ties.
            score = abs(gap) + 0.5 * abs(pattern.cy - name.cy)
            if gap < 0:
                score += 0.25 * abs(gap)
            if best is None or score < best[0]:
                best = (score, i, gap)

        if best is None:  # nothing on the same row -> nearest name overall
            for i, name in enumerate(names):
                if i in used:
                    continue
                dist = ((pattern.cx - name.cx) ** 2 + (pattern.cy - name.cy) ** 2) ** 0.5
                if best is None or dist < best[0]:
                    best = (dist, i, name.x1 - pattern.x2)

        if best is None:  # every name already taken -> reuse the closest one
            for i, name in enumerate(names):
                dist = ((pattern.cx - name.cx) ** 2 + (pattern.cy - name.cy) ** 2) ** 0.5
                if best is None or dist < best[0]:
                    best = (dist, i, name.x1 - pattern.x2)

        if best is None:
            continue
        score, idx, gap = best
        used.add(idx)
        results.append((pattern, names[idx], gap, score))

    return results


# --------------------------------------------------------------------------- #
# Cropping / OCR
# --------------------------------------------------------------------------- #
def crop_box(
    image: Image.Image,
    box: Box,
    padding: int = 0,
    scale: float = 1.0,
) -> Image.Image:
    """Crop a box out of an image, with optional padding and upscaling."""
    left = max(0, int(round(box.x1)) - padding)
    top = max(0, int(round(box.y1)) - padding)
    right = min(image.width, int(round(box.x2)) + padding)
    bottom = min(image.height, int(round(box.y2)) + padding)
    if right <= left or bottom <= top:
        right, bottom = left + 1, top + 1
    crop = image.crop((left, top, right, bottom))
    if scale != 1.0:
        crop = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
            Image.LANCZOS,
        )
    return crop


def ocr_text(
    image: Image.Image,
    tesseract: str = "tesseract",
    psm: int = 7,
    scale: int = 4,
) -> str:
    """Read the text inside a crop with the tesseract CLI (optional)."""
    if shutil.which(tesseract) is None:
        return ""
    prepared = ImageOps.autocontrast(image.convert("L"))
    prepared = prepared.resize(
        (prepared.width * scale, prepared.height * scale), Image.LANCZOS
    )
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    try:
        prepared.save(tmp)
        proc = subprocess.run(
            [tesseract, str(tmp), "stdout", "--psm", str(psm)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return " ".join(proc.stdout.split())
    except (OSError, subprocess.SubprocessError):
        return ""
    finally:
        tmp.unlink(missing_ok=True)


def compose_pair(pattern: Image.Image, name: Image.Image, gap: int = 12) -> Image.Image:
    """Side-by-side image of a swatch and its name, for quick visual checks."""
    height = max(pattern.height, name.height)
    canvas = Image.new("RGB", (pattern.width + gap + name.width, height), "white")
    canvas.paste(pattern, (0, (height - pattern.height) // 2))
    canvas.paste(name, (pattern.width + gap, (height - name.height) // 2))
    return canvas


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
@dataclass
class ImageResult:
    """Everything extracted from one legend image."""

    image: Path
    split: str
    pairs: list[Pair] = field(default_factory=list)
    n_patterns: int = 0
    n_names: int = 0
    has_labels: bool = True

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def process_image(
    image_path: Path,
    split: str,
    class_names: Sequence[str],
    pattern_cls: int,
    name_cls: int,
    out_dir: Path | None = None,
    padding: int = 0,
    scale: float = 1.0,
    do_ocr: bool = False,
    compose: bool = False,
    min_v_overlap: float = 0.5,
) -> ImageResult:
    """Crop and match every pattern/name pair of a single legend image."""
    result = ImageResult(image=image_path, split=split)
    label_path = find_label_file(image_path)
    if label_path is None:
        result.has_labels = False
        return result

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        boxes = read_boxes(label_path, img.width, img.height)
        patterns = [b for b in boxes if b.cls == pattern_cls]
        names = [b for b in boxes if b.cls == name_cls]
        result.n_patterns = len(patterns)
        result.n_names = len(names)
        if not patterns:
            return result

        matches = match_pairs(patterns, names, min_v_overlap=min_v_overlap)
        pairs: list[Pair] = []

        for index, (pattern, name, gap, score) in enumerate(matches):
            pair = Pair(
                image=str(image_path),
                split=split,
                index=index,
                pattern=pattern,
                name=name,
                gap=gap,
                score=score,
                image_width=img.width,
                image_height=img.height,
            )

            pattern_img = crop_box(img, pattern, padding=padding, scale=scale)
            name_img = crop_box(img, name, padding=padding, scale=scale)

            if do_ocr:
                pair.name_text = ocr_text(crop_box(img, name, padding=1))

            if out_dir is not None:
                stem = f"{image_path.stem[:60]}_{index:02d}"
                pattern_dir = out_dir / "patterns"
                name_dir = out_dir / "names"
                pattern_dir.mkdir(parents=True, exist_ok=True)
                name_dir.mkdir(parents=True, exist_ok=True)

                pattern_file = pattern_dir / f"{stem}.png"
                name_file = name_dir / f"{stem}.png"
                pattern_img.save(pattern_file)
                name_img.save(name_file)
                pair.pattern_crop = str(pattern_file)
                pair.name_crop = str(name_file)

                if compose:
                    pair_dir = out_dir / "pairs"
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    pair_file = pair_dir / f"{stem}.png"
                    compose_pair(pattern_img, name_img).save(pair_file)
                    pair.pair_crop = str(pair_file)

            pairs.append(pair)

    result.pairs = pairs
    return result


def write_json(pairs: Sequence[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(pairs),
        "pairs": [p.to_dict() for p in pairs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(pairs: Sequence[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "split",
        "index",
        "name_text",
        "pattern_bbox",
        "name_bbox",
        "gap_px",
        "match_score",
        "pattern_crop",
        "name_crop",
        "pair_crop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = pair.to_dict()
            row["pattern_bbox"] = " ".join(str(v) for v in row["pattern_bbox"])
            row["name_bbox"] = " ".join(str(v) for v in row["name_bbox"])
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _has_lithology_term(text: str, cutoff: float = 0.8) -> bool:
    """True when any word in `text` is a lithology term, allowing OCR typos."""
    tokens = re.findall(r"[a-z]+", text.lower())
    for token in tokens:
        if token in LITHOLOGY_TERMS:
            return True
        if len(token) >= 3 and difflib.get_close_matches(
            token, LITHOLOGY_TERMS, n=1, cutoff=cutoff
        ):
            return True
    return False


def validate_pairs(
    results: Sequence[ImageResult],
    gap_outlier_factor: float = 3.0,
) -> dict:
    """
    Run structural checks over the extracted pairs.

    Hard checks (a failure means the extraction is wrong):
      * coverage      - every pattern box produced a pair
      * uniqueness    - no name box is used by two patterns
      * right_side    - the name sits to the right of its pattern
      * vertical      - pattern and name share a row
      * in_bounds     - every box lies inside the image

    Soft checks (reported as warnings):
      * gap_outliers  - horizontal gap far above the median
      * ocr_known     - OCR text contains a known lithology term
    """
    pairs = [p for r in results for p in r.pairs]
    n_patterns = sum(r.n_patterns for r in results)
    n_names = sum(r.n_names for r in results)
    images_without_labels = [str(r.image) for r in results if not r.has_labels]

    report: dict = {
        "images": len(results),
        "images_without_labels": images_without_labels,
        "patterns": n_patterns,
        "names": n_names,
        "pairs": len(pairs),
        "checks": {},
        "warnings": [],
    }
    checks = report["checks"]

    # coverage
    missing = n_patterns - len(pairs)
    checks["coverage"] = {
        "ok": missing == 0,
        "detail": f"{len(pairs)}/{n_patterns} pattern boxes matched",
    }

    # uniqueness of name boxes per image
    seen: dict[tuple[str, tuple[int, ...]], int] = {}
    for pair in pairs:
        key = (pair.image, tuple(pair.name.as_list()))
        seen[key] = seen.get(key, 0) + 1
    reused = {k: v for k, v in seen.items() if v > 1}
    checks["uniqueness"] = {
        "ok": not reused,
        "detail": f"{len(reused)} name box(es) reused",
    }

    # name to the right of the pattern
    left = [p for p in pairs if p.gap < -0.5 * min(p.pattern.width, p.name.width)]
    checks["right_side"] = {
        "ok": not left,
        "detail": f"{len(left)} pair(s) with the name left of the pattern",
    }

    # same row: the name's centre must fall inside the pattern's vertical span
    # (a little slack, because text is often centred lower in a tall swatch)
    off_row = [
        p
        for p in pairs
        if not (
            p.pattern.y1 - 0.25 * p.pattern.height
            <= p.name.cy
            <= p.pattern.y2 + 0.25 * p.pattern.height
        )
    ]
    checks["vertical"] = {
        "ok": not off_row,
        "detail": f"{len(off_row)} pair(s) not vertically aligned",
    }

    # boxes inside the image
    outside = [
        p
        for p in pairs
        if p.image_width
        and (
            p.pattern.x1 < -1
            or p.pattern.y1 < -1
            or p.pattern.x2 > p.image_width + 1
            or p.pattern.y2 > p.image_height + 1
            or p.name.x1 < -1
            or p.name.y1 < -1
            or p.name.x2 > p.image_width + 1
            or p.name.y2 > p.image_height + 1
        )
    ]
    checks["in_bounds"] = {
        "ok": not outside,
        "detail": f"{len(outside)} pair(s) with a box outside the image",
    }

    # names that no pattern claimed (usually a missing pattern annotation)
    unmatched = n_names - len(pairs)
    checks["name_usage"] = {
        "ok": unmatched <= 0,
        "soft": True,
        "detail": f"{len(pairs)}/{n_names} name boxes used"
        + (f", {unmatched} left over" if unmatched > 0 else ""),
    }
    if unmatched > 0:
        report["warnings"].append(
            f"{unmatched} name box(es) had no pattern on their row "
            "(likely a missing pattern annotation)"
        )

    # gap outliers (soft)
    gaps = [p.gap for p in pairs]
    median_gap = _median(gaps)
    outliers = [p for p in pairs if p.gap > max(median_gap * gap_outlier_factor, 40)]
    checks["gap_outliers"] = {
        "ok": not outliers,
        "soft": True,
        "detail": (
            f"median gap {median_gap:.0f}px, "
            f"{len(outliers)} pair(s) above {max(median_gap * gap_outlier_factor, 40):.0f}px"
        ),
    }
    for pair in outliers[:10]:
        report["warnings"].append(
            f"wide gap {pair.gap:.0f}px: {Path(pair.image).name} #{pair.index}"
        )

    # OCR sanity (soft, only when OCR ran)
    ocr_pairs = [p for p in pairs if p.name_text]
    if ocr_pairs:
        unknown = [p for p in ocr_pairs if not _has_lithology_term(p.name_text)]
        checks["ocr_known"] = {
            "ok": len(unknown) <= 0.1 * len(ocr_pairs),
            "soft": True,
            "detail": (
                f"{len(ocr_pairs) - len(unknown)}/{len(ocr_pairs)} OCR names "
                f"contain a known lithology term"
            ),
        }
        for pair in unknown[:10]:
            report["warnings"].append(
                f"unrecognised OCR '{pair.name_text}': "
                f"{Path(pair.image).name} #{pair.index}"
            )
    else:
        checks["ocr_known"] = {
            "ok": True,
            "soft": True,
            "detail": "skipped (run with --ocr to check name text)",
        }

    # Only hard checks decide pass/fail; soft ones are data-quality warnings.
    report["ok"] = all(c["ok"] for c in checks.values() if not c.get("soft"))
    return report


def print_report(report: dict) -> None:
    print("\nValidation")
    print("-" * 62)
    print(
        f"  images {report['images']}   patterns {report['patterns']}   "
        f"names {report['names']}   pairs {report['pairs']}"
    )
    if report["images_without_labels"]:
        print(f"  images without labels: {len(report['images_without_labels'])}")
    print()
    for name, check in report["checks"].items():
        if check["ok"]:
            status = "PASS"
        else:
            status = "WARN" if check.get("soft") else "FAIL"
        print(f"  [{status}] {name:<14} {check['detail']}")
    if report["warnings"]:
        print("\n  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
    print("-" * 62)
    print(f"  result: {'PASS' if report['ok'] else 'FAIL'}")


def contact_sheet(
    pairs: Sequence[Pair],
    path: Path,
    columns: int = 6,
    cell: int = 150,
    max_pairs: int = 120,
) -> Path | None:
    """Render a grid of pattern+name pairs so a human can eyeball the matching."""
    selected = [p for p in pairs if p.pattern_crop and p.name_crop][:max_pairs]
    if not selected:
        return None

    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
    for i, pair in enumerate(selected):
        try:
            pattern = Image.open(pair.pattern_crop).convert("RGB")
            name = Image.open(pair.name_crop).convert("RGB")
        except OSError:
            continue
        tile = compose_pair(pattern, name, gap=6)
        tile.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
        x = (i % columns) * cell + (cell - tile.width) // 2
        y = (i // columns) * cell + (cell - tile.height) // 2
        sheet.paste(tile, (x, y))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def print_table(pairs: Sequence[Pair], show_paths: bool) -> None:
    if not pairs:
        print("No pattern/name pairs found.")
        return
    header = f"{'image':<34} {'#':>3}  {'name':<22} {'gap':>6}  {'pattern bbox':<22}"
    print(header)
    print("-" * len(header))
    for pair in pairs:
        image = Path(pair.image).name
        if len(image) > 33:
            image = image[:30] + "..."
        label = pair.name_text or "(no ocr)"
        bbox = ",".join(str(v) for v in pair.pattern.as_list())
        print(f"{image:<34} {pair.index:>3}  {label:<22} {pair.gap:>6.0f}  {bbox:<22}")
        if show_paths and pair.pattern_crop:
            print(f"{'':<34}      pattern -> {pair.pattern_crop}")
            print(f"{'':<34}      name    -> {pair.name_crop}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop lithology pattern swatches and their names from the pshhYolo "
            "YOLO dataset, matching each pattern with the name beside it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset root containing data.yaml and train/valid/test folders",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test", "all"],
        help="Which split to process",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Process a single image instead of a split (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write cropped pattern/name images into",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the matched pairs to a JSON manifest",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the matched pairs to a CSV manifest",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Read the name text with the tesseract CLI (tesseract 5.x on PATH)",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Also save a side-by-side pattern+name image per pair",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Pixels of padding around each crop",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Upscale factor for saved crops",
    )
    parser.add_argument(
        "--min-v-overlap",
        type=float,
        default=0.5,
        help="Minimum vertical overlap (0-1) for a name to match a pattern",
    )
    parser.add_argument(
        "--pattern-class",
        type=int,
        default=1,
        help="Class id of the pattern swatch",
    )
    parser.add_argument(
        "--name-class",
        type=int,
        default=0,
        help="Class id of the name text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run structural checks on the matches and print a PASS/FAIL report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when --validate reports a failure",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=None,
        help="Write a grid image of pattern+name pairs for visual inspection",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary line",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_dir: Path = args.dataset
    if not dataset_dir.is_dir():
        print(f"error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    class_names = load_class_names(dataset_dir)
    if args.pattern_class >= len(class_names) or args.name_class >= len(class_names):
        print(
            f"warning: class ids {args.pattern_class}/{args.name_class} exceed "
            f"data.yaml names {class_names}",
            file=sys.stderr,
        )

    if args.image:
        targets = [("custom", path) for path in args.image]
    else:
        targets = list(iter_images(dataset_dir, args.split))
    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print("error: no images found", file=sys.stderr)
        return 1

    results: list[ImageResult] = []
    for split, image_path in targets:
        results.append(
            process_image(
                image_path=image_path,
                split=split,
                class_names=class_names,
                pattern_cls=args.pattern_class,
                name_cls=args.name_class,
                out_dir=args.out,
                padding=args.padding,
                scale=args.scale,
                do_ocr=args.ocr,
                compose=args.compose,
                min_v_overlap=args.min_v_overlap,
            )
        )
    all_pairs = [pair for result in results for pair in result.pairs]

    if not args.quiet:
        print_table(all_pairs, show_paths=args.out is not None)

    if args.json:
        write_json(all_pairs, args.json)
    if args.csv:
        write_csv(all_pairs, args.csv)

    report: dict | None = None
    if args.validate:
        report = validate_pairs(results)
        print_report(report)

    if args.contact_sheet:
        sheet = contact_sheet(all_pairs, args.contact_sheet)
        if sheet:
            print(f"contact sheet -> {sheet}")
        else:
            print(
                "contact sheet skipped (needs --out so crops exist)",
                file=sys.stderr,
            )

    print(
        f"\n{len(all_pairs)} pattern/name pairs from {len(targets)} image(s)"
        + (f" -> crops in {args.out}" if args.out else "")
        + (f", json {args.json}" if args.json else "")
        + (f", csv {args.csv}" if args.csv else "")
    )

    if report is not None and not report["ok"] and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
