#!/usr/bin/env python3
"""
Crop every annotated class out of the mudlog dataset into per-class folders.

Two annotation sources are handled, because the classes live in different
places:

  1. Dataset/dataset/bbox_annotation/*.json
     Absolute-pixel boxes [x1, y1, x2, y2] over the full-resolution .tif
     mudlogs. Classes: lithology_legend, depth_column,
     interpreted_lithology_column.

  2. Dataset/pshhYolo.v1i.yolo26/<split>/labels/*.txt
     YOLO normalised boxes over the legend preview images. Classes:
     material_name, material_texture.

No name/pattern matching is performed - every box is simply cropped and saved.

Output layout
-------------
    <out>/
        lithology_legend/
        depth_column/
        interpreted_lithology_column/
        lithology/
            name/          <- material_name
            patterns/      <- material_texture

Usage
-----
    # everything, default output folder
    python "crop_dataset_classes.py"

    # explicit output folder
    python "crop_dataset_classes.py" --out crops

    # only the three column/legend classes
    python "crop_dataset_classes.py" --out crops --classes lithology_legend depth_column

    # only the lithology name/pattern crops, all splits
    python "crop_dataset_classes.py" --out crops --classes material_name material_texture --split all

    # add a manifest and upscale the crops
    python "crop_dataset_classes.py" --out crops --json crops/manifest.json --scale 2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency ({exc.name}):  pip install -r requirements.txt")

# Mudlogs are gigapixel images (up to ~4864 x 93524); lift Pillow's guard.
Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DATASET = REPO_ROOT / "Dataset" / "dataset"
DEFAULT_YOLO_DATASET = REPO_ROOT / "Dataset" / "pshhYolo.v1i.yolo26"
DEFAULT_OUT = REPO_ROOT / "crops"

# class name -> output sub-folder (relative to --out)
CLASS_FOLDERS = {
    # from bbox_annotation/*.json
    "lithology_legend": "lithology_legend",
    "depth_column": "depth_column",
    "interpreted_lithology_column": "interpreted_lithology_column",
    # from the YOLO labels
    "material_name": "lithology/name",
    "material_texture": "lithology/patterns",
}

JSON_CLASSES = ["lithology_legend", "depth_column", "interpreted_lithology_column"]
YOLO_CLASSES = ["material_name", "material_texture"]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    """One saved crop."""

    source: str  # "json" or "yolo"
    image: str
    cls: str
    folder: str
    bbox: list[int]
    width: int
    height: int
    path: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "image": self.image,
            "class": self.cls,
            "folder": self.folder,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            "path": self.path,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_stem(name: str, limit: int = 80) -> str:
    """Turn an image stem into a filesystem-safe name."""
    stem = _UNSAFE.sub("_", Path(name).stem).strip("_")
    return (stem or "image")[:limit]


def clamp_box(
    box: Sequence[float], width: int, height: int, padding: int = 0
) -> tuple[int, int, int, int]:
    """Clamp a box to the image and apply padding."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    left = max(0, int(round(min(x1, x2))) - padding)
    top = max(0, int(round(min(y1, y2))) - padding)
    right = min(width, int(round(max(x1, x2))) + padding)
    bottom = min(height, int(round(max(y1, y2))) + padding)
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def save_crop(
    image: Image.Image,
    box: Sequence[float],
    dest: Path,
    padding: int = 0,
    scale: float = 1.0,
) -> tuple[int, int]:
    """Crop, optionally upscale, and save as PNG. Returns the saved size."""
    left, top, right, bottom = clamp_box(box, image.width, image.height, padding)
    crop = image.crop((left, top, right, bottom))
    if crop.mode not in ("L", "RGB"):
        crop = crop.convert("L" if crop.mode in ("1", "I", "I;16", "F") else "RGB")
    if scale != 1.0:
        crop = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
            Image.LANCZOS,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)
    return crop.width, crop.height


# --------------------------------------------------------------------------- #
# Source 1: bbox_annotation/*.json over the full-resolution .tif mudlogs
# --------------------------------------------------------------------------- #
def crop_json_dataset(
    dataset_dir: Path,
    out_dir: Path,
    classes: Sequence[str],
    padding: int = 0,
    scale: float = 1.0,
    limit: int | None = None,
) -> list[Crop]:
    """Crop the absolute-pixel boxes stored in bbox_annotation/*.json."""
    ann_dir = dataset_dir / "bbox_annotation"
    img_dir = dataset_dir / "images"
    if not ann_dir.is_dir():
        print(f"warning: no bbox_annotation folder in {dataset_dir}", file=sys.stderr)
        return []

    wanted = [c for c in classes if c in JSON_CLASSES]
    if not wanted:
        return []

    annotations = sorted(ann_dir.glob("*.json"))
    if limit is not None:
        annotations = annotations[:limit]

    crops: list[Crop] = []
    for ann_path in annotations:
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {ann_path.name}: {exc}", file=sys.stderr)
            continue

        image_name = data.get("image_name") or f"{ann_path.stem}.tif"
        image_path = img_dir / image_name
        if not image_path.is_file():
            print(f"warning: missing image {image_name}", file=sys.stderr)
            continue

        boxes = data.get("bounding_boxes") or {}
        present = [c for c in wanted if boxes.get(c)]
        if not present:
            continue

        try:
            with Image.open(image_path) as img:
                img.load()
                for cls in present:
                    folder = CLASS_FOLDERS[cls]
                    dest = out_dir / folder / f"{safe_stem(image_name)}.png"
                    width, height = save_crop(
                        img, boxes[cls], dest, padding=padding, scale=scale
                    )
                    crops.append(
                        Crop(
                            source="json",
                            image=str(image_path),
                            cls=cls,
                            folder=folder,
                            bbox=[int(round(v)) for v in boxes[cls][:4]],
                            width=width,
                            height=height,
                            path=str(dest),
                        )
                    )
        except OSError as exc:
            print(f"warning: cannot read {image_name}: {exc}", file=sys.stderr)

    return crops


# --------------------------------------------------------------------------- #
# Source 2: YOLO labels over the legend preview images
# --------------------------------------------------------------------------- #
def load_yolo_class_names(yolo_dir: Path) -> list[str]:
    """Read `names:` from data.yaml without requiring PyYAML."""
    yaml_path = yolo_dir / "data.yaml"
    if not yaml_path.is_file():
        return list(YOLO_CLASSES)
    lines = yaml_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        if raw.startswith("["):
            return [
                n.strip().strip("'\"") for n in raw.strip("[]").split(",") if n.strip()
            ]
        names: list[str] = []
        for follow in lines[i + 1 :]:
            follow = follow.strip()
            if not follow.startswith("-"):
                break
            names.append(follow.lstrip("-").strip().strip("'\""))
        return names or list(YOLO_CLASSES)
    return list(YOLO_CLASSES)


def iter_yolo_images(yolo_dir: Path, split: str) -> Iterable[tuple[str, Path]]:
    """Yield (split, image_path) for the requested split(s)."""
    splits = ["train", "valid", "test"] if split == "all" else [split]
    for name in splits:
        img_dir = yolo_dir / name / "images"
        if not img_dir.is_dir():
            continue
        for path in sorted(img_dir.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield name, path


def read_yolo_boxes(
    label_path: Path, width: int, height: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse a YOLO label file into (class_id, absolute pixel box) tuples."""
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
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
            (
                cls,
                (
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ),
            )
        )
    return boxes


def crop_yolo_dataset(
    yolo_dir: Path,
    out_dir: Path,
    classes: Sequence[str],
    split: str = "all",
    padding: int = 0,
    scale: float = 1.0,
    limit: int | None = None,
) -> list[Crop]:
    """Crop the YOLO-labelled name/pattern boxes from the legend previews."""
    wanted = [c for c in classes if c in YOLO_CLASSES]
    if not wanted:
        return []

    class_names = load_yolo_class_names(yolo_dir)
    wanted_ids = {class_names.index(c): c for c in wanted if c in class_names}
    if not wanted_ids:
        print(
            f"warning: none of {wanted} found in {yolo_dir / 'data.yaml'} "
            f"(names: {class_names})",
            file=sys.stderr,
        )
        return []

    targets = list(iter_yolo_images(yolo_dir, split))
    if limit is not None:
        targets = targets[:limit]

    crops: list[Crop] = []
    for _split, image_path in targets:
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                boxes = read_yolo_boxes(label_path, img.width, img.height)
                counters: dict[str, int] = {}
                for cls_id, box in boxes:
                    cls = wanted_ids.get(cls_id)
                    if cls is None:
                        continue
                    folder = CLASS_FOLDERS[cls]
                    index = counters.get(cls, 0)
                    counters[cls] = index + 1
                    dest = (
                        out_dir
                        / folder
                        / f"{safe_stem(image_path.name)}_{index:02d}.png"
                    )
                    width, height = save_crop(
                        img, box, dest, padding=padding, scale=scale
                    )
                    crops.append(
                        Crop(
                            source="yolo",
                            image=str(image_path),
                            cls=cls,
                            folder=folder,
                            bbox=[int(round(v)) for v in box],
                            width=width,
                            height=height,
                            path=str(dest),
                        )
                    )
        except OSError as exc:
            print(f"warning: cannot read {image_path.name}: {exc}", file=sys.stderr)

    return crops


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def write_json(crops: Sequence[Crop], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(crops),
        "by_class": _count_by_class(crops),
        "crops": [c.to_dict() for c in crops],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(crops: Sequence[Crop], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "image",
        "class",
        "folder",
        "bbox",
        "width",
        "height",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for crop in crops:
            row = crop.to_dict()
            row["bbox"] = " ".join(str(v) for v in row["bbox"])
            writer.writerow(row)


def _count_by_class(crops: Sequence[Crop]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for crop in crops:
        counts[crop.cls] = counts.get(crop.cls, 0) + 1
    return counts


def print_summary(crops: Sequence[Crop], out_dir: Path) -> None:
    counts = _count_by_class(crops)
    print(f"\n{len(crops)} crops written to {out_dir}")
    if not counts:
        return
    width = max(len(c) for c in counts)
    print()
    for cls in sorted(counts):
        print(f"  {cls:<{width}}  {counts[cls]:>5}  ->  {out_dir / CLASS_FOLDERS[cls]}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop every annotated class from the mudlog dataset into per-class "
            "folders. No name/pattern matching is performed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json-dataset",
        type=Path,
        default=DEFAULT_JSON_DATASET,
        help="Dataset root with bbox_annotation/ and images/ (absolute-pixel JSON)",
    )
    parser.add_argument(
        "--yolo-dataset",
        type=Path,
        default=DEFAULT_YOLO_DATASET,
        help="YOLO dataset root with data.yaml and <split>/labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output root; one sub-folder per class is created inside",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=JSON_CLASSES + YOLO_CLASSES,
        choices=JSON_CLASSES + YOLO_CLASSES,
        help="Which classes to crop",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "valid", "test", "all"],
        help="YOLO split to read (ignored for the JSON dataset)",
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
        "--limit",
        type=int,
        default=None,
        help="Process at most N images per source",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write a JSON manifest of every crop",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write a CSV manifest of every crop",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out
    classes = list(dict.fromkeys(args.classes))  # de-duplicate, keep order

    crops: list[Crop] = []

    json_classes = [c for c in classes if c in JSON_CLASSES]
    if json_classes:
        if not args.json_dataset.is_dir():
            print(
                f"warning: JSON dataset not found: {args.json_dataset}", file=sys.stderr
            )
        else:
            crops += crop_json_dataset(
                args.json_dataset,
                out_dir,
                json_classes,
                padding=args.padding,
                scale=args.scale,
                limit=args.limit,
            )

    yolo_classes = [c for c in classes if c in YOLO_CLASSES]
    if yolo_classes:
        if not args.yolo_dataset.is_dir():
            print(
                f"warning: YOLO dataset not found: {args.yolo_dataset}", file=sys.stderr
            )
        else:
            crops += crop_yolo_dataset(
                args.yolo_dataset,
                out_dir,
                yolo_classes,
                split=args.split,
                padding=args.padding,
                scale=args.scale,
                limit=args.limit,
            )

    if not crops:
        print("error: nothing was cropped", file=sys.stderr)
        return 1

    if args.json:
        write_json(crops, args.json)
    if args.csv:
        write_csv(crops, args.csv)

    print_summary(crops, out_dir)
    if args.json:
        print(f"  manifest -> {args.json}")
    if args.csv:
        print(f"  manifest -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
