"""Load a whole page and shrink it to model size, with a disk cache.

The model input is the WHOLE page (never cropped), resized with separate x and y
scales: width -> OUT_W, height -> min(original, MAX_H) rounded to a multiple of 32.
Why not keep the aspect ratio: pages are up to 313800 x 1680. At max side 6144 with
the aspect ratio kept, that page would be 33 px wide and the 63 px lithology column
would shrink to ~1 px. Boxes are tall and thin, so x needs more resolution than y.

Cached files (work/cache/<stem>.png) are full pages, resized, RGB.
"""
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
CACHE = WORK / "cache"
OUT_W = 768
MAX_H = 6144
BOXES = ("depth_column", "interpreted_lithology_column", "lithology_legend")
COLORS = {"depth_column": (0, 160, 255), "interpreted_lithology_column": (255, 0, 200),
          "lithology_legend": (0, 200, 0)}


def target_size(w, h):
    """(out_w, out_h) for an original page of w x h."""
    out_h = min(h, MAX_H)
    out_h = max(32, int(round(out_h / 32)) * 32)
    return OUT_W, out_h


def _decode_full(path):
    """Decode a page into a uint8 array (H, W) or (H, W, 3). One page at a time."""
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        if p.compression.name.startswith("CCITT"):  # 1-bit fax: imagecodecs can't, Pillow can
            im = Image.open(path)
            im.load()
            return np.array(im, dtype=np.uint8) * 255  # mode '1' -> 0/255 (57 MB -> 455 MB max)
        arr = p.asarray()
        if p.photometric.name == "PALETTE":
            cmap = (p.colormap >> 8).astype(np.uint8).T  # (256, 3)
            if np.all(cmap[:, 0] == cmap[:, 1]) and np.all(cmap[:, 1] == cmap[:, 2]):
                return cmap[arr, 0]
            return cmap[arr]
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr


def load_resized(path, stem, w, h, cache=True):
    """Return the resized whole page as uint8 RGB (out_h, OUT_W, 3); cached by stem.
    cache=False neither reads nor writes the cache (for pages from outside the dataset,
    where two different files can share a name)."""
    out_w, out_h = target_size(w, h)
    cpath = CACHE / f"{stem}.png"
    if cache and cpath.exists():
        a = np.array(Image.open(cpath).convert("RGB"))
        if a.shape[:2] == (out_h, out_w):
            return a
    arr = _decode_full(path)
    assert arr.shape[:2] == (h, w), (stem, arr.shape, (h, w))
    small = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    del arr
    gc.collect()
    if small.ndim == 2:
        small = np.stack([small] * 3, -1)
    if cache:
        CACHE.mkdir(exist_ok=True)
        Image.fromarray(small).save(cpath)
    return small


def boxes_to_small(boxes, w, h):
    """Scale {name: [x1,y1,x2,y2]} from original pixels to resized pixels."""
    out_w, out_h = target_size(w, h)
    sx, sy = out_w / w, out_h / h
    return {k: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for k, b in boxes.items()}


def boxes_to_orig(boxes, w, h):
    out_w, out_h = target_size(w, h)
    sx, sy = w / out_w, h / out_h
    return {k: [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)] for k, b in boxes.items()}


def read_splits():
    rows = list(csv.DictReader(open(WORK / "splits.csv", encoding="utf-8")))
    for r in rows:
        r["width"], r["height"] = int(r["width"]), int(r["height"])
        r["boxes"] = {k: json.load(open(ROOT / r["json"], encoding="utf-8"))["bounding_boxes"][k] for k in BOXES}
    return rows
