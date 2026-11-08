"""Check the SUPPLIED boxes against the original full-resolution pixels.

For each page:
  1. Numbers: for the two column boxes, find the nearest strong vertical ruled line
     to each side (x1, x2) inside the column's height range. A box that follows the
     column border should have a line within a few pixels of each side.
  2. Picture: work/previews/fullres/<stem>.png with full-resolution views of each box
     at its top edge, middle and bottom edge (box drawn on top). Inspection only;
     these images are never used for training.
Writes work/box_check.csv.
Usage: work/.venv/Scripts/python work/check_boxes.py [stem-prefix ...]
"""
import csv
import gc
import sys

import cv2
import numpy as np

from pages import BOXES, COLORS, ROOT, WORK, _decode_full, read_splits

SEARCH = 25          # px to search on each side of a box edge for a ruled line
LINE_DARK = 0.6      # a column counts as a "line" if >= 60% of its rows are dark
TILE_H = 300         # rows shown above/below an edge
MARGIN = 120         # px shown left/right of a column box
NAMES = {"depth_column": "depth", "interpreted_lithology_column": "lithology", "lithology_legend": "legend"}


def grey(a):
    return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def nearest_line(dark_frac, x, w):
    """Offset (line_x - x) of the strongest dark column within +-SEARCH of x, or None."""
    lo, hi = max(0, x - SEARCH), min(w, x + SEARCH + 1)
    seg = dark_frac[lo:hi]
    cand = np.where(seg >= LINE_DARK)[0]
    if len(cand) == 0:
        return None, float(seg.max()) if len(seg) else 0.0
    off = cand + lo - x
    j = np.argmin(np.abs(off))
    return int(off[j]), float(seg[cand[j]])


def line_check(g, b):
    x1, y1, x2, y2 = b
    h, w = g.shape
    # middle 80% of the box height, sampled every 4th row (memory/speed)
    ya, yb = y1 + (y2 - y1) // 10, y2 - (y2 - y1) // 10
    band = g[ya:yb:4] < 128
    dark = band.mean(0)
    return nearest_line(dark, x1, w), nearest_line(dark, x2, w)


def tile(img, b, cy, rows, cols_pad, color):
    """Full-res window around (box x-range, row cy), with the box drawn."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    xa, xb = max(0, x1 - cols_pad), min(w, x2 + cols_pad)
    ya, yb = max(0, cy - rows), min(h, cy + rows)
    t = img[ya:yb, xa:xb]
    t = np.stack([t] * 3, -1) if t.ndim == 2 else t.copy()
    cv2.rectangle(t, (x1 - xa, y1 - ya), (x2 - 1 - xa, y2 - 1 - ya), color, 2)
    return t


def fit(t, width):
    s = width / t.shape[1]
    return cv2.resize(t, (width, max(1, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)


def label(t, text):
    bar = np.full((26, t.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return np.vstack([bar, t])


def pad_w(t, width):
    if t.shape[1] < width:
        t = np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3), 255, np.uint8)])
    return t


def main():
    prefixes = sys.argv[1:]
    rows = [r for r in read_splits() if not prefixes or any(r["stem"].startswith(p) for p in prefixes)]
    out_dir = WORK / "previews" / "fullres"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for r in rows:
        img = _decode_full(ROOT / r["image"])
        g = grey(img)
        res = {"stem": r["stem"], "split": r["split"]}
        for k in ("depth_column", "interpreted_lithology_column"):
            (o1, d1), (o2, d2) = line_check(g, r["boxes"][k])
            res[f"{NAMES[k]}_x1_line_off"] = o1
            res[f"{NAMES[k]}_x2_line_off"] = o2
        results.append(res)

        # picture: per column box -> top / middle / bottom windows side by side; legend -> whole box
        col_tiles = []
        for k in ("depth_column", "interpreted_lithology_column"):
            b = r["boxes"][k]
            parts = [label(tile(img, b, cy, TILE_H, MARGIN, COLORS[k]), f"{NAMES[k]} {tag} y={cy}")
                     for cy, tag in ((b[1], "top"), ((b[1] + b[3]) // 2, "middle"), (b[3], "bottom"))]
            hh = max(p.shape[0] for p in parts)
            parts = [np.vstack([p, np.full((hh - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in parts]
            col_tiles.append(np.hstack([np.hstack([p, np.full((hh, 8, 3), 200, np.uint8)]) for p in parts]))
        L = r["boxes"]["lithology_legend"]
        leg = tile(img, L, (L[1] + L[3]) // 2, (L[3] - L[1]) // 2 + 80, 40, COLORS["lithology_legend"])
        width = max(t.shape[1] for t in col_tiles)
        leg = label(fit(leg, min(width, leg.shape[1])), f"legend {L}")
        canvas = np.vstack([pad_w(t, width) for t in col_tiles] + [pad_w(leg, width)])
        title = label(np.zeros((0, width, 3), np.uint8), f"{r['stem'][:80]}  ({r['width']}x{r['height']})")
        canvas = np.vstack([title, canvas])
        cv2.imwrite(str(out_dir / f"{r['stem']}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(r["stem"][:50], {k: v for k, v in res.items() if k not in ("stem", "split")}, flush=True)
        del img, g
        gc.collect()

    with open(WORK / "box_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()
