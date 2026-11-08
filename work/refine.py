"""Edge snapping: move each predicted box edge onto the nearest printed ruled line
in the ORIGINAL full-resolution scan. No learning, no training.

A pixel counts as "ink" if it is dark and colourless-ish (max(R,G,B) < INK), so
coloured lithology fills (e.g. green 0,200,0) are not mistaken for lines.
  vertical line   at column x = ink in >= V_FRAC of the box's rows
  horizontal line at row y    = ink in >= H_FRAC of the box's columns
Each edge looks only within a small window around the model's edge and moves to the
nearest line; if there is none, the edge stays where the model put it.

The ink mask of each page is cached bit-packed in work/cache/ink/<stem>.npz
(a 1538 x 212600 page -> ~40 MB), so each scan is decoded only once.

CLI (evaluate snapping on saved predictions, CPU only):
  work/.venv/Scripts/python work/refine.py outputs/v0.2/val [--mode xy]
writes <dir>_snap_<mode>/ with snapped JSONs + metrics.csv and prints the summary.
"""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

from model import metrics
from pages import BOXES, MAX_H, OUT_W, ROOT, WORK, _decode_full, read_splits

INK = 110
V_FRAC = 0.5
H_FRAC = 0.85
COLUMNS = ("depth_column", "interpreted_lithology_column")


def ink_mask(stem, path, cache=True):
    """Bool (H, W) ink mask of the original page, cached by stem (cache=False: no cache)."""
    c = WORK / "cache" / "ink" / f"{stem}.npz"
    if cache and c.exists():
        z = np.load(c)
        return np.unpackbits(z["bits"], axis=1, count=int(z["w"])).astype(bool)
    a = _decode_full(path)
    h, w = a.shape[:2]
    bits = []
    for s in range(0, h, 4096):  # chunked so the temporary stays small
        blk = a[s:s + 4096]
        ink = (blk.max(axis=2) if blk.ndim == 3 else blk) < INK
        bits.append(np.packbits(ink, axis=1))
    del a
    gc.collect()
    bits = np.concatenate(bits)
    if cache:
        c.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(c, bits=bits, w=w)
    return np.unpackbits(bits, axis=1, count=w).astype(bool)


def _nearest(idx, target):
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(np.abs(idx - target))])


def snap_x(ink, b, R):
    """Snap x1, x2 to vertical lines within +-R px. Box convention: x1 on the left
    line, x2 one past the right line (matches the supplied boxes)."""
    h, w = ink.shape
    x1, y1, x2, y2 = b
    ya, yb = y1 + (y2 - y1) // 10, y2 - (y2 - y1) // 10
    step = max(1, (yb - ya) // 4000)
    lo, hi = max(0, x1 - R), min(w, x2 + R)
    frac = ink[ya:yb:step, lo:hi].mean(0)
    lines = np.where(frac >= V_FRAC)[0] + lo
    n1 = _nearest(lines[np.abs(lines - x1) <= R], x1)
    n2 = _nearest(lines[np.abs(lines - (x2 - 1)) <= R], x2 - 1)
    nx1 = n1 if n1 is not None else x1
    nx2 = n2 + 1 if n2 is not None else x2
    return [nx1, y1, nx2, y2] if nx2 - nx1 >= 4 else b


def snap_y(ink, b, R):
    """Snap y1, y2 to horizontal lines within +-R rows (over the box's columns)."""
    h, w = ink.shape
    x1, y1, x2, y2 = b
    xa, xb = x1 + 3, max(x1 + 4, x2 - 3)
    out = [y1, y2]
    for i, y in enumerate((y1, y2 - 1)):
        lo, hi = max(0, y - R), min(h, y + R + 1)
        frac = ink[lo:hi, xa:xb].mean(1)
        rows = np.where(frac >= H_FRAC)[0] + lo
        n = _nearest(rows, y)
        if n is not None:
            out[i] = n if i == 0 else n + 1
    return [x1, out[0], x2, out[1]] if out[1] - out[0] >= 4 else b


def snap_boxes(pred, ink, mode="xy"):
    """pred {class: [x1,y1,x2,y2]} in original px -> snapped copy.
    mode: any combination of 'x' (vertical edges) and 'y' (horizontal edges);
    'L' adds the legend (otherwise only the two columns are snapped)."""
    h, w = ink.shape
    rx = max(12, int(3 * w / OUT_W))  # ~3 model pixels across
    ry = max(30, int(3 * h / MAX_H))  # ~3 model pixels down
    out = {}
    for k in BOXES:
        b = [int(v) for v in pred[k]]
        if k in COLUMNS or "L" in mode:
            if "x" in mode:
                b = snap_x(ink, b, rx)
            if "y" in mode:
                b = snap_y(ink, b, ry)
        out[k] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--mode", default="xy")
    a = ap.parse_args()
    pdir = (WORK / a.pred_dir).resolve() if not Path(a.pred_dir).is_absolute() else Path(a.pred_dir)
    odir = pdir.parent / f"{pdir.name}_snap_{a.mode}"
    odir.mkdir(parents=True, exist_ok=True)
    rows = {r["stem"]: r for r in read_splits()}
    res = []
    for f in sorted(pdir.glob("*.json")):
        r = rows.get(f.stem)
        if r is None:
            continue
        pred = json.load(open(f, encoding="utf-8"))["bounding_boxes"]
        ink = ink_mask(r["stem"], ROOT / r["image"])
        snapped = snap_boxes(pred, ink, a.mode)
        del ink
        json.dump({"image_name": Path(r["image"]).name, "width": r["width"], "height": r["height"],
                   "bounding_boxes": snapped}, open(odir / f.name, "w", encoding="utf-8"), indent=2)
        m0 = metrics(pred, r["boxes"], r["width"], r["height"])
        m1 = metrics(snapped, r["boxes"], r["width"], r["height"])
        for k in BOXES:
            res.append({"stem": r["stem"], "class": k, "iou_before": round(m0[k]["iou"], 4),
                        "iou_after": round(m1[k]["iou"], 4), "edge_before": round(m0[k]["edge_px"], 1),
                        "edge_after": round(m1[k]["edge_px"], 1)})
    with open(odir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res[0]))
        wr.writeheader()
        wr.writerows(res)
    print(f"mode {a.mode}: {pdir.name} -> {odir.name}")
    for k in BOXES:
        rs = [x for x in res if x["class"] == k]
        print(f"  {k:30s} IoU {np.mean([x['iou_before'] for x in rs]):.3f} -> {np.mean([x['iou_after'] for x in rs]):.3f}"
              f"   median edge px {np.median([x['edge_before'] for x in rs]):.0f} -> {np.median([x['edge_after'] for x in rs]):.0f}")


if __name__ == "__main__":
    main()
