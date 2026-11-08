"""Draw boxes on whole pages so you can see if they are right.

Ground truth (the supplied boxes) = solid lines. Predictions, if
work/outputs/<split>/<stem>.json exists, = dashed lines of the same colour, and
the per-box IoU and edge errors (original pixels) are written on the image.

Each preview has: left = whole page squashed to 1600 px tall; right = zooms on
the top and bottom edge of each box (from the cached model-size page).

Usage: work/.venv/Scripts/python work/draw.py [split ...]   (default: val)
Output: work/previews/<split>/<stem>.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pages import BOXES, COLORS, ROOT, WORK, boxes_to_small, load_resized, read_splits

VIEW_H = 1600
TILE_W, TILE_H, ZOOM = 256, 160, 2
NAMES = {"depth_column": "depth", "interpreted_lithology_column": "lithology", "lithology_legend": "legend"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dashed_rect(img, b, color, t, dash=8):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    for (xa, ya, xb, yb) in ((x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)):
        n = max(1, int(np.hypot(xb - xa, yb - ya) // dash))
        for i in range(0, n, 2):
            p = (int(xa + (xb - xa) * i / n), int(ya + (yb - ya) * i / n))
            q = (int(xa + (xb - xa) * min(i + 1, n) / n), int(ya + (yb - ya) * min(i + 1, n) / n))
            cv2.line(img, p, q, color, t)


def draw_boxes(img, gt, pred, sx, sy, t):
    for k in BOXES:
        g = [gt[k][0] * sx, gt[k][1] * sy, gt[k][2] * sx, gt[k][3] * sy]
        cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), COLORS[k], t)
        if pred and k in pred:
            p = [pred[k][0] * sx, pred[k][1] * sy, pred[k][2] * sx, pred[k][3] * sy]
            dashed_rect(img, p, (255, 0, 0), t)


def preview(r, split, pred_dir=None):
    w, h = r["width"], r["height"]
    page = load_resized(ROOT / r["image"], r["stem"], w, h)
    gt_s = boxes_to_small(r["boxes"], w, h)
    pjson = (pred_dir or WORK / "outputs" / split) / f"{r['stem']}.json"
    pred_o = json.load(open(pjson, encoding="utf-8"))["bounding_boxes"] if pjson.exists() else None
    pred_s = boxes_to_small(pred_o, w, h) if pred_o else None

    # overview: whole page squashed to VIEW_H tall
    vw = max(120, int(page.shape[1] * VIEW_H / page.shape[0] * 4))  # keep some width so columns stay visible
    vw = min(vw, page.shape[1])
    view = cv2.resize(page, (vw, VIEW_H), interpolation=cv2.INTER_AREA).copy()
    draw_boxes(view, gt_s, pred_s, vw / page.shape[1], VIEW_H / page.shape[0], 2)

    # zoom tiles on the top and bottom edge of each box
    tiles = []
    for k in BOXES:
        b = gt_s[k]
        cx = (b[0] + b[2]) / 2
        for edge_y, tag in ((b[1], "top"), (b[3], "bottom")):
            x0 = int(np.clip(cx - TILE_W / 2, 0, max(0, page.shape[1] - TILE_W)))
            y0 = int(np.clip(edge_y - TILE_H / 2, 0, max(0, page.shape[0] - TILE_H)))
            crop = page[y0:y0 + TILE_H, x0:x0 + TILE_W]
            tile = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
            tile[:crop.shape[0], :crop.shape[1]] = crop
            tile = cv2.resize(tile, (TILE_W * ZOOM, TILE_H * ZOOM), interpolation=cv2.INTER_NEAREST)
            shift = lambda bb: {kk: [(v - (x0 if i % 2 == 0 else y0)) for i, v in enumerate(vv)] for kk, vv in bb.items()}
            draw_boxes(tile, shift(gt_s), shift(pred_s) if pred_s else None, ZOOM, ZOOM, 2)
            cv2.putText(tile, f"{NAMES[k]} {tag}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, 6, 2)]
    grid = np.vstack(rows)

    # text panel
    lines = [r["stem"][:60], f"{split}  {w}x{h}  model input {page.shape[1]}x{page.shape[0]}",
             "solid = supplied box, dashed red = prediction"]
    for k in BOXES:
        line = f"{NAMES[k]:9s} gt {r['boxes'][k]}"
        if pred_o:
            g, p = r["boxes"][k], pred_o[k]
            err = [abs(a - b) for a, b in zip(g, p)]
            line += f"  IoU {iou(g, p):.3f}  |err| x1,y1,x2,y2 = {err}"
        lines.append(line)
    text = np.full((30 * len(lines) + 10, grid.shape[1], 3), 255, np.uint8)
    for i, s in enumerate(lines):
        col = COLORS[BOXES[i - 3]] if i >= 3 else (0, 0, 0)
        cv2.putText(text, s, (8, 26 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    right = np.vstack([text, grid])

    H = max(view.shape[0], right.shape[0])
    canvas = np.full((H, view.shape[1] + 12 + right.shape[1], 3), 230, np.uint8)
    canvas[:view.shape[0], :view.shape[1]] = view
    canvas[:right.shape[0], view.shape[1] + 12:] = right
    sub = pred_dir.relative_to(WORK / "outputs") if pred_dir else Path(split)
    out = WORK / "previews" / sub / f"{r['stem']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out


def main():
    """draw.py val                       -> supplied boxes only, previews/val/
       draw.py outputs/v0.2/val_snap_xy  -> with those predictions, previews/v0.2/val_snap_xy/"""
    args = sys.argv[1:] or ["val"]
    for a in args:
        if "/" in a or "\\" in a:
            pdir = (WORK / a).resolve()
            split = next(s for s in ("train", "val", "test") if pdir.name.startswith(s))
        else:
            pdir, split = None, a
        for r in read_splits():
            if r["split"] == split:
                print(preview(r, split, pdir).relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
