"""Predict the three regions on whole pages and (when supplied boxes exist) score them.

Usage (defaults: --tag v0.2-final, --snap xyL, GPU if available else CPU):
  work/.venv/Scripts/python work/predict.py --images page1.tif page2.tif --out-name demo
  work/.venv/Scripts/python work/predict.py --tag v0.2 --split val --snap ""      # raw model, no snapping
  work/.venv/Scripts/python work/predict.py --split test                         # run ONCE, at the end
Model: work/checkpoints/<tag>/best.pt (or last.pt if there is no best.pt).
Per page, in work/outputs/<tag>/<split>[_snap_<mode>]/:
  <stem>.json          predicted boxes in original pixels (same format as the supplied JSONs)
  <stem>_boundary.png  model-size page with the predicted boxes drawn
  <stem>_mask.png      model-size probability maps (R = depth, G = lithology, B = legend)
With supplied boxes: metrics.csv (per page and class) and summary printed.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from model import UNet, metrics, probs_to_boxes
from pages import BOXES, COLORS, ROOT, WORK, boxes_to_small, load_resized, read_splits
from train import predict_probs

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0.2-final")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--images", nargs="*", help="arbitrary page images (no scoring)")
    ap.add_argument("--out-name", default="heldout")
    ap.add_argument("--snap", default="xyL", help="edge snapping mode (refine.snap_boxes); '' = off")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    a = ap.parse_args()

    if a.split:
        pages = [r for r in read_splits() if r["split"] == a.split]
        name = a.split
    else:
        pages = []
        for p in a.images:
            p = Path(p).resolve()
            w, h = Image.open(p).size
            pages.append({"stem": p.stem, "image": p, "width": w, "height": h, "boxes": None})
        name = a.out_name
    out = WORK / "outputs" / a.tag / (name + (f"_snap_{a.snap}" if a.snap else ""))
    out.mkdir(parents=True, exist_ok=True)

    ck = WORK / "checkpoints" / a.tag
    ckpt = ck / "best.pt" if (ck / "best.pt").exists() else ck / "last.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model = UNet().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    model.eval()
    print(f"model {ckpt.relative_to(WORK)} -> {out.relative_to(WORK)}")
    if a.snap:
        from refine import ink_mask, snap_boxes

    rows = []
    for r in pages:
        w, h = r["width"], r["height"]
        page = load_resized(ROOT / r["image"], r["stem"], w, h)
        prob = predict_probs(model, page, dev)
        pred = probs_to_boxes(prob, w, h)
        if a.snap:
            pred = snap_boxes(pred, ink_mask(r["stem"], ROOT / r["image"]), a.snap)
        json.dump({"image_name": Path(r["image"]).name, "width": w, "height": h, "bounding_boxes": pred},
                  open(out / f"{r['stem']}.json", "w", encoding="utf-8"), indent=2)

        vis = page.copy()
        for k, b in boxes_to_small(pred, w, h).items():
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]) - 1, int(b[3]) - 1), COLORS[k], 3)
        cv2.imwrite(str(out / f"{r['stem']}_boundary.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        mask = (np.moveaxis(prob, 0, -1) * 255).astype(np.uint8)  # R=depth, G=lithology, B=legend
        cv2.imwrite(str(out / f"{r['stem']}_mask.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        if r["boxes"]:
            m = metrics(pred, r["boxes"], w, h)
            for k in BOXES:
                rows.append({"stem": r["stem"], "class": k, **{kk: round(v, 5) for kk, v in m[k].items()},
                             "pred": pred[k], "gt": r["boxes"][k]})
        print(r["stem"][:60], flush=True)

    if rows:
        with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n{'class':32s} {'IoU mean':>9s} {'IoU min':>8s} {'edge px median':>15s} {'pixel err':>10s}")
        for k in BOXES:
            rs = [x for x in rows if x["class"] == k]
            ious = [x["iou"] for x in rs]
            print(f"{k:32s} {np.mean(ious):9.3f} {np.min(ious):8.3f} "
                  f"{np.median([x['edge_px'] for x in rs]):15.1f} {np.mean([x['pixel_err'] for x in rs]):10.5f}")


if __name__ == "__main__":
    main()
