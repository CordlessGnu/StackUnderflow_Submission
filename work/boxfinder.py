"""Python API for the main pipeline: page image in, three boxes out.

    import sys; sys.path.insert(0, "work")
    from boxfinder import BoxFinder
    bf = BoxFinder("work/export/boxfinder_v0.2-final.pt")   # GPU if available, else CPU
    boxes = bf.predict("Datasets/images/some_page.tif")
    # {"depth_column": [x1, y1, x2, y2], "interpreted_lithology_column": [...], "lithology_legend": [...]}

Boxes are in original page pixels, the same format as the supplied
Datasets/bbox_annotation/*.json "bounding_boxes", so they can replace those directly.
Command line: work/.venv/Scripts/python work/boxfinder.py work/export/boxfinder_v0.2-final.pt page.tif
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from model import UNet, probs_to_boxes
from pages import load_resized
from train import predict_probs

Image.MAX_IMAGE_PIXELS = None


class BoxFinder:
    def __init__(self, weights, device=None):
        ck = torch.load(weights, map_location="cpu")
        self.meta = ck.get("meta", {})
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = UNet().to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def predict(self, image_path, snap=True, return_probs=False, cache=True):
        """Boxes for one page. snap=True moves edges onto printed lines in the full-res scan
        (slower on the first run of a page: the scan is decoded once and cached in work/cache/)."""
        d = self.detect(image_path, snap=snap, cache=cache)
        return (d["boxes"], d["probs"]) if return_probs else d["boxes"]

    def detect(self, image_path, snap=True, cache=True):
        """Like predict, but returns everything: {"boxes" (final), "raw_boxes" (before
        snapping), "probs", "page" (model-size RGB page), "width", "height"}.
        cache=False: do not read or write work/cache (use for pages from outside the dataset)."""
        p = Path(image_path).resolve()
        with Image.open(p) as im:
            w, h = im.size
        page = load_resized(p, p.stem, w, h, cache=cache)
        prob = predict_probs(self.model, page, self.dev)
        raw = probs_to_boxes(prob, w, h, thr=self.meta.get("threshold", 0.5))
        boxes = raw
        if snap:
            from refine import ink_mask, snap_boxes
            boxes = snap_boxes(raw, ink_mask(p.stem, p, cache=cache), self.meta.get("snap_mode", "xyL"))
        return {"boxes": {k: [int(v) for v in b] for k, b in boxes.items()},
                "raw_boxes": {k: [int(v) for v in b] for k, b in raw.items()},
                "probs": prob, "page": page, "width": w, "height": h}


if __name__ == "__main__":
    bf = BoxFinder(sys.argv[1])
    print(json.dumps({k: v for k, v in bf.meta.items() if k in ("version", "trained_on", "test_iou")}))
    for img in sys.argv[2:]:
        print(Path(img).name, json.dumps(bf.predict(img)))
