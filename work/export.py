"""Export a trained checkpoint as ONE self-describing .pt file for teammates.

Usage:
  work/.venv/Scripts/python work/export.py --tag v0.2-final
  work/.venv/Scripts/python work/export.py --tag v0.2
Writes work/export/boxfinder_<tag>.pt containing
  "model": the U-Net weights (state_dict, float32)
  "meta":  everything needed to use them correctly (classes, input size, threshold,
           snapping mode, what it was trained on, measured scores)
Load it with boxfinder.py (BoxFinder("work/export/boxfinder_<tag>.pt")).
The .pt holds weights only, not code: model.py / pages.py / refine.py are still needed.
"""
import argparse
import datetime

import torch

from model import CHANNELS, UNet
from pages import BOXES, MAX_H, OUT_W, WORK

# measured results, copied from work/README.md (mean box IoU: depth / lithology / legend, with snapping)
INFO = {
    "v0.2": {"trained_on": "train split (35 pages), checkpoint picked by val IoU",
             "val_iou": [0.831, 0.904, 0.722], "test_iou": [0.510, 0.830, 0.377]},
    "v0.2-final": {"trained_on": "train + val splits (43 pages), 125 epochs, last checkpoint",
                   "val_iou": None, "test_iou": [0.499, 0.589, 0.346]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0.2-final")
    a = ap.parse_args()
    ck = WORK / "checkpoints" / a.tag
    src = ck / "best.pt" if (ck / "best.pt").exists() else ck / "last.pt"
    state = torch.load(src, map_location="cpu")["model"]
    UNet().load_state_dict(state)  # fails loudly if the weights do not match the code

    meta = {
        "name": "StackUnderflow mud-log region finder",
        "version": a.tag,
        "source_checkpoint": str(src.relative_to(WORK)),
        "exported": datetime.date.today().isoformat(),
        "torch_version": str(torch.__version__),  # plain str so torch.load(weights_only=True) works
        "architecture": {"class": "model.UNet", "in_channels": 5, "out_channels": 3, "channels": list(CHANNELS)},
        "classes": list(BOXES),
        "input": f"whole page resized to {OUT_W} wide x min(height, {MAX_H}) tall (multiple of 32), "
                 "RGB scaled to [-1, 1] + x/y coordinate channels (model.make_input)",
        "threshold": 0.5,
        "snap_mode": "xyL",
        "box_format": "[x1, y1, x2, y2] in original page pixels, same as Datasets/bbox_annotation",
        **INFO.get(a.tag, {}),
    }
    out = WORK / "export" / f"boxfinder_{a.tag}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"model": state, "meta": meta}, out)
    print(f"{src.relative_to(WORK)} -> {out.relative_to(WORK)} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
