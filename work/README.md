# Bonus model: find the three regions on a mud-log page

Given only a page image (TIFF), predict the three boxes the organisers otherwise supply:
`depth_column`, `interpreted_lithology_column`, `lithology_legend`, each `[x1, y1, x2, y2]`
in original pixels. Output JSON has the same format as `Datasets/bbox_annotation/*.json`,
so the main interval pipeline can use predicted boxes exactly like supplied ones.

## The bonus task, and how we got here

**Task.** The main pipeline reads intervals from three regions of each mud-log page. For the
challenge pages those boxes are supplied by hand (`Datasets/bbox_annotation`). The bonus is to
find them from the page image alone, so the pipeline also works on pages that have no boxes.

**1. YOLO (first attempt, `Model training/train.ipynb`).** We fine-tuned a pretrained
Ultralytics YOLO26s detector at `imgsz=640`. That ran into a pixel limit: YOLO resizes the whole
image so its longest side is 640 px, and our pages are extremely tall (up to 313,800 × 1,680 px;
typical ~1,500 × 40,000). Keeping the aspect ratio, a 1,275 × 44,550 page becomes about
18 × 640 px, so a 60–80 px wide column shrinks to about one pixel and can't be found. Raising
`imgsz` enough to keep the columns visible (thousands of px) runs out of GPU memory, and tiling
the page breaks each column into many pieces a box detector sees only parts of.

**2. U-Net on the whole page (v0.1).** Instead of predicting boxes directly, we shrink the
page with *separate* x and y scales (768 wide × up to 6,144 tall), so columns stay ~35 px wide.
A small U-Net (4.3 M weights) colours in every pixel that belongs to each region, and the boxes are
the bounding rectangles of those coloured areas. Training targets are the supplied boxes filled
in, so nothing was drawn by hand. Trained 60 min on train pages only.
Result: lithology good, depth OK, legend weak (val 0.64 / 0.81 / 0.39, test 0.49 / 0.70 / 0.24).
Problems seen in the previews: the legend (the smallest region) was often missed; columns with
empty stretches were split into pieces; edges were a few pixels off because the model works
on a shrunk page.

**3. v0.2: four fixes, one per problem.**

| Problem in v0.1 | Fix in v0.2 |
|---|---|
| Legend missed (small region, little weight in the loss) | legend counts ×3 in the loss |
| Columns broken where a stretch is empty | "blank column" augmentation: random stretches inside columns are painted white during training, answer unchanged; plus a post-process step that joins pieces stacked in the same column |
| Edges a few pixels off | edge snapping (`refine.py`): each edge moves onto the nearest printed ruled line in the full-resolution scan |
| Training slow (CPU-bound input preparation) | inputs and augmentation built on the GPU: ~2× more epochs in the same 60 min |

Result: val 0.83 / 0.90 / 0.72; test 0.51 / 0.83 / 0.38.

**4. v0.2-final.** The same recipe retrained on train + val (43 pages, fixed 125 epochs), the usual
last step when data is scarce. It was chosen as the final model before test was run. On test it
scored 0.50 / 0.59 / 0.35: depth and legend about the same as v0.2, lithology lower because of
two pages where it picked the wrong column. With 8 test pages this gap is weak evidence either
way. Both are exported (`export/`).

**Where it still fails (mc761 test pages, unseen layout).**

- *Legend*: this layout has two symbol keys side by side: the lithology key and, just below it,
  an "engineering" key (Bit Trip, Mud Loss, ...). They look the same to the model (small symbol
  boxes with labels), and it picked the wrong one (IoU 0). Telling them apart needs reading the text.
- *Column tops*: the ruled column lines continue up through the header. The model and the
  "join pieces in the same column" step extend the depth/lithology box into the header, starting
  600–2,000 rows too high. Edge snapping only searches a few pixels, so it cannot fix that.

## Quick start (demo on any page)

```bash
# Python 3.12, from the repository root
py -3.12 -m venv work/.venv
work/.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
work/.venv/Scripts/python -m pip install -r work/requirements.txt

work/.venv/Scripts/python work/predict.py --images "path/to/page.tif" --out-name demo
```

Results land in `work/outputs/v0.2-final/demo_snap_xyL/`:

| File | What it is |
|---|---|
| `<stem>.json` | the three predicted boxes, original pixels |
| `<stem>_boundary.png` | page (768 px wide) with the boxes drawn: blue depth, pink lithology, green legend |
| `<stem>_mask.png` | the model's raw maps: red = depth, green = lithology, blue = legend |

Runs on an NVIDIA GPU if there is one, otherwise on the CPU (slower; add `--cpu` to force it).
The model file is `work/checkpoints/v0.2-final/last.pt` (17 MB). Pages are cached once in
`work/cache/` (resized page + ink mask), so the first run on a page is the slow one.

## Using the model from the main pipeline (.pt files)

The trained models are in `work/export/` (16 MB each, weights + a `meta` dict describing them):

| File | Trained on | Test IoU (depth / lithology / legend) |
|---|---|---|
| `boxfinder_v0.2-final.pt` | train + val (43 pages); chosen as final before test | 0.50 / 0.59 / 0.35 |
| `boxfinder_v0.2.pt` | train only (35 pages) | 0.51 / 0.83 / 0.38 |

```python
import sys; sys.path.insert(0, "work")
from boxfinder import BoxFinder
bf = BoxFinder("work/export/boxfinder_v0.2-final.pt")     # GPU if available, else CPU
boxes = bf.predict("Datasets/images/page.tif")            # same dict as bbox_annotation "bounding_boxes"
print(bf.meta)                                            # version, classes, input size, scores, ...
```

A `.pt` holds weights only, not code, so `model.py`, `pages.py`, `train.py` and `refine.py`
must come along with it. Re-export after retraining: `work/.venv/Scripts/python work/export.py --tag <tag>`.

## How it works

1. **Shrink the whole page** to 768 wide × up to 6144 tall (`pages.py`). Width and height are
   scaled separately so thin columns on very tall pages stay visible. Nothing is cropped.
2. **U-Net** (`model.py`, 4.3 M weights) reads the page plus two x/y position channels and outputs,
   for every pixel, the probability that it belongs to each of the three regions.
3. **Maps to boxes**: threshold at 0.5, keep the largest blob per region, join blobs stacked in
   the same column, take the bounding rectangle, scale back to original pixels.
4. **Edge snapping** (`refine.py`): open the original full-resolution scan, and move each edge
   onto the nearest printed ruled line within ~3 model pixels. Columns: left/right edges.
   Legend: all four edges. Coloured fills are not counted as lines.

Training (`train.py`): the supplied boxes, filled in, are the per-pixel answer key. Loss =
BCE + Dice per region, legend weighted ×3. AdamW, 1 min warm-up + cosine decay, bf16.
Augmentation: brightness/contrast/noise and "blank column" stripes (random stretches inside the
columns painted white, answer unchanged). No flips.

## Versions and scores (mean box IoU: depth / lithology / legend)

| Version | What changed | Val | Test |
|---|---|---|---|
| v0.1 | first U-Net, 60 min, train split only | 0.64 / 0.81 / 0.39 | 0.49 / 0.70 / 0.24 |
| v0.2 + snapping | legend ×3, blank-column stripes, GPU input prep, edge snapping | 0.83 / 0.90 / 0.72 | 0.51 / 0.83 / 0.38 |
| v0.2-final + snapping | same settings, retrained on train + val (43 pages), 125 epochs | – (val used for training) | 0.50 / 0.59 / 0.35 |

Val scores are optimistic: val was used to choose the checkpoint, the epoch count and the
snapping mode. Test (8 pages, 5 wells) is the honest number. v0.2-final was chosen as the
final model before test was run; v0.2 is shown for comparison.

What the test run shows:

- **Val gains only partly carried over to test.** Legend 0.24 → 0.35–0.38, lithology 0.70 → 0.59–0.83,
  depth about flat (0.49 → 0.50).
- **Snapping helps on test too** (raw → snapped, v0.2-final: 0.47 → 0.50 / 0.58 → 0.59 / 0.33 → 0.35).
- **The failures are wrong choices, not imprecise edges**: on Trail Mtn and Freeman the lithology
  box is on a different column (IoU 0); on Trail Mtn and one mc761 page the legend is found in the
  wrong place; on the mc761 pages the depth box starts ~2,000 rows too high, in the header.
  Snapping cannot fix a wrong column.
- **v0.2 vs v0.2-final differ mainly on two pages** (Trail Mtn, Freeman). With 8 test pages and one
  training run each, this gap is not strong evidence that adding val pages hurts.
- Good pages are very good: Critter Creek 202 scores 0.99 / 0.97 / 0.93.

## Known limits

- **Unseen layouts**: only 35–43 training pages. A new company's layout can make it pick the
  wrong column (test pages Trail Mtn and Freeman) or a box two columns wide (val page dc222, depth 0.46).
- **Empty column tops**: where logging starts lower than the header, the lithology box can start
  too low; there is no printed line nearby for snapping to fix it.
- **Black-and-white scans are untested**: all 9 one-bit pages come from one document and are in
  training; val and test are colour only.
- **Tilted scans**: a straight box cannot follow a slightly rotated column exactly.

## Files

`prepare_data.py` (pairing + split) · `pages.py` (loading, resizing, cache) · `model.py` (U-Net,
loss, post-processing, metrics) · `train.py` · `predict.py` · `refine.py` (edge snapping) ·
`export.py` (checkpoint → `export/boxfinder_<tag>.pt`) · `boxfinder.py` (Python API) ·
`draw.py` (previews vs supplied boxes) · `check_boxes.py` (checks supplied boxes at full res) ·
`splits.csv`, `data_report.md` · `versions/v0.1/` (frozen v0.1).
