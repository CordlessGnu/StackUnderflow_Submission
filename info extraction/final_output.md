# Pixel-to-depth calibration and final output

For a full-document TIFF and bbox JSON, use **`pipeline.py`**; see `pipeline.md`.
This guide documents the lower-level depth and cropped-input export components.

The last two nodes of `Flow Diagram.drawio` are implemented by:

- **`depth_distance.py`** — OCR depth labels and fit a validated pixel-to-depth mapping.
- **`final_output.py`** — combine that mapping with `interval_lithology.py` results,
  preserving cluster IDs, lithology labels, and confidence/matching diagnostics.

## Run the full flow

From the repository root:

```bash
python 'info extraction/final_output.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --output crops/final_output
```

The script finds the matching depth and legend images in the sibling
`depth_column/` and `lithology_legend/` directories. It locates the source bounding
box annotation under `Dataset/dataset/bbox_annotation/`, including unique matches
for filenames sanitized or truncated during cropping. Use `--depth-column`,
`--legend`, `--legend-manifest`, and `--annotation` to override these paths.

This lower-level entry point operates on the **cropped column**. The two-input
`pipeline.py` now handles cropping the annotated regions from a full TIFF and
calls these components directly.

### Reuse the existing lithology output

```bash
python 'info extraction/final_output.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --labelled-csv 'crops/labelled_examples/Rio_Lavaca_1748_Unit_1_241_MUD.csv' \
  --output crops/final_output
```

This avoids re-running boundary detection, clustering, and legend OCR. Extra
columns in the input CSV are retained. A matching classification JSON sidecar is
also retained for cluster summaries and warnings; stale sidecars are ignored.

A reviewed calibration can be reused with:

```bash
  --calibration 'crops/depth_calibration/Rio_Lavaca_1748_Unit_1_241_MUD.json'
```

or with the `<document_stem>_depth.json` produced by the final pipeline. Its pixel
resolution must match the depth crop. Do not reuse a calibration from another
log just because it has the same height.

### Run depth extraction alone

```bash
python 'info extraction/depth_distance.py' \
  'crops/depth_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --output 'crops/depth_calibration/Rio_Lavaca_1748_Unit_1_241_MUD'
```

`--output` is a file prefix: this writes `.json` calibration data and a `.png`
with accepted anchors in green and excluded labels in red.

## How the depth extraction works

The original script's idea—read depths and calculate depth per pixel—works on
these examples. Its hard-coded `real_test1.png`, old `rapidocr_onnxruntime` import,
4-digit-only filter, and immediate stop after two numbers were replaced with:

1. The project's already installed **`rapidocr`** package, loaded lazily.
2. Twelve OCR tiles spread from the top to the bottom of the crop. This avoids
   shrinking a 100,000+ pixel log into an unreadable image. A pilot selects the
   best 0/90/270-degree orientation; all OCR coordinates are mapped back to the
   original depth crop, undoing padding, rotation, and upscaling.
3. Strict numeric parsing of integers, 5+ digit numbers, decimal numbers, and
   proper thousands separators (`1,610`, `6 910`). Letters are not guessed into
   digits. Low-confidence OCR and duplicate physical detections are excluded.
4. A deterministic robust positive affine fit:

   ```
   depth = slope * depth_crop_y + intercept
   ```

   Widely separated anchor pairs seed candidate fits. Inliers are refined with
   confidence-weighted least squares. Defaults require at least three distinct
   anchors spanning 25% of the crop and agreement from at least 60% of the
   selected labels, within eight pixels. Weak/inconsistent fits raise an error
   rather than writing an apparently valid final depth table.
5. By default, integer labels are preferred when enough exist, to avoid mixing
   the main integer track with a smaller decimal/secondary track. Decimal-only
   scales work too. `--depth-label-mode integer|decimal|all|auto` makes this
   explicit. This is a formatting heuristic, **not MD/TVD semantic recognition**.

`--depth-samples` controls sampling density; `--depth-rotation` overrides the
orientation. Standalone depth extraction uses `--samples`, `--rotation`, and
`--label-mode`. The Python API also exposes tile size and residual tolerance.

No new dependencies were added. `depth_distance.py` is now safe to import without
opening a hard-coded image or initializing OCR.

## Aligning the two crops

The interval and depth images need not share an origin or resize scale:

```
document_y = column_top + column_crop_y / column_scale
depth_crop_y = (document_y - depth_top) * depth_scale
depth = calibration.depth_at(depth_crop_y)
```

Scale means **crop pixels per original-document pixel**. Default source bboxes
provide the origins; crop dimensions provide the uniform scale. Geometry was
checked successfully for all 51 supplied crop pairs.

For custom, padded, or stretched crops, supply the actual geometry:

```bash
  --column-top 1000 --depth-top 900 --column-scale 2 --depth-scale 0.5
```

If no annotation is available, these explicit origins/scales are required unless
you deliberately pass `--assume-aligned` for equal-height, vertically aligned,
unscaled crops. In that mode original-document coordinates are left blank,
not invented. Merely sharing an image height does not prove alignment.

## Final output

Already generated under **`crops/final_output/`** for Rio Lavaca:

- `<stem>.csv`: the six requested columns first:
  **`top_y,bottom_y,cluster_id,lithology,top_depth,base_depth`**.
  Existing `match_distance`, `match_margin`, `runner_up`, and `match_status` (or
  other input confidence columns) follow. Depth diagnostics add `depth_unit`,
  `depth_extrapolated`, `document_top_y`, and `document_bottom_y`.
- `<stem>.json`: intervals, available cluster summaries, full calibration and
  accepted/excluded anchors, crop alignment, and warnings.
- `<stem>.png`: original column with cluster-coloured boundaries, lithology/ID
  labels, and interval top/base depths. Text is omitted in very thin intervals
  where it would be illegible; those intervals still appear in CSV/JSON.
- `<stem>_depth.json` / `.png`: reviewable calibration and depth-anchor overlay.
- `<stem>_legend/`: RGB swatches/OCR names when automatic legend extraction ran.
- `Rio_Lavaca_1748_Unit_1_241_MUD.preview.png`: a small annotated section for
  convenient viewing (rows 15000–16600, generated during validation).

**Coordinates:** `top_y,bottom_y` remain crop-local, half-open pixel edges, exactly
as in the classifier output. `document_*_y` give full-image pixel edges when the
crop geometry is known. Adjacent interval boundaries map to identical depths.

**Units:** numeric OCR does not reliably identify units. The default is
`unspecified`; pass **`--depth-unit ft`** or **`--depth-unit m`** only when known
from the document. The values are not silently assumed to be feet or converted
between units.

**Confidence:** lithology distance/margin/status are retained, not represented as
calibrated probabilities. Calibration RMSE is reported in depth-crop pixels in
JSON. It measures agreement with recognized label centres, not absolute depth
accuracy or boundary accuracy. Text/tick alignment can introduce systematic
bias even with a small fit residual.

## Validation and limitations

The original depth/output regressions remain in the now-expanded suite (see
`pipeline.md` for the current test count), including parsing, outlier rejection, secondary
tracks, tile/orientation coordinate recovery, affine fitting, short-span and
broken-scale rejection, filename lookup, origin/scale alignment, CSV confidence
preservation, export, cached CLI joining, and the earlier lithology/boundary tests:

```bash
python -m unittest discover -s tests -v
```

Real OCR checks:

| Log | Accepted labels | Depth units / pixel | Fit RMSE (px) |
|---|---:|---:|---:|
| Rio Lavaca | 108 | 0.09999955 | 0.61 |
| d-582 log02 | 16 | 0.04989130 | 0.70 |
| 60811407570000… | 123 | 0.09986327 | 0.33 |

The complete uncached Rio Lavaca flow ran in about **35 seconds**, producing
**1,082 fully depth-populated intervals in 51 visual clusters**. Classification
IDs and labels were identical to the earlier standalone classification output;
all adjacent depths were consistent. Two intervals extend beyond accepted
anchors and are explicitly flagged as extrapolated. These checks are **not a
manually labelled depth or lithology accuracy evaluation**.

Remaining limits:

- One affine mapping assumes a uniform depth scale. Page breaks, missing image
  sections, or changing scales may fail the consistency checks. Piecewise depth
  calibration is not implemented; do not increase residual tolerance blindly.
- Extrapolation beyond the first/last anchor uses the same line and is flagged.
  Review it, especially when it extends far beyond the accepted support range.
- Multiple integer depth tracks, unusual numeric formatting, or OCR errors can
  require explicit track selection or a reviewed calibration.
- Lithology/boundary limitations remain unchanged: symbols can cause false cuts,
  legend detections can be missing, and ambiguous lithologies remain `unknown`.
