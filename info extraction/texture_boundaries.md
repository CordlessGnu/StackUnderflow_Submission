# Texture-based interval boundary detection

Implementation: `info extraction/texture_boundaries.py`.
Dependencies: NumPy and Pillow (already in `requirements.txt`). No trained model,
legend extraction, OCR, SciPy, or additional dependency is required.

This is a separate experimental baseline. The existing `boundary_detection.py`
and its local edits have not been changed.

## Why not just convolve the image with an edge detector?

A horizontal-edge filter detects every repeated dash and every gridline, not just
lithology changes. Instead, use **texture change-point detection**:

1. Trim the sides of the crop to reduce border/gutter interference.
2. Extract dark ink, mostly independently of a bright coloured background.
3. Compute per-row gradient orientation histograms, ink density, and horizontal
   and vertical pixel differences at several offsets. Pool across column width.
4. At every possible boundary, average these features in windows immediately
   above and below it. Cumulative sums keep this efficient even on very tall logs.
5. Compare the pooled statistics. Orientation histograms use Hellinger distance;
   other features capture density, spacing, blank regions, and solid fills.
6. Combine scores from two window sizes, smooth the result, and select separated
   peaks above a threshold.

Repeated marks in one lithology should have similar *pooled statistics*, even if
individual pixels do not align. Horizontal and vertical dashes remain distinct;
the descriptors intentionally are not rotation invariant. Full vertical image
resolution is preserved. There is no CNN training step.

The window operations can also be expressed as convolutions over feature maps.
The important distinction is **filtering/comparing texture statistics**, not
looking for pixel edges alone.

## Run on a supplied crop

From the repository root:

```bash
python 'info extraction/texture_boundaries.py' \
  'crops/interpreted_lithology_column/0000000009_d-582_0000020435-0001_log02_fe_5in_md.png' \
  --output crops/boundary_examples
```

Produces three files with the source stem:

- `.csv`: `top_y,bottom_y`, one row per interval.
- `.png`: original crop with internal boundaries drawn in red.
- `.scores.npy`: the diagnostic change-score curve, length `image_height + 1`.

Coordinates are **crop-local** and intervals are **half-open**:
`[top_y, bottom_y)`. Intervals cover the entire input crop without gaps or overlaps,
including blank portions. Image top and bottom are interval edges, not detected
internal changes. Scores are not probabilities or calibrated confidence values.

## Python API

When importing from elsewhere, put `info extraction/` on `sys.path` or load the
module using your application's module loader:

```python
import sys
sys.path.insert(0, 'info extraction')
from texture_boundaries import detect_boundaries

result = detect_boundaries(
    'crops/interpreted_lithology_column/0000000009_d-582_0000020435-0001_log02_fe_5in_md.png',
    window=96,
    threshold=0.16,
)
print(result.boundaries)  # internal y coordinates only
print(result.intervals)   # [(0, first_cut), ..., (last_cut, image_height)]
```

Inputs may also be PIL images or uint8 grayscale/RGB/RGBA arrays.

## Tuning

- **`window`**: rows on each side of a candidate; the second scale uses twice
  this size. Use at least 2–3 vertical pattern repetitions. Increase it when
  repeated strokes produce false cuts; decrease it for thinner beds. The default
  is `max(24, min(96, round(crop_width * 0.6)))`, a heuristic, **not** automatic
  measurement of repeat pitch.
- **`threshold`**: default `0.16`. Raise it for fewer, stronger cuts; lower it
  for subtle changes. It is an empirical score threshold, not a probability.
- **`min_interval`**: default `window // 2`. Effective minimum cut spacing and
  distance from image edges is `max(min_interval, window)`. This intentionally
  avoids duplicate peaks, but limits thin-bed detection.
- **`margin`**: default `0.12` of the crop width removed from **each** side.
  Adjust this for crops containing borders or neighbouring tracks.
- **`use_color` / `--use-color`**: off by default, so a bright fill-colour change
  alone does not create a boundary. Enable if colour differences are meaningful.
- **`ignore_rules`**: on by default; removes short nearly full-width dark rules.
  Use `--keep-rules` if these might instead be genuine thin solid beds.

## Validation and limits

Nine synthetic regression tests cover repeating textures, orientation changes
and returns, colour-only changes, blank-to-texture transitions, solid beds,
horizontal rules, input formats, short images, and invalid parameters:

```bash
python -m unittest discover -s tests -v
```

A smoke test completed on all **51** supplied column crops without errors and
verified finite scores and complete interval coverage (about 30 seconds total
in the development environment). This tests execution, **not boundary accuracy**.
The first 2,200 rows of the d-582 `log02_fe_5in_md` crop were visually inspected;
the predicted lines track the obvious dash/dot pattern transitions there. No
manual ground-truth interval set or held-out precision/recall has been measured.

Known limitations:

- Overprinted symbols, text, page breaks, and scan defects can produce false
  boundaries. Some Rio Lavaca examples have symbols over otherwise repeated
  texture that trigger extra cuts.
- Similar textures with identical pooled features can be merged. Large windows
  can miss thin beds or move the peak away from the exact contact.
- Dark backgrounds and coloured strokes violate the default dark-ink/bright-fill
  assumption. Colour invariance is approximate, not universal.
- The fixed side trim does not deskew the column or follow drifting borders.
- Sparse textures leave ambiguity between the last mark of one pattern and the
  first mark of the next. Reported y positions should not be assumed pixel-exact.
- This detects changes only: no clustering, lithology naming, depth conversion,
  header removal, or full-document region detection is implemented here.

## Integrating with the eventual document-image pipeline

Keep region location and interval segmentation separate:

```
full document image
  -> locate interpreted-lithology bounding box
  -> crop at native resolution
  -> detect_boundaries(crop)
  -> cluster interval textures / match the document's legend
  -> calibrate y against depth labels
  -> export final CSV and annotated image
```

If the crop starts at document row `crop_top`, add `crop_top` to every local
boundary to obtain document-image y coordinates. If the crop was resized, also
undo that scale. The function already accepts a PIL crop, so a later detector can
call it without saving an intermediate image.

For a scored evaluation, manually label several representative logs, match
predictions to reference boundaries within a declared pixel tolerance, and report
precision/recall and localization error. Tune on development logs and keep other
logs held out. Legend-informed classification or a trained segmentation model
can be added later if these failure cases dominate.
