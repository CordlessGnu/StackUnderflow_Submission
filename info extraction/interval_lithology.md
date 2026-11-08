# Interval clustering and lithology identification

Implementation: **`info extraction/interval_lithology.py`**.
For the complete TIFF + bbox JSON flow, see **`pipeline.md`** / `pipeline.py`.
This implements the **“Match and identify intervals with lithology”** node in
`Flow Diagram.drawio`, downstream of the RGB boundary detector and legend swatch
extraction. Existing boundary/depth implementations are not modified.

## What it does

```
cropped interpreted lithology column -> interval boundaries -> RGB/texture descriptors
                                                              |
                                                              v
                                                   document-local visual clusters
                                                              |
legend image -> swatch_detection.pt -> RGB swatches + OCR names |
                                      |                       |
                                      +---- legend matching --+
                                                              |
                                               cluster_id + lithology per interval
```

1. Reuse an existing `top_y,bottom_y` CSV or run `detect_boundaries`.
2. Describe each interval with RGB gradient orientations, RGB means, a soft joint
   RGB histogram, ink density, and spatial differences. Long intervals are sampled
   at up to seven locations. Width is normalized to 64 pixels, **not** by squashing
   an entire interval into a square; local pattern structure is retained.
3. Group intervals using deterministic bounded-diameter clustering. An interval
   joins a group only if it is within the threshold of **every** member. This
   prevents A≈B≈C chaining from accidentally combining A and C when they differ.
   Long intervals seed groups first; IDs are assigned by first appearance.
4. Extract swatches and name boxes with the existing YOLO model. Full-legend OCR
   supplements missing text boxes; the existing right/below pairing logic is
   reused. Swatches remain **RGB**, unlike the older grayscale library loader.
   Implausible or guessed pairings are dropped with warnings.
5. Compare a height-weighted group representative against the **same document's**
   legend. Swatch descriptors are computed at five vertical scale factors
   (0.5, 0.75, 1, 1.5, 2) to accommodate renderer stretching. The best and runner-up
   must be distinct names, not just two scale variants of the same swatch.
6. Give all members of a group the same `cluster_id` and lithology assignment.
   Poor, ambiguous, blank, or unavailable matches are `unknown`.

IDs such as `C001` are **document-local visual IDs**, not lithology codes and not
cross-document IDs. A returning pattern reuses its ID even after intervening
intervals. Different visual clusters can map to the same legend name; identical
names do not force visually different groups to merge. Original boundaries are
not changed or merged by this stage.

## Run the automatic flow

From the project root:

```bash
python 'info extraction/interval_lithology.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --legend 'crops/lithology_legend/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --output crops/labelled_examples
```

If `--legend` is omitted, the script looks for the same filename under
`crops/lithology_legend/` (`--legend-dir` changes this location). Input is a native
resolution **column crop**, not the whole document; full-document region detection
and depth calibration remain separate stages in the diagram.

Reuse boundaries rather than detecting them again:

```bash
python 'info extraction/interval_lithology.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --intervals 'crops/boundary_examples_rgb/Rio_Lavaca_1748_Unit_1_241_MUD.csv' \
  --output crops/labelled_examples
```

## Outputs

`<output>/<document_stem>` has:

- **`.csv`**: one row per input interval:
  `top_y,bottom_y,cluster_id,lithology,top_depth,base_depth`, plus match distance,
  margin, runner-up name, and status.
- **`.json`**: intervals, group membership, representative indices, top three
  legend candidates per cluster, and extraction warnings.
- **`.png`**: original column with cluster-coloured boundaries and an adjacent
  label rail. Repeated IDs have the same colour.
- **`.clusters.png`**: compact review sheet showing a representative of every
  visual cluster beside its assigned legend swatch. Unknown groups show their
  nearest candidate explicitly marked **candidate only**, not a named assignment.
- **`<document_stem>_legend/legend.json`** and RGB swatch PNGs: extracted legend
  entries, raw OCR names, source boxes and warnings, for review and correction.

Intervals use crop-local, half-open coordinates `[top_y, bottom_y)`. Depth fields
are deliberately **blank/None**: this stage does not fabricate depth values.
Add the crop's document-y offset before combining with document-level depth
calibration (also undo any resizing).

`match_status` values:

| Status | Meaning |
|---|---|
| `matched` | Best candidate passes absolute distance and distinct-name margin tests |
| `ambiguous` | Top two distinct lithology names are too similar to choose safely |
| `poor_match` | Even the closest legend entry is too far away |
| `blank` | Near-white region without appreciable texture; not assigned a rock |
| `no_legend` | API clustering without a supplied legend |

Distances and margins are **not calibrated confidence probabilities**. The CLI
requires a usable legend; the Python API also permits clustering without one.

## Review or supply your own swatches

Inspect the extracted names and RGB crops. OCR spelling is intentionally preserved
rather than silently replaced with a guessed rock name. You can edit a manifest:

```json
{
  "swatches": [
    {"name": "Sandstone", "image": "sandstone.png"},
    {"name": "Shale", "image": "shale.png"}
  ]
}
```

Image paths are relative to the manifest. Crop swatches **inside** their border,
without the label or surrounding page. Multiple examples with the same name are
allowed. Load a reviewed manifest to bypass both YOLO and OCR:

```bash
python 'info extraction/interval_lithology.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --legend-manifest 'crops/labelled_examples/Rio_Lavaca_1748_Unit_1_241_MUD_legend/legend.json' \
  --output crops/labelled_examples
```

Only NumPy and Pillow are needed with a manifest. Automatic extraction additionally
uses the project's existing ultralytics/torch and RapidOCR/ONNX dependencies.

## Python integration

```python
import sys
from pathlib import Path
sys.path.insert(0, 'info extraction')
from texture_boundaries import detect_boundaries
from interval_lithology import extract_legend, label_intervals

column = Path('crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png')
legend = Path('crops/lithology_legend') / column.name
boundaries = detect_boundaries(column)
swatches, warnings = extract_legend(legend)
result = label_intervals(column, boundaries, swatches)
for interval in result.intervals:
    print(interval.top_y, interval.bottom_y,
          interval.cluster_id, interval.lithology, interval.match_status)
```

`label_intervals` accepts paths, PIL images, or uint8 arrays. Its interval argument
can be a `BoundaryResult` or a sorted list of non-overlapping integer `(top, bottom)`
pairs. It leaves gaps unchanged if only a subset of the column is supplied.

## Tuning and validation

- `--cluster-threshold` (default **0.20**): higher combines more visually different
  intervals; lower creates tighter, more numerous clusters.
- `--max-match-distance` (default **0.32**): lower rejects more distant candidates.
- `--min-match-margin` (default **0.06**): higher rejects more ambiguous names.
  Both default matching thresholds are deliberately conservative.
- `--margin` (default **0.12** per side): adjust for column borders/gutters.
- Boundary controls: `--window`, `--boundary-threshold`, `--min-interval`.
- Model controls: `--model`, `--swatch-conf`.

Run the offline tests:

```bash
python -m unittest discover -s tests -v
```

The **30 matching/boundary tests** pass (16 matching/clustering tests and 14 boundary
regressions); the complete suite also includes depth/final-output tests. Tests
cover returning IDs, colour distinctions, black-and-white
orientations, long beds, duplicate-name variants, ambiguous/poor/blank matches,
non-chaining clusters, deterministic IDs, manifests, export, and direct integration
with the boundary detector.

A full automatic run on **Rio Lavaca** completed in about 21 seconds and produced
**1,082 intervals -> 51 visual clusters**, with **621 named and 461 unknown
intervals** at the conservative defaults. Its legend supplied 22 usable named
swatches. Outputs are in `crops/labelled_examples/`. These are **execution and
assignment counts, not accuracy measurements**; no manually labelled evaluation
set has been scored.

Known limitations:

- Upstream YOLO extraction is currently unreliable on some legend styles. In the
  development checks the d-582 legend yielded no usable named swatches, and the
  first `60811407570000...` legend only four. The CLI reports failure/warnings;
  use a corrected manifest or improve the swatch detector. Full-legend OCR helps
  with missing labels but cannot recover all missing pattern detections.
- Short, low-resolution swatches can make related patterns hard to distinguish
  (for example brick-like limestone/claystone). The scale search is a heuristic,
  not a guarantee of renderer invariance.
- False/missed interval boundaries, overprinted symbols, mixed beds, and scan
  colour changes can split one lithology into several visual clusters or produce
  an incorrect/unknown name. Clustering cannot repair a mixed input interval.
- Matching is constrained to detected/supplied names; a genuinely absent swatch
  can sometimes resemble a present one. `matched` is not semantic proof.
- Thresholds need tuning against manually checked development intervals and
  legends, then evaluation on held-out logs. Depth conversion is a separate stage; see `final_output.md` for the combined
  CSV/annotated image with calibrated `top_depth` and `base_depth`.
