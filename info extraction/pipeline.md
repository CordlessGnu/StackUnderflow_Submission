# Complete TIFF + JSON pipeline

**Entry point: `info extraction/pipeline.py`**

Exactly two input files are required:

1. A full-resolution `.tif` or `.tiff` document.
2. Its JSON annotation containing the three region bounding boxes.

No previously generated crops, interval CSVs, or depth calibrations are needed.
The supplied swatch model is loaded from `models/swatch_detection.pt`.

## Run

From the repository root:

```bash
python 'info extraction/pipeline.py' \
  'Dataset/dataset/images/Rio_Lavaca_1748_Unit_1_241_MUD.tif' \
  'Dataset/dataset/bbox_annotation/Rio_Lavaca_1748_Unit_1_241_MUD.json'
```

The default output directory is:

```
crops/pipeline_output/<TIFF stem>/
```

Use `--output /path/to/run-directory` to choose an exact output directory. It may
be outside the repository; the input files may also be anywhere on disk. The
TIFF passed on the command line is authoritative; the script does not search the
dataset for another image based on the JSON filename.

### Annotation format

The existing dataset JSON files are supported directly:

```json
{
  "image_name": "example.tiff",
  "width": 1700,
  "height": 195800,
  "bounding_boxes": {
    "interpreted_lithology_column": [661, 6861, 731, 194865],
    "depth_column": [280, 6861, 321, 194865],
    "lithology_legend": [34, 3299, 1667, 3644]
  }
}
```

Boxes are `[left, top, right, bottom]` in original TIFF pixels, with half-open
right/bottom edges. Finite fractional values are rounded to pixel edges. Empty,
reversed, missing, and wholly out-of-image boxes are rejected. Partially out-of-
bounds boxes are clipped with a warning; the actual crop origin is recorded.
Optional width/height metadata must match the selected TIFF page.

### Six stages

1. **Crop the TIFF** at native resolution into the three annotated regions.
2. **Detect interval boundaries** with multiscale texture proposals, independent
   short-context colour contacts, local refinement, and split/merge verification.
3. **Extract legend swatches and OCR names** with the supplied YOLO model.
4. **Cluster and name intervals**, reusing document-local IDs for returning patterns.
5. **OCR and calibrate depth**, checking multiple labels across the depth crop.
6. **Export the final table and annotated column**, accounting for differing crop origins.

The TIFF is decoded once and closed **before** loading the ML/OCR models. The
provided large image explicitly bypasses Pillow's ordinary decompression-bomb
limit only within the crop operation; the previous global limit is restored on
both success and failure. A separate `--max-pixels` limit defaults to one billion
pixels. Pillow still decodes the full raster, so large documents require memory
(e.g. roughly 1.3 GB for the Rio Lavaca RGB raster, plus temporary buffers).
Multi-page TIFFs process page 0 by default; `--page N` selects a zero-based page.
Non-upright TIFF orientation tags are rejected rather than silently misaligning
coordinates and annotations.

## Outputs

```
<run-directory>/
  intervals.csv                 # FINAL table; six required columns first
  intervals.json                # intervals, cluster membership, calibration, warnings
  intervals.png                 # annotated original lithology COLUMN, with depths
  run.json                      # provenance, parameters, stage timings, status, counts
  annotation.json               # unchanged copy of the supplied annotation
  boundary_scores.npy           # combined diagnostic boundary score curve
  boundary_color_scores.npy     # independent fill-colour change evidence
  depth_calibration.json        # accepted/rejected OCR depth anchors and fitted mapping
  depth_calibration.png         # depth column with accepted/excluded label overlays
  crops/
    interpreted_lithology_column.png
    depth_column.png
    lithology_legend.png
  legend/                       # when usable named swatches exist
    legend.json                 # review/correct OCR labels here
    swatch_000.png
    ...
  classification/
    intervals.csv               # crop-local pre-depth assignments and diagnostics
    intervals.json              # visual clusters and legend candidate rankings
    intervals.png               # intermediate classification overlay
    intervals.clusters.png      # compact cluster-versus-swatch review sheet
```

### Coordinate conventions

Because the input is the full document, this entry point defaults to
**original-document `top_y` and `bottom_y`** in its final CSV/JSON. The extra
`crop_top_y,crop_bottom_y` columns retain the original classifier's local edges.
`document_top_y,document_bottom_y` are also retained. All pixel intervals are
half-open `[top_y,bottom_y)`.

For example, Rio Lavaca's final intervals run from document row **6861** to
**194865**, corresponding to crop-local rows **0** to **188004**.

Use **`--coordinate-system crop`** if the downstream consumer expects the earlier
crop-local convention. The intermediate `classification/` files always use crop-
local coordinates. The annotated image shows the lithology column, not the whole
large TIFF; labels are drawn at crop-local positions regardless of CSV convention.
Depth conversion is identical in either export convention.

### Depth and confidence

Final CSV columns start with:

```
top_y,bottom_y,cluster_id,lithology,top_depth,base_depth
```

Extra columns preserve matching distance, distinct-name margin, runner-up, and
status, plus depth units, extrapolation flags, and alternate pixel coordinates.
Distances and margins are diagnostic scores, **not calibrated probabilities**.
A type can be `unknown` while its interval still has valid depth values.

Units default to `unspecified`; use `--depth-unit ft` or `--depth-unit m` only when
known from the document. No implicit feet/metres conversion is performed.

## Failure handling, review, and reruns

- Every run records `run.json`: `completed`, `completed_with_warnings`, or
  `failed`. Failed runs retain crops/intermediates for inspection and include the
  failing stage and error. A failed depth fit does not produce fake final depths.
- If legend detection yields no usable named swatches, the default is to keep
  visual cluster IDs and depths but output lithology `unknown`, with explicit
  warnings. **`--strict-legend`** makes this a hard failure instead. Missing model
  files/dependencies and unexpected model/OCR errors are not silently swallowed.
- Provide **`--legend-manifest path/to/legend.json`** for reviewed RGB swatches,
  bypassing automatic legend detection/OCR. Provide **`--calibration file.json`**
  to reuse an inspected depth calibration at the same native crop resolution.
  Neither extra file is required for a normal two-input run.
- Nonempty output directories are refused by default. **`--overwrite`** archives
  the previous directory as `<name>.previous-1` (then `-2`, etc.) before starting
  a clean run. Existing results are not deleted or left mixed with a failed rerun.
  Output directories containing input/model/cache files are rejected.

Example rerun:

```bash
python 'info extraction/pipeline.py' document.tiff annotation.json \
  --output crops/my_run --overwrite --depth-unit ft
```

Detection/matching options remain available: `--window`, `--min-interval`,
`--boundary-threshold`, `--color-threshold`, `--refine-radius`, `--margin`,
`--cluster-threshold`, `--max-match-distance`,
`--min-match-margin`, `--depth-samples`, `--depth-rotation`, `--depth-label-mode`,
and `--model`. See `--help` and the individual stage guides for details.

## Python API

```python
import sys
sys.path.insert(0, 'info extraction')
from pipeline import PipelineConfig, run_pipeline

result = run_pipeline(
    'document.tiff',
    'annotation.json',
    'crops/my_run',
    config=PipelineConfig(depth_unit='unspecified'),
    progress=print,
)
print(result.csv_path, result.interval_count, result.cluster_count)
```

## Validation

```bash
python -B -m unittest discover -s tests -v
```

All **90 tests** pass: the original 71 pipeline/stage regressions plus 19 thin-bed
and boundary-refinement tests (two require optional local development crops).
Integration tests exercise actual TIFF cropping, boundary detection,
clustering, crop-to-document coordinate conversion, depth joining, export,
non-destructive reruns and failure handling; expensive OCR/model calls are mocked
or bypassed using reviewed caches in those offline tests.

Initial two-input integration baselines were completed **after archiving the legacy
scripts**, before the subsequent thin-bed boundary refinement:

- **Rio Lavaca:** original 1700×195800 TIFF + JSON -> **1,082 intervals, 51 visual
  clusters, 621 named and 461 unknown intervals**, 108 depth anchors, fit RMSE
  0.61 px, about 39 seconds end-to-end. All three new crops are pixel-identical
  to the supplied example crops. Cluster IDs, names and depths match the prior
  cropped-input pipeline exactly. Two intervals require flagged extrapolation.
- **d-582 log02:** original black-and-white TIFF + JSON -> **22 intervals, four
  visual clusters, 16 depth anchors**. The current detector cannot extract usable
  names from this legend, so all 22 lithologies are explicitly `unknown` while
  clustering and depth output still complete.

Current refined GB506 and Hadrian complete runs are saved under
`crops/boundary_refinement/gb506_pipeline/` and `hadrian_pipeline/`. See
`texture_boundaries.md` for the focused colour-contact comparisons and unchanged
log05 regression control. The minimum interval spacing now defaults to 2 pixels,
independently of texture context; effective scales/settings are recorded in `run.json`.

These are smoke/integration checks and assignment counts, **not measured boundary,
lithology, or absolute-depth accuracy**. Existing limitations (false cuts from
symbols, missed legend swatches, ambiguous patterns, affine depth-scale assumptions)
remain; this change stitches the stages together rather than retraining them.

## Clean active modules and archive

Only active modules and their stage guides remain in `info extraction/`:

| Module | Purpose |
|---|---|
| `pipeline.py` | Two-input orchestration, native cropping, provenance and failure handling |
| `texture_boundaries.py` | RGB/texture boundary detection |
| `interval_lithology.py` | Visual grouping, legend swatch extraction and naming |
| `legend_utils.py` | Shared OCR, box pairing and name filtering |
| `depth_distance.py` | Depth OCR and robust affine calibration |
| `final_output.py` | Crop alignment, depth joining and final export |

Superseded `boundary_detection.py`, `crop_dataset_classes.py`,
`crop_legend_patterns.py`, and stale extraction caches were moved to
**`wastebin/info extraction/`**, unchanged. They are not runtime dependencies.
See `wastebin/README.md` for the inventory and original checksums.
