# GeoLens — Geological Data Intelligence & Comparison Platform

GeoLens turns automated lithology extraction outputs (structured CSVs) into interactive geological insights. 

It lets geoscientists analyse a single document, compare an entire collection, and export traceable data and reports without manually opening each CSV.

> GeoLens can consume existing CSVs or run this repository's extraction pipeline locally from TIFF + bbox JSON inputs, or from a TIFF alone (the U-Net predicts the boxes). It does not retrain models or claim that predictions are geologically correct.

---

## Quick start

From the **repository root**, install the combined pipeline/UI dependencies and launch:

```bash
python -m pip install -r requirements.txt
python run_geolens.py
```

Then open <http://127.0.0.1:8501>. The launcher loads this directory's dark theme,
loopback-only server configuration, and upload settings. To use another port:
`python run_geolens.py --server.port 8502`.

Alternatively, from this directory run `python -m streamlit run app.py` after
installing the root dependencies. Installing only this directory's
`requirements.txt` supports CSV analytics but may lack the ML packages required
for extraction. The worker uses the **same Python interpreter** as Streamlit.

Run the GeoLens test suite from this directory:

```bash
cd Viridien-GeoLens  # when starting at the repository root
python -m pytest
```

**Integration validation:** 126 GeoLens tests pass, including real subprocess
lifecycle/cancellation tests and Streamlit CSV-handoff tests. An uncached GB506
TIFF + JSON run was also exercised through the actual UI: all six stages
completed, 201 intervals loaded as `Ready`, and four existing analytics charts
rendered. The selected output directory was honoured; the source's two blank
intervals and 62 runner-up flags were preserved. This validates integration, not
geological accuracy.

The separate pipeline suite currently retains one pre-existing rounding mismatch
in `tests/test_depth_output.py`: its JSON assertion expects 1004, while the current
rounding setting exports 1000. This predates the UI changes; pipeline rounding
logic was not altered. The UI exposes rounding explicitly, including 0 for full
precision.

### Local extraction: TIFF + JSON

In **Single Analysis**, select **Input source → TIFF + JSON**:

1. Choose **Local file paths** (recommended for large TIFFs) and paste/select the
   `.tif`/`.tiff` and its bounding-box `.json`. Relative paths resolve from the
   repository root. Alternatively choose **Upload files** and upload both files.
2. Select an **Output directory** using **Browse…**, or enter a local path. Each
   run creates a new `<document>_<job-id>` subfolder there; previous outputs are
   not overwritten. You may enter a new directory and it will be created.
3. Optionally adjust the matcher, minimum interval height, texture context,
   units and depth rounding. Rounding defaults to the pipeline's **10 units**;
   set it to **0** to preserve full precision (recommended for thin beds).
   Reviewed legend/depth calibration paths are optional, not required inputs.
4. Click **Run extraction**. A separate process performs the six pipeline stages
   while the UI remains usable. Progress, warnings and worker logs are shown;
   **Cancel extraction** stops the process and retains partial outputs for review.
5. On success the final `intervals.csv` automatically enters the **existing**
   profile, filters, statistics, quality checks and interval tables. Use **Add
   this extraction to collection** for Batch Analysis and Comparison Studio.

### Local extraction: TIFF only (U-Net boxes)

Select **Input source → TIFF only (U-Net boxes)** when there is no bounding-box
JSON. Choose the TIFF by path or upload, an output directory and the same
settings as above, then **Run extraction**. The worker
(`info extraction/unet_pipeline.py`) first runs the bonus U-Net
(`work/export/boxfinder_v0.2.pt`, see `work/README.md`) to predict the depth
column, interpreted lithology column and lithology legend, snaps their edges to
the printed lines, writes them as a normal bbox JSON, and then runs the unchanged
six-stage `pipeline.py` on those boxes. The run folder additionally contains
`predicted_boxes.png`, and `run.json` gains a `box_source` entry (model, boxes,
timing); `annotation.json` holds the predicted boxes. The **Predicted bounding
boxes (U-Net)** panel shows both, so the boxes can be checked before trusting the
intervals: on unfamiliar layouts the model can pick the wrong column (see the
test results in `work/README.md`). This mode needs the U-Net's extra packages
(`tifffile`, `imagecodecs`, OpenCV), which are in the root `requirements.txt`.

Only one extraction runs per GeoLens server to limit TIFF/model memory usage.
Navigating to another page does not cancel it; return to Single Analysis to see
results. If another browser session started the active job, **View active
extraction** attaches its progress. A normal server shutdown stops active workers.

**Local picker:** Browse opens a native dialog on the machine running GeoLens,
not a remote browser filesystem. This is intended for the local demo. If Tk or a
graphical display is unavailable, enter paths manually; the app remains usable.
There is no separate API server to start. The app binds to localhost by default.

The selected run folder contains native crops, `intervals.csv`, `intervals.json`,
`intervals.png`, legend review files, depth calibration and the pipeline's
`run.json`. CSV/JSON/annotated-column download buttons are also provided. The
Export Centre continues to offer its existing browser downloads for analytics.
Temporary uploaded inputs and worker logs are retained under
`crops/geolens_jobs/`; remove old job folders only when no job is running. Failed
or cancelled runs are never automatically loaded as successful analytics.

**Blank and fallback policy:** original pipeline CSV values are preserved in
`source_df` and original exports. Empty lithology cells with `match_status=blank`
are displayed as the unassigned **Blank** category for charts, not as a missing
rock prediction. `lithology_fallback` is parsed into a derived boolean flag;
runner-up assignments are explicitly described as **not verified matches** in
the interval detail and data quality views. Extra pipeline diagnostic columns
remain available in tables and processed exports.

### Appearance

GeoLens is **dark-mode only** by design. The palette lives in
`geolens/components/theme.py`, and `.streamlit/config.toml` pins Streamlit's own
widgets to the same dark palette so no text renders black on a dark background.
There is no light mode and no user-facing theme switch.

Lithology and cluster colours come from a bright, high-contrast palette in
`geolens/utils/colors.py` (luminance ≈ 0.4–0.85) so adjacent intervals separate
clearly on the dark canvas. Intervals are outlined in **white**; the selected
interval gets a thicker amber outline.

### Uploading a folder

Streamlit's uploader cannot open a native folder picker, so Batch Analysis
accepts a **ZIP of the folder** instead:

- Windows: right-click the folder → *Send to* → *Compressed (zipped) folder*
- macOS: right-click the folder → *Compress*

Sub-folders are searched recursively, and `__MACOSX`, `.DS_Store`, hidden files
and path-traversal entries are ignored. Only `.csv` files are extracted.

---

## What it does

### 1. Single Analysis
- Upload one extraction CSV, or run the TIFF + JSON pipeline and load its CSV automatically.
- Overview metrics: intervals, unique lithologies, unique clusters, depth
  coverage, average thickness, thickest interval.
- **Interactive geological profile** (Plotly): layer heights are proportional to
  `base_depth − top_depth`, with hover details, click-to-select, lithology and
  cluster filters, depth-range zoom, and a lithology/cluster colour toggle.
- Interval detail panel showing all six source fields plus calculated thickness.
- Lithology statistics (counts, thickness, percentage, clusters per lithology).
- Cluster explorer with depth ranges and associated lithologies.
- Data quality findings.

### 2. Batch Analysis
- **Upload a whole folder** as a `.zip` (sub-folders are searched automatically),
  or select many CSVs at once.
- Consolidated dataset with a `document_id` / `document_name` column added.
- Organisation-wide statistics and per-document summaries.
- **Cross-document lithology comparison matrix** (occurrences, thickness, or
  percentage of document thickness) — the core business-value feature.
- Presence matrix, distribution charts, and lithology → document lookup.
- Exploratory lithology sequence comparison.
- Collection-wide data quality findings.

### 3. Comparison Studio
- Select two or more documents and compare side-by-side profiles, distributions,
  interval counts, depth coverage, presence and sequence similarity.
- Highlight a single lithology across the selected documents.
- Flags potentially incomparable depth references instead of implying alignment.

### 4. Export Centre
- Single document: original CSV, processed CSV, lithology stats, cluster stats,
  Excel workbook.
- Batch: consolidated CSV, document summaries, comparison tables, multi-sheet
  Excel workbook, data quality CSV.
- Plain-text summary report with an explicit non-verification disclaimer.
- Exports respect the filters currently applied and record what they were.

---

## Design principles

**Dynamic lithology and cluster handling.** No lithology name or cluster ID is
hardcoded anywhere. `geolens/utils/colors.py` hashes each label onto a curated
palette and generates additional hues for unseen labels, so a brand-new
lithology gets a stable, distinct colour with zero code changes.

**Source data is never overwritten.** `DocumentRecord` keeps `source_df` (the
untouched CSV) separate from `processed_df` (source columns plus calculated
`thickness`, `document_id`, `document_name`, `is_valid`, `issues`). Exports can
always be traced back to the original values.

**Comparison stays honest.** The comparison matrix is lithology-based. Cluster
IDs are document-scoped and are never compared across documents. Sequence
similarity is labelled exploratory, not geological equivalence.

---

## Project structure

```
app.py                          # entry point + navigation
.streamlit/config.toml          # dark-only theme for native widgets
geolens/
  pages/                        # Streamlit page renderers (UI only)
    single_analysis.py
    batch_analysis.py
    comparison.py
    export_centre.py
  components/                   # reusable UI pieces
    theme.py                    # dark palette, CSS + Plotly layout defaults
    sidebar.py                  # branding, navigation, document status
    metrics.py                  # metric cards
    lithology_profile.py        # interactive geological column
    interval_details.py         # detail panel + cluster explorer
    filters.py                  # filter controls
    charts.py                   # Plotly chart builders
    uploader.py                 # CSV / TIFF+JSON input modes and session state
    extraction.py               # output picker, background progress and CSV handoff
  services/                     # all business logic
    csv_loader.py               # parsing, schema validation, column aliases
    extraction_service.py       # isolated local worker jobs and input validation
    local_picker.py             # optional native file/folder dialogs
    archive_loader.py           # folder (ZIP) upload extraction
    data_processor.py           # calculated fields, consolidation, filtering
    statistics.py               # single-document + collection statistics
    comparison.py               # comparison matrices, sequence similarity
    validation.py               # structural consistency checks
    export_service.py           # CSV / Excel / report generation
  models/
    schemas.py                  # column names, required fields, findings
    data_models.py              # DocumentRecord, DocumentCollection
  utils/
    colors.py                   # dynamic colour generation
    formatting.py               # number/label formatting
    helpers.py                  # ids and file helpers
  tests/                        # analytics, subprocess and Streamlit integration tests
  sample_data/                  # illustrative synthetic CSVs
```

---

## Data schema

Required CSV columns:

| Column | Meaning |
| --- | --- |
| `top_y` | Top boundary in source-image pixels |
| `bottom_y` | Bottom boundary in source-image pixels |
| `cluster_id` | Visual pattern cluster within the column |
| `lithology` | Lithology label matched against the legend |
| `top_depth` | Top depth of the interval |
| `base_depth` | Base depth of the interval |

### Alternative column names

Real extraction outputs are not always named identically. The company's own
export uses **`bottom_depth`** instead of `base_depth`, so GeoLens maps known
alternative spellings onto the canonical schema automatically. Matching is
case-insensitive and ignores spaces/underscores/hyphens.

| Canonical | Also accepted |
| --- | --- |
| `base_depth` | `bottom_depth`, `base`, `bottom`, `to_depth`, `end_depth`, `depth_base`, `basedepth`, `bottomdepth`, `base_depth_m` |
| `top_depth` | `top`, `from_depth`, `start_depth`, `depth_top`, `topdepth`, `top_depth_m` |
| `lithology` | `lith`, `lithology_name`, `lith_type`, `lithtype`, `label` |
| `cluster_id` | `cluster`, `cluster_label`, `clusterid`, `cluster_no` |
| `top_y` | `top_pixel`, `y_top`, `topy`, `top_px` |
| `bottom_y` | `bottom_pixel`, `y_bottom`, `bottomy`, `bottom_px` |

Only column *names* are mapped — values are never modified. A canonical name
always wins, so a file containing both `base_depth` and `bottom_depth` keeps
`base_depth`. The mapping is shown in the UI and recorded in the schema check,
and the **Original CSV** export restores the uploaded file's own headers.

Calculated columns added by GeoLens: `thickness`, `document_id`,
`document_name`, `source_row_index`, `is_valid`, `issues`,
`lithology_missing`, `cluster_missing`.

---

## Notes and limitations

- **Depth units are not assumed.** GeoLens never labels depths as metres. If
  document depth ranges differ by orders of magnitude, it warns that units may be
  inconsistent rather than comparing absolute depths.
- **Pixel coordinates are not depth.** `top_y` / `bottom_y` are retained as
  source provenance only. They are never treated as depth and never converted
  into a spatial position.
- **Data quality ≠ accuracy.** Findings describe structural consistency only.
  Depth gaps and clusters mapping to multiple lithologies are reported as
  observations, not errors.
- **Sample data is synthetic.** Files in `geolens/sample_data/` are illustrative
  and must not be used for geological interpretation.

---

## Testing

```bash
python -m pytest
```

Covers schema validation, CSV loading, missing-value handling, dynamic lithology
and cluster discovery, thickness and depth-coverage calculation, lithology and
cluster statistics, multi-document consolidation, comparison matrices, sequence
comparison, data consistency checks, colour generation and export generation.

Tests explicitly verify that:
- a previously unseen lithology requires no source change;
- adding a document does not corrupt existing documents;
- absent lithologies appear as zero in the comparison matrix;
- cluster IDs are never compared across documents.
