# GeoLens — Geological Data Intelligence & Comparison Platform

GeoLens turns automated lithology extraction outputs (structured CSVs) into interactive geological insights. 

It lets geoscientists analyse a single document, compare an entire collection, and export traceable data and reports without manually opening each CSV.

> GeoLens consumes existing extraction outputs. It does **not** run or retrain an AI model, and it does not claim that extracted predictions are geologically correct.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

Run the test suite:

```bash
python -m pytest
```

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
- Upload one extraction CSV.
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
    uploader.py                 # upload widgets + session state
  services/                     # all business logic
    csv_loader.py               # parsing, schema validation, column aliases
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
  tests/                        # pytest suite (102 tests)
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
