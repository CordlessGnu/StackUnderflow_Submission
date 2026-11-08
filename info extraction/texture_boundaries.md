# Thin-bed-aware RGB and texture boundaries

Implementation: **`info extraction/texture_boundaries.py`**. The complete TIFF +
bbox JSON entry point, **`pipeline.py`**, uses this detector automatically.
Dependencies remain NumPy and Pillow; no training, OCR, or legend is needed here.

## Why the detector changed

The original detector combined two texture windows and suppressed cuts closer
than the base window. That often found only one side of a thin yellow bed and
could place a broad score peak 5–15 pixels from the contact.

The updated detector separates **texture context**, **colour-contact evidence**,
**minimum bed thickness**, and **contact localization**. No filename-specific
parameters or lithology palettes are used in the implementation.

### 1. Conservative multiscale texture proposals

The default base context is still width-derived:

```
max(24, min(96, round(crop_width * 0.6)))
```

Compare row descriptors above/below each edge at half/base/double/quadruple this
context. Adjacent scales support texture changes. Prefer the established base
scale where it works; supplementary proposals require stronger three-scale
support. Overlapping texture hypotheses for one broad contact are collapsed.

This is a scale bank, **not automatic measurement of texture repeat pitch**.
Texture-only thin beds can still need more context than their thickness permits.

### 2. Independent short-context colour proposals

A per-row dominant-fill profile reduces the influence of black/dark strokes.
It selects bright pixels by luminance, requires spatial colour agreement, and
interpolates short drawing-line gaps. Long unsupported dark regions cannot
propose colour contacts. Small, mixed-colour overprints are often rejected by
spatial agreement, but this does not eliminate every symbol/text artefact.

Compare this profile with short windows of roughly 2, 4 and 8 pixels by default.
Unlike a wide texture window, these can see **both** sides of a 2–8 pixel colour
bed. Fine-scale contacts take priority over wide-window duplicates.

Safeguards reject repeated signed colour edges in a continuing pattern and
require extra wide-context evidence for mostly shading/brightness-only changes.
Achromatic drawing-line variation does not activate the colour rescue. Otherwise,
a brick pattern's repeated horizontal strokes could become hundreds of false
intervals. Short dark dips are treated as drawing strokes, so very thin dark
beds remain a limitation.

### 3. Local contact refinement

Within a bounded neighbourhood, choose the split that minimizes within-region
RGB fill error on its two sides. Neighbouring proposed contacts constrain the
search, preventing refinement from crossing a thin bed's other edge.

Where no reliable colour contrast exists, retain the texture estimate instead
of snapping to an arbitrary dash or dot. This deliberately preserves the working
black-and-white control.

### 4. Split/merge verification

Pool descriptors inside adjacent proposed intervals, excluding a small edge
margin. Remove weak cuts whose neighbouring interiors agree in both colour and
texture. Keep a coherent colour change even when one bed is very short. Do not
simultaneously erase both sides of a candidate bed in one verification pass.

The final minimum interval height is independent of texture context, defaulting
to **2 pixels**. This does not guarantee every 2-pixel texture-only bed is visible;
it prevents a wide texture window from automatically deleting a strong colour
contact merely because another contact is nearby.

## Run on a column crop

```bash
python 'info extraction/texture_boundaries.py' \
  'crops/interpreted_lithology_column/60807402860000_gb506_2_md_mudlog.png' \
  --output crops/boundary_refinement
```

Produces files with the source image stem:

- `.csv`: `top_y,bottom_y`, one row per contiguous interval.
- `.png`: native-resolution column with red internal boundary lines.
- `.scores.npy`: combined diagnostic scores at pixel edges, length H+1.
- `.color_scores.npy`: separate fill-colour change scores.
- `.parameters.json`: base/effective scales, minimum thickness, thresholds and
  refinement radius.

Coordinates are **crop-local**, half-open `[top_y,bottom_y)`. Intervals cover the
entire input, including blanks. The full `pipeline.py` may export document-level
coordinates separately; this stage's CSV remains crop-local.

## API and tuning

```python
import sys
sys.path.insert(0, 'info extraction')
from texture_boundaries import detect_boundaries

result = detect_boundaries('column.png')
print(result.boundaries, result.intervals)
print(result.scales, result.min_interval, result.refine_radius)
```

Inputs may be paths, PIL images, or uint8 grayscale/RGB/RGBA arrays. Existing
`BoundaryResult.boundaries`, `.intervals`, `.scores`, `.threshold` and `.window`
remain available. Additional diagnostic fields expose effective settings.

| Parameter / CLI flag | Default | Purpose |
|---|---|---|
| `window` / `--window` | width-derived | Base texture context, not a minimum bed size |
| `min_interval` / `--min-interval` | **2** | Final minimum spacing/thickness in pixels |
| `threshold` / `--threshold` | 0.16 | Texture-change score threshold |
| `color_threshold` / `--color-threshold` | **0.10** | RMS RGB fill-change threshold in normalized RGB units |
| `refine_radius` / `--refine-radius` | **up to 24 px** | Bounded colour-contact localization search; 0 disables refinement |
| `margin` / `--margin` | 0.12 per side | Trim borders/gutters |
| `use_color` | True | RGB texture and short-context colour evidence; `--no-color` disables |
| `ignore_rules` | True | Suppress short nearly full-width dark rules in texture features; `--keep-rules` disables |

Increase `color_threshold` to reject weaker colour fluctuations. Increase
`min_interval` only when intentionally discarding thinner beds. Large texture
windows no longer impose their size on colour-contact spacing. Lower thresholds
or tiny minimum heights can increase oversegmentation.

The same `--color-threshold` and `--refine-radius` controls are available in
`pipeline.py`, `interval_lithology.py`, and `final_output.py`; those entry points
call the shared detector. The full pipeline records effective scales, spacing,
and refinement settings in `run.json` and saves `boundary_color_scores.npy`.

Scores are diagnostic, **not probabilities**. The combined curve includes
rescaled colour evidence; not every curve peak is an accepted boundary because
periodicity, refinement, spacing and split/merge checks follow proposal scoring.

## Before/after examples

Generated outputs are in **`crops/boundary_refinement/`**:

- **`comparison.png`**: old/new overlays for GB506, Hadrian and log05, with row
  rulers. Each panel has old on the left and new on the right.
- Per-document CSVs, full overlays and diagnostic arrays.
- `comparison_metrics.json`: section-level development checks.
- `gb506_pipeline/` and `hadrian_pipeline/`: complete new TIFF + JSON runs,
  including final cluster/lithology/depth CSVs and annotated columns.

Small, visually reviewed development sections use exact native fill-palette
contacts as references. This checks colour contacts, **not comprehensive semantic
lithology ground truth**. Source palettes are only in test fixtures, never in the
algorithm.

| Section | Reference contacts | Old within 3 px | New within 3 px |
|---|---:|---:|---:|
| GB506 rows 500–1100 | 23 | 11 | **23** |
| Hadrian rows 16800–17400 | 29 | 8 | **29** |

All 52 contacts in those sections are localized within **1 pixel** of the
palette-contact reference (one reference is a half-pixel contact through a dark
separator). Complete interior colour intervals recovered at both ends improve
from 2/22 to 22/22 for GB506 and 1/28 to 28/28 for Hadrian.

The user-approved **log05** control retains **all 23 internal boundaries at
exactly the previous coordinates**. This is a regression check, not independent
localization ground truth.

## Validation and limits

```bash
python -B -m unittest discover -s tests -v
```

The expanded suite contains **125 tests**, including 19 boundary-refinement tests: 17 synthetic
checks and two optional local-crop checks. Real-crop tests are skipped in bare CI
without the supplied images. Synthetic coverage includes 2–30 pixel colour beds,
window changes, contacts near image edges, multiple short beds, grid rules,
periodic drawings, local refinement, unsupported black regions and parameter
validation. Existing clustering/depth/full-document tests remain in the suite.

All 51 supplied crop images were also smoke-tested for finite scores, contiguous
intervals, and complete image coverage. This verifies execution, **not accuracy
across all logs**. The new detector is more sensitive to short colour changes;
other coloured artwork, noise or overprints may still cause extra cuts.

Remaining limitations:

- Same-colour, very thin texture changes still require enough pattern evidence.
- Very dark fills, subtle same-hue shading changes and 1-pixel beds are not
  reliably rescued by the dominant bright-fill channel.
- Truly regular alternating colour beds can resemble one repeating texture;
  the periodicity veto may suppress some of them.
- Symbols/text can still produce false cuts. Fixed side trimming does not deskew
  columns or follow drifting borders.
- Gradual contacts and sparse marks may not define a unique exact pixel edge.
- Local development checks are not a held-out accuracy evaluation; broader manual
  annotation and precision/recall versus bed thickness remain future work.

No clustering, legend matching or depth calibration is performed in this module;
those remain separate pipeline stages. A correctly detected thin colour bed may
still lack enough texture context for the hybrid matcher to name it confidently. The legacy detector is preserved under
`wastebin/info extraction/` and is not imported.
