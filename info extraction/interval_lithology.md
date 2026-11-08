# Hybrid interval grouping and legend matching

Implementation: **`interval_lithology.py`** and **`texture_seam.py`**.
The complete TIFF + bbox JSON entry point, `pipeline.py`, uses the hybrid matcher
by default. Boundary detection and depth calibration are unchanged by this stage.

**Default label policy:** whenever the original decision is `unknown`, replace
`lithology` with its nonempty **`runner_up`** (the second candidate). Mark the row
with `lithology_fallback=True` and retain the original match status/scores. If no
usable runner-up exists, the label stays `unknown`. **Exception:** rows with
`match_status=blank` have an empty `lithology` cell, no fallback flag, and no
fallback distance. This also clears runner-up names assigned by older exports.

## The hybrid approach

This implements the proposed stacking idea **at the artificial seam**, without
building growing stacked images or repeatedly scanning every pixel for boundaries.

### Cached preparation

For every interval, sample at most three local patches, normalize their width to
64 pixels while preserving aspect ratio, and cap each patch at 128 rows. Reuse
the boundary detector's RGB histograms, gradient orientations, ink density and
spatial differences. Cache both a pooled descriptor and multiscale head/tail
summaries at contexts of 4, 8, 16, 32 and 64 rows when enough actual data exists.

Very short images may be padded for fixed-length feature extraction, but their
real native/normalized evidence height is retained. Padding and upscaling do
**not** make a two-pixel interval a well-supported texture sample.

### Incremental grouping

1. Process longer intervals first, so tiny/noisy intervals do not automatically
   become permanent prototypes. Even the first two intervals are compared; they
   are not assumed to represent different lithologies.
2. Use cached descriptors to shortlist the closest three existing groups.
3. Apply a cheap all-member descriptor guard, limiting the group's diameter to
   `1.5 * cluster_threshold` (0.30 by default). This prevents single-linkage drift.
4. Verify the interval against at most three group representatives: the exemplar
   nearest the weighted centroid plus diverse, preferably longer exemplars.
5. Require positive seam agreement with **all** representatives, not just one
   lucky matching patch. Retain an uncertain result instead of forcing membership.
6. If two groups are indistinguishable candidates, consolidate them only after
   direct cross-group representative comparisons and the descriptor diameter
   guard. A third interval matching A and B alone is not enough to merge A and B.
7. Assign final `C001`, `C002`, … IDs by first occurrence in the column. Track all
   member and representative interval indices in JSON.

Returning patterns reuse their group ID. Uncertain intervals can remain separate
provisional groups, so **more groups does not necessarily mean more rock types**.
IDs are document-local and deterministic for the same inputs/settings, not
permanent identifiers across matcher versions or parameter changes.

### What the seam test measures

Compare the cached features immediately above/below the artificial join at
multiple scales, allowing small phase shifts within each patch. Verify **both
stacking orders**. Features are extracted independently on each image, so the
act of concatenating pixels cannot create an artificial gradient edge.

The distance uses symmetric coverage of the available windows rather than one
minimum pair. The larger supported contexts determine the aggregate score.
Results are explicitly **`same`**, **`different`**, or **`uncertain`**:

- `same`: enough evidence and consistent multiscale appearance.
- `different`: strong colour/texture change or a supported blank-versus-pattern
  distinction.
- `uncertain`: too little context or a borderline score. No detected change alone
  is not treated as proof of equality.

The minimum-context heuristic requires at least eight real native rows, eight
normalized rows and two usable contexts. It is a heuristic, not a guarantee that
multiple texture repetitions are present. Thin intervals can therefore remain
uncertain even when their boundary was correctly detected from colour.

### Legend naming

Keep swatches in RGB and prepare five vertical rendering scales (0.5, 0.75, 1,
1.5 and 2). Group duplicate crops/scales by **distinct name** for runner-up tests.

For each visual group:

1. Descriptor-shortlist the closest three names (at least two when available,
   even if `shortlist_size=1`, to retain a distinct-name runner-up).
2. Seam-verify up to two crop/scale variants per name against the group's
   representatives. Include a usable-context variant when available instead of
   spending both checks on undersized variants.
3. Rank with a heuristic fused distance:
   `0.45 * descriptor_distance + 0.55 * worst_representative_seam_score`.
4. Require verified seam agreement, the absolute distance threshold, and a
   sufficient margin over the next distinct name before accepting a **verified**
   name. Otherwise use the runner-up fallback when one is available.

The legend seam acceptance threshold uses `max_match_distance` (default 0.32),
allowing for rendering differences. `seam_threshold` (default 0.20) governs
within-column group verification. These scores/margins are **not probabilities**.
Unshortlisted names retain their descriptor diagnostics; if an unverified entry
outranks checked candidates, the original status remains `unverified_match`.
The public label can still use the flagged runner-up fallback. Shortlisting is
approximate and can still miss the correct candidate.

## Run

Existing component commands now select the hybrid matcher automatically:

```bash
python 'info extraction/interval_lithology.py' \
  'crops/interpreted_lithology_column/Rio_Lavaca_1748_Unit_1_241_MUD.png' \
  --output crops/labelled_intervals
```

The script looks for the same filename under `crops/lithology_legend/`. Override
with `--legend`, `--legend-dir`, or `--legend-manifest`.

Reuse boundaries and reviewed RGB swatches:

```bash
python 'info extraction/interval_lithology.py' \
  'crops/boundary_refinement/gb506_pipeline/crops/interpreted_lithology_column.png' \
  --intervals 'crops/boundary_refinement/gb506_pipeline/classification/intervals.csv' \
  --legend-manifest 'crops/boundary_refinement/gb506_pipeline/legend/legend.json' \
  --output crops/my_hybrid_review
```

Or run the full document pipeline as usual. It accepts the same matching controls:

| Flag / API keyword | Default | Purpose |
|---|---:|---|
| `--matcher` / `matcher` | `hybrid` | `descriptor` retains the earlier baseline |
| `--shortlist-size` / `shortlist_size` | 3 | Candidate groups/names, range 1–10 |
| `--max-representatives` / `max_representatives` | 3 | Exemplar cap per group, range 1–5 |
| `--seam-threshold` / `seam_threshold` | 0.20 | Within-column seam agreement |
| `--cluster-threshold` / `cluster_threshold` | 0.20 | Descriptor screening; hybrid diameter guard is 1.5x |
| `--max-match-distance` / `max_match_distance` | 0.32 | Absolute naming distance and legend seam threshold |
| `--min-match-margin` / `min_match_margin` | 0.06 | Required distinct-name separation |

Raising thresholds may combine more appearances, not necessarily more correct
ones. `--matcher descriptor` is useful for coverage/runtime comparisons and to
retain earlier behavior while the hybrid thresholds are evaluated against labels.

## Python seam API

```python
import sys
sys.path.insert(0, 'info extraction')
from texture_seam import compare_interval_patches, prepare_texture, SeamVerifier

comparison = compare_interval_patches('interval_a.png', 'interval_b.png')
print(comparison.status, comparison.score, comparison.reason)
print(comparison.scales, comparison.scale_scores)

# Repeated comparisons: prepare once rather than re-reading/re-extracting features.
a = prepare_texture('interval_a.png')
b = prepare_texture('interval_b.png')
verifier = SeamVerifier()
comparison = verifier.compare(a, b)
```

`label_intervals(column, boundaries, swatches)` still accepts a `BoundaryResult`
or sorted non-overlapping `(top,bottom)` pairs. It never changes pixel boundaries
or invents depths. Different visual groups may map to the same legend name.

## Output and uncertainty

The component writes CSV/JSON, an annotated column and a `.clusters.png` review
sheet. All original fields remain. Additional interval columns are:

- `match_method`: hybrid or descriptor.
- `grouping_status`: new, verified, uncertain, or descriptor_only.
- `grouping_seam_score`: admission score when supported.
- `match_descriptor_distance`: raw descriptor distance for the chosen variant.
- `match_seam_score`, `match_seam_status`: explicit seam evidence.

In hybrid mode `match_distance` is the fused heuristic score for checked variants;
unchecked candidates retain their screening distance. It is not directly a
calibrated confidence or necessarily comparable with an old descriptor-only run.

Existing `match_status` values remain: matched, ambiguous, poor_match, blank,
no_legend. New statuses include:

- `insufficient_context`: not enough real texture data to verify a name.
- `seam_rejected`: a plausible descriptor candidate fails seam verification.
- `unverified_match`: borderline or unverified shortlist evidence.

Original unknown decisions use the runner-up label when available; all members
of a group share that assignment and fallback flag. Original `match_status`
values such as `ambiguous`, `poor_match`, or `insufficient_context` remain
unchanged and do not become `matched`. `lithology_fallback=True` identifies the
substitution; `fallback_distance` is the second candidate's score when known.
Existing `match_*` scores still describe the original best-candidate decision,
not confidence in the replacement. No runner-up means the label stays `unknown`. The JSON records representatives, candidate scores/reasons/scales and
comparison counts/cache hits. The full pipeline carries these diagnostics into
`run.json` and final `intervals.json`; CSV diagnostic columns survive depth joining.

Automatic legend extraction still uses YOLO plus OCR and exports reviewable
`legend.json`/swatch PNGs. A reviewed manifest has the format:

```json
{"swatches": [{"name": "Sandstone", "image": "sandstone.png"}]}
```

Paths are relative to the manifest. Crop inside swatch borders, without labels.
Duplicate examples of the same name are allowed. Missing/unreadable legend
patterns are not recovered by the matcher and can still lead to unknown/wrong
nearest candidates; OCR spelling is intentionally retained for review.

## Cost and validation

Let N be intervals, K groups, L legend entries, T shortlist size, R representative
cap and S the five rendering scales. Feature preparation is bounded per image.
Cheap descriptor screening/diameter guards can still be quadratic in the worst
case; there is no subquadratic guarantee. However, expensive seam tests are
bounded approximately by **N*(T*R + R²)** for grouping/consolidation and
**K*max(2,T)*2*R** for naming, rather than full image scans of growing composites.
Cached summaries and symmetric pair caching avoid repeated feature extraction.

All **125 tests** pass, including runner-up fallback and blank-exception checks: phase shifts,
both stacking orders, tiny-context uncertainty, bounded caches/work, direct
cross-group checks, anti-chaining, duplicate names, export and end-to-end pipeline
propagation. Tests run offline without YOLO/OCR when caches/mocks are used:

```bash
python -B -m unittest discover -s tests -v
```

Review examples are in **`crops/hybrid_matching/`**:

- `gb506/hybrid/intervals.clusters.png` and `hadrian/hybrid/intervals.clusters.png`.
- Corresponding `descriptor/` folders for the earlier baseline.
- `comparison.json` records assignment counts, timing and comparison diagnostics.
- `gb506_pipeline/` is an actual two-input TIFF + JSON hybrid run with final depths.

Using the same existing boundaries and reviewed extracted legends:

| Log | Intervals | Descriptor groups / verified | Hybrid groups / verified | Hybrid matching time |
|---|---:|---:|---:|---:|
| GB506 | 201 | 15 / 182 | 49 / 137 | ~3.5 s |
| Hadrian | 547 | 17 / 448 | 47 / 238 | ~16.6 s |

These are **verification counts, not accuracy measurements**. The hybrid keeps
additional uncertain/provisional groups and rejects more original decisions,
especially on tiny intervals or similar short legend swatches. The current
runner-up policy fills their public labels without increasing verified counts.
Updated complete examples are under `crops/runner_up_fallback/`. It has not
been shown to be more accurate on a manually labelled held-out set. GB506 used
1,320 unique seam comparisons; Hadrian used 3,282. The full raw-TIFF GB506 run
preserved every interval boundary and depth exactly relative to its previous run.

Remaining limits include visually indistinguishable swatches, inadequate native
context, wrong upstream boundaries, mixed beds, scale-bank mismatch and incorrect
legend detections/OCR. Scores require development/held-out calibration before
being treated as accuracy claims; the seam test is evidence, not semantic proof.
