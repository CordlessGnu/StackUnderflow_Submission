# notification.md — boundary detection rewrite

Working notes while rebuilding `boundary_detection.py` around your two rules.
I'll append findings as I go; questions are marked **QUESTION**.

Status: **implemented and validated** - see Findings 7-9 at the bottom.

## Finding 7 - a real bug the instrumentation caught: look-alike ties manufacture boundaries

On hadrian (47,921 rows) the first implementation produced **143 intervals
alternating Clay(300px)/Halite(225px) with confidence 0.02-0.03**. Instrumented
distances showed why: Clay wins at 0.045, Halite at 0.046 - a statistical tie,
decided by noise, repeating every ~525px down the column. The old algorithm had
called this whole stretch one Halite interval.

Fix: hysteresis. A label change is only accepted when the new lithology beats
the runner-up by at least `--min-margin` (default 0.01); tied windows keep the
previous label and are counted in a warning. hadrian now yields **11
intervals**, with "79/185 windows sat between two lithologies and kept the
previous label" reported instead of fake boundaries.

## Finding 8 - full-dataset results, old vs new

| | old (change-point) | new (two-pass) |
|---|---|---|
| intervals | 3,796 | **2,031** |
| columns segmented | 35/51 | 35/51 (16 skipped by guards) |
| blank top on d-582 | labelled `DACITE` (conf 0.95) | **`empty`** |
| blank top on gb506 | labelled `Core` | **`empty`** |
| hadrian's brick stretch | 1 interval `Halite` 0.87 | 1 interval `Clay` 0.02 + tie warning |
| wall time | ~90s | ~90s |

The interval count dropping by ~1,800 is mostly stripe-flicker boundaries and
blank regions mislabelled as lithologies disappearing.

## Finding 9 - what the two passes now are, in one paragraph each

**Pass 1:** per-band strip-edge detection (follows the skew, ignores the
gutter), a centre band of 5% of the strip width, one bit per row ("any ink in
band", vectorised with prefix sums), runs >= `--stable` become anchors
(blank/solid), bursts of short runs stay one texture region.

**Pass 2:** every legend swatch is resized over a scale grid (0.5-3.0x),
described, and cached; each sliding window of a texture region is classified
by descriptor distance at the best scale; hysteresis decides label changes;
the median winning scale is reported as the column's general swatch size
(gb506: 1.2, hadrian: 1.7, Rio_Lavaca: 0.9).

## Validation added

`--validate` now also checks that `empty` intervals are actually blank (<5%
ink over the full strip) and `solid` intervals are actually mostly ink (>80%)
- both failures mean the label is wrong information, and they fail the run.
Each interval's ink fraction and matched scale are in the JSON/CSV so bad
labels are auditable.

## Still open (unchanged)

* 16 columns skipped by the guards - 12 legends produce <2 usable swatches
  (d-582: the model reads the "LITHOLOGY SYMBOLS" title as swatches and the
  swatch interiors as names, so the library comes out as `['1']`), 4 have no
  swatches detected at all.
* Boundary positions from pass-2 subdivision are quantised to `--step`
  (default 75px) and uncertain by about half a window.

---

## Finding 1 — the centre-line signal is viable, and the stripe problem is exactly where you said it was

Measured on 51 columns (per-row band detection + centre band = 5% of band
width, state = "any ink in band"):

| column | height | raw flips | median run | notes |
|---|---|---|---|---|
| d-582 log01 | 88,276 | 6,068 | 9px | stripes run 5–15px on / 13–15px off |
| gb506 | 8,417 | 2,134 | **1px** | ultra-fine texture, band is only 2px wide |
| Trail_Mtn | 60,901 | 4,261 | ~30px | coarse stripes, period ~60–90px |
| FREEMAN | 13,932 | 4 | — | essentially all blank |

So raw colour changes fire every few pixels — thousands of false boundaries,
as you predicted. Grouping is mandatory.

## Finding 2 — run lengths are (mostly) strongly bimodal, which makes grouping tractable

For d-582 log01 there are **zero runs between 50px and 1000px**: everything is
either stripe-level (1–50px) or stable-level (>1000px, the blank top is 11,945px).
8 of 51 columns have no runs at all in 100–1000px. A handful of columns
(Callaway, Rio_Lavaca, mc761) have 20–170 runs in that middle range — those
need a larger threshold or will split more.

Consequence: a stable-run threshold T ≈ 100–150px separates "stripe" from
"real state" for most columns. I'll expose it as `--stable` rather than
hardcode it, and make the interval merge report how much it absorbed.

## Finding 3 — pass 1 alone cannot see striped→striped boundaries, by construction

If two neighbouring intervals are both striped and there's no white gap at the
join, the centre line never flips there. Measured: gb506's blank top ends at
532px and the *entire rest* of the column (532–8,116) is one continuous burst —
your old algorithm's 2,210 boundary is invisible to the on/off signal.

So pass 1 gives: blank/solid anchors + texture bursts. The burst-internal
boundaries have to come from pass 2 (sliding swatch matching — see Finding 5).
This is a property of the signal, not a bug; flagging it so we agree on it.

## Finding 4 — blank handling

d-582 log01: rows 0–8 have ink (column header mark), 8–11,953 is blank,
texture starts at 11,953, and the bottom 1,398px is blank again.
Plan: long white runs become explicit `empty` intervals (never given a
lithology name); tiny ink fragments at region edges get absorbed by
`--min-interval` and reported.

## Finding 5 — pass 2 "resize until it matches": what works and what provably doesn't

I tested three matchers on gb506 (same-document legend, clean detection):

| matcher | synthetic tiles | real column |
|---|---|---|
| 2D NCC, swatch tiled at swept scale | **44/44 correct** | **fails** — best NCC 0.19–0.37, top-5 scores nearly identical |
| 1D row-profile, folded per period | — | no discrimination (everything 0.9+) |
| texture descriptor, swatch **resized to the column's scale first** | — | **discriminates** — winner 0.03–0.06 vs runner-up 0.13–0.27 |

Why 2D NCC fails on real data: the legend swatch and the column fill are
*rendered differently* (staggered vs aligned dashes, different dash weights),
so pixel-level correlation is low even at the correct scale. It only worked on
synthetic tiles because there the column literally is the swatch.

Why descriptor-after-resize works: resizing fixes the scale blindness of the
descriptor; the descriptor itself only uses re-rendering-stable statistics
(orientation histogram, run lengths, density).

**So pass 2 becomes: estimate the column's stripe period, resize each swatch by
period_ratio (plus a bounded scale grid as fallback), describe, compare.** The
winning scale, averaged over windows, is the "general swatch size" you asked
pass 2 for. On gb506 it comes out at 0.8–1.2 — sensible for that log.

## Finding 6 — burst subdivision gives the missing boundaries

Classify the burst in sliding windows (default 300px window, 75px stride),
median-filter the label sequence, merge same-label runs → internal boundaries.
Boundary placement = first window that sees the new pattern, so position
uncertainty is about ±½ window. Trade-off is tunable via `--window/--step`.

---

## QUESTION 1 — what should `empty` mean, exactly?

Blank runs longer than `--stable` become `empty` intervals. Do you want
*short* blank gaps (say 130px, like Trail_Mtn's two internal blanks) to also be
`empty`, or should only "big" blanks (top-of-column, > 500px?) count? Current
plan: any blank run ≥ `--stable` (120px default) is `empty`. Cheap to change.

## QUESTION 2 — solid runs

Long all-ink runs (e.g. Rio_Lavaca has a 2,219px solid black run — coal) become
intervals classified like any other. Alternative: report them as `solid`
without naming. I'm naming them (coal matches by density). OK?


## Finding 11 - prototype verdict and integration (the convolution rewrite)

Built the self-referential continuity test and validated it against boundaries
I could check by eye before integrating anything.

**The test:** for every candidate row y, cut the strip's row profile into an
upper half [y-120, y) and a lower half [y, y+120); search +/-40 rows for the
lag that best continues the upper half into the lower half (the continuation
lag of a period-P pattern is -120 + k*P, which is usually POSITIVE - the first
prototype silently skipped all positive lags, a real bug the numbers caught).
d(y) = 1 - best correlation. Inside one continuous texture d hugs the burst's
own noise floor; where the pattern stops continuing it spikes.

**Visual scorecard on d-582 (boundaries judged by eye at 2x zoom):**

| candidate | d | verdict |
|---|---|---|
| y=31,153 | 0.60 | REAL - dots -> bracket pattern, obvious |
| y=67,743 | 0.50 | REAL - brackets -> dashes |
| y=55,363 | 0.50 | REAL |
| y=50,293 / 25,083 | ~0.47 | plausible, subtle |
| y=20,530 (my earlier visual reading) | 0.29 | REAL but GRADUAL - dashes shrink progressively into dots; no single boundary row exists |

**Negative control (gb506):** the continuity test finds no peaks above its
noise floor - and at 2x zoom there is genuinely no boundary at the old pass-2
positions. The column is one continuous dash texture; the old label-change
boundaries were over-splitting. Same period + same waveform = same texture,
as agreed.

**One borderline case, left alone on purpose:** hadrian has a grey->orange
brick-fill change at y=23,962 that scores 0.83 against a threshold of 0.82.
It is reachable with `--boundary-z 3`, but gb506's colour-mottling artefacts
(also checked visually: not boundaries) start firing at z=3, so the default
stays 4. Colour-only changes are half-visible to a binary pipeline by design.

**Integrated results (all 51 columns):**

| engine | intervals | columns |
|---|---|---|
| original change-point | 3,796 | 35 |
| label-change + hysteresis | 2,031 | 35 |
| **continuity test (current)** | **2,606** | 35 |

2,606 > 2,031 because hysteresis was suppressing real splits along with fake
ones; the continuity test fires on actual pattern discontinuities. gb506: 3
intervals (empty / one Clay/Shale 532-8116 / empty, confidence 0.47, was 8
fragments at 0.03-0.52). hadrian: 4 (burst 1 stays one interval; the orange
band case sits at the threshold). Rio_Lavaca's dense stretches are single
intervals instead of oscillating fragments.

**Known miss:** gradual transitions (T1) score below the threshold - a
gradual change has no single boundary row. If you want those flagged, the
option is a "soft transition zone" report (regions where d stays elevated for
300+ rows); not built.
