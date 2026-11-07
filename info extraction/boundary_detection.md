# Boundary detection — how the matching works

Companion notes for `boundary_detection.py`. Two questions are answered here:

1. [How is the swatch matching performed?](#1-swatch-matching)
2. [How is the legend detection done?](#2-legend-detection)

A short [pipeline overview](#0-where-this-sits) and a [limitations](#5-known-limitations)
section follow, because the matching only makes sense in context.

---

## 0. Where this sits

```
interpreted_lithology_column.png ──┐
                                   ├──► segment_column() ──► [(top, bottom, lithology), ...]
pattern library (swatches+names) ──┘
```

The pattern library is always built from the legend of the **same document**.
The legend is run through `models/swatch_detection.pt`, which returns the swatch
boxes and their labels; each swatch becomes a library entry.

Legends are paired with columns by file stem, so
`crops/interpreted_lithology_column/foo.png` uses
`crops/lithology_legend/foo.png`. `--legend` forces one legend for every
column, `--legend-dir` points at a different folder.

There is no global pattern pool: a swatch and the column it is matched against
must come from the same scan, because that is the only way their rendering and
scale agree.

---

## 1. Swatch matching

### 1.1 Why not template matching

The obvious approach — slide the swatch over the column and take the best
normalised cross-correlation — **does not work here**. Measured on a real pair:

| approach | best NCC |
|---|---|
| best over all scales, phase searched | **0.22** |
| swatch stretched to column width *and* height | 0.11 |
| swatch stretched to column width only | −0.03 |
| no resize | −0.03 |

The reason is visible when you put them side by side: the legend swatch draws
its dashes in a **staggered** brick-like layout, while the same pattern inside
the column is drawn as a **regular aligned grid** with longer dashes. The pixels
differ even though the texture is the same.

So the matcher compares *texture statistics*, not pixels.

### 1.2 Step 1 — binarise

```python
def binarize(image):
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return (array < 128).astype(np.float32)   # 1.0 = ink, 0.0 = paper
```

The mudlog TIFFs are 1-bit, so this is lossless for them and also handles the
8-bit PNG crops.

### 1.3 Step 2 — normalise scale onto a fixed grid

```python
DESCRIPTOR_GRID = 64
binary = block_mean(binary, 64, 64)
```

`block_mean` averages blocks of pixels onto a 64×64 raster using
`np.add.reduceat` (vectorised, no Python loop):

```python
y_edges = np.arange(rows) * height // rows
x_edges = np.arange(cols) * width  // cols
sums   = np.add.reduceat(np.add.reduceat(binary, y_edges, axis=0), x_edges, axis=1)
counts = np.outer(np.diff(np.append(y_edges, height)),
                  np.diff(np.append(x_edges, width)))
return sums / counts
```

**This is the key to scale invariance.** A 56×40 legend swatch and a 245×100
column band both become 64×64, so their descriptors are directly comparable
even though the column pattern is stretched ~4× horizontally.

It also makes the descriptor ~3× faster (2.76 ms → 0.88 ms per patch).

### 1.4 Step 3 — the 7-element descriptor

```python
def texture_descriptor(binary, grid=64):
    binary = block_mean(binary, grid, grid)

    # central differences -> gradient direction
    gy[1:-1, :] = binary[2:, :] - binary[:-2, :]
    gx[:, 1:-1] = binary[:, 2:] - binary[:, :-2]
    magnitude = np.hypot(gx, gy)
    angle     = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    mask      = magnitude > 0.5

    # 4-bin orientation histogram, normalised to sum 1
    for i, (low, high) in enumerate(ORIENTATION_BINS):
        histogram[i] = np.count_nonzero(mask & (angle >= low) & (angle < high))
    histogram /= histogram.sum()

    mean_h = mean(horizontal run lengths) / width
    mean_v = mean(vertical   run lengths) / height

    return [*histogram, mean_h, mean_v, binary.mean()]
```

| # | feature | what it separates |
|---|---|---|
| 0 | orientation 0–22.5° | horizontal dashes (clay, shale) |
| 1 | orientation 22.5–67.5° | diagonal hatching (dolomite, anhydrite) |
| 2 | orientation 67.5–112.5° | vertical dashes / bricks (siltstone, halite, limestone) |
| 3 | orientation 112.5–157.5° | the other diagonal |
| 4 | mean horizontal run / width | dash length vs dots |
| 5 | mean vertical run / height | vertical line length |
| 6 | ink density | solid fills (coal ≈ 0.9) vs line art (≈ 0.1) |

**Why the gradient, not the stroke direction.** The gradient of a binary stroke
points *across* it, so a horizontal dash produces a *vertical* gradient. The
angle is folded into 0–180° because a stroke and its gradient are the same line
modulo 180°.

**Why orientation dominates.** It is the only feature that survived the
rendering difference. Measured on the d-582 legend swatches vs the column:

| pattern | orientation bin 0 | bin 2 |
|---|---|---|
| Siltstone (vertical dashes) | 0.64 | 0.26 |
| Clay / Shale (horizontal dashes) | 0.14 | 0.72–0.74 |
| Limestone (brick) | 0.29 | 0.56 |

The column regions land in the same places: vertical-dash regions measured
0.63–0.67 in bin 0, horizontal-dash regions 0.13–0.33. That is what makes the
match work.

### 1.5 Step 4 — weighted Euclidean distance

```python
FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.3])

def descriptor_distance(a, b):
    return float(np.linalg.norm((a - b) * FEATURE_WEIGHTS))
```

Orientation gets full weight; run lengths half (they shift with the horizontal
stretch); density only 0.3 (it is the least transferable feature — the column
draws denser than the legend, so density is only trusted to catch solid fills).

### 1.6 Step 5 — nearest neighbour + confidence

```python
def classify(descriptor, library):
    scored = sorted((descriptor_distance(descriptor, p.descriptor), p.name)
                    for p in library)
    best_distance, best_name = scored[0]
    second_distance = scored[1][0]

    margin     = (second_distance - best_distance) / (second_distance + 1e-6)
    confidence = clip(margin, 0, 1) * exp(-best_distance / 2.0)
    return best_name, confidence, runner_up
```

Confidence is deliberately two-sided:

- **margin** — how much better the winner is than the runner-up. A tie gives 0.
- **`exp(-distance/2)`** — damps the score when the absolute distance is large,
  so a confident-looking win between two bad matches still scores low.

`--min-confidence` turns anything below the floor into `unknown`, so the output
degrades honestly instead of confidently wrong.

### 1.7 Step 6 — library preparation

- **`collapse_library`** averages the descriptors of every swatch that shares a
  name, so 619 swatches become 228 distinct lithologies.
- **`--filter-names`** drops non-lithology legend entries (`Casing shoe`,
  `Not Available`, `Void`, `Survey`, …). These are blank or generic swatches
  that otherwise win against sparse textures. A name is kept if it contains a
  lithology word, or if it contains no tool word.

### 1.8 Measured behaviour

Same-well matching (gb506 column vs gb506 legend swatches):

| column interval | nearest swatch | distance |
|---|---|---|
| 440–2210 | **Shale** | **0.06** |
| 2210–8417 | **Shale** | 0.16 |
| 0–440 (mostly blank) | Clay/Shale | 0.71 |

This is why the library is built per document: matching a column against
swatches from a *different* well gives nonsense (`PORCELANITE`, `OBSIDIAN`),
because every well draws its patterns slightly differently.

---

## 2. Legend detection

`load_library_from_legend()` uses a YOLO detector — `models/swatch_detection.pt` —
to find the swatches and their labels. There is no hand-written rectangle
detection any more.

### 2.1 The model

| property | value |
|---|---|
| file | `models/swatch_detection.pt` |
| task | object detection (Ultralytics YOLO) |
| classes | `material_name` (0), `material_texture` (1) |
| trained on | the pshhYolo legend dataset |

It returns both boxes we need in one pass: the swatch (`material_texture`) and
the text label beside it (`material_name`).

### 2.2 Step 1 — run inference

```python
result = model.predict(legend_path, conf=0.25, imgsz=960)[0]
```

**`imgsz=960` matters.** Legends are wide and short (e.g. 1513×407), so the
Ultralytics default of 640 downsamples them too aggressively and small swatches
are lost. Measured on the gb506 legend:

| `imgsz` | swatches found (conf 0.25) |
|---|---|
| 640 | 13 |
| **960** | **16** |
| 1280 | 11 |

`--swatch-conf` and `--swatch-imgsz` expose both knobs.

### 2.3 Step 2 — split the detections by class

```python
for box in result.boxes:
    label = model.names[int(box.cls)]
    xyxy  = tuple(float(v) for v in box.xyxy[0])
    if   label == "material_texture": textures.append(xyxy)
    elif label == "material_name":    names.append(xyxy)
```

If no `material_texture` box is found the function exits with a message
suggesting a lower `--swatch-conf`.

### 2.4 Step 3 — pair each swatch with its name

The model gives two independent lists, so they have to be paired. In a legend
the name sits to the **right** of its swatch on the **same row**:

```python
overlap = min(texture[3], name[3]) - max(texture[1], name[1])
if overlap / min(height(texture), height(name)) < 0.5:
    continue                                   # different row
gap   = name[0] - texture[2]                   # horizontal distance
score = abs(gap) + 0.5 * abs(centre_y(texture) - centre_y(name))
if gap < 0:
    score += 0.25 * abs(gap)                   # prefer names on the right
```

Two details that matter:

- **`abs(gap)`, not `max(gap, 0)`.** Clamping negative gaps to zero lets a name
  far to the *left* outscore the correct adjacent one. This was a real bug: in
  one legend a swatch at `x=505` was matched to `CLAY` at `x=293` instead of
  `GRANITE` at `x=545`.
- **Greedy with a used-set**, so one name cannot be claimed by two swatches.
  If nothing overlaps vertically, the nearest name overall is used as a
  fallback.

### 2.5 Step 4 — read the label

The paired `material_name` box is cropped and read with RapidOCR in
**recognition-only** mode (`use_det=False, use_cls=False, use_rec=True`) — the
box is already known, so running the detector again is slower and less reliable
on tight single-line crops.

If OCR returns nothing the swatch is named `swatch_<x>_<y>`.

### 2.6 Step 5 — describe the swatch

The swatch is cropped **2 px inside** the detected box, so the border does not
pollute the texture statistics, then reduced to the 7-element descriptor from
[§1.4](#14-step-3--the-7-element-descriptor).

### 2.7 Measured behaviour

| legend | swatches | names | after `--filter-names` |
|---|---|---|---|
| gb506#2_md | 16 | 17 | **13** clean lithologies |
| d-582 log01 | 6 | 23 | 7, all `swatch_*` fallbacks |

The gb506 library comes out correct:

```
Clay/Shale, Shale, Siltstone, Sandstone, Chert, Limestone,
Dolomite, Halite, Anhydrite, Gypsum, Core, Plastic Clay, Diatomite
```

and the descriptors are sensible — `Clay/Shale` scores 0.91 in the
horizontal-edge bin, `Siltstone` 0.50/0.50 across the horizontal and vertical
bins, `Core` 0.87 in the horizontal bin (it is drawn as vertical rules).

The d-582 legend is the known failure case: the model finds only 6 of its 10
swatches and reads 23 text fragments as names, so the pairing is unreliable.
The function warns when `len(names) > 2 * len(textures)`.

## 3. How the two passes work

### 3.1 Pass 1 - the centre line

**Strip edges, per band.** The crop is skewed (the d-582 strip drifts ~52px
over 88,000 rows) and sometimes includes the depth-tick gutter or a neighbour's
edge. The image is cut into 200-row bands; each band's left/right rules are the
columns whose ink density exceeds 80%. Bands with no rules (blank, or a solid
interval that fills the strip) reuse the nearest band's edges, and every reuse
is reported. The per-band edges are interpolated linearly down the image, which
follows the skew.

**The centre band.** A band 5% of the strip width is centred on the strip.
Each row becomes one bit: any ink in the band or not. A one-pixel column would
sit in the white gap of striped patterns; a band this wide catches the stripes
while staying inside the pattern. The test is vectorised with prefix sums, so
an 88,000-row column costs one pass.

**Runs to regions.** Runs of one state longer than `--stable` (default 120
rows) are anchors: paper -> `empty`, ink -> a `solid` interval. Everything
shorter is stripe-level flipping, so consecutive short runs merge into one
`texture` region. Measured on d-582 log01 there are *zero* runs between 50px
and 1000px - stripe runs and stable runs are cleanly separated - but a handful
of columns (Callaway, Rio_Lavaca, mc761) have runs in that middle range, so
the threshold is a flag, not a constant.

Regions shorter than `--min-interval` are absorbed into the taller neighbour;
how many boundaries this removed is reported.

**What pass 1 cannot see.** If two neighbouring intervals are both striped and
no white gap separates them, the centre line never flips there and pass 1
returns one texture region. This is a property of the signal. On gb506 the
blank top ends at 532px and the remaining 7,584 rows are a single burst; the
internal boundaries come from pass 2.

### 3.2 Pass 2 - self-referential boundaries, then swatch naming

**The continuity test.** For every candidate row y inside a texture region,
cut the strip's row profile into an upper half `[y-120, y)` and a lower half
`[y, y+120)`, then search +/-40 rows for the lag that best continues the
upper half into the lower half. d(y) = 1 minus that best correlation.

The continuation lag of a period-P pattern is `-120 + k*P`, which is usually
**positive** - the first prototype silently skipped every positive lag (its
windows were 2H wide, so positive shifts fell outside) and the numbers caught
it. The window is `2H + 2*lags` wide so both halves can slide.

Inside one continuous texture d hugs the burst's own noise floor; where the
pattern stops continuing it spikes. Peaks clear `z` robust deviations above
the burst's median (`--boundary-z`, default 4, with a small absolute floor so
a perfectly uniform burst cannot turn scan wobble into "boundaries"), are at
least 80 rows apart, and never within 120 rows of the burst edges - pass 1
already put boundaries there.

Why self-referential: the test compares the column with itself, never with
the legend, so the rendering mismatch that kills 2D template matching (§1.1)
cannot occur.

**Verified against eyes, not just numbers** (d-582, boundaries judged at 2x
zoom): y=31,153 dots->brackets REAL, y=67,743 and y=55,363 brackets->dashes
REAL, y=20,530 real-but-gradual (dashes shrink progressively into dots; scored
below threshold - a gradual change has no single boundary row). Negative
control: gb506 shows no boundary at its old pass-2 positions, and at 2x zoom
there is genuinely none - the column is one continuous dash texture.

**Naming.** Each segment between boundaries is classified by the scale library
(swatch resized over the scale grid, descriptor distance). Because boundaries
no longer come from label changes, look-alike ties cannot manufacture
boundaries; a tie only weakens the segment's confidence, which is reported.

The median winning scale across the column is reported as the general swatch
size (gb506 1.2, hadrian 1.5, Rio_Lavaca 0.9).

Blank regions skip everything: they are `empty`, confidence 1.0.

### 3.3 What the output carries

Each interval records `kind` (texture/solid/empty), `ink` (the ink fraction of
the full strip, not just the centre band), `scale`, `confidence` and
`runner_up`. The ink fraction makes labels auditable: an `empty` interval with
20% ink is visibly wrong, and `--validate` fails the run when that happens.

---

## 4. Tuning knobs

| flag | default | effect |
|---|---|---|
| `--centre-band` | 0.05 | centre band width, fraction of the strip |
| `--band-height` | 200 | rows per band when locating the strip edges |
| `--stable` | 120 | runs longer than this are anchors, not stripes |
| `--boundary-z` | 4.0 | continuity-peak threshold (lower = more boundaries) |
| `--min-interval` | 30 | shorter regions are absorbed into a neighbour |
| `--min-confidence` | 0.0 | below this the label becomes `unknown` |
| `--filter-names` | off | drop non-lithology legend entries |
| `--legend-dir` | `crops/lithology_legend` | folder paired with columns by stem |
| `--legend` | - | force one legend for every column |
| `--swatch-conf` / `--swatch-imgsz` | 0.25 / 960 | swatch detector knobs |

## 5. Known limitations

1. **Striped-to-striped boundaries are invisible to pass 1.** Where two
   neighbours both stripe and no white gap separates them, the centre line
   never flips; the boundary comes only from pass 2's label changes, which are
   quantised to `--step` and uncertain by about half a window.

2. **Every column needs its legend.** If no legend with the same file stem
   exists in `--legend-dir`, the column is skipped with an error. Use
   `--legend` to force one.

3. **The swatch model is trained on pshhYolo-style legends.** 12 legends
   produce fewer than 2 usable swatches - on d-582 the model reads the
   "LITHOLOGY SYMBOLS" title as swatches and swatch interiors as names, and
   the resulting library is rejected by the guards.

4. **Look-alike patterns tie.** Clay vs Halite-style pairs can sit 0.001 apart
   in descriptor space; hysteresis keeps the first label and reports the tie
   instead of oscillating, but the resulting label is weakly supported - check
   the confidence column.

5. **Scale is swept, not derived.** The winning scale comes from a 0.1-step
   grid, so window-to-window scale varies by up to 0.05. The median scale per
   column is reported so systematic drift is visible.
