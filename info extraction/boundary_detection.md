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

The pattern library comes from one of two places:

| mode | flag | source |
|---|---|---|
| global | `--patterns` / `--names` | `crops/lithology/patterns` + `crops/lithology/name` (from `crop_dataset_classes.py`) |
| per-document | `--legend` | swatches detected inside the legend of the **same** scan |

The per-document mode is more accurate because the swatch and the column are
then rendered by the same scan at the same scale.

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

Cross-well matching (d-582 column vs the global pshhYolo library) gives
nonsense (`PORCELANITE`, `OBSIDIAN`) because d-582 is not in that library —
see [limitations](#5-known-limitations).

---

## 2. Legend detection

`detect_swatches()` finds the bordered swatch rectangles in a legend image.
It is used by `--legend` to build a per-document pattern library.

### 2.1 The problem

A legend is a grid of small bordered rectangles, each containing a pattern, with
the lithology name written to the right:

```
┌────────┐
│ ▬ ▬ ▬  │  Clay/Shale
└────────┘
┌────────┐
│ ▭▭▭▭▭  │  Limestone
└────────┘
```

The borders are **1 px lines** in a scanned image, and they are not reliable:

- the **left/right** rules are continuous,
- the **top/bottom** rules are often **clipped** where the pattern touches them,
- a **solid** swatch (coal) makes every interior column look like a rule.

### 2.2 Step 1 — collect long runs

```python
def _runs_1d(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
```

For every column, keep vertical ink runs of at least `min_height` (30 px).
For every row, keep horizontal ink runs of at least `min_width` (50 px).

### 2.3 Step 2 — pair the vertical rules

```python
for i, (x1, top1, bottom1) in enumerate(verticals):
    for x2, top2, bottom2 in verticals[i + 1:]:
        if not (50 <= x2 - x1 <= 400):            continue   # plausible width
        if abs(top1 - top2) > 4:                  continue   # same top
        if abs(bottom1 - bottom2) > 4:            continue   # same bottom
        if not (30 <= bottom - top <= 300):       continue   # plausible height
        if has_horizontal(top, x1, x2) and has_horizontal(bottom, x1, x2):
            found.append((x1, top, x2, bottom))
```

**Verticals first, horizontals as confirmation.** This is the important design
choice. Pairing horizontals first fails on real legends because the bottom rule
of the Limestone swatch spans `x = 89–244` while the top spans `x = 75–243` — a
14 px difference that no small tolerance absorbs. The vertical rules at `x = 75`
and `x = 242` are both continuous over `y = 121–241`, so they pair cleanly.

### 2.4 Step 3 — confirm with a tolerant horizontal check

```python
def has_horizontal(y, x1, x2):
    span = x2 - x1
    for dy in range(-4, 5):
        for left, right in horizontals.get(y + dy, ()):
            overlap = min(right, x2) - max(left, x1)
            if overlap >= 0.85 * span:
                return True
    return False
```

The **85 %** threshold is load-bearing:

- the correct pairing (75→242) is confirmed by the clipped bottom rule at
  `89–244`, which overlaps **92 %** of the span → accepted;
- a wrong pairing that spans two adjacent swatches (75→327) is only covered
  **67 %** by the top rule → rejected.

A looser 60 % threshold merged Limestone+Dolomite into one 252 px box; 85 %
separates them.

### 2.5 Step 4 — de-duplicate by area

A solid swatch (coal) makes every interior column a valid "vertical rule", so
pairing produces dozens of nested slices of the same box. Keep the largest and
drop anything that overlaps it:

```python
found.sort(key=lambda r: -((r[2]-r[0]) * (r[3]-r[1])))   # biggest first
unique = []
for rect in found:
    if not any(overlaps(rect, other) for other in unique):
        unique.append(rect)
return sorted(unique, key=lambda r: (r[1], r[0]))        # reading order
```

### 2.6 Step 5 — read the label

The name sits to the right of the swatch, bounded by the next swatch on the
same row:

```python
label_right = min(image.width, x2 + 220)
for other_x1, other_y1, _, _ in swatches:
    if other_x1 > x2 and abs(other_y1 - y1) < 20:
        label_right = min(label_right, other_x1 - 4)

name = _ocr_name(image.crop((x2 + 4, y1, label_right, y2)))
```

OCR is RapidOCR in **recognition-only** mode (`use_det=False, use_cls=False,
use_rec=True`) — the box is already known, so running the detector again is
slower and less reliable on tight single-line crops. If OCR returns nothing the
swatch is named `swatch_<x>_<y>`.

The swatch itself is cropped 2 px inside its border before describing it, so the
border does not pollute the texture statistics.

### 2.7 Measured behaviour

| legend | swatches found | expected | notes |
|---|---|---|---|
| gb506#2_md | **25** | 25 | clean 5×5 grid, all correct |
| d-582 log01 | **3** | 10 | borders broken by the pattern |

The d-582 legend is the known failure case: several swatches have both their
top/bottom *and* one vertical rule interrupted, so no rectangle can be closed.
The function warns when it finds fewer than 3 swatches and suggests `--patterns`
instead.

---

## 3. How the two combine

`segment_column()` runs four stages:

### 3.1 Straighten (per-band border detection)

The scanned column is **skewed** — its rules drift ~52 px left over 88 000 px:

| y | left rule | right rule |
|---|---|---|
| 0 | x = 52–59 | x = 289–292 |
| 20 000 | x = 41–48 | x = 278–281 |
| 40 000 | x = 22–29 | x = 259–262 |
| 80 000 | x = 0–7 | x = 236–239 |

So the rules are found **per band**, not once globally:

```python
density = band.mean(axis=0)
solid   = np.flatnonzero(density > 0.80)     # RULE_DENSITY
left    = end of the first solid run
right   = start of the last solid run
```

If a band has no clear rules (a solid coal interval), the previous band's
borders are reused.

### 3.2 Describe

Every band (100 px tall, 10 px step) is cut inside its rules and reduced to the
7-element descriptor from [§1.4](#14-step-3--the-7-element-descriptor).

### 3.3 Detect boundaries

Per-band labels flicker, so boundaries are found from **change points** instead:

```python
for i in range(window, count - window):
    above = features[i - window : i].mean(axis=0)
    below = features[i : i + window].mean(axis=0)
    difference[i] = norm((above - below) * FEATURE_WEIGHTS)

threshold = difference.mean() + sensitivity * difference.std()
peaks     = local maxima of difference above threshold
peaks     = non-maximum suppression with radius `window`
```

`--sensitivity` (default 1.0) is the knob: lower finds more boundaries.

### 3.4 Classify

Each finished segment is classified using the **mean descriptor of the whole
segment**, not a single band — which is why the confidence values are stable.

`_merge_short()` then absorbs intervals shorter than `--min-interval` into the
taller neighbour and re-joins neighbours that ended up with the same name.

---

## 4. Tuning knobs

| flag | default | effect |
|---|---|---|
| `--band` | 100 | texture window height (px) |
| `--step` | 10 | vertical step between windows |
| `--window` | 8 | bands compared on each side of a boundary |
| `--sensitivity` | 1.0 | boundary threshold in std devs (lower = more) |
| `--min-interval` | 30 | shorter intervals are absorbed |
| `--min-confidence` | 0.0 | below this the label becomes `unknown` |
| `--filter-names` | off | drop non-lithology legend entries |
| `--legend` | — | build the library from the same document (best) |

---

## 5. Known limitations

1. **Library coverage.** 18 of 51 wells have no pshhYolo legend, including all
   the wide d-582 columns. Matching those against other wells' patterns is
   unreliable. Use `--legend` for those.

2. **Broken legend borders.** The d-582 legend defeats `detect_swatches()`
   (3/10). A more tolerant detector — or hand-supplied swatches — is needed.

3. **Similar textures.** `Sand` vs `Medium sand` vs `Fine sandstone` are
   genuinely hard to separate at this descriptor resolution; expect them to be
   confused. The `runner_up` field reports what the second choice was.

4. **Density does not transfer.** The column draws denser than the legend, so
   density is weighted low (0.3) and only trusted for solid fills.

5. **Boundaries are more reliable than names.** The change-point stage is
   stable and contiguous; the naming is only as good as the library.
