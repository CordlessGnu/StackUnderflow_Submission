"""Part A: pair JSONs with images, validate pairs, split by well, write a report.

Dataset files are only read, never modified. Everything is written to work/.
Usage:  work/.venv/Scripts/python work/prepare_data.py
"""
import csv
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
SEED = 42
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
IMG_EXT = (".tif", ".tiff")
BOXES = ("depth_column", "interpreted_lithology_column", "lithology_legend")


# ---------------------------------------------------------------- discovery
def discover():
    """Find JSONs and images on disk and inside zips (excluding work/ and .git)."""
    jsons, images = [], []  # jsons: (ref, data) ; images: ref (str path, or "zip::member")
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in ("work", ".git") or not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(".json"):
            jsons.append((rel.as_posix(), json.loads(p.read_text(encoding="utf-8"))))
        elif low.endswith(IMG_EXT):
            images.append(rel.as_posix())
        elif low.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for m in z.namelist():
                    ml = m.lower()
                    if ml.endswith(".json"):
                        jsons.append((f"{rel.as_posix()}::{m}", json.loads(z.read(m))))
                    elif ml.endswith(IMG_EXT):
                        images.append(f"{rel.as_posix()}::{m}")
    return jsons, images


def basename(ref):
    return ref.split("::")[-1].split("/")[-1]


def stem_of(name):
    return re.sub(r"\.(tiff?|json)$", "", name, flags=re.I)


def open_image(ref):
    if "::" in ref:
        zpath, member = ref.split("::", 1)
        with zipfile.ZipFile(ROOT / zpath) as z:
            return Image.open(io.BytesIO(z.read(member)))
    return Image.open(ROOT / ref)


# ---------------------------------------------------------------- A1 pairing
def pair(jsons, images):
    by_name = defaultdict(list)
    by_stem = defaultdict(list)
    for ref in images:
        by_name[basename(ref)].append(ref)
        by_stem[stem_of(basename(ref))].append(ref)
    pairs, unmatched_json, notes = [], [], []
    used = set()
    for jref, data in jsons:
        name = data.get("image_name", "")
        hit, how = by_name.get(name, []), "exact"
        if not hit:
            hit, how = by_stem.get(stem_of(name), []), "stem"
        if not hit:
            hit, how = [r for r in images if basename(r).endswith(name)], "suffix"
        if not hit:
            unmatched_json.append((jref, name))
            continue
        if len(hit) > 1:
            notes.append(f"{jref}: {len(hit)} candidate images, took {hit[0]}")
        pairs.append({"json": jref, "image": hit[0], "match": how, "data": data})
        used.add(hit[0])
    unmatched_img = [r for r in images if r not in used]
    return pairs, unmatched_json, unmatched_img, notes


# ---------------------------------------------------------------- A2 checks
def mode_kind(mode):
    if mode == "1":
        return "1bit"
    if mode in ("L", "I;16", "I", "F", "LA"):
        return "grey"
    return "colour"  # RGB, RGBA, P, CMYK, YCbCr ...


def check(p):
    d = p["data"]
    problems = []
    im = open_image(p["image"])  # header only, pixels are not decoded
    w, h = im.size
    p.update(width=w, height=h, pil_mode=im.mode, mode=mode_kind(im.mode))
    if im.mode == "P":  # palette images: grey if the palette is grey
        pal = im.getpalette() or []
        rgb = [pal[i:i + 3] for i in range(0, len(pal), 3)]
        if rgb and all(r == g == b for r, g, b in rgb):
            p["mode"] = "grey"
    if (d.get("width"), d.get("height")) != (w, h):
        problems.append(f"size mismatch: json {d.get('width')}x{d.get('height')} vs image {w}x{h}")
    bb = d.get("bounding_boxes") or {}
    for k in BOXES:
        b = bb.get(k)
        if b is None:
            problems.append(f"missing box {k}")
            continue
        if len(b) != 4:
            problems.append(f"{k} has {len(b)} values")
            continue
        x1, y1, x2, y2 = b
        if not (x1 < x2 and y1 < y2):
            problems.append(f"{k} not ordered {b}")
        if not (0 <= x1 and 0 <= y1 and x2 <= w and y2 <= h):
            problems.append(f"{k} outside image {b} (image {w}x{h})")
    p["problems"] = problems
    return p


# ---------------------------------------------------------------- A3 grouping / split
def group_of(stem):
    """Well id. API-numbered names -> first 10 digits (state+county+well, so
    sidetracks/bypasses of one well share a group). Other names -> first two
    underscore tokens, lowercased (conservative: may merge neighbouring wells)."""
    m = re.match(r"^(\d{10})", stem)
    if m:
        return m.group(1)
    return "_".join(stem.split("_")[:2]).lower().strip()


def split_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    names = sorted(groups)
    n = len(rows)
    colour_frac = sum(r["mode"] != "1bit" for r in rows) / n
    tall_cut = sorted(r["height"] for r in rows)[int(0.75 * n)]  # top quartile height = "very tall"
    tall_frac = sum(r["height"] >= tall_cut for r in rows) / n
    rng = random.Random(SEED)
    best = None
    for _ in range(5000):  # random group assignments, keep the most balanced one
        order = names[:]
        rng.shuffle(order)
        assign, count = {}, Counter()
        for g in order:  # greedy: put group where the page deficit is largest
            s = max(TARGET, key=lambda s: TARGET[s] * n - count[s])
            assign[g] = s
            count[s] += len(groups[g])
        cost = 0.0
        for s in TARGET:
            rs = [r for g in names if assign[g] == s for r in groups[g]]
            if not rs:
                cost += 100
                continue
            cost += abs(len(rs) - TARGET[s] * n) / n * 10
            cost += abs(sum(r["mode"] != "1bit" for r in rs) / len(rs) - colour_frac)
            cost += abs(sum(r["height"] >= tall_cut for r in rs) / len(rs) - tall_frac)
        if best is None or cost < best[0]:
            best = (cost, assign)
    for r in rows:
        r["split"] = best[1][r["group"]]
    return tall_cut


# ---------------------------------------------------------------- report
def write_report(rows, bad, unmatched_json, unmatched_img, notes, match_counts, tall_cut):
    L = ["# Data report (Part A)", ""]
    L += [f"Seed {SEED}. Split by well group, target 70/15/15 by page count.", ""]
    L += ["## Pairing", "",
          f"- JSON files found: {sum(match_counts.values()) + len(unmatched_json)}",
          f"- Paired: {sum(match_counts.values())} (match type: {dict(match_counts)})",
          f"- JSON without image: {len(unmatched_json)}"]
    L += [f"  - `{j}` (image_name `{n}`)" for j, n in unmatched_json]
    L += [f"- Images without JSON: {len(unmatched_img)}"]
    L += [f"  - `{i}`" for i in unmatched_img]
    L += [f"- Note: {x}" for x in notes]
    L += ["", "## Validation", "",
          f"- Bad pairs excluded: {len(bad)}"]
    L += [f"  - `{b['stem']}`: " + "; ".join(b["problems"]) for b in bad]
    L += [f"- Valid pairs used for splitting: {len(rows)}", ""]

    L += ["## Splits", "",
          "| split | pages | groups | 1-bit | grey | colour | very tall (h >= %d) | median h | max h | median w |" % tall_cut,
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in TARGET:
        rs = [r for r in rows if r["split"] == s]
        hs = sorted(r["height"] for r in rs)
        ws = sorted(r["width"] for r in rs)
        mc = Counter(r["mode"] for r in rs)
        L.append(f"| {s} | {len(rs)} ({len(rs) / len(rows):.0%}) | {len({r['group'] for r in rs})} | "
                 f"{mc['1bit']} | {mc['grey']} | {mc['colour']} | {sum(h >= tall_cut for h in hs)} | "
                 f"{hs[len(hs) // 2]} | {hs[-1]} | {ws[len(ws) // 2]} |")
    bit_groups = sorted({r["group"] for r in rows if r["mode"] == "1bit"})
    L += ["", "## Balance notes", "",
          f"- 1-bit pages all come from {len(bit_groups)} group(s): {', '.join(f'`{g}`' for g in bit_groups)}. "
          "A group never crosses splits, so val/test can only hold 1-bit pages if that whole group goes there. "
          "Here it is in train, which means **val and test contain no 1-bit pages**; scores on them say nothing "
          "about 1-bit scans.",
          "- No greyscale pages exist; every other page is colour.",
          "- Very tall pages (top quartile by height) are spread across all three splits, but the single tallest "
          "page is in test, so test resizing/memory must handle it.",
          "- Grouping by API well id merges sidetracks/bypasses (e.g. `6081140580` has 4 logs). Different wells in "
          "the same block (e.g. three GC743 wells, MC776-5 and MC776-6) stay separate groups, so some "
          "operator/style similarity across splits remains.",
          "- Validation only reads the image header (size, mode); pixel data was not decoded here."]
    L += ["", "## Groups per split", ""]
    for s in TARGET:
        gs = Counter(r["group"] for r in rows if r["split"] == s)
        L.append(f"- **{s}**: " + ", ".join(f"`{g}` ({c})" for g, c in sorted(gs.items())))
    L += ["", "## All pages", "", "| split | group | stem | mode | width | height |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (list(TARGET).index(r["split"]), r["group"], r["stem"])):
        L.append(f"| {r['split']} | `{r['group']}` | {r['stem']} | {r['pil_mode']} | {r['width']} | {r['height']} |")
    (WORK / "data_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    jsons, images = discover()
    print(f"found {len(jsons)} json, {len(images)} images")
    pairs, unmatched_json, unmatched_img, notes = pair(jsons, images)
    match_counts = Counter(p["match"] for p in pairs)

    with open(WORK / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["json", "image", "match"])
        for p in pairs:
            w.writerow([p["json"], p["image"], p["match"]])

    for p in pairs:
        p["stem"] = stem_of(basename(p["image"]))
        check(p)
    bad = [p for p in pairs if p["problems"]]
    rows = [p for p in pairs if not p["problems"]]

    print("\nfile-name patterns (proposed group = well id):")
    for p in sorted(rows, key=lambda p: p["stem"]):
        p["group"] = group_of(p["stem"])
        print(f"  {p['group']:<22} {p['stem']}")

    tall_cut = split_groups(rows)
    with open(WORK / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "json", "image", "split", "group", "width", "height", "mode"])
        for r in sorted(rows, key=lambda r: r["stem"]):
            w.writerow([r["stem"], r["json"], r["image"], r["split"], r["group"], r["width"], r["height"], r["mode"]])

    write_report(rows, bad, unmatched_json, unmatched_img, notes, match_counts, tall_cut)
    print(f"\npairs {len(pairs)}, bad {len(bad)}, json w/o image {len(unmatched_json)}, image w/o json {len(unmatched_img)}")
    print(Counter(r["split"] for r in rows))


if __name__ == "__main__":
    main()
