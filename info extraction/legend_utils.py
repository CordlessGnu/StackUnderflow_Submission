"""Shared legend OCR, name filtering, and box pairing (no legacy detector import)."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]
_OCR_ENGINE = None


class OCRError(RuntimeError):
    """A legend name crop could not be read."""


class LegendExtractionError(ValueError):
    """The detector/OCR produced no usable named legend swatches."""


NON_LITHOLOGY = {
    'not', 'available', 'casing', 'shoe', 'void', 'survey', 'core', 'cutting',
    'cuttings', 'sample', 'loss', 'show', 'others', 'other', 'test', 'fit',
    'deviation', 'sidewall', 'mud', 'gas', 'oil', 'water', 'total', 'depth',
    'date', 'company', 'well', 'field', 'hole', 'size', 'diameter', 'bit',
}
LITHOLOGY_WORDS = {
    'sand', 'sandstone', 'sandy', 'shale', 'shaly', 'clay', 'claystone',
    'silt', 'siltstone', 'mud', 'mudstone', 'lime', 'limestone', 'dolomite',
    'dolomitic', 'dolostone', 'chert', 'cherty', 'halite', 'salt', 'anhydrite',
    'gypsum', 'coal', 'lignite', 'marl', 'chalk', 'tuff', 'till', 'gravel',
    'conglomerate', 'breccia', 'granite', 'basalt', 'gneiss', 'schist', 'slate',
    'quartz', 'feldspar', 'mica', 'pyrite', 'glauconite', 'fossil', 'fossils',
    'shell', 'wackestone', 'packstone', 'grainstone', 'boundstone', 'calcareous',
    'siliceous', 'argillaceous', 'ferruginous', 'carbonate', 'evaporite',
    'igneous', 'metamorphic', 'sedimentary', 'andesite', 'gabbro', 'diorite',
    'rhyolite', 'dacite', 'syenite', 'tonalite', 'trachyte', 'peridotite',
    'serpentine', 'hornfels', 'marble', 'quartzite', 'amphibolite', 'phyllite',
    'pegmatite', 'felsic', 'mafic', 'dike', 'dyke', 'sill', 'vein', 'fault',
    'altered', 'alteration', 'sericitization', 'biotitization', 'chloritization',
    'silicification', 'dolomitization', 'ash', 'pumice', 'scoria', 'lapilli',
    'agglomerate', 'ignimbrite', 'welded', 'tephra', 'diatomite', 'bitumen',
    'bituminous', 'tar', 'pebble', 'pebbles', 'extrusive', 'intrusive',
    'foraminifera', 'obsidian', 'paleosol', 'phosphate', 'glassy', 'wash',
    'porcelanite', 'porcelaneous', 'calcilutite', 'calcarenite', 'calcirudite',
    'coquina', 'pyroclastic', 'pyroclastics', 'volcaniclastic', 'volcaniclastics',
    'clast', 'clasts', 'clastic', 'sediment', 'sediments', 'seds', 'hornblende',
    'sylvite', 'bentonite', 'trona', 'potash', 'zeolite', 'barite', 'celestite',
    'fluorite', 'phosphatic', 'gypsiferous', 'siderite', 'limonite', 'hematite',
    'magnetite', 'goethite', 'kaolinite', 'illite', 'smectite', 'chlorite',
    'cement', 'volcanic', 'volcanics', 'tuffaceous', 'micrite', 'sparite',
    'loess', 'diamictite', 'aragonite', 'calcite', 'peat', 'rubble', 'cavings',
}


def is_lithology_name(name: str) -> bool:
    words = set(re.findall(r'[a-z]+', name.lower()))
    if not any(len(word) >= 3 for word in words):
        return False
    if words & LITHOLOGY_WORDS:
        return True
    return not (words & NON_LITHOLOGY)


def ocr_name(image: Image.Image) -> str:
    global _OCR_ENGINE
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRError('Install rapidocr and onnxruntime for legend OCR') from exc
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    try:
        result = _OCR_ENGINE(np.asarray(image.convert('RGB')), use_det=False,
                             use_cls=False, use_rec=True)
    except Exception as exc:
        raise OCRError(f'OCR failed: {exc}') from exc
    return ' '.join(' '.join(getattr(result, 'txts', None) or ()).split())


def merge_overlapping_boxes(boxes: Sequence[Box], overlap_threshold: float = 0.3
                            ) -> tuple[list[Box], int]:
    def area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def intersection(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0.0, min(a[3], b[3]) - max(a[1], b[1]))

    result, merged = list(boxes), 0
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a, b = result[i], result[j]
                smaller = min(area(a), area(b))
                if smaller > 0 and intersection(a, b) / smaller > overlap_threshold:
                    result[i] = (min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3]))
                    result.pop(j)
                    merged += 1
                    changed = True
                    break
            if changed:
                break
    return result, merged


def pair_boxes(textures: Sequence[Box], names: Sequence[Box],
               min_v_overlap: float = 0.5, min_h_overlap: float = 0.5
               ) -> tuple[list[tuple[Box, Box]], list[Box], list[tuple[Box, Box]], str]:
    """Pair right/below labels; report nearest-only guesses separately."""
    if not names:
        return [], list(textures), [], 'none'
    width = lambda b: b[2] - b[0]
    height = lambda b: b[3] - b[1]
    cx = lambda b: (b[0] + b[2]) / 2
    cy = lambda b: (b[1] + b[3]) / 2
    ordered = sorted(textures, key=lambda b: (round(cy(b) / 10), b[0]))
    used, pairs, fallbacks, unmatched = set(), [], [], []
    right_hits = below_hits = 0
    for texture in ordered:
        best = None
        for i, name in enumerate(names):
            if i in used:
                continue
            overlap = min(texture[3], name[3]) - max(texture[1], name[1])
            reference = min(height(texture), height(name))
            gap = name[0] - texture[2]
            if reference <= 0 or overlap / reference < min_v_overlap or gap < -0.5 * min(width(texture), width(name)):
                continue
            score = abs(gap) + 0.5 * abs(cy(texture) - cy(name)) + (0.25 * abs(gap) if gap < 0 else 0)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((texture, names[best[1]]))
            right_hits += 1
            continue
        best = None
        for i, name in enumerate(names):
            if i in used:
                continue
            overlap = min(texture[2], name[2]) - max(texture[0], name[0])
            reference = min(width(texture), width(name))
            gap = name[1] - texture[3]
            if reference <= 0 or overlap / reference < min_h_overlap or gap < -0.5 * min(height(texture), height(name)):
                continue
            score = abs(gap) + 0.5 * abs(cx(texture) - cx(name)) + (0.25 * abs(gap) if gap < 0 else 0)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((texture, names[best[1]]))
            below_hits += 1
            continue
        nearest = None
        for i, name in enumerate(names):
            if i not in used:
                distance = ((cx(texture) - cx(name)) ** 2 + (cy(texture) - cy(name)) ** 2) ** 0.5
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, i)
        if nearest is None:
            unmatched.append(texture)
        else:
            used.add(nearest[1])
            fallbacks.append((texture, names[nearest[1]]))
    return pairs, unmatched, fallbacks, 'right' if right_hits >= below_hits else 'below'
