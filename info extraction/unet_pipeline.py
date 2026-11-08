"""TIFF-only extraction: the U-Net predicts the three bounding boxes, then the
normal two-input pipeline (pipeline.py) runs on those boxes, unchanged.

Example:
  python 'info extraction/unet_pipeline.py' document.tif --output crops/run [pipeline options]

Stage 0 (here): work/boxfinder.py finds interpreted_lithology_column, depth_column and
lithology_legend on the whole page and snaps their edges to printed lines. The boxes are
written to a temporary bbox JSON in the same format as Datasets/bbox_annotation.
Stages 1-6: pipeline.main() with that JSON; every pipeline option is passed through.
Afterwards run.json gets a "box_source" entry (model, predicted boxes, timing) and the
output folder gets predicted_boxes.png (the page with the predicted boxes drawn).
The pipeline itself copies the predicted JSON to <output>/annotation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / 'work'))

DEFAULT_UNET = ROOT / 'work/export/boxfinder_v0.2.pt'
STAGE = '0/6 U-Net region detection'
PREVIEW_MAX_H = 1600


def detect_regions(tiff: Path, weights: Path, snap: bool = True) -> dict:
    """Run the U-Net on one page. No work/cache reads or writes: uploaded pages can share names."""
    import torch
    from boxfinder import BoxFinder
    finder = BoxFinder(weights)
    detection = finder.detect(tiff, snap=snap, cache=False)
    detection['meta'] = dict(finder.meta)
    detection['device'] = str(finder.dev)
    del finder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # leave the GPU to the swatch detector
    return detection


def preview_image(detection: dict):
    """Model-size page with the predicted boxes drawn, squashed to at most PREVIEW_MAX_H rows."""
    import cv2
    from pages import COLORS, boxes_to_small
    page = detection['page'].copy()
    small = boxes_to_small(detection['boxes'], detection['width'], detection['height'])
    for key, box in small.items():
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(page, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), COLORS[key], 3)
    height = min(PREVIEW_MAX_H, page.shape[0])
    width = max(240, round(page.shape[1] * height / page.shape[0] * 4)) if height < page.shape[0] else page.shape[1]
    return cv2.resize(page, (min(width, page.shape[1]), height), interpolation=cv2.INTER_AREA)


def _is_empty_dir(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _write_failure(output: Path, tiff: Path, message: str) -> None:
    """Report a stage-0 failure in run.json, but only into a folder this run may use."""
    if not _is_empty_dir(output):
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='failed', error=message, inputs=dict(tiff=str(tiff), annotation=None),
                    stages={STAGE: dict(status='failed')}, warnings=[], box_source=dict(method='unet'))
    (output / 'run.json').write_text(json.dumps(manifest, indent=2))


def _record(output: Path, annotation: Path, detection: dict, weights: Path, seconds: float, preview) -> None:
    """Add the U-Net details to the pipeline's run.json, only if this run created it."""
    manifest_path = output / 'run.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return
    if Path(str(manifest.get('inputs', {}).get('annotation', ''))) != annotation.resolve():
        return  # not our run (e.g. the output folder already held another run)
    meta = detection['meta']
    manifest['box_source'] = dict(
        method='unet', model=str(weights), version=meta.get('version'),
        snap_mode=meta.get('snap_mode', 'xyL') if detection['snapped'] else None,
        bounding_boxes=detection['boxes'], raw_boxes=detection['raw_boxes'],
        image_size=[detection['width'], detection['height']], device=detection['device'],
        seconds=round(seconds, 3), preview='predicted_boxes.png',
        note='Boxes predicted from the image alone; check predicted_boxes.png. '
             'An unfamiliar layout can make the model pick the wrong column.')
    stages = manifest.get('stages')
    if isinstance(stages, dict):
        manifest['stages'] = {STAGE: dict(status='complete', seconds=round(seconds, 3)), **stages}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    if preview is not None:
        import cv2
        cv2.imwrite(str(output / 'predicted_boxes.png'), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tiff', type=Path)
    parser.add_argument('--output', type=Path, help='run directory; default crops/pipeline_output/<TIFF stem>')
    parser.add_argument('--unet-model', type=Path, default=DEFAULT_UNET, help='exported U-Net (.pt)')
    parser.add_argument('--no-snap', action='store_true', help='skip edge snapping of the predicted boxes')
    args, passthrough = parser.parse_known_args(argv)
    tiff = args.tiff.resolve()
    output = (args.output if args.output is not None else ROOT / 'crops/pipeline_output' / tiff.stem).resolve()
    weights = args.unet_model.resolve()
    if tiff.suffix.lower() not in ('.tif', '.tiff') or not tiff.is_file():
        parser.exit(1, f'Pipeline failed: TIFF not found or not .tif/.tiff: {tiff}\n')
    if not weights.is_file():
        parser.exit(1, f'Pipeline failed: U-Net model not found: {weights}\n')
    if not _is_empty_dir(output) and '--overwrite' not in passthrough:
        parser.exit(1, f'Pipeline failed: Output is not empty: {output}; choose another folder or use --overwrite\n')

    print(STAGE, flush=True)
    started = time.perf_counter()
    try:
        detection = detect_regions(tiff, weights, snap=not args.no_snap)
        detection['snapped'] = not args.no_snap
        preview = preview_image(detection)
    except Exception as exc:  # report it like a pipeline stage failure
        traceback.print_exc()
        _write_failure(output, tiff, f'U-Net region detection failed: {type(exc).__name__}: {exc}')
        parser.exit(1, f'Pipeline failed: U-Net region detection failed: {exc}\n')
    seconds = time.perf_counter() - started
    print(f"U-Net boxes ({detection['device']}, {seconds:.1f}s): {json.dumps(detection['boxes'])}", flush=True)

    with tempfile.TemporaryDirectory(prefix='unet_boxes_') as tmp:
        annotation = Path(tmp) / f'{tiff.stem}.json'
        annotation.write_text(json.dumps(dict(
            image_name=tiff.name, width=detection['width'], height=detection['height'], bbox_format='xyxy',
            coordinate_system='pixels, origin at top-left, x_max and y_max exclusive',
            source='unet', model=detection['meta'].get('version'),
            bounding_boxes=detection['boxes']), indent=2))
        import pipeline
        old_argv = sys.argv
        sys.argv = [str(HERE / 'pipeline.py'), str(tiff), str(annotation), '--output', str(output), *passthrough]
        try:
            pipeline.main()
        finally:
            sys.argv = old_argv
            _record(output, annotation, detection, weights, seconds, preview)


if __name__ == '__main__':
    main()
