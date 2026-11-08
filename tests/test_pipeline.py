"""Full-document integration tests; ML/OCR are mocked or use reviewed caches."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'info extraction'))
from depth_distance import DepthAnchor, DepthCalibration
from interval_lithology import LegendSwatch, save_swatch_manifest
from legend_utils import (LegendExtractionError, is_lithology_name,
                           merge_overlapping_boxes, pair_boxes)
from pipeline import PipelineConfig, crop_document, run_pipeline


class DocumentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        y, x = np.indices((180, 80))
        sand = np.full((180, 80, 3), [255, 255, 0], dtype=np.uint8)
        sand[(y % 12 < 2) & (x % 14 < 10)] = 0
        lime = np.full((180, 80, 3), [0, 255, 255], dtype=np.uint8)
        lime[(x % 12 < 2) & (y % 14 < 10)] = 0
        self.swatches = [LegendSwatch('Sandstone', Image.fromarray(sand)),
                         LegendSwatch('Limestone', Image.fromarray(lime))]
        image = np.full((640, 240, 3), 255, dtype=np.uint8)
        image[40:580, 20:100] = np.vstack([sand, lime, sand])
        image[:180, 150:230] = sand
        self.original = image
        self.tiff = self.root / 'Example.tiff'
        Image.fromarray(image).save(self.tiff, compression='tiff_lzw')
        self.annotation = self.root / 'Example.json'
        self.data = dict(image_name='Example.tiff', width=240, height=640, bounding_boxes={
            'interpreted_lithology_column': [20, 40, 100, 580],
            'depth_column': [110, 20, 130, 620],
            'lithology_legend': [150, 0, 230, 180]})
        self.annotation.write_text(json.dumps(self.data))
        self.calibration = DepthCalibration(0.5, 1000, 600,
                                            [DepthAnchor(1000, 0), DepthAnchor(1150, 300),
                                             DepthAnchor(1300, 600)], [], 0, 0)
        self.output = self.root / 'result'

    def run_mocked(self, **kwargs):
        config = PipelineConfig(window=48, **kwargs)
        with patch('pipeline.extract_legend', return_value=(self.swatches, [])), \
             patch('pipeline.calibrate_depth_column', return_value=self.calibration):
            return run_pipeline(self.tiff, self.annotation, self.output, config=config)

    def test_native_crops_match_the_supplied_tiff_exactly(self):
        crops = crop_document(self.tiff, self.annotation, self.root / 'crops')
        for name, bbox in crops.boxes.items():
            with Image.open(crops.paths[name]) as image:
                left, top, right, bottom = bbox
                np.testing.assert_array_equal(np.asarray(image), self.original[top:bottom, left:right])
                self.assertEqual(image.size, (right - left, bottom - top))

    def test_global_pillow_guard_is_restored_on_success_and_failure(self):
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 1
            crop_document(self.tiff, self.annotation, self.root / 'crops')
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)
            with self.assertRaises(ValueError):
                crop_document(self.tiff, self.annotation, self.root / 'bad', max_pixels=10)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)
        finally:
            Image.MAX_IMAGE_PIXELS = previous

    def test_dimension_mismatch_is_rejected_before_cropping(self):
        self.data['height'] = 641
        self.annotation.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            crop_document(self.tiff, self.annotation, self.output)
        self.assertFalse(self.output.exists())

    def test_malformed_missing_reversed_and_outside_boxes_rejected(self):
        for box in [[], [1, 2, 3], [1, 2, 3, float('nan')], [100, 40, 20, 580],
                    [300, 50, 400, 100]]:
            self.data['bounding_boxes']['depth_column'] = box
            self.annotation.write_text(json.dumps(self.data))
            with self.subTest(box=box), self.assertRaises(ValueError):
                crop_document(self.tiff, self.annotation, self.output)
        del self.data['bounding_boxes']['depth_column']
        self.annotation.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            crop_document(self.tiff, self.annotation, self.output)

    def test_clipped_boxes_record_the_actual_origin(self):
        self.data['bounding_boxes']['depth_column'] = [110, -10, 130, 620]
        self.annotation.write_text(json.dumps(self.data))
        crops = crop_document(self.tiff, self.annotation, self.output)
        self.assertEqual(crops.boxes['depth_column'], (110, 0, 130, 620))
        self.assertTrue(crops.warnings)

    def test_wrong_page_and_non_tiff_rejected(self):
        with self.assertRaises(ValueError):
            crop_document(self.tiff, self.annotation, self.output, page=1)
        png = self.root / 'example.png'
        Image.fromarray(self.original).save(png)
        with self.assertRaises(ValueError):
            crop_document(png, self.annotation, self.output)

    def test_two_input_pipeline_preserves_ids_and_produces_document_coordinates(self):
        before = hashlib.sha256(self.tiff.read_bytes()).hexdigest()
        result = self.run_mocked()
        self.assertEqual(result.interval_count, 3)
        self.assertEqual(result.cluster_count, 2)
        self.assertEqual(result.matched_intervals, 3)
        with result.csv_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['cluster_id'] for r in rows], ['C001', 'C002', 'C001'])
        self.assertEqual([r['lithology'] for r in rows], ['Sandstone', 'Limestone', 'Sandstone'])
        self.assertEqual(int(rows[0]['top_y']), 40)
        self.assertEqual(int(rows[-1]['bottom_y']), 580)
        self.assertEqual(int(rows[0]['crop_top_y']), 0)
        self.assertEqual(int(rows[-1]['crop_bottom_y']), 540)
        self.assertEqual(float(rows[0]['top_depth']), 1010)
        self.assertEqual(float(rows[-1]['base_depth']), 1280)
        self.assertTrue(all(a['base_depth'] == b['top_depth'] for a, b in zip(rows, rows[1:])))
        self.assertEqual(hashlib.sha256(self.tiff.read_bytes()).hexdigest(), before)
        with Image.open(self.output / 'intervals.png') as image:
            self.assertEqual(image.height, 540)  # overlay is crop-local, even with document-coordinate CSV
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(len(manifest['stages']), 6)
        self.assertEqual(manifest['boundary_scales'], [24, 48, 96, 192])
        self.assertEqual(manifest['effective_min_interval'], 2)
        self.assertEqual(manifest['effective_refine_radius'], 24)
        self.assertEqual(manifest['matching_diagnostics']['matcher'], 'hybrid')
        final = json.loads((self.output / 'intervals.json').read_text())
        self.assertEqual(final['matching_diagnostics']['matcher'], 'hybrid')
        self.assertEqual(final['intervals'][0]['match_seam_status'], 'same')
        self.assertTrue((self.output / 'boundary_color_scores.npy').exists())
        self.assertTrue((self.output / 'classification/intervals.clusters.png').exists())

    def test_runner_up_fallback_is_not_counted_as_a_verified_match(self):
        self.swatches.append(LegendSwatch('Alternative sandstone', self.swatches[0].image))
        result = self.run_mocked()
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(result.fallback_intervals, 2)
        self.assertEqual(result.matched_intervals, 1)
        self.assertEqual(manifest['named_intervals'], 3)
        self.assertEqual(manifest['verified_intervals'], 1)
        self.assertEqual(manifest['fallback_intervals'], 2)
        self.assertEqual(manifest['unknown_intervals'], 0)
        final = json.loads((self.output / 'intervals.json').read_text())
        self.assertEqual(sum(r['lithology_fallback'] for r in final['intervals']), 2)

    def test_blank_intervals_are_empty_and_counted_separately(self):
        self.original[40:220, 20:100] = 255
        Image.fromarray(self.original).save(self.tiff, compression='tiff_lzw')
        result = self.run_mocked()
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(result.blank_intervals, 1)
        self.assertEqual(manifest['blank_intervals'], 1)
        self.assertEqual(manifest['named_intervals'], 2)
        self.assertEqual(manifest['unknown_intervals'], 0)
        rows = json.loads((self.output / 'intervals.json').read_text())['intervals']
        blank = next(r for r in rows if r['match_status'] == 'blank')
        self.assertEqual(blank['lithology'], '')
        self.assertFalse(blank['lithology_fallback'])

    def test_crop_coordinate_export_is_available(self):
        self.run_mocked(coordinate_system='crop')
        data = json.loads((self.output / 'intervals.json').read_text())
        self.assertEqual(data['intervals'][0]['top_y'], 0)
        self.assertEqual(data['intervals'][-1]['bottom_y'], 540)
        self.assertEqual(data['intervals'][0]['document_top_y'], 40)

    def test_failed_legend_degrades_explicitly_to_unknown(self):
        with patch('pipeline.extract_legend', side_effect=LegendExtractionError('no usable swatches')), \
             patch('pipeline.calibrate_depth_column', return_value=self.calibration):
            result = run_pipeline(self.tiff, self.annotation, self.output, config=PipelineConfig(window=48))
        self.assertEqual(result.interval_count, 3)
        self.assertEqual(result.matched_intervals, 0)
        data = json.loads((self.output / 'intervals.json').read_text())
        self.assertTrue(all(row['lithology'] == 'unknown' for row in data['intervals']))
        self.assertTrue(any('Legend unavailable' in w for w in result.warnings))

    def test_strict_legend_failure_is_logged_without_fake_final_output(self):
        with patch('pipeline.extract_legend', side_effect=LegendExtractionError('bad legend')):
            with self.assertRaises(LegendExtractionError):
                run_pipeline(self.tiff, self.annotation, self.output,
                             config=PipelineConfig(window=48, strict_legend=True))
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('legend', manifest['current_stage'])
        self.assertFalse((self.output / 'intervals.csv').exists())

    def test_depth_failure_is_logged_without_fake_depth_values(self):
        with patch('pipeline.extract_legend', return_value=(self.swatches, [])), \
             patch('pipeline.calibrate_depth_column', side_effect=ValueError('inconsistent depth scale')):
            with self.assertRaises(ValueError):
                run_pipeline(self.tiff, self.annotation, self.output, config=PipelineConfig(window=48))
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('depth', manifest['current_stage'])
        self.assertFalse((self.output / 'intervals.csv').exists())

    def test_reviewed_caches_bypass_model_and_ocr(self):
        legend = save_swatch_manifest(self.swatches, self.root / 'reviewed_legend')
        calibration = self.root / 'reviewed_depth.json'
        calibration.write_text(json.dumps(self.calibration.to_dict()))
        with patch('pipeline.extract_legend', side_effect=AssertionError('must not invoke YOLO')), \
             patch('pipeline.calibrate_depth_column', side_effect=AssertionError('must not invoke OCR')):
            result = run_pipeline(self.tiff, self.annotation, self.output,
                                  config=PipelineConfig(window=48, legend_manifest=legend, calibration=calibration))
        self.assertEqual(result.matched_intervals, 3)
        # The installed entry point also works outside the repository cwd.
        cli_output = self.root / 'cli_result'
        completed = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'info extraction/pipeline.py'), str(self.tiff),
             str(self.annotation), '--legend-manifest', str(legend), '--calibration', str(calibration),
             '--window', '48', '--output', str(cli_output)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((cli_output / 'intervals.csv').exists())

    def test_wrong_resolution_cache_is_rejected(self):
        calibration = self.root / 'bad_depth.json'
        data = self.calibration.to_dict()
        data['image_height'] = 700
        calibration.write_text(json.dumps(data))
        with patch('pipeline.extract_legend', return_value=(self.swatches, [])):
            with self.assertRaises(ValueError):
                run_pipeline(self.tiff, self.annotation, self.output,
                             config=PipelineConfig(window=48, calibration=calibration))
        self.assertFalse((self.output / 'intervals.csv').exists())

    def test_existing_output_requires_opt_in_and_is_archived_not_deleted(self):
        self.run_mocked()
        original = (self.output / 'intervals.csv').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_mocked()
        self.run_mocked(overwrite=True)
        self.assertEqual((self.root / 'result.previous-1/intervals.csv').read_bytes(), original)

    def test_output_cannot_move_or_overwrite_its_inputs(self):
        with self.assertRaises(ValueError):
            run_pipeline(self.tiff, self.annotation, self.root, config=PipelineConfig(overwrite=True))
        self.assertTrue(self.tiff.exists())
        self.assertTrue(self.annotation.exists())

    def test_config_errors_rejected(self):
        for kwargs in [dict(window=0), dict(min_interval=0), dict(margin=0.5), dict(page=-1),
                       dict(depth_samples=1), dict(coordinate_system='invalid'),
                       dict(boundary_threshold=float('nan')), dict(color_threshold=0),
                       dict(refine_radius=-1), dict(matcher='invalid'), dict(shortlist_size=0),
                       dict(max_representatives=0), dict(seam_threshold=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)


class LegendUtilityTests(unittest.TestCase):
    def test_right_below_and_missing_labels(self):
        texture = [(0, 0, 20, 20)]
        name_right = [(25, 5, 80, 15)]
        name_below = [(0, 25, 20, 35)]
        self.assertEqual(pair_boxes(texture, name_right)[0], [(texture[0], name_right[0])])
        self.assertEqual(pair_boxes(texture, name_below)[3], 'below')
        self.assertEqual(pair_boxes(texture, [])[1], texture)

    def test_overlapping_detections_merge_without_changing_unique_boxes(self):
        boxes, count = merge_overlapping_boxes([(0, 0, 20, 20), (5, 0, 25, 20), (40, 0, 60, 20)])
        self.assertEqual(count, 1)
        self.assertEqual(boxes, [(0, 0, 25, 20), (40, 0, 60, 20)])

    def test_lithology_name_filter(self):
        self.assertTrue(is_lithology_name('Sandstone'))
        self.assertFalse(is_lithology_name('Casing shoe'))
        self.assertFalse(is_lithology_name('1'))


if __name__ == '__main__':
    unittest.main()
