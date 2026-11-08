"""Offline depth OCR geometry, robust calibration, and final-output tests."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'info extraction'))
from depth_distance import (DepthAnchor, DepthCalibration, _read_tile, extract_depth_anchors,
                             fit_depth_calibration, load_calibration, parse_depth, save_calibration)
from final_output import (CropAlignment, alignment_from_annotation, attach_depths,
                           find_annotation, read_labelled_csv, write_final)


class DepthCalibrationTests(unittest.TestCase):
    def anchors(self):
        return [DepthAnchor(2000 + y * 0.1, y, text=str(2000 + y * 0.1))
                for y in np.linspace(100, 9900, 12)]

    def test_numeric_formats(self):
        for text, expected in [('1,610', 1610), ('23,520', 23520), ('23520', 23520),
                               ('6 910', 6910), ('6910.0', 6910), ('11879.5', 11879.5), ('100', 100)]:
            self.assertEqual(parse_depth(text)[0], expected)
        for text in ['23O00', 'depth 23520', '1,23', '-20', '12/34', 'NaN', '']:
            self.assertIsNone(parse_depth(text))

    def test_affine_fit(self):
        result = fit_depth_calibration(self.anchors(), 10000)
        self.assertAlmostEqual(result.slope, 0.1)
        self.assertAlmostEqual(result.intercept, 2000)
        self.assertAlmostEqual(result.depth_at(1000), 2100)
        self.assertLess(result.rmse_px, 1e-8)

    def test_ocr_outliers_are_rejected(self):
        anchors = self.anchors() + [DepthAnchor(25, 3500), DepthAnchor(99999, 8500)]
        result = fit_depth_calibration(anchors, 10000)
        self.assertEqual(len(result.anchors), 12)
        self.assertEqual(len(result.rejected), 2)
        self.assertAlmostEqual(result.depth_at(5000), 2500)

    def test_integer_track_preferred_to_secondary_decimal_track(self):
        anchors = self.anchors()
        anchors += [DepthAnchor(1900 + y * 0.095, y + 12, text='1900.0', is_decimal=True)
                    for y in np.linspace(100, 9800, 12)]
        result = fit_depth_calibration(anchors, 10000)
        self.assertAlmostEqual(result.slope, 0.1)
        self.assertTrue(all(not a.is_decimal for a in result.anchors))
        self.assertTrue(any('integer' in w for w in result.warnings))

    def test_decimal_only_scale_and_explicit_track_selection(self):
        decimals = [DepthAnchor(500.5 + y * 0.05, y, is_decimal=True)
                    for y in np.linspace(100, 9900, 12)]
        auto = fit_depth_calibration(decimals, 10000)
        self.assertAlmostEqual(auto.slope, 0.05)
        selected = fit_depth_calibration(self.anchors() + decimals, 10000, label_mode='decimal')
        self.assertAlmostEqual(selected.intercept, 500.5)

    def test_insufficient_or_decreasing_anchors_rejected(self):
        for points in [self.anchors()[:2], self.anchors()[::11] * 5,
                       [DepthAnchor(1000, 500)] * 4,
                       [DepthAnchor(2000 - y * 0.1, y) for y in (100, 1000, 5000, 9000)]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                fit_depth_calibration(points, 10000)

    def test_two_anchor_mode_is_explicit_and_warns(self):
        result = fit_depth_calibration([DepthAnchor(2000, 100), DepthAnchor(2900, 9100)],
                                        10000, min_anchors=2)
        self.assertTrue(any('two' in warning for warning in result.warnings))

    def test_short_span_cannot_calibrate_a_long_log(self):
        with self.assertRaises(ValueError):
            fit_depth_calibration([DepthAnchor(2000 + y * 0.1, y) for y in (100, 200, 300)], 10000)

    def test_changing_or_broken_scale_is_not_silently_accepted(self):
        points = [DepthAnchor(1000 + y * 0.1 + (100 if y >= 6000 else 0), y)
                  for y in range(0, 12000, 1000)]
        with self.assertRaises(ValueError):
            fit_depth_calibration(points, 12000)

    def test_rotation_and_resize_coordinates_return_to_original_crop(self):
        tile = Image.new('RGB', (40, 100), 'white')
        for angle in (0, 90, 270):
            # Original y=25. The OCR canvas has 12px padding, then 2x scaling.
            x, y = (20, 74) if angle == 0 else ((74, 20) if angle == 90 else (174, 20))
            polygon = np.array([[x - 3, y - 3], [x + 3, y - 3], [x + 3, y + 3], [x - 3, y + 3]])
            def engine(*args, **kwargs):
                return SimpleNamespace(txts=['1,234'], boxes=[polygon], scores=[0.99])
            result = _read_tile(tile, 1000, angle, engine, 0.8)
            self.assertAlmostEqual(result[0].y, 1025)
            self.assertEqual(result[0].depth, 1234)

    def test_tiled_ocr_spans_full_height_without_global_downsampling(self):
        polygon = np.array([[35, 220], [45, 220], [45, 228], [35, 228]])  # y=100 before pad/scale
        def engine(*args, **kwargs):
            return SimpleNamespace(txts=['1234'], boxes=[polygon], scores=[0.99])
        points, height, rotation = extract_depth_anchors(Image.new('RGB', (40, 5000), 'white'),
                                                         samples=3, tile_height=1000,
                                                         rotation=0, engine=engine)
        self.assertEqual(height, 5000)
        self.assertEqual(rotation, 0)
        self.assertEqual([a.y for a in points], [100, 2100, 4100])

    def test_automatic_orientation_uses_increasing_depth_order(self):
        def engine(image, **kwargs):
            if image.shape[0] > image.shape[1]:
                return SimpleNamespace(txts=[], boxes=[], scores=[])
            polygons = [np.array([[x - 3, 20], [x + 3, 20], [x + 3, 26], [x - 3, 26]])
                        for x in (224, 1824)]
            return SimpleNamespace(txts=['2000', '2100'], boxes=polygons, scores=[1, 1])
        points, _, angle = extract_depth_anchors(Image.new('RGB', (40, 5000), 'white'),
                                                  samples=3, tile_height=1000, engine=engine)
        self.assertEqual(angle, 90)
        self.assertEqual([p.y for p in points], [100, 900, 2100, 2900, 4100, 4900])

    def test_cached_calibration_roundtrip_preserves_dotted_filenames(self):
        calibration = fit_depth_calibration(self.anchors(), 10000)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'sample.pdf_depth'
            save_calibration(calibration, Image.new('RGB', (40, 10000), 'white'), prefix)
            loaded = load_calibration(Path(f'{prefix}.json'))
            self.assertAlmostEqual(loaded.depth_at(1234), calibration.depth_at(1234))
            self.assertTrue(Path(f'{prefix}.png').exists())


class FinalDepthOutputTests(unittest.TestCase):
    def setUp(self):
        self.calibration = DepthCalibration(0.2, 1000, 1000,
                                            [DepthAnchor(1000, 0), DepthAnchor(1200, 1000)],
                                            [], 0, 0)
        self.rows = [dict(top_y=20, bottom_y=40, cluster_id='C001', lithology='Shale',
                          match_distance='0.12', confidence='0.8')]

    def test_offsets_and_different_resize_scales_are_respected(self):
        alignment = CropAlignment(column_top=1000, depth_top=900, column_scale=2, depth_scale=0.5)
        rows = attach_depths(self.rows, self.calibration, alignment, 100, depth_unit='ft')
        self.assertEqual(rows[0]['document_top_y'], 1010)
        self.assertEqual(rows[0]['document_bottom_y'], 1020)
        self.assertEqual(rows[0]['top_depth'], 1011)
        self.assertEqual(rows[0]['base_depth'], 1012)
        self.assertEqual(rows[0]['confidence'], '0.8')
        self.assertEqual(rows[0]['match_distance'], '0.12')
        self.assertEqual(rows[0]['top_y'], 20)
        self.assertNotIn('top_depth', self.rows[0])  # original is not mutated

    def test_adjacent_boundaries_have_identical_depths(self):
        rows = [dict(top_y=0, bottom_y=100, cluster_id='A', lithology='Shale'),
                dict(top_y=100, bottom_y=200, cluster_id='B', lithology='Sandstone')]
        output = attach_depths(rows, self.calibration, CropAlignment(), 200)
        self.assertEqual(output[0]['base_depth'], output[1]['top_depth'])

    def test_extrapolation_is_flagged(self):
        calibration = DepthCalibration(0.2, 1000, 1000,
                                        [DepthAnchor(1020, 100), DepthAnchor(1180, 900)], [], 0, 0)
        rows = attach_depths(self.rows, calibration, CropAlignment(), 1000)
        self.assertTrue(rows[0]['depth_extrapolated'])

    def test_assumed_alignment_does_not_invent_document_coordinates(self):
        rows = attach_depths(self.rows, self.calibration,
                             CropAlignment(source='assumed', document_coordinates_known=False), 100)
        self.assertIsNone(rows[0]['document_top_y'])

    def test_annotation_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bbox.json'
            path.write_text(json.dumps(dict(width=1000, height=10000, bounding_boxes={
                'interpreted_lithology_column': [100, 500, 150, 1500],
                'depth_column': [10, 400, 30, 1600]})))
            alignment = alignment_from_annotation(path, (100, 2000), (10, 600))
            self.assertEqual(alignment.column_top, 500)
            self.assertEqual(alignment.depth_top, 400)
            self.assertEqual(alignment.column_scale, 2)
            self.assertEqual(alignment.depth_scale, 0.5)
            self.assertEqual(alignment.depth_y(20), 55)
            with self.assertRaises(ValueError):
                alignment_from_annotation(path, (200, 2000), (10, 600))

    def test_sanitized_annotation_lookup_is_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = dict(image_name='Example-Well 1.tif')
            (path / 'Example-Well 1.json').write_text(json.dumps(data))
            found = find_annotation('Example_Well_1.png', path)
            self.assertEqual(found.name, 'Example-Well 1.json')
            (path / 'example-well_1.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                find_annotation('Example_Well_1.png', path)

    def test_invalid_geometry_and_intervals_rejected(self):
        for kwargs in [dict(column_scale=0), dict(depth_scale=-1), dict(column_top=float('nan'))]:
            with self.assertRaises(ValueError):
                CropAlignment(**kwargs)
        with self.assertRaises(ValueError):
            attach_depths([dict(self.rows[0], bottom_y=2000)], self.calibration, CropAlignment(), 1000)

    def test_final_export_required_columns_and_annotations(self):
        rows = attach_depths(self.rows, self.calibration, CropAlignment(), 100)
        with tempfile.TemporaryDirectory() as directory:
            write_final(rows, Image.new('RGB', (40, 100), 'white'), self.calibration,
                         CropAlignment(), directory, 'example')
            path = Path(directory)
            with (path / 'example.csv').open() as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames[:6], ['top_y', 'bottom_y', 'cluster_id', 'lithology',
                                                         'top_depth', 'base_depth'])
                record = next(reader)
            self.assertEqual(record['confidence'], '0.8')
            self.assertEqual(float(record['top_depth']), 1004)
            data = json.loads((path / 'example.json').read_text())
            self.assertEqual(data['calibration']['slope'], 0.2)
            self.assertTrue((path / 'example.png').exists())
            loaded = read_labelled_csv(path / 'example.csv')
            self.assertEqual(loaded[0]['cluster_id'], 'C001')

    def test_cached_cli_joins_existing_outputs_without_running_ocr_or_yolo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new('RGB', (40, 1000), 'white').save(path / 'column.png')
            Image.new('RGB', (20, 1000), 'white').save(path / 'depth.png')
            (path / 'labels.csv').write_text('top_y,bottom_y,cluster_id,lithology,confidence\n0,1000,C001,Shale,0.8\n')
            (path / 'calibration.json').write_text(json.dumps(self.calibration.to_dict()))
            command = [sys.executable, '-B', str(ROOT / 'info extraction/final_output.py'),
                       str(path / 'column.png'), '--depth-column', str(path / 'depth.png'),
                       '--labelled-csv', str(path / 'labels.csv'), '--calibration', str(path / 'calibration.json'),
                       '--column-top', '0', '--depth-top', '0', '--output', str(path / 'out')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = read_labelled_csv(path / 'out/column.csv')
            self.assertEqual(float(rows[0]['base_depth']), 1200)
            self.assertEqual(rows[0]['confidence'], '0.8')


if __name__ == '__main__':
    unittest.main()
