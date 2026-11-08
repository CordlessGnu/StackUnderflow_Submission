"""Offline regression tests for clustering, legend matching and export."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from interval_lithology import (LegendSwatch, _distances, cluster_descriptors,
                                describe_patch, label_intervals, load_swatch_manifest,
                                read_intervals, save_swatch_manifest, write_results)
from texture_boundaries import detect_boundaries


class IntervalLithologyTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((180, 80))
        self.horizontal = (y % 12 < 2) & (x % 14 < 10)
        self.vertical = (x % 12 < 2) & (y % 14 < 10)
        self.sand = self.patch([255, 255, 0], self.horizontal)
        self.lime = self.patch([0, 255, 255], self.vertical)
        self.dolo = self.patch([255, 0, 255], self.horizontal)
        self.legend = [LegendSwatch('Sandstone', self.sand),
                       LegendSwatch('Limestone', self.lime),
                       LegendSwatch('Dolomite', self.dolo)]

    def patch(self, color, mask):
        rgb = np.full((180, 80, 3), color, dtype=np.uint8)
        rgb[mask] = 0
        return Image.fromarray(rgb)

    def column(self, patches):
        return np.vstack([np.asarray(p) for p in patches])

    def test_returning_patterns_share_ids_and_names(self):
        image = self.column([self.sand, self.lime, self.sand, self.dolo, self.lime])
        intervals = [(i * 180, (i + 1) * 180) for i in range(5)]
        result = label_intervals(image, intervals, self.legend)
        self.assertEqual([r.cluster_id for r in result.intervals], ['C001', 'C002', 'C001', 'C003', 'C002'])
        self.assertEqual([r.lithology for r in result.intervals],
                         ['Sandstone', 'Limestone', 'Sandstone', 'Dolomite', 'Limestone'])
        self.assertTrue(all(r.match_status == 'matched' for r in result.intervals))
        self.assertEqual([(r.top_y, r.bottom_y) for r in result.intervals], intervals)
        self.assertTrue(all(r.top_depth is None and r.base_depth is None for r in result.intervals))

    def test_same_orientation_different_rgb_is_not_one_cluster(self):
        result = label_intervals(self.column([self.sand, self.dolo]), [(0, 180), (180, 360)], self.legend)
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual([r.lithology for r in result.intervals], ['Sandstone', 'Dolomite'])

    def test_black_white_orientation_is_retained(self):
        horizontal = self.patch([255, 255, 255], self.horizontal)
        vertical = self.patch([255, 255, 255], self.vertical)
        legend = [LegendSwatch('horizontal', horizontal), LegendSwatch('vertical', vertical)]
        result = label_intervals(self.column([horizontal, vertical, horizontal]),
                                 [(0, 180), (180, 360), (360, 540)], legend)
        self.assertEqual([r.cluster_id for r in result.intervals], ['C001', 'C002', 'C001'])
        self.assertEqual([r.lithology for r in result.intervals], ['horizontal', 'vertical', 'horizontal'])

    def test_tall_intervals_do_not_squash_texture(self):
        tall = Image.fromarray(self.column([self.sand] * 8))
        a, b = describe_patch(self.sand), describe_patch(tall)
        self.assertLess(_distances(a, b[None, :])[0], 0.10)

    def test_duplicate_swatch_names_do_not_make_false_runner_up(self):
        legend = self.legend + [LegendSwatch('Sandstone', self.sand.copy())]
        result = label_intervals(self.sand, [(0, 180)], legend)
        self.assertEqual(result.intervals[0].lithology, 'Sandstone')
        self.assertNotEqual(result.intervals[0].runner_up, 'Sandstone')

    def test_identical_swatches_different_names_are_ambiguous(self):
        legend = [LegendSwatch('Clay', self.sand), LegendSwatch('Shale', self.sand.copy())]
        result = label_intervals(self.sand, [(0, 180)], legend)
        self.assertEqual(result.intervals[0].lithology, 'unknown')
        self.assertEqual(result.intervals[0].match_status, 'ambiguous')
        self.assertAlmostEqual(result.intervals[0].match_margin, 0)

    def test_unrelated_legend_does_not_force_a_lithology(self):
        result = label_intervals(self.sand, [(0, 180)], [LegendSwatch('Limestone', self.lime)])
        self.assertEqual(result.intervals[0].lithology, 'unknown')
        self.assertEqual(result.intervals[0].match_status, 'poor_match')

    def test_blank_is_not_assigned_a_rock(self):
        blank = Image.new('RGB', (80, 180), 'white')
        result = label_intervals(blank, [(0, 180)], self.legend)
        self.assertEqual(result.intervals[0].match_status, 'blank')
        self.assertEqual(result.intervals[0].lithology, 'unknown')

    def test_without_legend_clustering_still_works(self):
        result = label_intervals(self.column([self.sand, self.sand]), [(0, 180), (180, 360)], [])
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.intervals[0].match_status, 'no_legend')
        self.assertTrue(result.warnings)

    def test_no_single_linkage_chaining(self):
        colors = [Image.new('RGB', (80, 180), (v, 255, 255)) for v in (0, 50, 100)]
        descriptors = np.stack([describe_patch(p) for p in colors])
        distances = np.stack([_distances(d, descriptors) for d in descriptors])
        threshold = (max(distances[0, 1], distances[1, 2]) + distances[0, 2]) / 2
        self.assertLess(distances[0, 1], threshold)
        self.assertLess(distances[1, 2], threshold)
        groups = cluster_descriptors(descriptors, [180, 180, 180], threshold)
        self.assertGreater(len(groups), 1)
        for members in groups:
            self.assertTrue((distances[np.ix_(members, members)] <= threshold + 1e-8).all())

    def test_ids_are_deterministic(self):
        image = self.column([self.sand, self.lime, self.sand])
        intervals = [(0, 180), (180, 360), (360, 540)]
        first = label_intervals(image, intervals, self.legend)
        second = label_intervals(image, intervals, self.legend)
        self.assertEqual([r.cluster_id for r in first.intervals], [r.cluster_id for r in second.intervals])

    def test_manifest_roundtrip_and_rgb_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_swatch_manifest(self.legend, directory)
            loaded = load_swatch_manifest(path)
            self.assertEqual([s.name for s in loaded], [s.name for s in self.legend])
            np.testing.assert_equal(np.asarray(loaded[0].image), np.asarray(self.sand))

    def test_exports_preserve_required_fields_and_cluster_ids(self):
        result = label_intervals(self.sand, [(0, 180)], self.legend)
        with tempfile.TemporaryDirectory() as directory:
            write_results(result, self.sand, directory, 'example')
            path = Path(directory)
            with (path / 'example.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['cluster_id'], 'C001')
            self.assertEqual(rows[0]['lithology'], 'Sandstone')
            self.assertEqual(rows[0]['top_depth'], '')
            self.assertEqual(read_intervals(path / 'example.csv'), [(0, 180)])
            data = json.loads((path / 'example.json').read_text())
            self.assertEqual(data['clusters'][0]['interval_count'], 1)
            self.assertTrue((path / 'example.png').exists())

    def test_boundary_result_can_feed_label_stage_directly(self):
        image = self.column([self.sand, self.lime, self.sand])
        boundaries = detect_boundaries(image, window=36)
        result = label_intervals(image, boundaries, self.legend)
        self.assertEqual([r.lithology for r in result.intervals], ['Sandstone', 'Limestone', 'Sandstone'])
        self.assertEqual(result.intervals[0].cluster_id, result.intervals[-1].cluster_id)

    def test_empty_intervals(self):
        result = label_intervals(self.sand, [], self.legend)
        self.assertEqual(result.intervals, [])
        self.assertEqual(result.clusters, [])

    def test_bad_inputs_are_rejected(self):
        for intervals in [[(10, 8)], [(-1, 5)], [(0, 181)], [(0, 100), (90, 180)], [(0.5, 100)]]:
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                label_intervals(self.sand, intervals, self.legend)
        for kwargs in [dict(cluster_threshold=-1), dict(max_match_distance=0),
                       dict(min_match_margin=-1), dict(margin=0.5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                label_intervals(self.sand, [(0, 180)], self.legend, **kwargs)
        with self.assertRaises(ValueError):
            label_intervals(self.sand, [(0, 180)], [LegendSwatch('', self.sand)])


if __name__ == '__main__':
    unittest.main()
