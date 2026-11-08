"""Hybrid grouping and name assignment; no YOLO or OCR dependencies."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from interval_lithology import (LegendSwatch, _distances, _hybrid_groups,
                                label_intervals, write_results)
from texture_seam import SeamComparison, prepare_texture


class HybridMatchingTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((180, 80))
        a = np.full((180, 80, 3), [255, 255, 0], dtype=np.uint8)
        a[(y % 12 < 2) & (x % 14 < 10)] = 0
        b = np.full((180, 80, 3), [0, 255, 255], dtype=np.uint8)
        b[(x % 12 < 2) & (y % 14 < 10)] = 0
        self.a, self.b = a, b
        self.legend = [LegendSwatch('Sandstone', Image.fromarray(a)),
                       LegendSwatch('Limestone', Image.fromarray(b))]

    def test_first_two_intervals_are_compared_not_assumed_different(self):
        image = np.vstack([self.a, self.a, self.b, self.a])
        result = label_intervals(image, [(i * 180, (i + 1) * 180) for i in range(4)], self.legend)
        self.assertEqual([r.cluster_id for r in result.intervals], ['C001', 'C001', 'C002', 'C001'])
        self.assertEqual([r.lithology for r in result.intervals], ['Sandstone', 'Sandstone', 'Limestone', 'Sandstone'])
        self.assertTrue(all(r.match_seam_status == 'same' for r in result.intervals))
        self.assertTrue(all(r.match_method == 'hybrid' for r in result.intervals))

    def test_representatives_and_expensive_work_are_bounded(self):
        image = np.vstack([np.roll(self.a, shift, axis=0) for shift in range(16)])
        result = label_intervals(image, [(i * 180, (i + 1) * 180) for i in range(16)], self.legend)
        self.assertEqual(len(result.clusters), 1)
        self.assertLessEqual(len(result.clusters[0]['representative_indices']), 3)
        stats = result.diagnostics
        self.assertLessEqual(stats['group_seam_evaluations'], 16 * (3 * 3 + 3 * 3))
        self.assertLessEqual(stats['legend_seam_evaluations'], len(result.clusters) * 3 * 2 * 3)
        self.assertGreater(stats['seam_evaluations'], 0)

    def test_descriptor_mode_remains_available_as_a_baseline(self):
        result = label_intervals(self.a, [(0, 180)], self.legend, matcher='descriptor')
        row = result.intervals[0]
        self.assertEqual(row.lithology, 'Sandstone')
        self.assertEqual(row.match_method, 'descriptor')
        self.assertIsNone(row.match_seam_score)
        self.assertEqual(result.diagnostics['seam_evaluations'], 0)

    def test_short_context_does_not_inherit_a_confident_name(self):
        short = np.full((2, 80, 3), [255, 255, 0], dtype=np.uint8)
        image = np.vstack([self.a, short])
        result = label_intervals(image, [(0, 180), (180, 182)], self.legend)
        row = result.intervals[1]
        self.assertEqual(row.grouping_status, 'uncertain')
        self.assertEqual(row.lithology, row.runner_up)
        self.assertTrue(row.lithology_fallback)
        self.assertNotEqual(row.match_status, 'matched')
        self.assertNotEqual(result.intervals[0].cluster_id, row.cluster_id)

    def test_seam_rejection_overrides_identical_descriptor_shortlist(self):
        prepared = [prepare_texture(self.a) for _ in range(3)]
        verifier = Mock()
        verifier.compare.return_value = SeamComparison(.4, .01, 'different', 'test_seam_change', (8, 16), (.4, .4))
        groups, _ = _hybrid_groups(prepared, [180] * 3, .2, verifier, 3, 3, .2)
        self.assertEqual(len(groups), 3)
        self.assertGreater(verifier.compare.call_count, 0)

    def test_uncertain_seam_does_not_force_membership(self):
        prepared = [prepare_texture(self.a) for _ in range(3)]
        verifier = Mock()
        verifier.compare.return_value = SeamComparison(.22, .01, 'uncertain', 'borderline', (8, 16), (.22, .22))
        groups, states = _hybrid_groups(prepared, [180] * 3, .2, verifier, 3, 3, .2)
        self.assertEqual(len(groups), 3)
        self.assertEqual(states[1]['status'], 'uncertain')

    def test_groups_cannot_merge_only_because_a_third_interval_matches_both(self):
        prepared = [prepare_texture(self.a) for _ in range(3)]
        verifier = Mock()
        def compare(a, b, **kwargs):
            same = prepared[2] in (a, b)
            return SeamComparison(.01 if same else .4, .01,
                                  'same' if same else 'different', 'test', (8, 16), (.01, .01))
        verifier.compare.side_effect = compare
        groups, states = _hybrid_groups(prepared, [180] * 3, .2, verifier, 3, 3, .2)
        self.assertEqual(len(groups), 3)
        self.assertEqual(states[2]['status'], 'uncertain')

    def test_complete_link_descriptor_guard_prevents_chaining(self):
        patches = [Image.new('RGB', (80, 180), (v, 255, 255)) for v in (0, 50, 100)]
        descriptors = np.stack([prepare_texture(p).descriptor for p in patches])
        distances = np.stack([_distances(d, descriptors) for d in descriptors])
        gate = (max(distances[0, 1], distances[1, 2]) + distances[0, 2]) / 2
        image = np.vstack([np.asarray(p) for p in patches])
        result = label_intervals(image, [(0, 180), (180, 360), (360, 540)], [], cluster_threshold=gate / 1.5)
        for group in result.clusters:
            indices = group['member_indices']
            self.assertTrue((distances[np.ix_(indices, indices)] <= gate + 1e-8).all())

    def test_different_names_with_identical_swatches_stay_ambiguous(self):
        legend = [LegendSwatch('Clay', Image.fromarray(self.a)), LegendSwatch('Shale', Image.fromarray(self.a))]
        result = label_intervals(self.a, [(0, 180)], legend, shortlist_size=1)
        self.assertEqual(result.intervals[0].match_status, 'ambiguous')
        self.assertEqual(result.intervals[0].lithology, result.intervals[0].runner_up)
        self.assertTrue(result.intervals[0].lithology_fallback)
        self.assertTrue(result.intervals[0].runner_up)

    def test_diagnostics_and_seam_evidence_are_exported(self):
        result = label_intervals(self.a, [(0, 180)], self.legend)
        with tempfile.TemporaryDirectory() as directory:
            write_results(result, Image.fromarray(self.a), directory, 'example', self.legend)
            data = json.loads((Path(directory) / 'example.json').read_text())
            self.assertEqual(data['diagnostics']['matcher'], 'hybrid')
            self.assertIn('seam_status', data['clusters'][0]['candidates'][0])
            self.assertIn('match_descriptor_distance', data['intervals'][0])

    def test_invalid_hybrid_settings_are_rejected(self):
        for kwargs in [dict(matcher='invalid'), dict(shortlist_size=0), dict(shortlist_size=11),
                       dict(max_representatives=0), dict(max_representatives=6), dict(seam_threshold=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                label_intervals(self.a, [(0, 180)], self.legend, **kwargs)


if __name__ == '__main__':
    unittest.main()
