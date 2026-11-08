"""Thin-bed and contact-localization regression tests for adaptive boundaries."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'info extraction'))
from texture_boundaries import (_fill_profile, _mean_windows, _periodic_color_cut,
                                 _refine_fill_cut, _row_features, _verify_cuts, detect_boundaries)


class BoundaryRefinementTests(unittest.TestCase):
    def make_bed(self, top=300, thickness=8, height=800):
        y, x = np.indices((height, 90))
        image = np.full((height, 90, 3), [128, 255, 128], dtype=np.uint8)
        image[(y % 10 < 2) & (x % 16 < 13)] = 0
        image[top:top + thickness] = [255, 255, 0]
        dots = (y % 6 < 2) & (x % 8 < 2)
        image[top:top + thickness][dots[top:top + thickness]] = 0
        return image

    def test_two_edges_of_beds_much_thinner_than_texture_context(self):
        for thickness in (2, 3, 4, 6, 8, 15, 30):
            with self.subTest(thickness=thickness):
                result = detect_boundaries(self.make_bed(thickness=thickness), window=96)
                self.assertEqual(result.boundaries, [300, 300 + thickness])

    def test_thin_bed_detection_is_stable_across_base_windows(self):
        for window in (24, 48, 96, 160):
            with self.subTest(window=window):
                result = detect_boundaries(self.make_bed(), window=window)
                self.assertEqual(result.boundaries, [300, 308])

    def test_near_image_edge_is_not_blocked_by_texture_guard(self):
        result = detect_boundaries(self.make_bed(top=4, thickness=8), window=96)
        self.assertEqual(result.boundaries, [4, 12])

    def test_several_short_intervals_retain_paired_edges(self):
        image = self.make_bed(top=300, thickness=4)
        image[312:315] = [255, 255, 0]
        image[321:329] = [255, 255, 0]
        result = detect_boundaries(image, window=96)
        self.assertEqual(result.boundaries, [300, 304, 312, 315, 321, 329])

    def test_explicit_minimum_spacing_is_respected(self):
        result = detect_boundaries(self.make_bed(), window=96, min_interval=12)
        edges = [0, *result.boundaries, 800]
        self.assertTrue(all(b - a >= 12 for a, b in zip(edges, edges[1:])))

    def test_colored_strokes_do_not_become_fill_changes(self):
        y, x = np.indices((800, 90))
        image = np.full((800, 90, 3), [255, 255, 0], dtype=np.uint8)
        image[(y % 12 < 2) & (x % 14 < 10)] = [255, 0, 0]
        self.assertEqual(detect_boundaries(image).boundaries, [])

    def test_horizontal_black_grid_rules_do_not_create_color_beds(self):
        image = self.make_bed()
        image[100:102] = 0
        image[500:502] = 0
        self.assertEqual(detect_boundaries(image, window=96).boundaries, [300, 308])

    def test_missing_fill_in_black_bed_does_not_create_an_interior_contact(self):
        image = np.full((800, 90, 3), [128, 255, 128], dtype=np.uint8)
        image[200:600] = 0
        image[600:] = [255, 255, 0]
        result = detect_boundaries(image, window=48)
        self.assertEqual(len(result.boundaries), 2)
        self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(result.boundaries, [200, 600])))

    def test_refinement_recovers_an_offset_contact(self):
        fill = np.full((600, 3), [0.5, 1, 0.5])
        fill[300:] = [1, 1, 0]
        for proposal in (285, 290, 307, 315):
            refined = _refine_fill_cut(proposal, fill, 0, 600, 24, 2, 0.07)
            self.assertEqual(refined, 300)

    def test_refinement_preserves_the_two_sides_of_a_short_bed(self):
        fill = np.full((600, 3), [0.5, 1, 0.5])
        fill[300:305] = [1, 1, 0]
        self.assertEqual(_refine_fill_cut(296, fill, 0, 305, 24, 2, 0.07), 300)
        self.assertEqual(_refine_fill_cut(310, fill, 300, 600, 24, 2, 0.07), 305)

    def test_refinement_does_not_move_a_color_free_texture_estimate(self):
        fill = np.ones((600, 3))
        self.assertEqual(_refine_fill_cut(290, fill, 0, 600, 24, 2, 0.07), 290)
        self.assertEqual(_refine_fill_cut(290, fill, 0, 600, 0, 2, 0.07), 290)

    def test_segment_verification_removes_identical_fragments(self):
        rgb = np.full((120, 30, 3), [1.0, 1.0, 0.0], dtype=np.float32)
        rows = _row_features(rgb, ignore_rules=True, use_color=True)
        fill, _ = _fill_profile(rgb, 2)
        self.assertEqual(_verify_cuts([30, 60, 90], rows, fill, 0.16, 0.10, True), [])

    def test_periodic_gray_drawing_lines_on_colored_fill_are_not_beds(self):
        image = np.full((1200, 60, 3), [155, 220, 230], dtype=np.uint8)
        for y in range(0, 1200, 6):
            image[y:y + 2] = [170, 170, 170]
        self.assertEqual(detect_boundaries(image).boundaries, [])

    def test_achromatic_crosshatching_does_not_trigger_color_rescue(self):
        y, x = np.indices((1200, 56))
        image = np.full((1200, 56, 3), 255, dtype=np.uint8)
        strokes = ((x + y) % 24 < 2) | ((x - y) % 24 < 2) | (y % 24 < 1)
        image[strokes] = [130, 130, 130]
        self.assertEqual(detect_boundaries(image).boundaries, [])

    def test_repeating_signed_color_edges_are_vetoed_but_isolated_beds_are_not(self):
        fill = np.full((400, 3), [0.6, 0.9, 0.9])
        for y in range(0, 400, 8):
            fill[y:y + 2] = [0.4, 0.4, 0.4]
        signatures = np.column_stack(_mean_windows(fill, 2))
        self.assertTrue(_periodic_color_cut(162, signatures, 24, 0.10))
        fill[:] = [0.5, 1, 0.5]
        fill[160:168] = [1, 1, 0]
        signatures = np.column_stack(_mean_windows(fill, 2))
        self.assertFalse(_periodic_color_cut(160, signatures, 24, 0.10))

    def test_result_exposes_scales_and_separate_color_scores(self):
        result = detect_boundaries(self.make_bed(), window=64)
        self.assertEqual(result.scales, (32, 64, 128, 256))
        self.assertEqual(result.min_interval, 2)
        self.assertEqual(result.color_scores.shape, result.scores.shape)
        self.assertTrue(np.isfinite(result.color_scores).all())

    def test_bad_new_parameters_are_rejected(self):
        image = self.make_bed()
        for kwargs in [dict(color_threshold=0), dict(color_threshold=float('nan')),
                       dict(refine_radius=-1), dict(refine_radius=2.5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_boundaries(image, **kwargs)


class OptionalRealBoundaryChecks(unittest.TestCase):
    """Development fixtures require the local supplied crops; skipped in bare CI."""
    def setUp(self):
        self.references = json.loads((ROOT / 'tests/fixtures/boundary_reference_sections.json').read_text())
        self.crops = ROOT / 'crops/interpreted_lithology_column'

    def test_reviewed_color_contacts(self):
        for section in self.references['color_sections']:
            path = self.crops / section['image']
            if not path.exists():
                self.skipTest('Local development crops are unavailable')
            with self.subTest(image=section['image']):
                result = detect_boundaries(path)
                lo, hi = section['range']
                predicted = [y for y in result.boundaries if lo < y < hi]
                expected = section['boundaries']
                self.assertEqual(len(predicted), len(expected))
                self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(predicted, expected)))

    def test_log05_control_boundaries_are_unchanged(self):
        control = self.references['regression_controls'][0]
        path = self.crops / control['image']
        if not path.exists():
            self.skipTest('Local development crops are unavailable')
        self.assertEqual(detect_boundaries(path).boundaries, control['boundaries'])


if __name__ == '__main__':
    unittest.main()
