"""Synthetic regression tests; run: python -m unittest discover -s tests -v."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from texture_boundaries import detect_boundaries


class TextureBoundaryTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((1200, 100))
        self.horizontal = (y % 12 < 2) & (x % 14 < 10)
        self.vertical = (x % 12 < 2) & (y % 14 < 10)
        self.image = np.full((1200, 100, 3), 255, dtype=np.uint8)
        self.image[self.horizontal] = 0

    def assertCutsNear(self, actual, expected, tolerance=12):
        self.assertEqual(len(actual), len(expected), actual)
        for a, e in zip(actual, expected):
            self.assertLessEqual(abs(a - e), tolerance, actual)

    def test_repeating_pattern_is_not_many_intervals(self):
        self.assertEqual(detect_boundaries(self.image, window=48).boundaries, [])

    def test_orientation_change_and_return(self):
        image = self.image.copy()
        image[400:800] = 255
        image[400:800][self.vertical[400:800]] = 0
        result = detect_boundaries(image, window=48)
        self.assertCutsNear(result.boundaries, [400, 800])
        self.assertEqual(result.intervals[0][0], 0)
        self.assertEqual(result.intervals[-1][1], 1200)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(result.intervals, result.intervals[1:])))
        self.assertEqual(result.scores.shape, (1201,))

    def test_same_pattern_different_color(self):
        image = self.image.copy()
        image[600:] = np.where(self.horizontal[600:, :, None], 0, [255, 255, 0])
        self.assertEqual(detect_boundaries(image, window=48).boundaries, [])
        self.assertCutsNear(detect_boundaries(image, window=48, use_color=True).boundaries, [600])

    def test_blank_to_texture(self):
        image = self.image.copy()
        image[:400] = 255
        self.assertCutsNear(detect_boundaries(image, window=48).boundaries, [400])

    def test_solid_bed_is_not_suppressed_as_rule(self):
        image = self.image.copy()
        image[400:800] = 0
        self.assertCutsNear(detect_boundaries(image, window=48).boundaries, [400, 800])

    def test_thin_full_width_rule_does_not_split_same_texture(self):
        image = self.image.copy()
        image[600:602] = 0
        self.assertEqual(detect_boundaries(image, window=48).boundaries, [])

    def test_blank_and_short_images(self):
        for shape in [(1200, 100, 3), (3, 3, 3), (10, 20, 3)]:
            image = np.full(shape, 255, dtype=np.uint8)
            result = detect_boundaries(image)
            self.assertEqual(result.intervals, [(0, shape[0])])
            self.assertTrue(np.isfinite(result.scores).all())

    def test_grayscale_pil_and_rgba_inputs(self):
        for image in [self.image[:, :, 0], Image.fromarray(self.image),
                      np.dstack((self.image, np.full((1200, 100), 255, dtype=np.uint8)))]:
            self.assertEqual(detect_boundaries(image, window=48).boundaries, [])

    def test_parameters_are_validated(self):
        for kwargs in [dict(window=0), dict(window=4.5), dict(min_interval=0),
                       dict(margin=0.5), dict(threshold=-1), dict(threshold=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_boundaries(self.image, **kwargs)
        with self.assertRaises(ValueError):
            detect_boundaries(self.image.astype(float))


if __name__ == '__main__':
    unittest.main()
