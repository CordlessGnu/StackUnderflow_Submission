"""Cached local seam evidence: phase tolerance, uncertainty and bounded work."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from texture_seam import SeamVerifier, compare_interval_patches, prepare_texture


class TextureSeamTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((240, 80))
        self.horizontal = np.full((240, 80, 3), 255, dtype=np.uint8)
        self.horizontal[(y % 12 < 2) & (x % 14 < 10)] = 0
        self.vertical = np.full_like(self.horizontal, 255)
        self.vertical[(x % 12 < 2) & (y % 14 < 10)] = 0

    def test_identical_and_phase_shifted_patterns_are_same(self):
        for shift in (0, 3, 7, 11):
            with self.subTest(shift=shift):
                result = compare_interval_patches(self.horizontal, np.roll(self.horizontal, shift, axis=0))
                self.assertEqual(result.status, 'same')
                self.assertLessEqual(result.score, 0.20)
                self.assertGreaterEqual(len(result.scales), 2)

    def test_orientation_changes_are_not_rotation_invariant(self):
        self.assertEqual(compare_interval_patches(self.horizontal, self.vertical).status, 'different')

    def test_color_difference_is_retained(self):
        a = Image.new('RGB', (80, 240), 'yellow')
        b = Image.new('RGB', (80, 240), 'cyan')
        self.assertEqual(compare_interval_patches(a, b).status, 'different')

    def test_two_pixel_patch_is_uncertain_not_a_proven_match(self):
        a = Image.new('RGB', (80, 2), 'yellow')
        b = Image.new('RGB', (80, 240), 'yellow')
        result = compare_interval_patches(a, b)
        self.assertEqual(result.status, 'uncertain')
        self.assertEqual(result.reason, 'insufficient_texture_context')
        self.assertIsNone(result.score)

    def test_padding_or_width_upscaling_cannot_invent_native_evidence(self):
        prepared = prepare_texture(Image.new('RGB', (2, 1000), 'yellow'))
        self.assertLess(prepared.native_height, 8)
        self.assertEqual(SeamVerifier().compare(prepared, prepared).status, 'uncertain')

    def test_only_seam_features_are_used_not_full_boundary_detection(self):
        with patch('texture_boundaries.detect_boundaries', side_effect=AssertionError('no full scan')):
            self.assertEqual(compare_interval_patches(self.horizontal, self.horizontal).status, 'same')

    def test_prepared_cache_is_bounded_for_very_tall_beds(self):
        prepared = prepare_texture(Image.new('RGB', (80, 20000), 'yellow'))
        self.assertLessEqual(prepared.samples, 3)
        self.assertLessEqual(prepared.normalized_height, 128)
        self.assertEqual(prepared.descriptor.shape, (47,))
        self.assertTrue(all(len(values) <= 12 for values in prepared.heads.values()))
        self.assertTrue(np.isfinite(prepared.descriptor).all())

    def test_verifier_reuses_symmetric_comparisons_and_reclassifies_threshold(self):
        a = prepare_texture(self.horizontal)
        b = prepare_texture(np.roll(self.horizontal, 5, axis=0))
        verifier = SeamVerifier()
        first = verifier.compare(a, b)
        reverse = verifier.compare(b, a)
        strict = verifier.compare(a, b, threshold=1e-8)
        self.assertEqual(first.score, reverse.score)
        self.assertEqual(first.status, 'same')
        self.assertNotEqual(strict.status, 'same')
        self.assertEqual(verifier.evaluations, 1)
        self.assertEqual(verifier.cache_hits, 2)

    def test_blank_and_visible_pattern_are_not_equivalent(self):
        blank = Image.new('RGB', (80, 240), 'white')
        result = compare_interval_patches(blank, self.horizontal)
        self.assertEqual(result.status, 'different')
        self.assertEqual(result.reason, 'blank_vs_texture')

    def test_mixed_interval_is_not_confirmed_by_one_lucky_window(self):
        mixed = self.horizontal.copy()
        mixed[120:] = self.vertical[120:]
        result = compare_interval_patches(self.horizontal, mixed)
        self.assertNotEqual(result.status, 'same')

    def test_paths_pil_grayscale_and_arrays_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.png'
            Image.fromarray(self.horizontal).save(path)
            self.assertEqual(compare_interval_patches(path, Image.fromarray(self.horizontal)).status, 'same')
        self.assertEqual(compare_interval_patches(self.horizontal[:, :, 0], self.horizontal).status, 'same')

    def test_inputs_are_not_mutated(self):
        original = self.horizontal.copy()
        prepare_texture(self.horizontal)
        np.testing.assert_array_equal(self.horizontal, original)

    def test_bad_arguments_are_rejected(self):
        for kwargs in [dict(y_scale=0), dict(y_scale=float('nan')), dict(max_samples=0), dict(max_samples=8)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_texture(self.horizontal, **kwargs)
        with self.assertRaises(ValueError):
            prepare_texture(self.horizontal.astype(float))
        prepared = prepare_texture(self.horizontal)
        with self.assertRaises(ValueError):
            SeamVerifier().compare(prepared, prepared, threshold=-1)


if __name__ == '__main__':
    unittest.main()
