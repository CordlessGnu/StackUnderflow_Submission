"""Requested label fallback preserves rejected-match evidence and geometry."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from depth_distance import DepthAnchor, DepthCalibration
from final_output import CropAlignment, attach_depths, read_labelled_csv, write_final
from interval_lithology import LegendSwatch, apply_runner_up_fallback, label_intervals


class RunnerUpFallbackTests(unittest.TestCase):
    def test_unknown_uses_second_candidate_and_retains_original_match_evidence(self):
        original = dict(lithology='unknown', runner_up='Shale', match_status='ambiguous',
                        match_distance=0.12, match_margin=0.04, match_seam_status='uncertain')
        result = apply_runner_up_fallback(original)
        self.assertEqual(result['lithology'], 'Shale')
        self.assertTrue(result['lithology_fallback'])
        self.assertAlmostEqual(result['fallback_distance'], 0.16)
        self.assertEqual(result['match_status'], 'ambiguous')
        self.assertEqual(result['match_seam_status'], 'uncertain')
        self.assertEqual(result['match_distance'], 0.12)
        self.assertEqual(original['lithology'], 'unknown')
        self.assertNotIn('lithology_fallback', original)

    def test_blank_status_clears_labels_and_previous_fallbacks(self):
        for label in ('unknown', '', 'Shale'):
            with self.subTest(label=label):
                row = dict(lithology=label, runner_up='Sandstone', match_status=' Blank ',
                           lithology_fallback='True', fallback_distance=.4)
                result = apply_runner_up_fallback(row)
                self.assertEqual(result['lithology'], '')
                self.assertFalse(result['lithology_fallback'])
                self.assertIsNone(result['fallback_distance'])
                self.assertEqual(result['runner_up'], 'Sandstone')
                self.assertEqual(apply_runner_up_fallback(result), result)

    def test_blank_cells_survive_cached_csv_and_final_export(self):
        calibration = DepthCalibration(.5, 1000, 200,
                                        [DepthAnchor(1000, 0), DepthAnchor(1100, 200)], [], 0, 0)
        row = dict(top_y=0, bottom_y=200, cluster_id='C001', lithology='Shale',
                   runner_up='Shale', match_status='blank', lithology_fallback=True)
        with tempfile.TemporaryDirectory() as directory:
            rows = attach_depths([row], calibration, CropAlignment(), 200)
            write_final(rows, Image.new('RGB', (40, 200), 'white'), calibration,
                         CropAlignment(), directory, 'example', clusters=[row])
            path = Path(directory)
            loaded = read_labelled_csv(path / 'example.csv')
            self.assertEqual(loaded[0]['lithology'], '')
            self.assertEqual(loaded[0]['lithology_fallback'], 'False')
            data = json.loads((path / 'example.json').read_text())
            self.assertEqual(data['clusters'][0]['lithology'], '')
            self.assertEqual(data['intervals'][0]['top_depth'], 1000)
            self.assertEqual(data['matching_diagnostics']['runner_up_fallback_intervals'], 0)
            # Empty nonblank labels still fail validation.
            (path / 'bad.csv').write_text('top_y,bottom_y,cluster_id,lithology,match_status\n0,200,C001,,ambiguous\n')
            with self.assertRaises(ValueError):
                read_labelled_csv(path / 'bad.csv')

    def test_verified_label_is_not_replaced(self):
        result = apply_runner_up_fallback(dict(lithology='Sandstone', runner_up='Shale', match_status='matched'))
        self.assertEqual(result['lithology'], 'Sandstone')
        self.assertFalse(result['lithology_fallback'])

    def test_missing_or_unknown_runner_up_keeps_unknown(self):
        for runner in (None, '', '  ', 'unknown', ' UNKNOWN '):
            with self.subTest(runner=runner):
                result = apply_runner_up_fallback(dict(lithology='unknown', runner_up=runner))
                self.assertEqual(result['lithology'], 'unknown')
                self.assertFalse(result['lithology_fallback'])

    def test_csv_strings_whitespace_and_case_are_supported(self):
        result = apply_runner_up_fallback(dict(lithology=' Unknown ', runner_up=' Shale ',
                                                match_distance='0.2', match_margin='0.1', lithology_fallback='False'))
        self.assertEqual(result['lithology'], 'Shale')
        self.assertAlmostEqual(result['fallback_distance'], .3)
        self.assertTrue(result['lithology_fallback'])
        self.assertFalse(apply_runner_up_fallback(dict(lithology='Sandstone', lithology_fallback='False'))['lithology_fallback'])

    def test_fallback_is_idempotent_and_tolerates_missing_scores(self):
        first = apply_runner_up_fallback(dict(lithology='unknown', runner_up='Shale'))
        self.assertEqual(apply_runner_up_fallback(first), first)
        self.assertIsNone(first['fallback_distance'])
        result = apply_runner_up_fallback(dict(lithology='unknown', runner_up='Shale',
                                                match_distance='unavailable', match_margin='.1'))
        self.assertEqual(result['lithology'], 'Shale')
        self.assertIsNone(result['fallback_distance'])

    def test_both_matchers_apply_fallback_consistently_to_group_members(self):
        patch = Image.new('RGB', (80, 180), 'yellow')
        image = np.vstack([np.asarray(patch), np.asarray(patch)])
        legend = [LegendSwatch('Clay', patch), LegendSwatch('Shale', patch)]
        for matcher in ('descriptor', 'hybrid'):
            with self.subTest(matcher=matcher):
                result = label_intervals(image, [(0, 180), (180, 360)], legend, matcher=matcher)
                self.assertEqual(len(result.clusters), 1)
                self.assertTrue(all(r.lithology == r.runner_up and r.lithology_fallback for r in result.intervals))
                self.assertTrue(all(r.match_status == 'ambiguous' for r in result.intervals))
                self.assertEqual(result.clusters[0]['lithology'], result.intervals[0].lithology)
                self.assertTrue(result.clusters[0]['lithology_fallback'])
                self.assertEqual(result.diagnostics['runner_up_fallback_intervals'], 2)

    def test_cached_rows_and_cluster_summaries_are_updated_in_final_export(self):
        calibration = DepthCalibration(.5, 1000, 200,
                                        [DepthAnchor(1000, 0), DepthAnchor(1100, 200)], [], 0, 0)
        row = dict(top_y=0, bottom_y=200, cluster_id='C001', lithology='unknown', runner_up='Shale',
                   match_status='poor_match', match_distance='.4', match_margin='.2')
        output = attach_depths([row], calibration, CropAlignment(), 200)
        self.assertEqual(output[0]['lithology'], 'Shale')
        self.assertEqual(output[0]['top_depth'], 1000)
        self.assertEqual(output[0]['base_depth'], 1100)
        with tempfile.TemporaryDirectory() as directory:
            write_final(output, Image.new('RGB', (40, 200), 'white'), calibration,
                         CropAlignment(), directory, 'example', clusters=[dict(row)])
            data = json.loads((Path(directory) / 'example.json').read_text())
            self.assertEqual(data['intervals'][0]['lithology'], 'Shale')
            self.assertEqual(data['clusters'][0]['lithology'], 'Shale')
            self.assertEqual(data['intervals'][0]['match_status'], 'poor_match')
            with (Path(directory) / 'example.csv').open() as file:
                result = next(csv.DictReader(file))
            self.assertEqual(result['lithology_fallback'], 'True')
            self.assertEqual(result['lithology'], result['runner_up'])


if __name__ == '__main__':
    unittest.main()
