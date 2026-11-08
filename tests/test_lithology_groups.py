"""Final cluster IDs follow resolved names, not provisional texture groups."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'info extraction'))
from lithology_groups import consolidate_lithology_clusters
from interval_lithology import (LegendSwatch, apply_runner_up_fallback, label_intervals, write_results)
from final_output import CropAlignment, attach_depths, write_final
from depth_distance import DepthAnchor, DepthCalibration


class LithologyGroupTests(unittest.TestCase):
    def row(self, index, name, **kwargs):
        return dict(top_y=index * 10, bottom_y=(index + 1) * 10,
                    cluster_id=f'C{index + 100:03d}', lithology=name,
                    match_status='matched', lithology_fallback=False, **kwargs)

    def test_same_final_name_shares_id_across_visual_groups(self):
        rows = [self.row(0, 'Shale'), self.row(1, 'Sandstone'), self.row(2, 'Shale')]
        output, groups = consolidate_lithology_clusters(rows)
        self.assertEqual([r['cluster_id'] for r in output], ['C001', 'C002', 'C001'])
        self.assertEqual([r['visual_cluster_id'] for r in output], ['C100', 'C101', 'C102'])
        self.assertEqual(groups[0]['member_indices'], [0, 2])
        self.assertEqual(rows[0]['cluster_id'], 'C100')

    def test_case_and_whitespace_do_not_create_extra_ids(self):
        rows = [self.row(0, 'Clay Shale'), self.row(1, ' CLAY   SHALE '), self.row(2, 'clay shale')]
        output, groups = consolidate_lithology_clusters(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(output[1]['lithology'], ' CLAY   SHALE ')  # original label retained
        self.assertTrue(all(r['cluster_id'] == 'C001' for r in output))

    def test_fallback_and_verified_labels_merge_without_upgrading_evidence(self):
        a = self.row(0, 'Shale')
        b = dict(self.row(1, 'unknown'), runner_up='Shale', match_status='ambiguous',
                 match_distance=.1, match_margin=.02)
        output, groups = consolidate_lithology_clusters([a, apply_runner_up_fallback(b)])
        self.assertEqual(output[0]['cluster_id'], output[1]['cluster_id'])
        self.assertEqual(output[1]['match_status'], 'ambiguous')
        self.assertTrue(output[1]['lithology_fallback'])
        self.assertEqual(groups[0]['verified_intervals'], 1)
        self.assertEqual(groups[0]['fallback_intervals'], 1)
        self.assertEqual(groups[0]['match_status'], 'mixed')
        self.assertEqual(groups[0]['representative_index'], 0)

    def test_blank_and_unresolved_have_separate_unassigned_buckets(self):
        rows = [dict(self.row(0, ''), match_status='blank'), self.row(1, 'unknown'),
                dict(self.row(2, ''), match_status='blank'), self.row(3, 'UNKNOWN'), self.row(4, 'Shale')]
        output, groups = consolidate_lithology_clusters(rows)
        self.assertEqual([r['cluster_id'] for r in output], ['BLANK', 'UNKNOWN', 'BLANK', 'UNKNOWN', 'C001'])
        self.assertEqual(len(groups), 3)
        self.assertEqual(output[0]['lithology'], '')

    def test_hundreds_of_provisional_ids_collapse_to_names(self):
        rows = [self.row(i, ['Shale', 'Sandstone', 'Limestone'][i % 3]) for i in range(500)]
        output, groups = consolidate_lithology_clusters(rows)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len({r['cluster_id'] for r in output}), 3)
        self.assertEqual(len({r['visual_cluster_id'] for r in output}), 500)
        self.assertEqual(sum(g['interval_count'] for g in groups), 500)

    def test_normalization_is_idempotent(self):
        first, groups = consolidate_lithology_clusters([self.row(0, 'Shale'), self.row(1, 'Shale')])
        second, second_groups = consolidate_lithology_clusters(first, groups)
        self.assertEqual(first, second)
        self.assertEqual(second_groups[0]['visual_cluster_ids'], ['C100', 'C101'])

    def test_preserves_geometry_depths_and_scores(self):
        original = dict(self.row(0, 'Shale'), top_depth=1000, base_depth=1010, match_distance=.13,
                        match_margin=.2, confidence='review-required')
        output, _ = consolidate_lithology_clusters([original])
        for key, value in original.items():
            if key != 'cluster_id': self.assertEqual(output[0][key], value)

    def test_source_candidate_metadata_comes_from_chosen_representative(self):
        rows = [self.row(0, 'Shale'), self.row(1, 'Shale')]
        source = [dict(cluster_id='C100', member_indices=[0], candidates=[{'lithology': 'Shale', 'distance': .1}]),
                  dict(cluster_id='C101', member_indices=[1], candidates=[{'lithology': 'Clay', 'distance': .2}])]
        _, groups = consolidate_lithology_clusters(rows, source)
        self.assertEqual(groups[0]['candidates'][0]['lithology'], 'Shale')

    def test_blank_component_export_supports_reserved_cluster_id(self):
        blank = Image.new('RGB', (80, 100), 'white')
        result = label_intervals(blank, [(0, 100)], [])
        self.assertEqual(result.intervals[0].cluster_id, 'BLANK')
        self.assertTrue(result.intervals[0].visual_cluster_id.startswith('C'))
        self.assertEqual(len(result.visual_clusters), 1)
        with tempfile.TemporaryDirectory() as directory:
            write_results(result, blank, directory, 'blank')
            data = json.loads((Path(directory) / 'blank.json').read_text())
            self.assertEqual(data['clusters'][0]['cluster_id'], 'BLANK')
            self.assertEqual(len(data['visual_clusters']), 1)

    def test_cached_export_reassigns_ids_after_runner_up(self):
        rows = [dict(self.row(i, 'unknown'), runner_up='Shale', match_status='ambiguous') for i in range(3)]
        calibration = DepthCalibration(1, 1000, 100, [DepthAnchor(1000, 0), DepthAnchor(1100, 100)], [], 0, 0)
        output = attach_depths(rows, calibration, CropAlignment(), 100, depth_rounding=0)
        self.assertTrue(all(r['cluster_id'] == 'C001' for r in output))
        with tempfile.TemporaryDirectory() as directory:
            write_final(output, Image.new('RGB', (30, 100), 'white'), calibration,
                         CropAlignment(), directory, 'example', depth_rounding=0)
            with (Path(directory) / 'example.csv').open() as file: exported = list(csv.DictReader(file))
            self.assertEqual({r['cluster_id'] for r in exported}, {'C001'})
            data = json.loads((Path(directory) / 'example.json').read_text())
            self.assertEqual(len(data['clusters']), 1)
            self.assertEqual(data['matching_diagnostics']['cluster_id_policy'], 'final_lithology')


if __name__ == '__main__':
    unittest.main()
