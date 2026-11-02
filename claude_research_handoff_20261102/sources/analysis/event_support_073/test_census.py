"""Synthetic checks: prevent overlap/off-label errors and missing-as-zero joins."""
import unittest
from census import event_components, optional_bool, patch_stages, structural_history


def row(i, start, end):
    return dict(video_name="v", gt_index=i, subject_tid=1, object_tid=2,
                predicate="hold", begin_fid=start, end_fid=end, fps=10,
                primary_cohort=False)


class CensusTests(unittest.TestCase):
    def test_missing_is_not_false(self):
        self.assertIsNone(optional_bool(""))
        self.assertIsNone(optional_bool(None))
        self.assertIs(optional_bool("0"), False)
        self.assertIs(optional_bool("True"), True)
        with self.assertRaises(ValueError):
            optional_bool("UNKNOWN")

    def test_overlapping_and_touching_annotations_are_merged(self):
        out = event_components([row(0, 0, 10), row(1, 5, 15), row(2, 15, 20), row(3, 30, 40)])
        self.assertEqual(out, [dict(begin_fid=0, end_fid=20, gt_indices=[0, 1, 2]),
                               dict(begin_fid=30, end_fid=40, gt_indices=[3])])

    def test_donor_is_earlier_than_previous_event_not_target_or_future(self):
        result = structural_history([row(0, 0, 10), row(1, 30, 40), row(2, 60, 70)])
        self.assertEqual(result["N_transition_candidates"], 2)
        first, second = result["transitions"]
        self.assertEqual(first["earlier_length_matched_unlabeled_donor_candidates"], [])
        donor = second["earlier_length_matched_unlabeled_donor_candidates"]
        self.assertEqual(donor, [dict(begin_fid=10, end_fid=30, replacement_frames=10,
                                     latest_length_matched_begin_fid=20)])
        self.assertIsNone(second["joint_support_in_gap_and_donors"])

    def test_corrections_must_match_frozen_original(self):
        source = {("v", 1): dict(video_name="v", gt_index="1", cohort="048", stage="numeric_unresolved")}
        correction = {"changes": [dict(video_name="v", gt_index=1, cohort="048",
                                        old_stage="numeric_unresolved", new_stage="grid_infeasible")]}
        patched = patch_stages(source, correction)
        self.assertEqual(source["v", 1]["stage"], "numeric_unresolved")
        self.assertEqual(patched["v", 1]["stage_056"], "numeric_unresolved")
        self.assertEqual(patched["v", 1]["stage"], "grid_infeasible")
        with self.assertRaises(ValueError):
            patch_stages(patched, correction)


if __name__ == "__main__":
    unittest.main()
