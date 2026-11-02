"""Synthetic tests only; does not load project predictions."""
import copy
import unittest

from run import classify_readout, decompose, support_status, validate_and_group


def records():
    common = dict(video_name="test", subject_tid=1, object_tid=2, fps=10,
                  coexist_sec=20, pair_annotation_count=3, video_GT_load=3,
                  source_scope="GT27_correct_entity_input", readout="target_predicate_maxclass_query_min5",
                  pred_available=True, chosen_pair_covers_gt=True, duration_ratio=1)
    return [dict(common, gt_index=0, predicate="hold", pair_predicate_annotation_count=2,
                 begin_fid=0, end_fid=10, relation_sec=1, pred_begin=0, pred_end=10, tiou=1, oracle_tiou=1),
            dict(common, gt_index=1, predicate="hold", pair_predicate_annotation_count=2,
                 begin_fid=20, end_fid=70, relation_sec=5, pred_begin=80, pred_end=130,
                 tiou=0, oracle_tiou=.8, chosen_pair_covers_gt=False),
            dict(common, gt_index=2, predicate="watch", pair_predicate_annotation_count=1,
                 begin_fid=100, end_fid=105, relation_sec=.5, pred_begin=None, pred_end=None,
                 pred_available=False, duration_ratio=None, tiou=0, oracle_tiou=.3)]


class PartitionTests(unittest.TestCase):
    def test_states_exclusive_with_threshold_tie(self):
        self.assertEqual(classify_readout(.5, .5), "selected_hit")
        self.assertEqual(classify_readout(.49, .5), "oracle_only_hit")
        self.assertEqual(classify_readout(.49, .49), "neither")
        with self.assertRaises(ValueError):
            classify_readout(.5, .49)

    def test_full_data_labels_before_population_filter(self):
        rows = validate_and_group(records())
        self.assertEqual([r["readout_state"] for r in rows], ["selected_hit", "oracle_only_hit", "neither"])
        self.assertTrue(all(r["pair_mix_group"] == "mixed_short_long" for r in rows))
        self.assertEqual([r["repeat_group"] for r in rows], ["repeated", "repeated", "unique"])
        result = decompose(rows, "chosen_pair_covers_gt")
        mixed = next(r for r in result if r["view"] == "pair_mix")
        self.assertEqual(mixed["pair_mix_group"], "mixed_short_long")
        self.assertEqual(mixed["N"], 2)

    def test_each_view_partitions_and_rates_have_denominators(self):
        result = decompose(validate_and_group(records()), "all")
        for view in {r["view"] for r in result}:
            self.assertEqual(sum(r["N"] for r in result if r["view"] == view), 3)
        total = next(r for r in result if r["view"] == "overall")
        for state in ("selected_hit", "oracle_only_hit", "neither"):
            self.assertEqual(total[state+"_count"], 1)
            self.assertEqual(total[state+"_denominator"], 3)
            self.assertAlmostEqual(total[state+"_rate"], 1/3)

    def test_incomplete_video_or_wrong_counts_stop(self):
        with self.assertRaises(ValueError):
            validate_and_group(records()[:2])
        for column in ("pair_annotation_count", "pair_predicate_annotation_count"):
            rows = records()
            rows[0][column] += 1
            with self.assertRaises(ValueError):
                validate_and_group(rows)

    def test_saved_selected_endpoints_must_match_tiou(self):
        rows = records()
        rows[0]["tiou"] = .6
        with self.assertRaises(ValueError):
            validate_and_group(rows)

    def test_false_prediction_not_zero_length_oracle(self):
        rows = records()
        rows[2]["duration_ratio"] = 0
        with self.assertRaises(ValueError):
            validate_and_group(rows)

    def test_duplicate_gt_and_wrong_source_rejected(self):
        rows = records()
        with self.assertRaises(ValueError):
            validate_and_group(rows + [copy.deepcopy(rows[0])])
        rows[0]["readout"] = "oracle_selected_by_gt"
        with self.assertRaises(ValueError):
            validate_and_group(rows)

    def test_support_gate_is_separate_from_mechanism(self):
        result = dict(N_common_support=60, N_videos_side0=20, N_videos_side1=20, N_missing_outcome=0,
                      bootstrap=dict(ci_low=-.2, ci_high=-.1, repetitions_valid=2000,
                                     repetitions_requested=2000, retained_original_support_mass_p05=.8))
        self.assertEqual(support_status(result), "support_gate_passed_no_prespecified_continuous_effect_threshold")
        result["N_videos_side0"] = 19
        self.assertEqual(support_status(result), "insufficient_support")
        result["N_videos_side0"] = 20
        result["bootstrap"]["retained_original_support_mass_p05"] = .7
        self.assertEqual(support_status(result), "unstable_or_incomplete")


if __name__ == "__main__":
    unittest.main()
